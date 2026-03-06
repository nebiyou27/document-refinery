"""Summarization layer for PageIndex trees."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from src.chunking.page_index import PageIndexTree
from src.models.chunking import LDU, PageIndexNode
from src.utils.hashing import canonicalize_text


@dataclass(frozen=True)
class SummaryInput:
    """Normalized summarization payload for a single PageIndex node."""

    node_id: str
    title: str
    section_path: tuple[str, ...]
    source_text: str


class SummaryBackend(Protocol):
    """Backend interface for generating short factual summaries."""

    def summarize(self, summary_input: SummaryInput) -> str:
        """Return a short factual summary for the provided node content."""


class SummaryBackendError(RuntimeError):
    """Raised when a summary backend fails to generate a summary."""


class OllamaClientProtocol(Protocol):
    """Minimal Ollama client surface needed by the summary backend."""

    def chat(self, **kwargs: Any) -> dict[str, Any]:
        """Submit a chat request and return the raw Ollama response."""


class OllamaSummaryBackend:
    """Ollama-backed summary backend with deterministic prompting."""

    def __init__(
        self,
        client: OllamaClientProtocol | None = None,
        *,
        model: str = "qwen3:1.7b",
        keep_alive: str | int = "0s",
    ) -> None:
        self.client = client or self._default_client()
        self.model = model
        self.keep_alive = keep_alive

    def summarize(self, summary_input: SummaryInput) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You write short factual retrieval summaries for document sections. "
                    "Use one sentence, no bullets, no speculation, no citations, and keep it under 30 words."
                ),
            },
            {
                "role": "user",
                "content": self._build_user_prompt(summary_input),
            },
        ]

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={"temperature": 0},
                keep_alive=self.keep_alive,
            )
        except Exception as exc:  # pragma: no cover - exercised via tests with fake client
            raise SummaryBackendError(f"Ollama summary request failed: {exc}") from exc

        content = self._extract_content(response)
        if not content:
            raise SummaryBackendError("Ollama summary response did not include message content")
        return canonicalize_text(content)

    def _build_user_prompt(self, summary_input: SummaryInput) -> str:
        section = " > ".join(summary_input.section_path) if summary_input.section_path else summary_input.title
        return (
            "Summarize this section for retrieval.\n"
            f"Section: {section}\n"
            "Requirements: factual only; focus on entities, metrics, actions, or outcomes when present; "
            "do not speculate; keep exactly one short sentence.\n"
            f"Source:\n{summary_input.source_text}"
        )

    def _extract_content(self, response: dict[str, Any]) -> str:
        message = response.get("message")
        if not isinstance(message, dict):
            return ""
        content = message.get("content")
        if not isinstance(content, str):
            return ""
        return content.strip()

    def _default_client(self) -> OllamaClientProtocol:
        try:
            from ollama import Client
        except ImportError as exc:  # pragma: no cover - depends on optional install
            raise SummaryBackendError("Ollama client is not installed") from exc
        return Client()


class PageIndexSummarizer:
    """Applies bottom-up summaries to an existing PageIndex tree."""

    def __init__(self, backend: SummaryBackend) -> None:
        self.backend = backend

    def summarize_tree(self, tree: PageIndexTree, ldus: list[LDU]) -> PageIndexTree:
        summarized_tree = tree.model_copy(deep=True)
        ldu_by_id = {ldu.ldu_id: ldu for ldu in ldus if ldu.ldu_id is not None}
        nodes_by_id = {node.node_id: node for node in summarized_tree.nodes}
        ordered_nodes = sorted(
            summarized_tree.nodes,
            key=lambda node: (-node.depth, node.order_index, node.node_id),
        )

        for node in ordered_nodes:
            if node.node_id == summarized_tree.root_id:
                continue

            source_text = self._source_text_for_node(node=node, nodes_by_id=nodes_by_id, ldu_by_id=ldu_by_id)
            if not source_text:
                continue

            try:
                summary = self.backend.summarize(
                    SummaryInput(
                        node_id=node.node_id,
                        title=node.title,
                        section_path=node.section_path,
                        source_text=source_text,
                    )
                )
            except SummaryBackendError:
                continue
            node.summary = summary

        return summarized_tree

    def _source_text_for_node(
        self,
        node: PageIndexNode,
        nodes_by_id: dict[str, PageIndexNode],
        ldu_by_id: dict[str, LDU],
    ) -> str:
        direct_texts = [ldu_by_id[ldu_id].text for ldu_id in node.ldu_ids if ldu_id in ldu_by_id]
        if direct_texts:
            return "\n\n".join(direct_texts)

        child_summaries = [
            nodes_by_id[child_id].summary
            for child_id in node.child_ids
            if child_id in nodes_by_id and nodes_by_id[child_id].summary
        ]
        return "\n\n".join(child_summaries)
