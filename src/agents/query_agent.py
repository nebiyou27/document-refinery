"""Deterministic Phase 4 query agent with retrieval-backed provenance."""

from __future__ import annotations

from dataclasses import dataclass

from src.chunking.page_index import PageIndexTree
from src.chunking.page_index_query import PageIndexMatch, PageIndexQueryEngine
from src.chunking.provenance import ProvenanceChainBuilder, ProvenanceChainError
from src.chunking.vector_store import VectorStoreMatch
from src.agents.structured_fact_query import StructuredFactQueryBackend
from src.models import ProvenanceChain
from src.utils.hashing import canonicalize_text


@dataclass(frozen=True)
class QueryAgentResult:
    """Query response with explicit verification status."""

    query: str
    status: str
    answer: str | None
    provenance_chain: ProvenanceChain | None
    retrieval_matches: tuple[VectorStoreMatch, ...]
    page_index_matches: tuple[PageIndexMatch, ...]
    route: str
    failure_reason: str | None = None


class QueryAgent:
    """Combines PageIndex traversal, vector retrieval, and provenance assembly."""

    def __init__(
        self,
        *,
        page_index_backend: PageIndexQueryEngine,
        vector_backend,
        provenance_builder: ProvenanceChainBuilder | None = None,
        structured_query_backend: StructuredFactQueryBackend | None = None,
    ) -> None:
        self.page_index_backend = page_index_backend
        self.vector_backend = vector_backend
        self.provenance_builder = provenance_builder or ProvenanceChainBuilder()
        self.structured_query_backend = structured_query_backend

    def answer(
        self,
        *,
        tree: PageIndexTree,
        query: str,
        top_k: int = 3,
        section_top_k: int = 3,
        record_type: str = "chunk",
    ) -> QueryAgentResult:
        if self.structured_query_backend is not None:
            structured_result = self.structured_query_backend.answer(query)
            if structured_result is not None:
                return QueryAgentResult(
                    query=query,
                    status="verified",
                    answer=structured_result.answer,
                    provenance_chain=structured_result.provenance_chain,
                    retrieval_matches=(),
                    page_index_matches=(),
                    route="structured_query",
                    failure_reason=None,
                )

        page_index_matches = tuple(self.page_index_backend.query(tree, query, top_k=section_top_k))
        assisted_matches = tuple(
            self._assisted_retrieve(
                query=query,
                page_index_matches=page_index_matches,
                top_k=top_k,
                record_type=record_type,
            )
        )

        route = "pageindex_assisted"
        retrieval_matches = assisted_matches
        if not retrieval_matches:
            route = "baseline_vector"
            retrieval_matches = tuple(
                self.vector_backend.query(
                    query,
                    top_k=top_k,
                    record_type=record_type,
                )
            )

        if not retrieval_matches:
            return QueryAgentResult(
                query=query,
                status="unverifiable",
                answer=None,
                provenance_chain=None,
                retrieval_matches=(),
                page_index_matches=page_index_matches,
                route=route,
                failure_reason="No supporting evidence retrieved",
            )

        try:
            provenance_chain = self.provenance_builder.build(retrieval_matches, query=query)
        except ProvenanceChainError as exc:
            return QueryAgentResult(
                query=query,
                status="unverifiable",
                answer=None,
                provenance_chain=None,
                retrieval_matches=retrieval_matches,
                page_index_matches=page_index_matches,
                route=route,
                failure_reason=str(exc),
            )

        answer = self._synthesize_answer(retrieval_matches)
        if not answer:
            return QueryAgentResult(
                query=query,
                status="unverifiable",
                answer=None,
                provenance_chain=None,
                retrieval_matches=retrieval_matches,
                page_index_matches=page_index_matches,
                route=route,
                failure_reason="Retrieved evidence did not yield an answerable snippet",
            )

        return QueryAgentResult(
            query=query,
            status="verified",
            answer=answer,
            provenance_chain=provenance_chain,
            retrieval_matches=retrieval_matches,
            page_index_matches=page_index_matches,
            route=route,
            failure_reason=None,
        )

    def _assisted_retrieve(
        self,
        *,
        query: str,
        page_index_matches: tuple[PageIndexMatch, ...],
        top_k: int,
        record_type: str,
    ) -> list[VectorStoreMatch]:
        collected: list[VectorStoreMatch] = []
        seen_record_ids: set[str] = set()

        for section_match in page_index_matches:
            vector_matches = self.vector_backend.query(
                query,
                top_k=top_k,
                section_path=section_match.section_path,
                record_type=record_type,
            )
            for vector_match in vector_matches:
                if vector_match.record_id in seen_record_ids:
                    continue
                seen_record_ids.add(vector_match.record_id)
                collected.append(vector_match)
                if len(collected) >= top_k:
                    return collected
        return collected

    def _synthesize_answer(self, retrieval_matches: tuple[VectorStoreMatch, ...]) -> str:
        snippets: list[str] = []
        seen: set[str] = set()
        for match in retrieval_matches:
            snippet = canonicalize_text(match.text)
            if not snippet or snippet in seen:
                continue
            seen.add(snippet)
            snippets.append(snippet)
            if len(snippets) == 2:
                break
        return " ".join(snippets)
