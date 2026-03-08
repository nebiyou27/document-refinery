"""Deterministic PageIndex tree builder for Stage 3/4."""

from __future__ import annotations

import hashlib
import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from src.models.chunking import LDU, LDUKind, PageIndexNode

if TYPE_CHECKING:
    from src.chunking.page_index_query import PageIndexMatch


class PageIndexTree(BaseModel):
    """Materialized page-index tree for traversal and later summarization."""

    doc_id: str
    root_id: str
    nodes: list[PageIndexNode] = Field(default_factory=list)

    def traverse(self, topic: str, top_k: int = 3) -> list[PageIndexMatch]:
        from src.chunking.page_index_query import PageIndexQueryEngine

        return PageIndexQueryEngine().query(tree=self, topic=topic, top_k=top_k)


class PageIndexBuilder:
    """Builds a deterministic section tree from LDU section paths."""

    _ENTITY_STOPWORDS = {
        "ROOT",
        "Section",
        "Summary",
        "Introduction",
        "Overview",
        "Results",
        "Table",
        "Figure",
    }

    def build(self, doc_id: str, ldus: list[LDU]) -> PageIndexTree:
        ordered_ldus = sorted(ldus, key=lambda ldu: (ldu.page_number, ldu.source_block_order, ldu.ldu_id or ""))
        if not ordered_ldus:
            raise ValueError("PageIndexBuilder requires at least one LDU")

        root = PageIndexNode(
            node_id=self._node_id(doc_id=doc_id, section_path=()),
            title="ROOT",
            section_path=(),
            parent_id=None,
            depth=0,
            start_page=ordered_ldus[0].page_number,
            end_page=ordered_ldus[0].page_number,
            order_index=0,
        )
        nodes_by_path: dict[tuple[str, ...], PageIndexNode] = {(): root}
        creation_order: list[tuple[str, ...]] = [()]

        for ldu in ordered_ldus:
            target_path = ldu.section_path
            parent_path: tuple[str, ...] = ()

            for depth in range(1, len(target_path) + 1):
                current_path = target_path[:depth]
                if current_path not in nodes_by_path:
                    parent_node = nodes_by_path[parent_path]
                    node = PageIndexNode(
                        node_id=self._node_id(doc_id=doc_id, section_path=current_path),
                        title=current_path[-1],
                        section_path=current_path,
                        parent_id=parent_node.node_id,
                        depth=depth,
                        start_page=ldu.page_number,
                        end_page=ldu.page_number,
                        order_index=len(creation_order),
                    )
                    nodes_by_path[current_path] = node
                    creation_order.append(current_path)
                    if node.node_id not in parent_node.child_ids:
                        parent_node.child_ids.append(node.node_id)
                    if node.title not in parent_node.child_sections:
                        parent_node.child_sections.append(node.title)
                parent_path = current_path

            leaf_path = target_path
            leaf_node = nodes_by_path[leaf_path]
            if ldu.ldu_id is None:
                raise ValueError("All LDUs must have ldu_id before page-index construction")
            entities = self._extract_key_entities(ldu.text, fallback_title=leaf_node.title)
            data_types = self._detect_data_types(ldu)
            leaf_node.ldu_ids.append(ldu.ldu_id)
            self._extend_node_range(leaf_node, page_number=ldu.page_number, bbox=ldu.bbox)
            self._extend_node_annotations(leaf_node, entities=entities, data_types=data_types)

            if leaf_path:
                ancestor_path = leaf_path[:-1]
                while True:
                    ancestor_node = nodes_by_path[ancestor_path]
                    self._extend_node_range(ancestor_node, page_number=ldu.page_number, bbox=ldu.bbox)
                    self._extend_node_annotations(ancestor_node, entities=entities, data_types=data_types)
                    if not ancestor_path:
                        break
                    ancestor_path = ancestor_path[:-1]

        nodes = [nodes_by_path[path] for path in creation_order]
        return PageIndexTree(doc_id=doc_id, root_id=root.node_id, nodes=nodes)

    def _node_id(self, doc_id: str, section_path: tuple[str, ...]) -> str:
        payload = "|".join([doc_id, *section_path])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _extend_node_range(
        self,
        node: PageIndexNode,
        page_number: int,
        bbox: tuple[float, float, float, float],
    ) -> None:
        node.start_page = min(node.start_page, page_number)
        node.end_page = max(node.end_page, page_number)
        node.bbox = self._merge_bbox(node.bbox, bbox)

    def _merge_bbox(
        self,
        current_bbox: tuple[float, float, float, float] | None,
        next_bbox: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        if current_bbox is None:
            return next_bbox
        return (
            min(current_bbox[0], next_bbox[0]),
            min(current_bbox[1], next_bbox[1]),
            max(current_bbox[2], next_bbox[2]),
            max(current_bbox[3], next_bbox[3]),
        )

    def _extend_node_annotations(self, node: PageIndexNode, *, entities: list[str], data_types: list[str]) -> None:
        for candidate in entities:
            if candidate not in node.key_entities:
                node.key_entities.append(candidate)
        for data_type in data_types:
            if data_type not in node.data_types_present:
                node.data_types_present.append(data_type)

    def _extract_key_entities(self, text: str, *, fallback_title: str) -> list[str]:
        candidates: list[str] = []
        title_candidate = re.sub(r"^\d+(?:\.\d+)*\s*", "", fallback_title).strip()
        if title_candidate and title_candidate not in self._ENTITY_STOPWORDS:
            candidates.append(title_candidate)

        for pattern in (
            r"\b[A-Z]{2,}(?:[/-][A-Z0-9]{2,})*\b",
            r"\b[A-Z][a-z]+(?: [A-Z][a-z]+){0,2}\b",
            r"\bEFY\d{4}\b",
            r"\b\d+(?:\.\d+)?\s*%",
            r"\b\d+(?:\.\d+)?\s*(?:percent|million|billion|birr|usd)\b",
        ):
            for match in re.finditer(pattern, text):
                candidate = match.group(0).strip()
                if candidate and candidate not in self._ENTITY_STOPWORDS and candidate not in candidates:
                    candidates.append(candidate)
                if len(candidates) == 8:
                    return candidates
        return candidates

    def _detect_data_types(self, ldu: LDU) -> list[str]:
        detected = [ldu.kind.value]
        if ldu.kind == LDUKind.text and self._looks_like_equation(ldu.text):
            detected.append("equation")
        return detected

    def _looks_like_equation(self, text: str) -> bool:
        return bool(
            re.search(
                r"\b[A-Za-z][A-Za-z0-9_]*\s*=\s*[-+*/()A-Za-z0-9.%]+|\b\d+(?:\.\d+)?\s*[%+\-/*]\s*\d+(?:\.\d+)?\b",
                text,
            )
        )
