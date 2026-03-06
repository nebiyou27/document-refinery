"""Deterministic query traversal for PageIndex trees."""

from __future__ import annotations

from dataclasses import dataclass
import re

from src.chunking.page_index import PageIndexTree
from src.models.chunking import PageIndexNode


@dataclass(frozen=True)
class PageIndexMatch:
    """Ranked PageIndex match returned for a topic query."""

    node_id: str
    title: str
    section_path: tuple[str, ...]
    score: int
    start_page: int
    end_page: int
    summary: str | None


class PageIndexQueryEngine:
    """Traverses and ranks non-root PageIndex nodes for a topic query."""

    def query(self, tree: PageIndexTree, topic: str, top_k: int = 3) -> list[PageIndexMatch]:
        query_tokens = self._tokenize(topic)
        if not query_tokens:
            return []

        nodes_by_id = {node.node_id: node for node in tree.nodes}
        root = nodes_by_id[tree.root_id]
        matches: list[PageIndexMatch] = []

        for child_id in root.child_ids:
            self._traverse(
                node_id=child_id,
                nodes_by_id=nodes_by_id,
                query_tokens=query_tokens,
                ancestor_bonus=0,
                matches=matches,
            )

        ranked = sorted(
            matches,
            key=lambda match: (-match.score, -len(match.section_path), self._order_index(match.node_id, nodes_by_id), match.node_id),
        )
        return ranked[:top_k]

    def _traverse(
        self,
        node_id: str,
        nodes_by_id: dict[str, PageIndexNode],
        query_tokens: set[str],
        ancestor_bonus: int,
        matches: list[PageIndexMatch],
    ) -> None:
        node = nodes_by_id[node_id]
        title_overlap = self._overlap_score(query_tokens, self._tokenize(node.title))
        summary_overlap = self._overlap_score(query_tokens, self._tokenize(node.summary or ""))
        path_overlap = self._overlap_score(query_tokens, self._tokenize(" ".join(node.section_path)))

        direct_score = (title_overlap * 4) + (summary_overlap * 3) + path_overlap
        score = direct_score + ancestor_bonus

        matches.append(
            PageIndexMatch(
                node_id=node.node_id,
                title=node.title,
                section_path=node.section_path,
                score=score,
                start_page=node.start_page,
                end_page=node.end_page,
                summary=node.summary,
            )
        )

        next_bonus = min(ancestor_bonus + title_overlap + path_overlap, 4)
        for child_id in node.child_ids:
            self._traverse(
                node_id=child_id,
                nodes_by_id=nodes_by_id,
                query_tokens=query_tokens,
                ancestor_bonus=next_bonus,
                matches=matches,
            )

    def _tokenize(self, text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def _overlap_score(self, query_tokens: set[str], node_tokens: set[str]) -> int:
        return len(query_tokens & node_tokens)

    def _order_index(self, node_id: str, nodes_by_id: dict[str, PageIndexNode]) -> int:
        return nodes_by_id[node_id].order_index
