"""Evaluation helpers for Phase 3 retrieval strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from src.chunking.page_index import PageIndexTree
from src.chunking.page_index_query import PageIndexMatch
from src.chunking.vector_store import VectorStoreMatch


class VectorRetrievalBackend(Protocol):
    """Minimal vector retrieval interface used by evaluation."""

    def query(
        self,
        topic: str,
        *,
        top_k: int = 3,
        section_path: tuple[str, ...] | None = None,
        record_type: str | None = None,
    ) -> list[VectorStoreMatch]:
        """Return vector matches for a topic."""


class PageIndexTraversalBackend(Protocol):
    """Minimal PageIndex traversal interface used by evaluation."""

    def query(self, tree: PageIndexTree, topic: str, top_k: int = 3) -> list[PageIndexMatch]:
        """Return ranked PageIndex matches for a topic."""


@dataclass(frozen=True)
class LabeledRetrievalQuery:
    """Single labeled query used for retrieval evaluation."""

    query_id: str
    topic: str
    relevant_record_ids: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalRunResult:
    """Per-query retrieval outcome."""

    query_id: str
    topic: str
    retrieved_record_ids: tuple[str, ...]
    relevant_record_ids: tuple[str, ...]
    precision_at_3: float
    hit: bool


@dataclass(frozen=True)
class RetrievalMetrics:
    """Aggregated retrieval metrics."""

    query_count: int
    precision_at_3: float
    hit_rate: float


@dataclass(frozen=True)
class RetrievalEvaluationReport:
    """Combined evaluation results."""

    per_query: tuple[RetrievalRunResult, ...]
    metrics: RetrievalMetrics


class RetrievalEvaluator:
    """Compares baseline and PageIndex-assisted retrieval deterministically."""

    def __init__(
        self,
        vector_backend: VectorRetrievalBackend,
        page_index_backend: PageIndexTraversalBackend,
    ) -> None:
        self.vector_backend = vector_backend
        self.page_index_backend = page_index_backend

    def evaluate_baseline(
        self,
        queries: list[LabeledRetrievalQuery],
        *,
        top_k: int = 3,
        record_type: str | None = None,
    ) -> RetrievalEvaluationReport:
        per_query = tuple(
            self._evaluate_baseline_query(query=query, top_k=top_k, record_type=record_type)
            for query in queries
        )
        return RetrievalEvaluationReport(
            per_query=per_query,
            metrics=self._aggregate_metrics(per_query),
        )

    def evaluate_pageindex_assisted(
        self,
        tree: PageIndexTree,
        queries: list[LabeledRetrievalQuery],
        *,
        section_top_k: int = 3,
        top_k: int = 3,
        record_type: str | None = None,
    ) -> RetrievalEvaluationReport:
        per_query = tuple(
            self._evaluate_assisted_query(
                tree=tree,
                query=query,
                section_top_k=section_top_k,
                top_k=top_k,
                record_type=record_type,
            )
            for query in queries
        )
        return RetrievalEvaluationReport(
            per_query=per_query,
            metrics=self._aggregate_metrics(per_query),
        )

    def _evaluate_baseline_query(
        self,
        *,
        query: LabeledRetrievalQuery,
        top_k: int,
        record_type: str | None,
    ) -> RetrievalRunResult:
        matches = self.vector_backend.query(
            query.topic,
            top_k=top_k,
            record_type=record_type,
        )
        return self._build_run_result(query=query, matches=matches, top_k=top_k)

    def _evaluate_assisted_query(
        self,
        *,
        tree: PageIndexTree,
        query: LabeledRetrievalQuery,
        section_top_k: int,
        top_k: int,
        record_type: str | None,
    ) -> RetrievalRunResult:
        section_matches = self.page_index_backend.query(tree, query.topic, top_k=section_top_k)
        collected: list[VectorStoreMatch] = []
        seen_record_ids: set[str] = set()

        for section_match in section_matches:
            vector_matches = self.vector_backend.query(
                query.topic,
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
                    break
            if len(collected) >= top_k:
                break

        return self._build_run_result(query=query, matches=collected, top_k=top_k)

    def _build_run_result(
        self,
        *,
        query: LabeledRetrievalQuery,
        matches: list[VectorStoreMatch],
        top_k: int,
    ) -> RetrievalRunResult:
        retrieved_record_ids = tuple(match.record_id for match in matches[:top_k])
        relevant = set(query.relevant_record_ids)
        hits = sum(1 for record_id in retrieved_record_ids if record_id in relevant)
        precision_at_3 = hits / top_k if top_k > 0 else 0.0
        hit = any(record_id in relevant for record_id in retrieved_record_ids)
        return RetrievalRunResult(
            query_id=query.query_id,
            topic=query.topic,
            retrieved_record_ids=retrieved_record_ids,
            relevant_record_ids=query.relevant_record_ids,
            precision_at_3=precision_at_3,
            hit=hit,
        )

    def _aggregate_metrics(self, per_query: tuple[RetrievalRunResult, ...]) -> RetrievalMetrics:
        query_count = len(per_query)
        if query_count == 0:
            return RetrievalMetrics(query_count=0, precision_at_3=0.0, hit_rate=0.0)
        precision = sum(result.precision_at_3 for result in per_query) / query_count
        hit_rate = sum(1.0 for result in per_query if result.hit) / query_count
        return RetrievalMetrics(
            query_count=query_count,
            precision_at_3=precision,
            hit_rate=hit_rate,
        )
