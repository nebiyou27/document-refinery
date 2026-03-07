"""Deterministic SQLite-backed fact query support for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import re
from typing import Literal

from src.models import ExtractionStrategy, ProvenanceChain, ProvenanceChainEntry, ProvenanceRef
from src.storage import canonicalize_fact_subject
from src.utils.hashing import canonicalize_text

_QUERY_PREFIX_RE = re.compile(
    r"^(?:what|which|show|give)\s+(?:were|was|are|is|do|did|me\s+)?\s*",
    flags=re.IGNORECASE,
)
_TRAILING_FILLER_RE = re.compile(
    r"\b(?:for|in|on|during)\s+\d{4}\b$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class StructuredFactQueryResult:
    """Structured fact answer assembled from SQLite rows."""

    answer: str
    provenance_chain: ProvenanceChain


class StructuredFactQueryBackend:
    """Answers numeric fact questions deterministically from SQLite fact rows."""

    def __init__(self, *, db_path: Path | None = None) -> None:
        self.db_path = db_path

    def configure(self, *, db_path: Path | None) -> None:
        self.db_path = db_path

    def answer(self, query: str) -> StructuredFactQueryResult | None:
        if self.db_path is None or not self.db_path.exists():
            return None
        metric_result = self._answer_metric_query(query)
        if metric_result is not None:
            return metric_result
        canonical_subject = self._canonical_subject_from_query(query)
        if not canonical_subject:
            return None
        rows = self._lookup_rows(canonical_subject)
        if not rows:
            return None
        entries = tuple(self._entry_from_row(row) for row in rows)
        answer = self._answer_from_rows(rows)
        return StructuredFactQueryResult(
            answer=answer,
            provenance_chain=ProvenanceChain(entries=entries, query=query),
        )

    def _answer_metric_query(self, query: str) -> StructuredFactQueryResult | None:
        parsed = self._parse_metric_query(query)
        if parsed is None:
            return None

        subject, period_hint, metrics = parsed
        rows = self._lookup_rows(subject)
        if not rows:
            return None

        metric_rows: list[sqlite3.Row] = []
        metric_value_by_name: dict[str, str] = {}
        normalized_period_hint = self._normalize_text(period_hint) if period_hint else ""

        for metric in metrics:
            matched = self._best_metric_row(rows=rows, metric=metric, period_hint=normalized_period_hint)
            if matched is None:
                continue
            metric_rows.append(matched)
            metric_value_by_name[metric] = str(matched["value_text"])

        if not metric_rows:
            return None

        subject_text = canonicalize_text(str(metric_rows[0]["subject"]))
        fragments: list[str] = []
        if "weight" in metric_value_by_name:
            fragments.append(f"CPI weight {metric_value_by_name['weight']}%")
        if "yoy_inflation" in metric_value_by_name:
            fragments.append(f"year-on-year inflation {metric_value_by_name['yoy_inflation']}%")
        if "mom_inflation" in metric_value_by_name:
            fragments.append(f"month-to-month inflation {metric_value_by_name['mom_inflation']}%")
        if not fragments:
            return None

        answer = f"{subject_text}: " + " and ".join(fragments)
        if period_hint:
            answer += f" ({canonicalize_text(period_hint)})."
        else:
            answer += "."

        deduped_rows = list(dict.fromkeys(str(row["fact_id"]) for row in metric_rows))
        row_by_fact_id = {str(row["fact_id"]): row for row in metric_rows}
        entries = tuple(self._entry_from_row(row_by_fact_id[fact_id]) for fact_id in deduped_rows)
        return StructuredFactQueryResult(
            answer=answer,
            provenance_chain=ProvenanceChain(entries=entries, query=query),
        )

    def _parse_metric_query(self, query: str) -> tuple[str, str | None, tuple[Literal["weight", "yoy_inflation", "mom_inflation"], ...]] | None:
        normalized = canonicalize_text(query).strip()
        lowered = normalized.lower()
        metrics: list[Literal["weight", "yoy_inflation", "mom_inflation"]] = []

        if "weight" in lowered and "cpi" in lowered:
            metrics.append("weight")
        elif "weight" in lowered:
            metrics.append("weight")

        if "year-on-year" in lowered or "year on year" in lowered or "yoy" in lowered:
            metrics.append("yoy_inflation")
        if "month-to-month" in lowered or "month to month" in lowered or "mom" in lowered:
            metrics.append("mom_inflation")

        if not metrics:
            return None

        subject_match = re.search(
            r"\b(?:weight|inflation(?:\s+rate)?)\s+of\s+(.+?)(?:\s+and\s+what|\s+and\s+its|\s+in\s+|[?.]|$)",
            lowered,
            flags=re.IGNORECASE,
        )
        if subject_match is None:
            return None
        subject = canonicalize_text(subject_match.group(1))
        if not subject:
            return None

        period_hint_match = re.search(r"\b(?:in|during|for)\s+([a-z]+\s+efy\d{4}|\w+\s+\d{4})\b", lowered, flags=re.IGNORECASE)
        period_hint = canonicalize_text(period_hint_match.group(1)) if period_hint_match else None

        ordered_metrics = tuple(dict.fromkeys(metrics))
        return (canonicalize_fact_subject(subject), period_hint, ordered_metrics)

    def _best_metric_row(
        self,
        *,
        rows: list[sqlite3.Row],
        metric: Literal["weight", "yoy_inflation", "mom_inflation"],
        period_hint: str,
    ) -> sqlite3.Row | None:
        scored: list[tuple[int, sqlite3.Row]] = []
        for row in rows:
            period_label = self._normalize_text(str(row["period_label"]))
            score = 0
            if metric == "weight":
                if "weight" in period_label or "wight" in period_label:
                    score += 5
                if "cpi" in period_label:
                    score += 2
            elif metric == "yoy_inflation":
                if "year on year" in period_label:
                    score += 5
                if "inflation" in period_label:
                    score += 2
                if period_hint and period_hint in period_label:
                    score += 2
            elif metric == "mom_inflation":
                if "month to month" in period_label:
                    score += 5
                if "inflation" in period_label:
                    score += 2
                if period_hint and period_hint in period_label:
                    score += 2

            if score > 0:
                scored.append((score, row))

        if not scored:
            return None
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1]

    def _normalize_text(self, value: str) -> str:
        normalized = canonicalize_text(value).lower()
        normalized = normalized.replace("-", " ")
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        return canonicalize_text(normalized)

    def _canonical_subject_from_query(self, query: str) -> str:
        normalized = canonicalize_text(query).strip(" ?.")
        normalized = _QUERY_PREFIX_RE.sub("", normalized)
        normalized = _TRAILING_FILLER_RE.sub("", normalized).strip()
        return canonicalize_fact_subject(normalized)

    def _lookup_rows(self, canonical_subject: str) -> list[sqlite3.Row]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            rows = connection.execute(
                """
                SELECT
                    fact_id,
                    subject,
                    canonical_subject,
                    period_label,
                    value_text,
                    value_number,
                    section_path,
                    provenance_json
                FROM fact_values
                WHERE canonical_subject = ?
                  AND normalized_predicate = 'value'
                  AND period_label <> 'Notes'
                ORDER BY period_label DESC, page_number ASC, fact_id ASC
                """,
                (canonical_subject,),
            ).fetchall()
            return list(rows)
        finally:
            connection.close()

    def _answer_from_rows(self, rows: list[sqlite3.Row]) -> str:
        subject = canonicalize_text(str(rows[0]["canonical_subject"]))
        if len(rows) == 1:
            return f"{subject} was {rows[0]['value_text']} ({rows[0]['period_label']})."
        parts = [f"{row['value_text']} ({row['period_label']})" for row in rows]
        return f"{subject} was " + "; ".join(parts) + "."

    def _entry_from_row(self, row: sqlite3.Row) -> ProvenanceChainEntry:
        provenance_payload = json.loads(str(row["provenance_json"]))
        strategy_used = ExtractionStrategy(provenance_payload["strategy_used"])
        section_path_raw = json.loads(str(row["section_path"]))
        snippet = f"{row['subject']} | {row['period_label']} | {row['value_text']}"
        return ProvenanceChainEntry(
            record_id=str(row["fact_id"]),
            record_type="fact",
            section_path=tuple(str(part) for part in section_path_raw),
            snippet=snippet,
            distance=0.0,
            provenance=ProvenanceRef(
                document_name=str(provenance_payload["document_name"]),
                doc_id=str(provenance_payload["doc_id"]),
                page_number=int(provenance_payload["page_number"]),
                bbox=tuple(float(value) for value in provenance_payload["bbox"]),
                content_hash=str(provenance_payload["content_hash"]),
                strategy_used=strategy_used,
                confidence_score=float(provenance_payload["confidence_score"]),
            ),
        )
