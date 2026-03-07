"""Deterministic SQLite-backed fact query support for Phase 4."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
import re

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
