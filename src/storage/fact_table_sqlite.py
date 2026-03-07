"""SQLite export for normalized fact-table entries."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
import re

from src.models import FactTable
from src.utils.hashing import canonicalize_text


class FactTableSqliteWriter:
    """Persist normalized fact-table entries into a query-friendly SQLite schema."""

    def write(self, *, fact_table: FactTable, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(db_path)
        try:
            self._create_schema(connection)
            self._replace_document_rows(connection, fact_table=fact_table)
            connection.commit()
        finally:
            connection.close()

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                document_name TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS fact_values (
                fact_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                page_number INTEGER NOT NULL,
                section_path TEXT NOT NULL,
                fact_type TEXT NOT NULL,
                subject TEXT NOT NULL,
                predicate TEXT NOT NULL,
                value_text TEXT NOT NULL,
                value_number REAL,
                unit TEXT,
                period_label TEXT,
                notes_ref TEXT,
                normalized_subject TEXT NOT NULL,
                normalized_predicate TEXT NOT NULL,
                provenance_json TEXT NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            );

            CREATE INDEX IF NOT EXISTS idx_fact_lookup
            ON fact_values(normalized_subject, normalized_predicate, period_label);

            CREATE INDEX IF NOT EXISTS idx_fact_number
            ON fact_values(value_number);
            """
        )

    def _replace_document_rows(self, connection: sqlite3.Connection, *, fact_table: FactTable) -> None:
        if not fact_table.entries:
            return
        document_name = fact_table.entries[0].document_name
        connection.execute(
            """
            INSERT INTO documents (doc_id, document_name)
            VALUES (?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET document_name = excluded.document_name
            """,
            (fact_table.doc_id, document_name),
        )
        connection.execute("DELETE FROM fact_values WHERE doc_id = ?", (fact_table.doc_id,))
        deduped_rows: dict[str, tuple[object, ...]] = {}
        for entry in fact_table.entries:
            deduped_rows[entry.fact_id] = (
                entry.fact_id,
                entry.doc_id,
                entry.page_number,
                json.dumps(list(entry.section_path)),
                "table_numeric",
                entry.row_label,
                "value",
                entry.raw_value,
                entry.numeric_value,
                entry.unit,
                entry.column_label,
                self._extract_notes_ref(entry.row_label),
                self._normalize_label(entry.row_label),
                "value",
                entry.provenance.model_dump_json(),
            )
        connection.executemany(
            """
            INSERT INTO fact_values (
                fact_id,
                doc_id,
                page_number,
                section_path,
                fact_type,
                subject,
                predicate,
                value_text,
                value_number,
                unit,
                period_label,
                notes_ref,
                normalized_subject,
                normalized_predicate,
                provenance_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            list(deduped_rows.values()),
        )

    def _normalize_label(self, value: str) -> str:
        normalized = canonicalize_text(value).lower()
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        return canonicalize_text(normalized)

    def _extract_notes_ref(self, row_label: str) -> str | None:
        match = re.search(r"\bnote(?:s)?\s+(\d+[a-z]?)\b", row_label, flags=re.IGNORECASE)
        return match.group(1) if match else None
