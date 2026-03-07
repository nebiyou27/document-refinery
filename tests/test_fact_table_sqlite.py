from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.fact_table_extractor import FactTableExtractor
from src.models.extracted_document import ExtractedDocument, ExtractedPage, ExtractionMetadata, TableBlock
from src.storage import FactTableSqliteWriter


def _metadata() -> ExtractionMetadata:
    return ExtractionMetadata(
        strategy_used="strategy_b",
        confidence_score=0.96,
        processing_time_sec=0.01,
        cost_estimate_usd=0.0,
        escalation_triggered=False,
    )


def _document_with_table(rows: list[list[str]]) -> ExtractedDocument:
    return ExtractedDocument(
        doc_id="doc123",
        file_name="finance.pdf",
        file_path="data/finance.pdf",
        page_count=1,
        metadata=_metadata(),
        pages=[
            ExtractedPage(
                doc_id="doc123",
                page_number=1,
                metadata=_metadata(),
                signals={"char_count": 0, "char_density": 0.0, "image_area_ratio": 0.0, "table_count": 1},
                table_blocks=[
                    TableBlock(
                        doc_id="doc123",
                        page_number=1,
                        bbox=(0.0, 50.0, 200.0, 160.0),
                        content_hash="table-hash-1",
                        table_index=0,
                        rows=rows,
                    )
                ],
                page_content_hash="page-hash-1",
            )
        ],
    )


def test_fact_table_sqlite_writer_persists_queryable_rows(tmp_path: Path) -> None:
    document = _document_with_table(
        [
            ["Category", "30June2022", "30June2021"],
            ["Cash and cash equivalents", "28,191,157", "15,194,080"],
            ["Margin", "12%", "14%"],
        ]
    )
    fact_table = FactTableExtractor().extract(document)
    db_path = tmp_path / "fact_table.sqlite"

    FactTableSqliteWriter().write(fact_table=fact_table, db_path=db_path)

    connection = sqlite3.connect(db_path)
    try:
        document_row = connection.execute(
            "SELECT doc_id, document_name FROM documents WHERE doc_id = ?",
            ("doc123",),
        ).fetchone()
        assert document_row == ("doc123", "finance.pdf")

        cash_row = connection.execute(
            """
            SELECT subject, predicate, period_label, value_text, value_number, normalized_subject, provenance_json
            FROM fact_values
            WHERE normalized_subject = ? AND normalized_predicate = ?
            ORDER BY period_label
            """,
            ("cash and cash equivalents", "value"),
        ).fetchone()
        assert cash_row is not None
        assert cash_row[0] == "Cash and cash equivalents"
        assert cash_row[1] == "value"
        assert cash_row[2] == "30June2021"
        assert cash_row[3] == "15,194,080"
        assert cash_row[4] == 15194080.0
        assert cash_row[5] == "cash and cash equivalents"

        provenance = json.loads(cash_row[6])
        assert provenance["document_name"] == "finance.pdf"
        assert provenance["page_number"] == 1
    finally:
        connection.close()


def test_fact_table_sqlite_writer_deduplicates_duplicate_fact_ids(tmp_path: Path) -> None:
    document = _document_with_table(
        [
            ["Category", "2024"],
            ["Revenue", "1,200"],
            ["Revenue", "1,200"],
        ]
    )
    fact_table = FactTableExtractor().extract(document)
    db_path = tmp_path / "fact_table.sqlite"

    FactTableSqliteWriter().write(fact_table=fact_table, db_path=db_path)

    connection = sqlite3.connect(db_path)
    try:
        row_count = connection.execute(
            "SELECT COUNT(*) FROM fact_values WHERE normalized_subject = ?",
            ("revenue",),
        ).fetchone()
        assert row_count == (1,)
    finally:
        connection.close()
