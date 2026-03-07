from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.fact_table_extractor import FactTableExtractor
from src.models.extracted_document import ExtractedDocument, ExtractedPage, ExtractionMetadata, TableBlock


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


def test_fact_table_extractor_emits_numeric_entries_with_provenance() -> None:
    document = _document_with_table(
        [
            ["Category", "2024", "2025"],
            ["Revenue", "1,200", "1,450"],
            ["Margin", "12%", "14%"],
        ]
    )

    fact_table = FactTableExtractor().extract(document)

    assert len(fact_table.entries) == 4
    first = fact_table.entries[0]
    assert first.document_name == "finance.pdf"
    assert first.row_label == "Revenue"
    assert first.column_label == "2024"
    assert first.raw_value == "1,200"
    assert first.numeric_value == 1200.0
    assert first.section_path
    assert first.provenance.page_number == 1
    assert first.provenance.bbox == (0.0, 50.0, 200.0, 160.0)
    percent_entry = next(entry for entry in fact_table.entries if entry.raw_value == "12%")
    assert percent_entry.numeric_value == 12.0
    assert percent_entry.unit == "percent"


def test_fact_table_extractor_skips_non_numeric_cells() -> None:
    document = _document_with_table(
        [
            ["Category", "2024", "Notes"],
            ["Revenue", "1,200", "Audited"],
            ["Status", "Complete", "Reviewed"],
        ]
    )

    fact_table = FactTableExtractor().extract(document)

    assert len(fact_table.entries) == 1
    assert fact_table.entries[0].row_label == "Revenue"
    assert fact_table.entries[0].column_label == "2024"


def test_fact_table_extractor_parses_negative_parenthetical_values() -> None:
    document = _document_with_table(
        [
            ["Category", "2025"],
            ["Operating loss", "(250.5)"],
        ]
    )

    fact_table = FactTableExtractor().extract(document)

    assert len(fact_table.entries) == 1
    assert fact_table.entries[0].numeric_value == -250.5


def test_fact_table_extractor_skips_numeric_cells_under_blank_column_headers() -> None:
    document = _document_with_table(
        [
            ["Category", "", "2025"],
            ["Revenue", "100", "250"],
        ]
    )

    fact_table = FactTableExtractor().extract(document)

    assert len(fact_table.entries) == 1
    assert fact_table.entries[0].column_label == "2025"
    assert fact_table.entries[0].numeric_value == 250.0
