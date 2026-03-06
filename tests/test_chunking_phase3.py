from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chunking import ChunkValidationError, ChunkValidator, ChunkingConfig, ChunkingEngine
from src.models.chunking import LDU, LDUKind
from src.models.extracted_document import (
    ExtractedDocument,
    ExtractedPage,
    ExtractionMetadata,
    FigureBlock,
    TableBlock,
    TextBlock,
)
from src.utils.hashing import ldu_content_hash


def _metadata() -> ExtractionMetadata:
    return ExtractionMetadata(
        strategy_used="strategy_b",
        confidence_score=0.95,
        processing_time_sec=0.01,
        cost_estimate_usd=0.0,
        escalation_triggered=False,
    )


def _document_from_pages(*pages: ExtractedPage) -> ExtractedDocument:
    return ExtractedDocument(
        doc_id="doc123",
        file_name="sample.pdf",
        file_path="data/sample.pdf",
        page_count=len(pages),
        metadata=_metadata(),
        pages=list(pages),
    )


def test_ldu_content_hash_is_stable_and_page_agnostic() -> None:
    first = LDU(
        doc_id="doc123",
        page_number=1,
        bbox=(0.0, 0.0, 10.0, 10.0),
        kind=LDUKind.text,
        text="Revenue  grew\n\n  25% ",
        section_path=("Financial Highlights",),
    )
    second = LDU(
        doc_id="doc123",
        page_number=9,
        bbox=(5.0, 5.0, 15.0, 15.0),
        kind=LDUKind.text,
        text="Revenue grew 25%",
        section_path=("  financial   highlights ",),
    )

    assert first.content_hash == second.content_hash
    assert first.ldu_id != second.ldu_id
    assert first.content_hash == ldu_content_hash("Revenue grew 25%", ("Financial Highlights",))


def test_chunk_validator_rejects_table_ldu_without_header_row() -> None:
    validator = ChunkValidator()
    invalid_table = LDU(
        doc_id="doc123",
        page_number=1,
        bbox=(0.0, 0.0, 20.0, 20.0),
        kind=LDUKind.table,
        text="100 | 200",
        metadata={"row_count": 1, "header_row": []},
    )

    with pytest.raises(ChunkValidationError):
        validator.raise_for_issues(validator.validate_ldu(invalid_table))


def test_chunk_validator_rejects_figure_ldu_without_caption_metadata() -> None:
    validator = ChunkValidator()
    invalid_figure = LDU(
        doc_id="doc123",
        page_number=1,
        bbox=(0.0, 0.0, 20.0, 20.0),
        kind=LDUKind.figure,
        text="[figure]",
        metadata={"caption": ""},
    )

    with pytest.raises(ChunkValidationError):
        validator.raise_for_issues(validator.validate_ldu(invalid_figure))


def test_chunking_engine_emits_deterministic_chunks_with_explicit_rules() -> None:
    page = ExtractedPage(
        doc_id="doc123",
        page_number=1,
        metadata=_metadata(),
        signals={"char_count": 40, "char_density": 0.1, "image_area_ratio": 0.0, "table_count": 1},
        text_blocks=[
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="Executive summary",
                bbox=(0.0, 0.0, 50.0, 10.0),
                reading_order=0,
                content_hash="t1",
            ),
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="Revenue increased year over year.",
                bbox=(0.0, 12.0, 50.0, 22.0),
                reading_order=1,
                content_hash="t2",
            ),
        ],
        table_blocks=[
            TableBlock(
                doc_id="doc123",
                page_number=1,
                bbox=(0.0, 25.0, 50.0, 45.0),
                content_hash="tb1",
                table_index=0,
                rows=[["Year", "Revenue"], ["2025", "100"]],
            )
        ],
        figure_blocks=[
            FigureBlock(
                doc_id="doc123",
                page_number=1,
                bbox=(0.0, 46.0, 50.0, 66.0),
                content_hash="fg1",
                caption="Revenue trend chart",
            )
        ],
        page_content_hash="page1",
    )
    document = _document_from_pages(page)

    engine = ChunkingEngine(config=ChunkingConfig(max_chunk_chars=200))
    first_run = engine.build_chunks(document)
    second_run = engine.build_chunks(document)

    assert len(first_run) == 3
    assert [chunk.text for chunk in first_run] == [
        "Executive summary\n\nRevenue increased year over year.",
        "Year | Revenue\n2025 | 100",
        "Revenue trend chart",
    ]
    assert first_run[0].metadata["kinds"] == ["text", "text"]
    assert first_run[1].metadata["kinds"] == ["table"]
    assert first_run[2].metadata["kinds"] == ["figure"]
    assert [chunk.chunk_id for chunk in first_run] == [chunk.chunk_id for chunk in second_run]
    assert [chunk.content_hash for chunk in first_run] == [chunk.content_hash for chunk in second_run]


def test_section_path_inference_for_numbered_headings() -> None:
    page = ExtractedPage(
        doc_id="doc123",
        page_number=1,
        metadata=_metadata(),
        signals={"char_count": 60, "char_density": 0.1, "image_area_ratio": 0.0, "table_count": 0},
        text_blocks=[
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="3 Results",
                bbox=(0.0, 0.0, 60.0, 16.0),
                reading_order=0,
                content_hash="t1",
            ),
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="The retrieval pipeline improved materially.",
                bbox=(0.0, 20.0, 60.0, 32.0),
                reading_order=1,
                content_hash="t2",
            ),
        ],
        page_content_hash="page1",
    )

    ldus = ChunkingEngine().build_ldus(_document_from_pages(page))

    assert [ldu.section_path for ldu in ldus] == [("3 Results",), ("3 Results",)]


def test_section_path_inference_for_nested_numbered_headings() -> None:
    page = ExtractedPage(
        doc_id="doc123",
        page_number=1,
        metadata=_metadata(),
        signals={"char_count": 120, "char_density": 0.1, "image_area_ratio": 0.0, "table_count": 0},
        text_blocks=[
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="3 Results",
                bbox=(0.0, 0.0, 80.0, 16.0),
                reading_order=0,
                content_hash="t1",
            ),
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="3.2 Retrieval Precision",
                bbox=(0.0, 18.0, 80.0, 34.0),
                reading_order=1,
                content_hash="t2",
            ),
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="Precision improved by 8%.",
                bbox=(0.0, 38.0, 80.0, 50.0),
                reading_order=2,
                content_hash="t3",
            ),
        ],
        page_content_hash="page1",
    )

    ldus = ChunkingEngine().build_ldus(_document_from_pages(page))

    assert ldus[0].section_path == ("3 Results",)
    assert ldus[1].section_path == ("3 Results", "3.2 Retrieval Precision")
    assert ldus[2].section_path == ("3 Results", "3.2 Retrieval Precision")


def test_section_context_is_inherited_when_body_blocks_have_no_heading() -> None:
    page = ExtractedPage(
        doc_id="doc123",
        page_number=1,
        metadata=_metadata(),
        signals={"char_count": 140, "char_density": 0.1, "image_area_ratio": 0.0, "table_count": 1},
        text_blocks=[
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="2 Methods",
                bbox=(0.0, 0.0, 60.0, 16.0),
                reading_order=0,
                content_hash="t1",
            ),
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="We evaluated the pipeline on 10 benchmark queries.",
                bbox=(0.0, 20.0, 60.0, 32.0),
                reading_order=1,
                content_hash="t2",
            ),
        ],
        table_blocks=[
            TableBlock(
                doc_id="doc123",
                page_number=1,
                bbox=(0.0, 36.0, 60.0, 56.0),
                content_hash="tb1",
                table_index=0,
                rows=[["Metric", "Value"], ["Queries", "10"]],
            )
        ],
        page_content_hash="page1",
    )

    ldus = ChunkingEngine().build_ldus(_document_from_pages(page))

    assert [ldu.section_path for ldu in ldus] == [
        ("2 Methods",),
        ("2 Methods",),
        ("2 Methods",),
    ]


def test_section_boundary_changes_across_pages_affect_paths_hashes_and_chunks() -> None:
    first_page = ExtractedPage(
        doc_id="doc123",
        page_number=1,
        metadata=_metadata(),
        signals={"char_count": 100, "char_density": 0.1, "image_area_ratio": 0.0, "table_count": 0},
        text_blocks=[
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="1 Introduction",
                bbox=(0.0, 0.0, 70.0, 16.0),
                reading_order=0,
                content_hash="p1t1",
            ),
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="Shared body text",
                bbox=(0.0, 20.0, 70.0, 32.0),
                reading_order=1,
                content_hash="p1t2",
            ),
        ],
        page_content_hash="page1",
    )
    second_page = ExtractedPage(
        doc_id="doc123",
        page_number=2,
        metadata=_metadata(),
        signals={"char_count": 100, "char_density": 0.1, "image_area_ratio": 0.0, "table_count": 0},
        text_blocks=[
            TextBlock(
                doc_id="doc123",
                page_number=2,
                text="2 Discussion",
                bbox=(0.0, 0.0, 70.0, 16.0),
                reading_order=0,
                content_hash="p2t1",
            ),
            TextBlock(
                doc_id="doc123",
                page_number=2,
                text="Shared body text",
                bbox=(0.0, 20.0, 70.0, 32.0),
                reading_order=1,
                content_hash="p2t2",
            ),
        ],
        page_content_hash="page2",
    )

    engine = ChunkingEngine(config=ChunkingConfig(max_chunk_chars=200))
    document = _document_from_pages(first_page, second_page)

    ldus = engine.build_ldus(document)
    chunks = engine.build_chunks(document)

    assert ldus[1].section_path == ("1 Introduction",)
    assert ldus[3].section_path == ("2 Discussion",)
    assert ldus[1].content_hash != ldus[3].content_hash
    assert len(chunks) == 2
    assert chunks[0].section_path == ("1 Introduction",)
    assert chunks[1].section_path == ("2 Discussion",)
