from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.extracted_document import (
    ExtractedDocument,
    ExtractedPage,
    ExtractionMetadata,
    TextBlock,
)
from src.utils.hashing import content_hash
from src.utils.ledger import append_ledger_entry


def test_extracted_document_constructs() -> None:
    block = TextBlock(
        doc_id="abc123",
        page_number=1,
        text="Hello",
        bbox=(0.0, 0.0, 10.0, 10.0),
        reading_order=0,
        content_hash=content_hash("Hello"),
    )
    page = ExtractedPage(
        doc_id="abc123",
        page_number=1,
        metadata=ExtractionMetadata(
            strategy_used="strategy_a",
            confidence_score=0.9,
            processing_time_sec=0.01,
            cost_estimate_usd=0.0,
            escalation_triggered=False,
        ),
        signals={"char_count": 5, "char_density": 0.1, "image_area_ratio": 0.0, "table_count": 0},
        text_blocks=[block],
        page_content_hash=content_hash("page"),
    )
    doc = ExtractedDocument(
        doc_id="abc123",
        file_name="sample.pdf",
        file_path="data/sample.pdf",
        page_count=1,
        metadata=ExtractionMetadata(
            strategy_used="strategy_a",
            confidence_score=0.9,
            processing_time_sec=0.1,
            cost_estimate_usd=0.0,
            escalation_triggered=False,
        ),
        pages=[page],
    )
    assert doc.page_count == 1
    assert doc.pages[0].text_blocks[0].content_hash == content_hash("Hello")


def test_content_hash_is_deterministic() -> None:
    value = "deterministic content"
    assert content_hash(value) == content_hash(value)


def test_ledger_writes_one_line_per_page(tmp_path) -> None:
    ledger_root = tmp_path / "ledger"
    for page in [1, 2, 3]:
        append_ledger_entry(
            doc_id="doc001",
            file_name="sample.pdf",
            page_number=page,
            strategy_used="strategy_a",
            confidence=0.8,
            signals={"char_count": 100},
            cost_estimate=0.0,
            processing_time=0.01,
            escalated_to=None,
            ledger_root=ledger_root,
        )

    ledger_file = ledger_root / "doc001.jsonl"
    lines = ledger_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    payload = [json.loads(line) for line in lines]
    assert [row["page_number"] for row in payload] == [1, 2, 3]
    assert all(row["cost_estimate_usd"] == 0.0 for row in payload)
    assert all(row["cost_units"]["type"] == "runtime_seconds" for row in payload)
    assert all("value" in row["cost_units"] for row in payload)
    assert all(row["vlm_used"] is False for row in payload)
    assert all(row["vlm_wall_time_sec"] == 0.0 for row in payload)
