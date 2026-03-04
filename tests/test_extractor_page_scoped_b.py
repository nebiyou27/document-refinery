from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.agents import extractor
from src.models.document_profile import (
    DocumentProfile,
    DomainHint,
    ExtractionStrategy,
    LayoutComplexity,
    OriginType,
)
from src.models.extracted_document import ExtractedDocument, ExtractedPage, ExtractionMetadata
from src.utils.hashing import content_hash


def _make_page(
    *,
    doc_id: str,
    page_number: int,
    strategy_used: str,
    confidence: float,
    text: str,
    escalation_triggered: bool = False,
    escalation_target: str | None = None,
) -> ExtractedPage:
    return ExtractedPage(
        doc_id=doc_id,
        page_number=page_number,
        status="ok",
        text=text,
        tables=[],
        metadata=ExtractionMetadata(
            strategy_used=strategy_used,
            confidence_score=confidence,
            processing_time_sec=0.01,
            cost_estimate_usd=0.0,
            escalation_triggered=escalation_triggered,
            escalation_target=escalation_target,
        ),
        signals={"char_count": 100, "char_density": 0.01, "image_area_ratio": 0.1, "table_count": 0},
        text_blocks=[],
        table_blocks=[],
        figure_blocks=[],
        page_content_hash=content_hash(f"{strategy_used}:{page_number}:{text}"),
    )


def test_replaces_only_escalated_page_and_ledger_has_no_duplicate_strategy_rows(
    monkeypatch,
    tmp_path: Path,
) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")

    doc_id = "doc123abc456"
    rules = {
        "strategy_routing": {
            "strategy_a": {
                "confidence_gates": {"min_confidence_score": 0.75},
                "escalation_target": "strategy_b",
                "escalation": {
                    "to_c_if": {
                        "image_area_ratio_gte": 0.85,
                        "char_density_lte": 0.001,
                        "char_count_lte": 80,
                    },
                    "default_target": "strategy_b",
                },
            },
            "strategy_b": {
                "confidence_gates": {"min_confidence_score": 0.65},
                "escalation_target": "strategy_c",
                "escalation_scope": "page_level",
            },
        }
    }
    profile = DocumentProfile(
        doc_id=doc_id,
        file_path=str(pdf_path),
        file_name=pdf_path.name,
        origin_type=OriginType.mixed,
        layout_complexity=LayoutComplexity.single_column,
        domain_hint=DomainHint.general,
        extraction_strategy=ExtractionStrategy.strategy_a,
        estimated_cost_usd=0.0,
        page_count=3,
        per_page_signals=[],
    )

    a_pages = [
        _make_page(doc_id=doc_id, page_number=1, strategy_used="strategy_a", confidence=0.92, text="a1"),
        _make_page(doc_id=doc_id, page_number=2, strategy_used="strategy_a", confidence=0.42, text="a2"),
        _make_page(doc_id=doc_id, page_number=3, strategy_used="strategy_a", confidence=0.91, text="a3"),
    ]
    extracted_a = ExtractedDocument(
        doc_id=doc_id,
        file_name=pdf_path.name,
        file_path=str(pdf_path),
        page_count=3,
        status="ok",
        metadata=ExtractionMetadata(
            strategy_used="strategy_a",
            confidence_score=0.75,
            processing_time_sec=0.1,
            cost_estimate_usd=0.0,
            escalation_triggered=False,
            escalation_target=None,
        ),
        pages=a_pages,
    )
    b_page2 = _make_page(
        doc_id=doc_id,
        page_number=2,
        strategy_used="strategy_b",
        confidence=0.88,
        text="b2",
    )

    monkeypatch.setattr(extractor, "load_rules", lambda rules_path=extractor.RULES_PATH: rules)
    monkeypatch.setattr(extractor, "_load_or_compute_profile", lambda _pdf_path: profile)
    monkeypatch.setattr(extractor, "_compute_doc_id", lambda _pdf_path: doc_id)
    monkeypatch.setattr(extractor, "extract_with_pdfplumber", lambda **kwargs: extracted_a)
    monkeypatch.setattr(
        extractor,
        "extract_pages_with_docling",
        lambda **kwargs: {2: b_page2},
    )

    ledger_root = tmp_path / "ledger"

    def _append_ledger_entry(
        *,
        doc_id: str,
        file_name: str,
        page_number: int,
        strategy_used: str,
        confidence: float,
        signals: dict[str, Any],
        cost_estimate: float,
        processing_time: float,
        escalated_to: str | None,
    ) -> Path:
        ledger_root.mkdir(parents=True, exist_ok=True)
        ledger_file = ledger_root / f"{doc_id}.jsonl"
        row = {
            "doc_id": doc_id,
            "file_name": file_name,
            "page_number": page_number,
            "strategy_used": strategy_used,
            "confidence": confidence,
            "signals": signals,
            "cost_estimate_usd": cost_estimate,
            "processing_time_sec": processing_time,
            "escalated_to": escalated_to,
        }
        with ledger_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        return ledger_file

    monkeypatch.setattr(extractor, "append_ledger_entry", _append_ledger_entry)
    monkeypatch.setattr(extractor, "LEDGER_DIR", ledger_root)
    monkeypatch.setattr(extractor, "EXTRACTED_DIR", tmp_path / "extracted")

    result = extractor.run_extraction(pdf_path)

    assert [p.metadata.strategy_used for p in result.pages] == ["strategy_a", "strategy_b", "strategy_a"]
    assert result.pages[1].text == "b2"
    assert result.pages[0].text == "a1"
    assert result.pages[2].text == "a3"

    ledger_file = ledger_root / f"{doc_id}.jsonl"
    rows = [json.loads(line) for line in ledger_file.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 4  # 3x A + 1x B replacement
    assert sum(1 for row in rows if row["strategy_used"] == "strategy_b") == 1
    assert any(row["page_number"] == 2 and row["strategy_used"] == "strategy_b" for row in rows)

    keys = [(row["doc_id"], row["page_number"], row["strategy_used"]) for row in rows]
    assert len(keys) == len(set(keys))
