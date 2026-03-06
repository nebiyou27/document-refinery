from __future__ import annotations

import json
from pathlib import Path

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


def _rules() -> dict:
    return {
        "strategy_routing": {
            "strategy_a": {
                "confidence_gates": {
                    "min_char_density": 0.001,
                    "max_image_area_ratio": 0.61,
                    "min_confidence_score": 0.75,
                },
                "escalation_target": "strategy_b",
                "escalation": {
                    "to_c_if": {
                        "image_area_ratio_gte": 0.70,
                        "char_density_lte": 0.001,
                        "char_count_lte": 200,
                    },
                    "default_target": "strategy_b",
                },
            },
            "strategy_b": {
                "confidence_gates": {"min_confidence_score": 0.65},
                "escalation_target": "strategy_c",
                "escalation_scope": "page_level",
            },
            "strategy_c": {
                "tools": {
                    "primary_ocr": "easyocr",
                    "fallback_vlm": {"provider": "ollama", "model": "minicpm-v"},
                },
                "execution_policy": {
                    "escalate_to_vlm_on_low_ocr_confidence": True,
                    "ocr_min_chars_per_page": 50,
                    "ocr_min_mean_confidence": 0.5,
                },
                "budget_guard": {
                    "cost_per_page_estimate_usd": 0.0,
                    "max_pages_per_document": 200,
                    "max_vlm_pages_per_document": 40,
                    "max_total_runtime_seconds": 900,
                },
            },
        }
    }


def _make_page(doc_id: str, page_number: int, strategy_used: str, confidence: float, text: str) -> ExtractedPage:
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
            escalation_triggered=False,
            escalation_target=None,
        ),
        signals={"char_count": len(text), "char_density": 0.01, "image_area_ratio": 0.9, "table_count": 0},
        text_blocks=[],
        table_blocks=[],
        figure_blocks=[],
        page_content_hash=content_hash(f"{strategy_used}:{page_number}:{text}"),
    )


def _make_error_page(doc_id: str, page_number: int, message: str) -> ExtractedPage:
    return ExtractedPage(
        doc_id=doc_id,
        page_number=page_number,
        status="error",
        text="",
        tables=[],
        metadata=ExtractionMetadata(
            strategy_used="strategy_c",
            confidence_score=0.0,
            processing_time_sec=45.0,
            cost_estimate_usd=0.0,
            escalation_triggered=False,
            escalation_target=None,
        ),
        signals={"char_count": 0, "char_density": 0.0, "image_area_ratio": 1.0, "table_count": 0},
        text_blocks=[],
        table_blocks=[],
        figure_blocks=[],
        page_content_hash=content_hash(f"strategy_c:error:{page_number}:{message}"),
        error_message=message,
    )


def test_extractor_forced_strategy_c_replaces_pages_and_logs_ledger(monkeypatch, tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    pdf_path = repo_root / "data/20191010_Pharmaceutical-Manufacturing-Opportunites-in-Ethiopia_VF.pdf"
    assert pdf_path.exists(), "Expected pharma PDF fixture to exist in data/"

    doc_id = "cfeedbeef123"
    profile = DocumentProfile(
        doc_id=doc_id,
        file_path=str(pdf_path),
        file_name=pdf_path.name,
        origin_type=OriginType.scanned_image,
        layout_complexity=LayoutComplexity.single_column,
        domain_hint=DomainHint.general,
        extraction_strategy=ExtractionStrategy.strategy_c,
        estimated_cost_usd=0.0,
        page_count=2,
        per_page_signals=[],
    )
    base_doc = ExtractedDocument(
        doc_id=doc_id,
        file_name=pdf_path.name,
        file_path=str(pdf_path),
        page_count=2,
        status="ok",
        metadata=ExtractionMetadata(
            strategy_used="strategy_a",
            confidence_score=0.5,
            processing_time_sec=0.1,
            cost_estimate_usd=0.0,
            escalation_triggered=False,
            escalation_target=None,
        ),
        pages=[
            _make_page(doc_id, 1, "strategy_a", 0.2, "a1"),
            _make_page(doc_id, 2, "strategy_a", 0.2, "a2"),
        ],
    )

    monkeypatch.setattr(extractor, "load_rules", lambda rules_path=extractor.RULES_PATH: _rules())
    monkeypatch.setattr(extractor, "_load_or_compute_profile", lambda _pdf_path: profile)
    monkeypatch.setattr(extractor, "_compute_doc_id", lambda _pdf_path: doc_id)
    monkeypatch.setattr(extractor, "extract_with_pdfplumber", lambda **kwargs: base_doc)
    monkeypatch.setattr(
        extractor,
        "extract_pages_with_vision",
        lambda **kwargs: {
            1: _make_page(doc_id, 1, "strategy_c", 0.83, "c1"),
            2: _make_page(doc_id, 2, "strategy_c", 0.79, "c2"),
        },
    )
    monkeypatch.setattr(extractor, "LEDGER_DIR", tmp_path / "ledger")
    monkeypatch.setattr(extractor, "EXTRACTED_DIR", tmp_path / "extracted")
    monkeypatch.setattr(
        extractor,
        "append_ledger_entry",
        lambda **kwargs: __import__("src.utils.ledger", fromlist=["append_ledger_entry"])
        .append_ledger_entry(**kwargs, ledger_root=tmp_path / "ledger"),
    )

    result = extractor.run_extraction(pdf_path, strategy="strategy_c")

    assert any(page.metadata.strategy_used == "strategy_c" for page in result.pages)
    ledger_file = (tmp_path / "ledger") / f"{doc_id}.jsonl"
    rows = [json.loads(line) for line in ledger_file.read_text(encoding="utf-8").splitlines()]
    assert rows
    assert any(row["strategy_used"] == "strategy_c" for row in rows)


def test_extractor_uses_profile_strategy_c_as_starting_strategy(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")

    doc_id = "startc123456"
    profile = DocumentProfile(
        doc_id=doc_id,
        file_path=str(pdf_path),
        file_name=pdf_path.name,
        origin_type=OriginType.scanned_image,
        layout_complexity=LayoutComplexity.single_column,
        domain_hint=DomainHint.general,
        extraction_strategy=ExtractionStrategy.strategy_c,
        estimated_cost_usd=0.0,
        page_count=2,
        per_page_signals=[],
    )

    monkeypatch.setattr(extractor, "load_rules", lambda rules_path=extractor.RULES_PATH: _rules())
    monkeypatch.setattr(extractor, "_load_or_compute_profile", lambda _pdf_path: profile)
    monkeypatch.setattr(extractor, "_compute_doc_id", lambda _pdf_path: doc_id)
    monkeypatch.setattr(
        extractor,
        "extract_with_pdfplumber",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("Strategy A should not run first")),
    )
    monkeypatch.setattr(
        extractor,
        "extract_pages_with_vision",
        lambda **kwargs: {
            1: _make_page(doc_id, 1, "strategy_c", 0.9, "c1"),
            2: _make_page(doc_id, 2, "strategy_c", 0.88, "c2"),
        },
    )
    monkeypatch.setattr(extractor, "extract_pages_with_docling", lambda **kwargs: {})
    monkeypatch.setattr(extractor, "LEDGER_DIR", tmp_path / "ledger")
    monkeypatch.setattr(extractor, "EXTRACTED_DIR", tmp_path / "extracted")
    monkeypatch.setattr(
        extractor,
        "append_ledger_entry",
        lambda **kwargs: __import__("src.utils.ledger", fromlist=["append_ledger_entry"])
        .append_ledger_entry(**kwargs, ledger_root=tmp_path / "ledger"),
    )

    result = extractor.run_extraction(pdf_path)

    assert all(page.metadata.strategy_used == "strategy_c" for page in result.pages)
    routing = json.loads((tmp_path / "extracted" / f"{doc_id}.routing.json").read_text(encoding="utf-8"))
    assert routing["document_level_strategy_suggestion"] == "strategy_c"
    assert routing["starting_strategy"] == "strategy_c"
    assert routing["strategy_counts"]["final_pages_by_strategy"]["strategy_c"] == 2
    assert routing["strategy_counts"]["executions_by_strategy"]["strategy_c"] == 2
    assert routing["strategy_c_fallback_to_b"]["attempted"] is False


def test_assemble_document_preserves_underlying_page_error_when_all_pages_fail(tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")

    doc_id = "allerr123456"
    profile = DocumentProfile(
        doc_id=doc_id,
        file_path=str(pdf_path),
        file_name=pdf_path.name,
        origin_type=OriginType.scanned_image,
        layout_complexity=LayoutComplexity.single_column,
        domain_hint=DomainHint.general,
        extraction_strategy=ExtractionStrategy.strategy_c,
        estimated_cost_usd=0.0,
        page_count=2,
        per_page_signals=[],
    )
    failing_page = ExtractedPage(
        doc_id=doc_id,
        page_number=1,
        status="error",
        text="",
        tables=[],
        metadata=ExtractionMetadata(
            strategy_used="strategy_c",
            confidence_score=0.0,
            processing_time_sec=120.0,
            cost_estimate_usd=0.0,
            escalation_triggered=False,
            escalation_target=None,
        ),
        signals={"char_count": 0, "char_density": 0.0, "image_area_ratio": 1.0, "table_count": 0},
        text_blocks=[],
        table_blocks=[],
        figure_blocks=[],
        page_content_hash=content_hash("page-1-error"),
        error_message="strategy_c_failed: vlm_failed: HTTPConnectionPool(host='localhost', port=11434): Read timed out. (read timeout=120)",
    )
    missing_page = extractor._missing_page(
        doc_id=doc_id,
        page_number=2,
        strategy_used="strategy_c",
        message="strategy_c_missing_page_output",
    )

    assembled = extractor._assemble_document(
        pdf_path=pdf_path,
        profile=profile,
        pages_by_number={1: failing_page, 2: missing_page},
        default_strategy="strategy_c",
    )

    assert assembled.status == "error"
    assert assembled.error_message == failing_page.error_message


def test_high_image_thin_page_escalates_from_a_to_c() -> None:
    rules = _rules()
    page_signals = {
        "image_area_ratio": 0.78,
        "char_density": 0.0008,
        "char_count": 150,
    }
    assert extractor.choose_escalation_target_for_page(page_signals, rules) == "strategy_c"


def test_strategy_c_timeout_falls_back_to_strategy_b_and_recovers(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")

    doc_id = "c2b123456789"
    profile = DocumentProfile(
        doc_id=doc_id,
        file_path=str(pdf_path),
        file_name=pdf_path.name,
        origin_type=OriginType.scanned_image,
        layout_complexity=LayoutComplexity.single_column,
        domain_hint=DomainHint.general,
        extraction_strategy=ExtractionStrategy.strategy_c,
        estimated_cost_usd=0.0,
        page_count=2,
        per_page_signals=[],
    )

    monkeypatch.setattr(extractor, "load_rules", lambda rules_path=extractor.RULES_PATH: _rules())
    monkeypatch.setattr(extractor, "_load_or_compute_profile", lambda _pdf_path: profile)
    monkeypatch.setattr(extractor, "_compute_doc_id", lambda _pdf_path: doc_id)
    monkeypatch.setattr(
        extractor,
        "extract_pages_with_vision",
        lambda **kwargs: {
            1: _make_error_page(doc_id, 1, "strategy_c_failed: vlm_failed(timeout=45.0s): Read timed out"),
            2: _make_error_page(doc_id, 2, "strategy_c_failed: vlm_failed(timeout=45.0s): Read timed out"),
        },
    )
    monkeypatch.setattr(
        extractor,
        "extract_pages_with_docling",
        lambda **kwargs: {
            1: _make_page(doc_id, 1, "strategy_b", 0.81, "budget table rows"),
            2: _make_page(doc_id, 2, "strategy_b", 0.79, "expense summary"),
        },
    )
    monkeypatch.setattr(extractor, "LEDGER_DIR", tmp_path / "ledger")
    monkeypatch.setattr(extractor, "EXTRACTED_DIR", tmp_path / "extracted")
    monkeypatch.setattr(
        extractor,
        "append_ledger_entry",
        lambda **kwargs: __import__("src.utils.ledger", fromlist=["append_ledger_entry"])
        .append_ledger_entry(**kwargs, ledger_root=tmp_path / "ledger"),
    )

    result = extractor.run_extraction(pdf_path)

    assert result.status == "ok"
    assert result.metadata.strategy_used == "strategy_b"
    assert all(page.metadata.strategy_used == "strategy_b" for page in result.pages)

    routing = json.loads((tmp_path / "extracted" / f"{doc_id}.routing.json").read_text(encoding="utf-8"))
    assert routing["starting_strategy"] == "strategy_c"
    assert routing["executed_strategy"] == "strategy_b"
    assert routing["strategy_counts"]["executions_by_strategy"]["strategy_c"] == 2
    assert routing["strategy_counts"]["executions_by_strategy"]["strategy_b"] == 2
    assert routing["strategy_c_fallback_to_b"]["attempted"] is True
    assert routing["strategy_c_fallback_to_b"]["trigger_reason"] == "strategy_c_timeout_or_error"
    assert routing["strategy_c_fallback_to_b"]["recovered"] is True


def test_strategy_c_no_usable_content_falls_back_to_strategy_b(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")

    doc_id = "cmeta1234567"
    profile = DocumentProfile(
        doc_id=doc_id,
        file_path=str(pdf_path),
        file_name=pdf_path.name,
        origin_type=OriginType.scanned_image,
        layout_complexity=LayoutComplexity.single_column,
        domain_hint=DomainHint.general,
        extraction_strategy=ExtractionStrategy.strategy_c,
        estimated_cost_usd=0.0,
        page_count=2,
        per_page_signals=[],
    )

    monkeypatch.setattr(extractor, "load_rules", lambda rules_path=extractor.RULES_PATH: _rules())
    monkeypatch.setattr(extractor, "_load_or_compute_profile", lambda _pdf_path: profile)
    monkeypatch.setattr(extractor, "_compute_doc_id", lambda _pdf_path: doc_id)
    monkeypatch.setattr(
        extractor,
        "extract_pages_with_vision",
        lambda **kwargs: {
            1: _make_page(doc_id, 1, "strategy_c", 0.9, "The extracted JSON is as follows: {\"plain_text\":\"...\"}"),
            2: _make_page(doc_id, 2, "strategy_c", 0.9, "The plain_text field contains the following bullets: - item"),
        },
    )
    monkeypatch.setattr(
        extractor,
        "extract_pages_with_docling",
        lambda **kwargs: {
            1: _make_page(doc_id, 1, "strategy_b", 0.77, "recoverable structured text"),
            2: _make_page(doc_id, 2, "strategy_b", 0.74, "procurement table summary"),
        },
    )
    monkeypatch.setattr(extractor, "LEDGER_DIR", tmp_path / "ledger")
    monkeypatch.setattr(extractor, "EXTRACTED_DIR", tmp_path / "extracted")
    monkeypatch.setattr(
        extractor,
        "append_ledger_entry",
        lambda **kwargs: __import__("src.utils.ledger", fromlist=["append_ledger_entry"])
        .append_ledger_entry(**kwargs, ledger_root=tmp_path / "ledger"),
    )

    result = extractor.run_extraction(pdf_path)

    assert result.status == "ok"
    assert result.metadata.strategy_used == "strategy_b"
    routing = json.loads((tmp_path / "extracted" / f"{doc_id}.routing.json").read_text(encoding="utf-8"))
    assert routing["strategy_c_fallback_to_b"]["attempted"] is True
    assert routing["strategy_c_fallback_to_b"]["trigger_reason"] == "strategy_c_no_usable_content"
    assert routing["strategy_c_fallback_to_b"]["recovered"] is True


def test_strategy_c_fallback_trace_is_visible_in_ledger(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")

    doc_id = "ctrace123456"
    profile = DocumentProfile(
        doc_id=doc_id,
        file_path=str(pdf_path),
        file_name=pdf_path.name,
        origin_type=OriginType.scanned_image,
        layout_complexity=LayoutComplexity.single_column,
        domain_hint=DomainHint.general,
        extraction_strategy=ExtractionStrategy.strategy_c,
        estimated_cost_usd=0.0,
        page_count=1,
        per_page_signals=[],
    )

    monkeypatch.setattr(extractor, "load_rules", lambda rules_path=extractor.RULES_PATH: _rules())
    monkeypatch.setattr(extractor, "_load_or_compute_profile", lambda _pdf_path: profile)
    monkeypatch.setattr(extractor, "_compute_doc_id", lambda _pdf_path: doc_id)
    monkeypatch.setattr(
        extractor,
        "extract_pages_with_vision",
        lambda **kwargs: {1: _make_error_page(doc_id, 1, "strategy_c_failed: vlm_failed(timeout=45.0s): Read timed out")},
    )
    monkeypatch.setattr(
        extractor,
        "extract_pages_with_docling",
        lambda **kwargs: {1: _make_page(doc_id, 1, "strategy_b", 0.82, "recovered text")},
    )
    monkeypatch.setattr(extractor, "LEDGER_DIR", tmp_path / "ledger")
    monkeypatch.setattr(extractor, "EXTRACTED_DIR", tmp_path / "extracted")
    monkeypatch.setattr(
        extractor,
        "append_ledger_entry",
        lambda **kwargs: __import__("src.utils.ledger", fromlist=["append_ledger_entry"])
        .append_ledger_entry(**kwargs, ledger_root=tmp_path / "ledger"),
    )

    extractor.run_extraction(pdf_path)

    ledger_file = (tmp_path / "ledger") / f"{doc_id}.jsonl"
    rows = [json.loads(line) for line in ledger_file.read_text(encoding="utf-8").splitlines()]
    assert [row["strategy_used"] for row in rows] == ["strategy_c", "strategy_b"]

    routing = json.loads((tmp_path / "extracted" / f"{doc_id}.routing.json").read_text(encoding="utf-8"))
    assert routing["strategy_c_fallback_to_b"] == {
        "attempted": True,
        "trigger_reason": "strategy_c_timeout_or_error",
        "recovered": True,
        "recovery_status": "ok",
        "recovery_usable_content": True,
    }
