from __future__ import annotations

from pathlib import Path

import fitz

from src.strategies import strategy_c


def _rules() -> dict:
    return {
        "strategy_routing": {
            "strategy_c": {
                "tools": {
                    "primary_ocr": "easyocr",
                    "fallback_vlm": {"provider": "ollama", "model": "minicpm-v"},
                },
                "execution_policy": {
                    "escalate_to_vlm_on_low_ocr_confidence": True,
                    "ocr_min_chars_per_page": 50,
                    "ocr_min_mean_confidence": 0.5,
                    "failure_policy": {
                        "ocr_unavailable": "vlm_only",
                        "ocr_failure": "vlm_only",
                        "vlm_failure": "error_page",
                    },
                },
                "budget_guard": {
                    "cost_per_page_estimate_usd": 0.0,
                    "max_pages_per_document": 200,
                    "max_vlm_pages_per_document": 40,
                    "max_total_runtime_seconds": 900,
                },
            }
        }
    }


def _make_pdf(path: Path, pages: int) -> None:
    doc = fitz.open()
    try:
        for i in range(pages):
            page = doc.new_page()
            page.insert_text((72, 72), f"Page {i + 1}")
        doc.save(path)
    finally:
        doc.close()


def test_strategy_c_extract_pages_schema_valid(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path, pages=2)

    def _fake_ocr_extract(*, image_path: Path, doc_id: str, page_number: int):
        _ = image_path, doc_id
        if page_number == 1:
            return ("weak", [], 0.1)
        return (
            "strong OCR page two text " * 4,
            [],
            0.95,
        )

    monkeypatch.setattr(strategy_c, "_ocr_extract", _fake_ocr_extract)
    monkeypatch.setattr(strategy_c, "_vlm_extract", lambda *, image_path, model: "VLM recovered text")

    pages = strategy_c.extract_pages_with_vision(
        pdf_path=pdf_path,
        page_numbers=[1, 2],
        rules=_rules(),
    )

    assert set(pages.keys()) == {1, 2}
    for page_number, page in pages.items():
        assert page.page_number == page_number
        assert page.metadata.strategy_used == "strategy_c"
        assert page.status in {"ok", "error"}
        assert 0.0 <= page.metadata.confidence_score <= 1.0
        assert page.page_content_hash
        assert page.metadata.cost_estimate_usd == 0.0
        assert "vlm_wall_time_sec" in page.signals


def test_strategy_c_uses_vlm_only_when_easyocr_unavailable(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path, pages=1)

    def _fake_ocr_extract(*, image_path: Path, doc_id: str, page_number: int):
        _ = image_path, doc_id, page_number
        raise RuntimeError("easyocr is required for strategy_c")

    monkeypatch.setattr(strategy_c, "_ocr_extract", _fake_ocr_extract)
    monkeypatch.setattr(
        strategy_c,
        "_vlm_extract",
        lambda *, image_path, model: "Recovered text from VLM-only fallback",
    )

    pages = strategy_c.extract_pages_with_vision(
        pdf_path=pdf_path,
        page_numbers=[1],
        rules=_rules(),
    )
    page = pages[1]
    assert page.status == "ok"
    assert page.metadata.strategy_used == "strategy_c"
    assert page.signals.get("used_vlm") == 1
    assert "Recovered text from VLM-only fallback" in page.text
    assert page.metadata.bbox_precision == "page_level"
    assert page.metadata.vlm_used is True
    assert page.metadata.vlm_wall_time_sec >= 0.0


def test_strategy_c_parses_structured_vlm_json(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path, pages=1)

    def _fake_ocr_extract(*, image_path: Path, doc_id: str, page_number: int):
        _ = image_path, doc_id, page_number
        return ("weak", [], 0.1)

    monkeypatch.setattr(strategy_c, "_ocr_extract", _fake_ocr_extract)
    monkeypatch.setattr(
        strategy_c,
        "_vlm_extract",
        lambda *, image_path, model: (
            '{"plain_text":"Main text","bullets":["one","two"],'
            '"tables":[{"columns":["a"],"rows":[["1"]]}],'
            '"figures":[{"caption":"Chart A"}]}'
        ),
    )

    pages = strategy_c.extract_pages_with_vision(
        pdf_path=pdf_path,
        page_numbers=[1],
        rules=_rules(),
    )
    page = pages[1]
    assert page.status == "ok"
    assert page.metadata.strategy_used == "strategy_c"
    assert page.metadata.bbox_precision == "page_level"
    assert "Main text" in page.text
    assert "- one" in page.text
    assert len(page.tables) == 1
    assert len(page.table_blocks) == 1
    assert page.table_blocks[0].rows == [["a"], ["1"]]
    assert len(page.figure_blocks) == 1


def test_strategy_c_marks_page_error_when_vlm_budget_exceeded(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path, pages=1)

    rules = _rules()
    rules["strategy_routing"]["strategy_c"]["budget_guard"]["max_vlm_pages_per_document"] = 0

    def _fake_ocr_extract(*, image_path: Path, doc_id: str, page_number: int):
        _ = image_path, doc_id, page_number
        return ("weak", [], 0.1)

    monkeypatch.setattr(strategy_c, "_ocr_extract", _fake_ocr_extract)

    pages = strategy_c.extract_pages_with_vision(
        pdf_path=pdf_path,
        page_numbers=[1],
        rules=rules,
    )
    page = pages[1]
    assert page.status == "error"
    assert page.error_message == "budget_exceeded: max_vlm_pages_per_document"
