from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import src.strategies.strategy_b as strategy_b


class _FakeBBox:
    def __init__(self, box: tuple[float, float, float, float]) -> None:
        self._box = box

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self._box


class _FakeProv:
    def __init__(self, page_no: int, box: tuple[float, float, float, float]) -> None:
        self.page_no = page_no
        self.bbox = _FakeBBox(box)


class _FakeCell:
    def __init__(self, text: str) -> None:
        self._text = text

    def _get_text(self, doc=None) -> str:
        _ = doc
        return self._text


class _FakeTextItem:
    def __init__(self, page_no: int, text: str) -> None:
        self.label = "text"
        self.text = text
        self.prov = [_FakeProv(page_no, (0.0, 0.0, 50.0, 10.0))]


class _FakeTableItem:
    def __init__(self, page_no: int) -> None:
        self.label = "table"
        self.prov = [_FakeProv(page_no, (0.0, 12.0, 50.0, 40.0))]
        self.data = SimpleNamespace(
            grid=[
                [_FakeCell("H1"), _FakeCell("H2")],
                [_FakeCell("v1"), _FakeCell("v2")],
            ]
        )


class _FakeDocument:
    def iterate_items(self, *, page_no: int, with_groups: bool = False):
        _ = with_groups
        if page_no == 1:
            return [(_FakeTextItem(1, "alpha"), 0), (_FakeTableItem(1), 0)]
        return [(_FakeTextItem(2, "beta"), 0)]


class _FakeConverter:
    def convert(self, *args, **kwargs):
        _ = args, kwargs
        confidence = SimpleNamespace(
            pages={
                1: SimpleNamespace(mean_score=0.8),
                2: SimpleNamespace(mean_score=0.7),
            }
        )
        return SimpleNamespace(document=_FakeDocument(), confidence=confidence)


def test_strategy_b_output_validates(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")

    monkeypatch.setattr(strategy_b, "_import_docling", lambda: _FakeConverter)
    monkeypatch.setattr(
        strategy_b,
        "_write_single_page_pdf",
        lambda _source_pdf, *, page_number, output_pdf: output_pdf.write_bytes(
            f"%PDF-page-{page_number}".encode("utf-8")
        ),
    )
    monkeypatch.setattr(
        strategy_b,
        "_page_signals_from_pdf",
        lambda _path: [
            {"char_count": 120, "char_density": 0.01, "image_area_ratio": 0.1, "table_count": 0},
            {"char_count": 80, "char_density": 0.007, "image_area_ratio": 0.2, "table_count": 0},
        ],
    )

    pages = strategy_b.extract_pages_with_docling(
        pdf_path=pdf_path,
        page_numbers=[1, 2],
        rules={"strategy_routing": {"strategy_b": {"confidence_gates": {"min_confidence_score": 0.65}}}},
        batch_size=1,
    )

    assert set(pages.keys()) == {1, 2}
    for page_number, page in pages.items():
        assert page.page_number == page_number
        assert page.metadata.strategy_used == "strategy_b"
        assert page.page_number >= 1
        assert "char_count" in page.signals
        assert "char_density" in page.signals
        assert "image_area_ratio" in page.signals
        assert "table_count" in page.signals
        assert 0.0 <= page.metadata.confidence_score <= 1.0
