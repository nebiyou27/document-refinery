from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import src.strategies.strategy_b as strategy_b
from src.models.extracted_document import ExtractedPage, ExtractionMetadata, TableBlock


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


def test_build_docling_converter_default_uses_existing_constructor(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class _RecordingConverter:
        def __init__(self, *args, **kwargs) -> None:
            calls.append({"args": args, "kwargs": kwargs})

    class _FakeInputFormat:
        PDF = "pdf"

    class _FakeAcceleratorOptions:
        def __init__(self, *, device: str) -> None:
            self.device = device

    class _FakePdfPipelineOptions:
        def __init__(self, *, accelerator_options=None) -> None:
            self.accelerator_options = accelerator_options

    class _FakePdfFormatOption:
        def __init__(self, *, pipeline_options) -> None:
            self.pipeline_options = pipeline_options

    monkeypatch.delenv("DOC_REFINERY_DOCLING_DEVICE", raising=False)
    monkeypatch.setattr(strategy_b, "_import_docling", lambda: _RecordingConverter)
    monkeypatch.setattr(
        strategy_b,
        "_import_docling_pdf_options",
        lambda: (
            _FakeAcceleratorOptions,
            _FakeInputFormat,
            _FakePdfFormatOption,
            _FakePdfPipelineOptions,
        ),
    )

    converter = strategy_b._build_docling_converter()

    assert isinstance(converter, _RecordingConverter)
    assert len(calls) == 1
    assert "format_options" in calls[0]["kwargs"]


def test_build_docling_converter_cuda_sets_pdf_pipeline_device(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class _RecordingConverter:
        def __init__(self, *args, **kwargs) -> None:
            calls.append({"args": args, "kwargs": kwargs})

    class _FakeInputFormat:
        PDF = "pdf"

    class _FakeAcceleratorOptions:
        def __init__(self, *, device: str) -> None:
            self.device = device

    class _FakePdfPipelineOptions:
        def __init__(self, *, accelerator_options) -> None:
            self.accelerator_options = accelerator_options

    class _FakePdfFormatOption:
        def __init__(self, *, pipeline_options) -> None:
            self.pipeline_options = pipeline_options

    monkeypatch.setenv("DOC_REFINERY_DOCLING_DEVICE", "cuda")
    monkeypatch.setattr(strategy_b, "_import_docling", lambda: _RecordingConverter)
    monkeypatch.setattr(
        strategy_b,
        "_import_docling_pdf_options",
        lambda: (
            _FakeAcceleratorOptions,
            _FakeInputFormat,
            _FakePdfFormatOption,
            _FakePdfPipelineOptions,
        ),
    )

    converter = strategy_b._build_docling_converter()

    assert isinstance(converter, _RecordingConverter)
    assert len(calls) == 1
    format_options = calls[0]["kwargs"]["format_options"]
    pdf_option = format_options[_FakeInputFormat.PDF]
    assert pdf_option.pipeline_options.accelerator_options.device == "cuda"


def test_strategy_b_output_validates(monkeypatch, tmp_path: Path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")

    monkeypatch.setattr(strategy_b, "_build_docling_converter", lambda: _FakeConverter())
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


def _page_with_tables(table_blocks: list[TableBlock]) -> ExtractedPage:
    return ExtractedPage(
        doc_id="doc1",
        page_number=1,
        status="ok",
        text="",
        tables=[{"table_index": block.table_index, "rows": block.rows} for block in table_blocks],
        metadata=ExtractionMetadata(
            strategy_used="strategy_b",
            confidence_score=0.8,
            processing_time_sec=0.1,
            cost_estimate_usd=0.0,
            escalation_triggered=False,
            escalation_target=None,
        ),
        signals={"char_count": 100, "char_density": 0.01, "image_area_ratio": 0.1, "table_count": len(table_blocks)},
        text_blocks=[],
        table_blocks=table_blocks,
        figure_blocks=[],
        page_content_hash="hash",
    )


def test_repair_table_boundaries_replaces_overlapping_truncated_table() -> None:
    existing = TableBlock(
        doc_id="doc1",
        page_number=1,
        bbox=(10.0, 10.0, 200.0, 70.0),
        content_hash="old",
        table_index=0,
        rows=[["Year", "Value"], ["2024", "100"]],
    )
    page = _page_with_tables([existing])
    candidates = [
        (
            (10.0, 10.0, 200.0, 70.0),
            [["Year", "Q1", "Q2"], ["2024", "100", "120"], ["2025", "110", "130"]],
        )
    ]

    strategy_b._repair_table_boundaries_for_page(page=page, table_candidates=candidates)

    assert len(page.table_blocks) == 1
    assert page.table_blocks[0].rows == [["Year", "Q1", "Q2"], ["2024", "100", "120"], ["2025", "110", "130"]]
    assert page.signals["table_count"] == 1


def test_repair_table_boundaries_appends_missing_table() -> None:
    page = _page_with_tables([])
    candidates = [
        (
            (20.0, 20.0, 180.0, 120.0),
            [["Category", "Amount"], ["Operating", "200"], ["Personnel", "150"]],
        )
    ]

    strategy_b._repair_table_boundaries_for_page(page=page, table_candidates=candidates)

    assert len(page.table_blocks) == 1
    assert page.table_blocks[0].table_index == 0
    assert page.table_blocks[0].rows[0] == ["Category", "Amount"]
    assert page.signals["table_count"] == 1


def test_recover_columns_from_single_cell_rows_preserves_expected_alignment() -> None:
    rows = [
        ["Month    Food Inflation    Non-Food Inflation"],
        ["June-EFY2016    12.3    8.1"],
        ["June-EFY2017    13.0    9.4"],
    ]

    recovered = strategy_b._recover_columns_from_single_cell_rows(rows)

    assert recovered[0] == ["Month", "Food Inflation", "Non-Food Inflation"]
    assert recovered[1] == ["June-EFY2016", "12.3", "8.1"]
    assert recovered[2] == ["June-EFY2017", "13.0", "9.4"]


def test_is_likely_tabular_rows_rejects_narrative_like_rows() -> None:
    rows = [
        ["Issue No. / Page", "24", "2017", "9"],
        [
            "Recent Improvements in CPI Price survey ... long narrative paragraph that is not tabular data.",
            "1963",
            "1955",
            "2000",
        ],
    ]

    assert strategy_b._is_likely_tabular_rows(rows) is False
