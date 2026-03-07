from __future__ import annotations

from pathlib import Path

import pytest

from src.ocr import tesseract_ocr


def test_preprocess_image_supports_threshold_modes() -> None:
    Image = pytest.importorskip("PIL.Image")

    image = Image.new("RGB", (12, 12), color="white")
    grayscale = tesseract_ocr.preprocess_image(image, "grayscale")
    threshold = tesseract_ocr.preprocess_image(image, "threshold")
    adaptive = tesseract_ocr.preprocess_image(image, "adaptive_threshold")

    assert grayscale.size == (12, 12)
    assert threshold.size == (12, 12)
    assert adaptive.size == (12, 12)


def test_ocr_path_on_image_uses_pytesseract_and_returns_boxes(monkeypatch, tmp_path: Path) -> None:
    Image = pytest.importorskip("PIL.Image")

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (20, 10), color="white").save(image_path)

    class _Output:
        DICT = object()

    class _FakePyTesseractModule:
        Output = _Output

        class pytesseract:
            tesseract_cmd = ""

        @staticmethod
        def image_to_string(image, *, lang: str, config: str) -> str:
            _ = image, config
            assert lang == "amh+eng"
            return "ገቢ Revenue"

        @staticmethod
        def image_to_data(image, *, lang: str, config: str, output_type: object) -> dict[str, list[object]]:
            _ = image, config, output_type
            assert lang == "amh+eng"
            return {
                "text": ["ገቢ", "Revenue", ""],
                "conf": ["88", "92", "-1"],
                "left": [1, 40, 0],
                "top": [2, 2, 0],
                "width": [20, 55, 0],
                "height": [8, 8, 0],
                "line_num": [1, 1, 0],
                "block_num": [1, 1, 0],
                "par_num": [1, 1, 0],
                "word_num": [1, 2, 0],
            }

    monkeypatch.setattr(tesseract_ocr, "_require_pytesseract", lambda: _FakePyTesseractModule)

    result = tesseract_ocr.ocr_path(
        image_path,
        lang="amh+eng",
        preprocess="grayscale",
        include_boxes=True,
    )

    assert result.source_type == "image"
    assert len(result.pages) == 1
    assert result.pages[0].text == "ገቢ Revenue"
    assert len(result.pages[0].boxes) == 2
    assert result.pages[0].mean_confidence == 90.0


def test_ocr_path_computes_mean_confidence_without_boxes(monkeypatch, tmp_path: Path) -> None:
    Image = pytest.importorskip("PIL.Image")

    image_path = tmp_path / "sample.png"
    Image.new("RGB", (20, 10), color="white").save(image_path)
    seen_configs: list[str] = []

    class _Output:
        DICT = object()

    class _FakePyTesseractModule:
        Output = _Output

        class pytesseract:
            tesseract_cmd = ""

        @staticmethod
        def image_to_string(image, *, lang: str, config: str) -> str:
            _ = image, lang
            seen_configs.append(config)
            return "Revenue"

        @staticmethod
        def image_to_data(image, *, lang: str, config: str, output_type: object) -> dict[str, list[object]]:
            _ = image, lang, output_type
            seen_configs.append(config)
            return {
                "text": ["Revenue", ""],
                "conf": ["87", "-1"],
                "left": [1, 0],
                "top": [2, 0],
                "width": [20, 0],
                "height": [8, 0],
                "line_num": [1, 0],
                "block_num": [1, 0],
                "par_num": [1, 0],
                "word_num": [1, 0],
            }

    monkeypatch.setattr(tesseract_ocr, "_require_pytesseract", lambda: _FakePyTesseractModule)

    result = tesseract_ocr.ocr_path(
        image_path,
        lang="eng",
        preprocess="grayscale",
        include_boxes=False,
        psm=6,
    )

    assert result.pages[0].boxes == []
    assert result.pages[0].mean_confidence == 87.0
    assert seen_configs == ["--psm 6", "--psm 6"]


def test_boxes_mean_confidence_ignores_negative_layout_confidence() -> None:
    boxes, mean_confidence = tesseract_ocr._boxes_from_data(
        {
            "text": ["Revenue", "Cost", "", "", ""],
            "conf": ["88", "92", "-1", "-1", "-1"],
            "left": [1, 40, 0, 0, 0],
            "top": [2, 2, 0, 0, 0],
            "width": [20, 55, 0, 0, 0],
            "height": [8, 8, 0, 0, 0],
            "line_num": [1, 1, 0, 0, 0],
            "block_num": [1, 1, 0, 0, 0],
            "par_num": [1, 1, 0, 0, 0],
            "word_num": [1, 2, 0, 0, 0],
        }
    )

    assert len(boxes) == 2
    assert mean_confidence == 90.0


def test_resolve_tesseract_config_appends_psm() -> None:
    assert tesseract_ocr._resolve_tesseract_config("", 6) == "--psm 6"
    assert tesseract_ocr._resolve_tesseract_config("--oem 1", 6) == "--oem 1 --psm 6"


def test_ocr_path_uses_pdf2image_for_pdf(monkeypatch, tmp_path: Path) -> None:
    Image = pytest.importorskip("PIL.Image")

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-FAKE")
    converted_calls: list[tuple[int, int, int]] = []

    def _fake_convert_from_path(
        path: str,
        *,
        dpi: int,
        first_page: int,
        last_page: int,
        fmt: str,
        poppler_path: str | None,
    ) -> list[object]:
        _ = path, fmt, poppler_path
        converted_calls.append((dpi, first_page, last_page))
        return [
            Image.new("RGB", (10, 10), color="white"),
            Image.new("RGB", (10, 10), color="white"),
        ]

    monkeypatch.setattr(tesseract_ocr, "_require_pdf2image", lambda: _fake_convert_from_path)
    monkeypatch.setattr(
        tesseract_ocr,
        "_ocr_image",
        lambda image, **kwargs: tesseract_ocr.OCRPageResult(
            page_number=kwargs["page_number"],
            text=f"page-{kwargs['page_number']}",
            boxes=[],
            mean_confidence=0.0,
            image_size=image.size,
        ),
    )

    result = tesseract_ocr.ocr_path(pdf_path, first_page=2, last_page=3, dpi=240)

    assert converted_calls == [(240, 2, 3)]
    assert [page.page_number for page in result.pages] == [2, 3]
    assert result.text == "page-2\n\npage-3"
