"""Minimal local Tesseract OCR adapter for PDFs and images."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
DEFAULT_LANG = "amh+eng"


@dataclass(slots=True)
class OCRBox:
    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int
    line_num: int
    block_num: int
    par_num: int
    word_num: int


@dataclass(slots=True)
class OCRPageResult:
    page_number: int
    text: str
    boxes: list[OCRBox]
    mean_confidence: float
    image_size: tuple[int, int]


@dataclass(slots=True)
class OCRResult:
    input_path: Path
    source_type: str
    lang: str
    preprocess: str
    pages: list[OCRPageResult]

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text.strip()).strip()


def _require_pillow() -> Any:
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required for local OCR helper") from exc
    return Image


def _require_pdf2image() -> Any:
    try:
        from pdf2image import convert_from_path
    except ModuleNotFoundError as exc:
        raise RuntimeError("pdf2image is required for PDF OCR helper") from exc
    return convert_from_path


def _require_cv2() -> Any:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("opencv-python is required for OCR preprocessing") from exc
    return cv2


def _require_pytesseract() -> Any:
    try:
        import pytesseract
    except ModuleNotFoundError as exc:
        raise RuntimeError("pytesseract is required for local OCR helper") from exc
    return pytesseract


def _configure_tesseract(tesseract_cmd: str) -> Any:
    pytesseract = _require_pytesseract()
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    return pytesseract


def _normalize_preprocess(preprocess: str) -> str:
    value = (preprocess or "none").strip().lower()
    allowed = {"none", "grayscale", "threshold", "adaptive_threshold"}
    if value not in allowed:
        raise ValueError(f"Unsupported preprocess mode: {preprocess}. Expected one of {sorted(allowed)}")
    return value


def _pil_to_cv_array(image: Any) -> Any:
    import numpy as np

    rgb_image = image.convert("RGB")
    rgb_array = np.array(rgb_image)
    return rgb_array[:, :, ::-1]


def preprocess_image(image: Any, preprocess: str) -> Any:
    """Apply the requested preprocessing to a PIL image."""
    Image = _require_pillow()
    cv2 = _require_cv2()
    mode = _normalize_preprocess(preprocess)

    if mode == "none":
        return image.convert("RGB")

    cv_image = _pil_to_cv_array(image)
    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    if mode == "grayscale":
        return Image.fromarray(gray)
    if mode == "threshold":
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return Image.fromarray(thresh)

    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )
    return Image.fromarray(adaptive)


def _coerce_int(values: dict[str, Any], key: str, index: int) -> int:
    raw = values.get(key, [])
    if not isinstance(raw, list) or index >= len(raw):
        return 0
    try:
        return int(float(raw[index]))
    except (TypeError, ValueError):
        return 0


def _coerce_float(values: dict[str, Any], key: str, index: int) -> float:
    raw = values.get(key, [])
    if not isinstance(raw, list) or index >= len(raw):
        return -1.0
    try:
        return float(raw[index])
    except (TypeError, ValueError):
        return -1.0


def _resolve_tesseract_config(config: str, psm: int | None) -> str:
    parts: list[str] = []
    if config.strip():
        parts.append(config.strip())
    if psm is not None:
        parts.append(f"--psm {psm}")
    return " ".join(parts)


def _boxes_from_data(data: dict[str, Any], *, include_boxes: bool = True) -> tuple[list[OCRBox], float]:
    texts = data.get("text", [])
    if not isinstance(texts, list):
        return [], 0.0

    boxes: list[OCRBox] = []
    raw_confidences = data.get("conf", [])
    confidences = (
        [float(conf) for conf in raw_confidences if float(conf) >= 0]
        if isinstance(raw_confidences, list)
        else []
    )
    for index, raw_text in enumerate(texts):
        text = str(raw_text or "").strip()
        conf = _coerce_float(data, "conf", index)
        if not text:
            continue
        if not include_boxes:
            continue
        boxes.append(
            OCRBox(
                text=text,
                confidence=conf,
                left=_coerce_int(data, "left", index),
                top=_coerce_int(data, "top", index),
                width=_coerce_int(data, "width", index),
                height=_coerce_int(data, "height", index),
                line_num=_coerce_int(data, "line_num", index),
                block_num=_coerce_int(data, "block_num", index),
                par_num=_coerce_int(data, "par_num", index),
                word_num=_coerce_int(data, "word_num", index),
            )
        )
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return boxes, round(mean_conf, 2)


def _ocr_image(
    image: Any,
    *,
    page_number: int,
    lang: str,
    tesseract_cmd: str,
    preprocess: str,
    config: str,
    psm: int | None,
    include_boxes: bool,
) -> OCRPageResult:
    pytesseract = _configure_tesseract(tesseract_cmd)
    processed_image = preprocess_image(image, preprocess)
    effective_config = _resolve_tesseract_config(config, psm)
    text = pytesseract.image_to_string(processed_image, lang=lang, config=effective_config).strip()

    boxes: list[OCRBox] = []
    mean_confidence = 0.0
    data = pytesseract.image_to_data(
        processed_image,
        lang=lang,
        config=effective_config,
        output_type=pytesseract.Output.DICT,
    )
    boxes, mean_confidence = _boxes_from_data(data, include_boxes=include_boxes)

    return OCRPageResult(
        page_number=page_number,
        text=text,
        boxes=boxes,
        mean_confidence=mean_confidence,
        image_size=processed_image.size,
    )


def _pdf_page_images(
    pdf_path: Path,
    *,
    dpi: int,
    first_page: int,
    last_page: int,
    poppler_path: str | None,
) -> list[Any]:
    convert_from_path = _require_pdf2image()
    return convert_from_path(
        str(pdf_path),
        dpi=dpi,
        first_page=first_page,
        last_page=last_page,
        fmt="png",
        poppler_path=poppler_path,
    )


def ocr_path(
    input_path: str | Path,
    *,
    lang: str = DEFAULT_LANG,
    tesseract_cmd: str = DEFAULT_TESSERACT_CMD,
    preprocess: str = "grayscale",
    dpi: int = 300,
    first_page: int = 1,
    last_page: int | None = None,
    include_boxes: bool = False,
    config: str = "",
    psm: int | None = None,
    poppler_path: str | None = None,
) -> OCRResult:
    """Run local Tesseract OCR on a PDF or image path."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    suffix = path.suffix.lower()
    pages: list[OCRPageResult] = []
    if suffix == ".pdf":
        final_last_page = last_page or first_page
        if final_last_page < first_page:
            raise ValueError("last_page must be greater than or equal to first_page")
        images = _pdf_page_images(
            path,
            dpi=dpi,
            first_page=first_page,
            last_page=final_last_page,
            poppler_path=poppler_path,
        )
        for offset, image in enumerate(images):
            page_number = first_page + offset
            pages.append(
                _ocr_image(
                    image,
                    page_number=page_number,
                    lang=lang,
                    tesseract_cmd=tesseract_cmd,
                    preprocess=preprocess,
                    config=config,
                    psm=psm,
                    include_boxes=include_boxes,
                )
            )
        source_type = "pdf"
    else:
        Image = _require_pillow()
        image = Image.open(path)
        try:
            pages.append(
                _ocr_image(
                    image,
                    page_number=1,
                    lang=lang,
                    tesseract_cmd=tesseract_cmd,
                    preprocess=preprocess,
                    config=config,
                    psm=psm,
                    include_boxes=include_boxes,
                )
            )
        finally:
            image.close()
        source_type = "image"

    return OCRResult(
        input_path=path,
        source_type=source_type,
        lang=lang,
        preprocess=_normalize_preprocess(preprocess),
        pages=pages,
    )
