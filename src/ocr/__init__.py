"""Optional local OCR helpers."""

from src.ocr.tesseract_ocr import (
    OCRBox,
    OCRPageResult,
    OCRResult,
    ocr_path,
)

__all__ = ["OCRBox", "OCRPageResult", "OCRResult", "ocr_path"]
