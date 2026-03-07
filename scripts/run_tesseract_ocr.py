"""Small CLI for local Tesseract OCR smoke tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ocr.tesseract_ocr import DEFAULT_LANG, DEFAULT_TESSERACT_CMD, ocr_path


def _trim(text: str, limit: int) -> str:
    compact = " ".join((text or "").split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _print_safe(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((text + "\n").encode("utf-8", errors="replace"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local Tesseract OCR on a PDF or image.")
    parser.add_argument("input_path", type=Path, help="PDF or image path")
    parser.add_argument("--first-page", type=int, default=1, help="First PDF page to OCR")
    parser.add_argument("--last-page", type=int, default=None, help="Last PDF page to OCR")
    parser.add_argument(
        "--lang",
        default=DEFAULT_LANG,
        help=f"OCR languages. Default: {DEFAULT_LANG}",
    )
    parser.add_argument(
        "--tesseract-cmd",
        default=DEFAULT_TESSERACT_CMD,
        help=f"Path to tesseract.exe. Default: {DEFAULT_TESSERACT_CMD}",
    )
    parser.add_argument(
        "--preprocess",
        choices=["none", "grayscale", "threshold", "adaptive_threshold"],
        default="grayscale",
        help="Simple preprocessing mode",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PDF rasterization DPI")
    parser.add_argument("--config", default="", help="Extra Tesseract config string")
    parser.add_argument(
        "--include-boxes",
        action="store_true",
        help="Include OCR boxes and confidence from image_to_data",
    )
    parser.add_argument(
        "--max-preview-chars",
        type=int,
        default=600,
        help="Characters to print per page preview",
    )
    parser.add_argument(
        "--poppler-path",
        default=None,
        help="Optional Poppler bin path for pdf2image on Windows",
    )
    args = parser.parse_args()

    try:
        result = ocr_path(
            args.input_path,
            lang=args.lang,
            tesseract_cmd=args.tesseract_cmd,
            preprocess=args.preprocess,
            dpi=args.dpi,
            first_page=args.first_page,
            last_page=args.last_page,
            include_boxes=args.include_boxes,
            config=args.config,
            poppler_path=args.poppler_path,
        )
    except Exception as exc:
        print(f"status=error message={exc}")
        if args.input_path.suffix.lower() == ".pdf" and "poppler" in str(exc).lower():
            print(
                "hint=pdf2image needs Poppler on Windows. Install it and pass --poppler-path "
                "\"C:\\path\\to\\poppler\\Library\\bin\" or add it to PATH."
            )
        return 1

    summary = {
        "input_path": str(result.input_path),
        "source_type": result.source_type,
        "lang": result.lang,
        "preprocess": result.preprocess,
        "pages": len(result.pages),
    }
    _print_safe(json.dumps(summary, indent=2, ensure_ascii=False))
    for page in result.pages:
        _print_safe(f"\n=== Page {page.page_number} ===")
        _print_safe(f"mean_confidence={page.mean_confidence}")
        _print_safe(_trim(page.text, args.max_preview_chars))
        if args.include_boxes:
            preview_boxes = [
                {
                    "text": box.text,
                    "confidence": box.confidence,
                    "left": box.left,
                    "top": box.top,
                    "width": box.width,
                    "height": box.height,
                }
                for box in page.boxes[:10]
            ]
            _print_safe(json.dumps({"box_preview": preview_boxes}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
