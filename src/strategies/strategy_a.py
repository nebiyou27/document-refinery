"""Strategy A extraction using pdfplumber."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pdfplumber

from src.models.extracted_document import (
    ExtractedDocument,
    ExtractedPage,
    ExtractionMetadata,
    FigureBlock,
    TableBlock,
    TextBlock,
)
from src.utils.hashing import content_hash


def _compute_doc_id(pdf_path: Path) -> str:
    return hashlib.md5(pdf_path.read_bytes()).hexdigest()[:12]


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _page_signals(page: pdfplumber.page.Page) -> dict[str, float | int]:
    area = max((page.width or 1.0) * (page.height or 1.0), 1.0)
    text = page.extract_text() or ""
    chars = page.chars or []
    char_count = len(text)
    char_density = char_count / area
    image_area = sum((img.get("width", 0.0) or 0.0) * (img.get("height", 0.0) or 0.0) for img in (page.images or []))
    image_area_ratio = min(image_area / area, 1.0)
    try:
        tables = page.extract_tables() or []
        table_count = len(tables)
    except Exception:
        table_count = 0

    return {
        "char_count": char_count,
        "char_density": round(char_density, 6),
        "image_area_ratio": round(image_area_ratio, 4),
        "table_count": table_count,
    }


def _page_confidence(signals: dict[str, float | int], rules: dict) -> float:
    gates = rules["strategy_routing"]["strategy_a"]["confidence_gates"]
    min_char_density = float(gates["min_char_density"])
    max_image_area_ratio = float(gates["max_image_area_ratio"])

    char_density = float(signals["char_density"])
    image_area_ratio = float(signals["image_area_ratio"])
    has_text = 1.0 if int(signals["char_count"]) > 0 else 0.0

    density_factor = _clamp(char_density / max(min_char_density, 1e-12))
    image_factor = 1.0 - _clamp(image_area_ratio / max(max_image_area_ratio, 1e-12))
    score = (0.55 * density_factor) + (0.35 * image_factor) + (0.10 * has_text)

    if char_density < min_char_density:
        score *= 0.60
    if image_area_ratio > max_image_area_ratio:
        score *= 0.75
    return round(_clamp(score), 3)


def _extract_text_blocks(
    *,
    doc_id: str,
    page_number: int,
    page: pdfplumber.page.Page,
) -> list[TextBlock]:
    words = page.extract_words() or []
    blocks: list[TextBlock] = []
    for idx, word in enumerate(words):
        text = (word.get("text") or "").strip()
        if not text:
            continue
        x0 = float(word.get("x0", 0.0) or 0.0)
        top = float(word.get("top", 0.0) or 0.0)
        x1 = float(word.get("x1", x0) or x0)
        bottom = float(word.get("bottom", top) or top)
        if x1 <= x0:
            x1 = x0 + 1.0
        if bottom <= top:
            bottom = top + 1.0
        blocks.append(
            TextBlock(
                doc_id=doc_id,
                page_number=page_number,
                text=text,
                bbox=(x0, top, x1, bottom),
                reading_order=idx,
                content_hash=content_hash(text),
            )
        )
    return blocks


def _extract_table_blocks(
    *,
    doc_id: str,
    page_number: int,
    page: pdfplumber.page.Page,
) -> list[TableBlock]:
    table_blocks: list[TableBlock] = []
    tables = page.extract_tables() or []
    page_bbox = (0.0, 0.0, float(page.width or 1.0), float(page.height or 1.0))
    for idx, table in enumerate(tables):
        rows: list[list[str]] = []
        for row in table or []:
            rows.append([(cell or "").strip() for cell in (row or [])])
        serialized = "\n".join(" | ".join(r) for r in rows)
        table_blocks.append(
            TableBlock(
                doc_id=doc_id,
                page_number=page_number,
                bbox=page_bbox,
                content_hash=content_hash(serialized),
                table_index=idx,
                rows=rows,
            )
        )
    return table_blocks


def _extract_figure_blocks(
    *,
    doc_id: str,
    page_number: int,
    page: pdfplumber.page.Page,
) -> list[FigureBlock]:
    blocks: list[FigureBlock] = []
    for img in (page.images or []):
        x0 = float(img.get("x0", 0.0) or 0.0)
        y0 = float(img.get("y0", 0.0) or 0.0)
        x1 = float(img.get("x1", x0) or x0)
        y1 = float(img.get("y1", y0) or y0)
        if x1 <= x0:
            x1 = x0 + float(img.get("width", 1.0) or 1.0)
        if y1 <= y0:
            y1 = y0 + float(img.get("height", 1.0) or 1.0)
        blocks.append(
            FigureBlock(
                doc_id=doc_id,
                page_number=page_number,
                bbox=(x0, y0, x1, y1),
                content_hash=content_hash(f"figure:{page_number}:{x0}:{y0}:{x1}:{y1}"),
                caption=None,
            )
        )
    return blocks


def extract_with_pdfplumber(pdf_path: Path, rules: dict) -> ExtractedDocument:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc_id = _compute_doc_id(pdf_path)
    doc_start = time.perf_counter()
    pages: list[ExtractedPage] = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        for idx, page in enumerate(pdf.pages, start=1):
            page_start = time.perf_counter()
            try:
                signals = _page_signals(page)
                confidence = _page_confidence(signals, rules)
                text_blocks = _extract_text_blocks(doc_id=doc_id, page_number=idx, page=page)
                table_blocks = _extract_table_blocks(doc_id=doc_id, page_number=idx, page=page)
                figure_blocks = _extract_figure_blocks(doc_id=doc_id, page_number=idx, page=page)
                page_hash_seed = " ".join(block.text for block in text_blocks)
                page_hash = content_hash(page_hash_seed or f"page:{idx}:empty")
                page_cost = 0.0
                tables = [
                    {"table_index": tb.table_index, "rows": tb.rows}
                    for tb in table_blocks
                ]
                pages.append(
                    ExtractedPage(
                        doc_id=doc_id,
                        page_number=idx,
                        status="ok",
                        text=page_hash_seed,
                        tables=tables,
                        metadata=ExtractionMetadata(
                            strategy_used="strategy_a",
                            confidence_score=confidence,
                            processing_time_sec=time.perf_counter() - page_start,
                            cost_estimate_usd=page_cost,
                            escalation_triggered=False,
                            escalation_target=None,
                        ),
                        signals=signals,
                        text_blocks=text_blocks,
                        table_blocks=table_blocks,
                        figure_blocks=figure_blocks,
                        page_content_hash=page_hash,
                    )
                )
            except Exception as exc:
                error_message = f"page_extraction_failed: {exc}"
                pages.append(
                    ExtractedPage(
                        doc_id=doc_id,
                        page_number=idx,
                        status="error",
                        text="",
                        tables=[],
                        metadata=ExtractionMetadata(
                            strategy_used="strategy_a",
                            confidence_score=0.0,
                            processing_time_sec=time.perf_counter() - page_start,
                            cost_estimate_usd=0.0,
                            escalation_triggered=True,
                            escalation_target=rules["strategy_routing"]["strategy_a"]["escalation_target"],
                        ),
                        signals={
                            "char_count": 0,
                            "char_density": 0.0,
                            "image_area_ratio": 0.0,
                            "table_count": 0,
                        },
                        text_blocks=[],
                        table_blocks=[],
                        figure_blocks=[],
                        page_content_hash=content_hash(error_message),
                        error_message=error_message,
                    )
                )

    avg_confidence = sum(page.metadata.confidence_score for page in pages) / max(len(pages), 1)
    return ExtractedDocument(
        doc_id=doc_id,
        file_name=pdf_path.name,
        file_path=str(pdf_path),
        page_count=total_pages,
        status="ok",
        metadata=ExtractionMetadata(
            strategy_used="strategy_a",
            confidence_score=round(avg_confidence, 3),
            processing_time_sec=time.perf_counter() - doc_start,
            cost_estimate_usd=0.0,
            escalation_triggered=False,
            escalation_target=None,
        ),
        pages=pages,
    )
