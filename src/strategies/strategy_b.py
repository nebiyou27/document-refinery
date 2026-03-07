"""Strategy B extraction using Docling (layout-aware)."""

from __future__ import annotations

import hashlib
import math
import os
import time
from collections.abc import Iterable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import fitz
import pdfplumber

from src.models.extracted_document import (
    ExtractedDocument,
    ExtractedPage,
    ExtractionMetadata,
    TableBlock,
    TextBlock,
)
from src.utils.hashing import content_hash


def _compute_doc_id(pdf_path: Path) -> str:
    return hashlib.md5(pdf_path.read_bytes()).hexdigest()[:12]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if math.isnan(out):
            return default
        return out
    except (TypeError, ValueError):
        return default


def _page_signals_from_pdf(pdf_path: Path) -> list[dict[str, float | int]]:
    signals: list[dict[str, float | int]] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            area = max((page.width or 1.0) * (page.height or 1.0), 1.0)
            text = page.extract_text() or ""
            char_count = len(text)
            image_area = sum(
                (img.get("width", 0.0) or 0.0) * (img.get("height", 0.0) or 0.0)
                for img in (page.images or [])
            )
            try:
                table_count = len(page.extract_tables() or [])
            except Exception:
                table_count = 0
            signals.append(
                {
                    "char_count": char_count,
                    "char_density": round(char_count / area, 6),
                    "image_area_ratio": round(min(image_area / area, 1.0), 4),
                    "table_count": int(table_count),
                }
            )
    return signals


def _import_docling():
    from docling.document_converter import DocumentConverter  # type: ignore

    return DocumentConverter


def _import_docling_pdf_options():
    from docling.datamodel.accelerator_options import AcceleratorOptions  # type: ignore
    from docling.datamodel.base_models import InputFormat  # type: ignore
    from docling.datamodel.pipeline_options import PdfPipelineOptions  # type: ignore
    from docling.document_converter import PdfFormatOption  # type: ignore

    return AcceleratorOptions, InputFormat, PdfFormatOption, PdfPipelineOptions


def _build_docling_converter():
    DocumentConverter = _import_docling()
    device = os.getenv("DOC_REFINERY_DOCLING_DEVICE", "").strip().lower()
    if device != "cuda":
        return DocumentConverter()

    AcceleratorOptions, InputFormat, PdfFormatOption, PdfPipelineOptions = _import_docling_pdf_options()
    pipeline_options = PdfPipelineOptions(
        accelerator_options=AcceleratorOptions(device="cuda")
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )


def _table_rows_from_item(item: Any, doc: Any) -> list[list[str]]:
    try:
        df = item.export_to_dataframe(doc=doc)
        if hasattr(df, "fillna") and hasattr(df, "astype"):
            filled = df.fillna("").astype(str)
            rows: list[list[str]] = [list(filled.columns.astype(str))]
            rows.extend(filled.values.tolist())
            return [[str(c) for c in row] for row in rows]
    except Exception:
        pass

    data = getattr(item, "data", None)
    grid = getattr(data, "grid", None)
    if not grid:
        return []

    rows: list[list[str]] = []
    for row in grid:
        cells: list[str] = []
        for cell in row or []:
            text = ""
            getter = getattr(cell, "_get_text", None)
            if callable(getter):
                try:
                    text = str(getter(doc=doc))
                except Exception:
                    text = ""
            cells.append((text or "").strip())
        rows.append(cells)
    return rows


def _bbox_from_item(item: Any) -> tuple[float, float, float, float]:
    prov = getattr(item, "prov", None) or []
    if not prov:
        return (0.0, 0.0, 1.0, 1.0)
    bbox = getattr(prov[0], "bbox", None)
    if bbox is None:
        return (0.0, 0.0, 1.0, 1.0)
    as_tuple = getattr(bbox, "as_tuple", None)
    if callable(as_tuple):
        raw = as_tuple()
    else:
        raw = (
            getattr(bbox, "l", 0.0),
            getattr(bbox, "t", 0.0),
            getattr(bbox, "r", 1.0),
            getattr(bbox, "b", 1.0),
        )
    x0, y0, x1, y1 = (_safe_float(v, 0.0) for v in raw)
    if x1 <= x0:
        x1 = x0 + 1.0
    if y1 <= y0:
        y1 = y0 + 1.0
    return (x0, y0, x1, y1)


class DoclingDocumentAdapter:
    """Adapter to normalize Docling output into ExtractedPage records."""

    def __init__(self, *, doc_id: str, file_name: str) -> None:
        self.doc_id = doc_id
        self.file_name = file_name

    def adapt(
        self,
        *,
        conversion_result: Any,
        per_page_signals: list[dict[str, float | int]],
    ) -> list[ExtractedPage]:
        page_count = len(per_page_signals)
        doc = getattr(conversion_result, "document", None)
        confidence_report = getattr(conversion_result, "confidence", None)
        page_conf = getattr(confidence_report, "pages", {}) if confidence_report else {}

        text_blocks_by_page: dict[int, list[TextBlock]] = {i: [] for i in range(1, page_count + 1)}
        table_blocks_by_page: dict[int, list[TableBlock]] = {i: [] for i in range(1, page_count + 1)}

        for page_no in range(1, page_count + 1):
            if doc is None:
                continue
            try:
                items = doc.iterate_items(page_no=page_no, with_groups=False)
            except Exception:
                items = []

            text_index = 0
            table_index = 0
            for item, _ in items:
                label = str(getattr(item, "label", "")).lower()
                if hasattr(item, "data") and "table" in label:
                    rows = _table_rows_from_item(item, doc)
                    serialized = "\n".join(" | ".join(r) for r in rows)
                    table_blocks_by_page[page_no].append(
                        TableBlock(
                            doc_id=self.doc_id,
                            page_number=page_no,
                            bbox=_bbox_from_item(item),
                            content_hash=content_hash(serialized or f"table:{page_no}:{table_index}"),
                            table_index=table_index,
                            rows=rows,
                        )
                    )
                    table_index += 1
                    continue

                text = (getattr(item, "text", None) or "").strip()
                if not text:
                    continue
                text_blocks_by_page[page_no].append(
                    TextBlock(
                        doc_id=self.doc_id,
                        page_number=page_no,
                        text=text,
                        bbox=_bbox_from_item(item),
                        reading_order=text_index,
                        content_hash=content_hash(text),
                    )
                )
                text_index += 1

        pages: list[ExtractedPage] = []
        for page_no in range(1, page_count + 1):
            signals = per_page_signals[page_no - 1]
            text_blocks = text_blocks_by_page[page_no]
            table_blocks = table_blocks_by_page[page_no]
            merged_text = " ".join(block.text for block in text_blocks).strip()
            page_hash = content_hash(
                merged_text
                or "\n".join(
                    " | ".join(row) for tb in table_blocks for row in tb.rows
                )
                or f"page:{page_no}:empty"
            )

            score = 0.0
            page_conf_item = page_conf.get(page_no) if isinstance(page_conf, dict) else None
            if page_conf_item is not None:
                score = _safe_float(getattr(page_conf_item, "mean_score", 0.0), 0.0)
            if score <= 0.0:
                has_text = 1.0 if int(signals.get("char_count", 0)) > 0 else 0.0
                score = min(1.0, 0.45 + (0.45 * has_text))

            table_count = len(table_blocks)
            enriched_signals = dict(signals)
            enriched_signals["table_count"] = table_count

            pages.append(
                ExtractedPage(
                    doc_id=self.doc_id,
                    page_number=page_no,
                    status="ok",
                    text=merged_text,
                    tables=[{"table_index": t.table_index, "rows": t.rows} for t in table_blocks],
                    metadata=ExtractionMetadata(
                        strategy_used="strategy_b",
                        confidence_score=round(max(0.0, min(1.0, score)), 3),
                        processing_time_sec=0.0,
                        cost_estimate_usd=0.0,
                        escalation_triggered=False,
                        escalation_target=None,
                    ),
                    signals=enriched_signals,
                    text_blocks=text_blocks,
                    table_blocks=table_blocks,
                    figure_blocks=[],
                    page_content_hash=page_hash,
                )
            )

        return pages


def _chunked(items: list[int], size: int) -> Iterable[list[int]]:
    for i in range(0, len(items), max(1, size)):
        yield items[i : i + max(1, size)]


def _write_single_page_pdf(
    source_pdf: Path,
    *,
    page_number: int,
    output_pdf: Path,
) -> None:
    page_index = page_number - 1
    src_doc = fitz.open(source_pdf)
    try:
        if page_index < 0 or page_index >= src_doc.page_count:
            raise ValueError(f"Invalid page_number={page_number} for {source_pdf.name}")
        page_doc = fitz.open()
        try:
            page_doc.insert_pdf(src_doc, from_page=page_index, to_page=page_index)
            page_doc.save(output_pdf)
        finally:
            page_doc.close()
    finally:
        src_doc.close()


def _remap_page_identity(
    *,
    page: ExtractedPage,
    doc_id: str,
    original_page_number: int,
) -> ExtractedPage:
    page.doc_id = doc_id
    page.page_number = original_page_number
    for block in page.text_blocks:
        block.doc_id = doc_id
        block.page_number = original_page_number
    for block in page.table_blocks:
        block.doc_id = doc_id
        block.page_number = original_page_number
    for block in page.figure_blocks:
        block.doc_id = doc_id
        block.page_number = original_page_number
    return page


def _error_page(
    *,
    doc_id: str,
    page_number: int,
    signals: dict[str, float | int],
    message: str,
    processing_time_sec: float,
) -> ExtractedPage:
    return ExtractedPage(
        doc_id=doc_id,
        page_number=page_number,
        status="error",
        text="",
        tables=[],
        metadata=ExtractionMetadata(
            strategy_used="strategy_b",
            confidence_score=0.0,
            processing_time_sec=processing_time_sec,
            cost_estimate_usd=0.0,
            escalation_triggered=True,
            escalation_target="strategy_c",
        ),
        signals=signals,
        text_blocks=[],
        table_blocks=[],
        figure_blocks=[],
        page_content_hash=content_hash(f"{message}:{page_number}"),
        error_message=message,
    )


def extract_pages_with_docling(
    pdf_path: Path,
    page_numbers: list[int],
    rules: dict,
    *,
    batch_size: int = 5,
) -> dict[int, ExtractedPage]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    _ = rules  # accepted by contract
    if not page_numbers:
        return {}

    page_signals = _page_signals_from_pdf(pdf_path)
    page_count = len(page_signals)
    normalized_pages = sorted(set(int(p) for p in page_numbers))
    invalid_pages = [p for p in normalized_pages if p < 1 or p > page_count]
    if invalid_pages:
        raise ValueError(
            f"Invalid page numbers for {pdf_path.name}: {invalid_pages} (page_count={page_count})"
        )

    doc_id = _compute_doc_id(pdf_path)
    file_name = pdf_path.name
    converter = _build_docling_converter()
    adapter = DoclingDocumentAdapter(doc_id=doc_id, file_name=file_name)

    pages_out: dict[int, ExtractedPage] = {}
    with TemporaryDirectory(prefix="strategy_b_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for page_group in _chunked(normalized_pages, batch_size):
            for page_number in page_group:
                page_start = time.perf_counter()
                page_signal = dict(page_signals[page_number - 1])
                single_page_pdf = tmp_root / f"{doc_id}_p{page_number}.pdf"
                try:
                    _write_single_page_pdf(
                        pdf_path,
                        page_number=page_number,
                        output_pdf=single_page_pdf,
                    )
                    conversion_result = converter.convert(single_page_pdf, raises_on_error=False)
                    adapted_pages = adapter.adapt(
                        conversion_result=conversion_result,
                        per_page_signals=[page_signal],
                    )
                    if not adapted_pages:
                        raise RuntimeError("Docling returned no pages")
                    page = adapted_pages[0]
                    page.metadata.processing_time_sec = time.perf_counter() - page_start
                    pages_out[page_number] = _remap_page_identity(
                        page=page,
                        doc_id=doc_id,
                        original_page_number=page_number,
                    )
                except Exception as exc:
                    elapsed = time.perf_counter() - page_start
                    message = f"strategy_b_docling_failed: {exc}"
                    pages_out[page_number] = _error_page(
                        doc_id=doc_id,
                        page_number=page_number,
                        signals=page_signal,
                        message=message,
                        processing_time_sec=elapsed,
                    )
    return pages_out


def extract_with_docling(pdf_path: Path, rules: dict) -> ExtractedDocument:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc_id = _compute_doc_id(pdf_path)
    doc_start = time.perf_counter()
    page_count = len(_page_signals_from_pdf(pdf_path))
    page_numbers = list(range(1, page_count + 1))
    pages_by_number = extract_pages_with_docling(
        pdf_path=pdf_path,
        page_numbers=page_numbers,
        rules=rules,
        batch_size=5,
    )
    pages = [pages_by_number[p] for p in page_numbers]

    avg_confidence = sum(page.metadata.confidence_score for page in pages) / max(len(pages), 1)
    all_error = all(page.status == "error" for page in pages) if pages else True
    return ExtractedDocument(
        doc_id=doc_id,
        file_name=pdf_path.name,
        file_path=str(pdf_path),
        page_count=page_count,
        status="error" if all_error else "ok",
        metadata=ExtractionMetadata(
            strategy_used="strategy_b",
            confidence_score=round(avg_confidence, 3),
            processing_time_sec=time.perf_counter() - doc_start,
            cost_estimate_usd=0.0,
            escalation_triggered=all_error,
            escalation_target="strategy_c" if all_error else None,
        ),
        pages=pages,
        error_message="strategy_b_failed" if all_error else None,
    )
