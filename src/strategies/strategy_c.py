"""Strategy C extraction using OCR with VLM fallback via local Ollama."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import fitz

from src.models.extracted_document import (
    ExtractedPage,
    ExtractionMetadata,
    FigureBlock,
    TableBlock,
    TextBlock,
)
from src.utils.hashing import content_hash

_EASYOCR_READER: Any | None = None


def _compute_doc_id(pdf_path: Path) -> str:
    return hashlib.md5(pdf_path.read_bytes()).hexdigest()[:12]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out):
        return default
    return out


def _require(rules: dict, path: tuple[str, ...]) -> Any:
    node: Any = rules
    traversed: list[str] = []
    for part in path:
        traversed.append(part)
        if not isinstance(node, dict) or part not in node:
            joined = ".".join(path)
            at = ".".join(traversed[:-1]) or "<root>"
            raise KeyError(f"Missing config key '{joined}' (stopped at '{at}')")
        node = node[part]
    return node


def _strategy_c_cfg(rules: dict) -> dict[str, Any]:
    policy_node = (
        rules.get("strategy_routing", {})
        .get("strategy_c", {})
        .get("execution_policy", {})
        .get("failure_policy", {})
    )
    if not isinstance(policy_node, dict):
        policy_node = {}
    ocr_unavailable_policy = str(policy_node.get("ocr_unavailable", "vlm_only")).strip().lower()
    ocr_failure_policy = str(policy_node.get("ocr_failure", "vlm_only")).strip().lower()
    vlm_failure_policy = str(policy_node.get("vlm_failure", "error_page")).strip().lower()
    model = _require(
        rules,
        ("strategy_routing", "strategy_c", "tools", "fallback_vlm", "model"),
    )
    ocr_min_chars = _require(
        rules,
        ("strategy_routing", "strategy_c", "execution_policy", "ocr_min_chars_per_page"),
    )
    ocr_min_mean_conf = _require(
        rules,
        ("strategy_routing", "strategy_c", "execution_policy", "ocr_min_mean_confidence"),
    )
    max_pages = _require(
        rules,
        ("strategy_routing", "strategy_c", "budget_guard", "max_pages_per_document"),
    )
    max_vlm_pages = _require(
        rules,
        ("strategy_routing", "strategy_c", "budget_guard", "max_vlm_pages_per_document"),
    )
    max_runtime = _require(
        rules,
        ("strategy_routing", "strategy_c", "budget_guard", "max_total_runtime_seconds"),
    )
    cost_per_page = _require(
        rules,
        ("strategy_routing", "strategy_c", "budget_guard", "cost_per_page_estimate_usd"),
    )
    return {
        "vlm_model": str(model),
        "ocr_min_chars_per_page": int(ocr_min_chars),
        "ocr_min_mean_confidence": float(ocr_min_mean_conf),
        "max_pages_per_document": int(max_pages),
        "max_vlm_pages_per_document": int(max_vlm_pages),
        "max_total_runtime_seconds": float(max_runtime),
        "cost_per_page_estimate_usd": float(cost_per_page),
        "failure_policy_ocr_unavailable": ocr_unavailable_policy,
        "failure_policy_ocr_failure": ocr_failure_policy,
        "failure_policy_vlm_failure": vlm_failure_policy,
    }


def _get_easyocr_reader() -> Any:
    global _EASYOCR_READER
    if _EASYOCR_READER is not None:
        return _EASYOCR_READER
    try:
        import easyocr  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("easyocr is required for strategy_c") from exc
    _EASYOCR_READER = easyocr.Reader(["en"], gpu=False)
    return _EASYOCR_READER


def _write_single_page_image(
    source_pdf: Path,
    *,
    page_number: int,
    output_image: Path,
    dpi: int = 220,
) -> tuple[float, float]:
    src = fitz.open(source_pdf)
    try:
        page_index = page_number - 1
        if page_index < 0 or page_index >= src.page_count:
            raise ValueError(f"Invalid page_number={page_number} for {source_pdf.name}")
        page = src.load_page(page_index)
        zoom = max(dpi / 72.0, 1.0)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        pix.save(output_image)
        return (float(page.rect.width), float(page.rect.height))
    finally:
        src.close()


def _bbox_from_easyocr(box: Any) -> tuple[float, float, float, float]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return (0.0, 0.0, 1.0, 1.0)
    xs: list[float] = []
    ys: list[float] = []
    for point in box:
        if isinstance(point, (list, tuple)) and len(point) == 2:
            xs.append(_safe_float(point[0], 0.0))
            ys.append(_safe_float(point[1], 0.0))
    if not xs or not ys:
        return (0.0, 0.0, 1.0, 1.0)
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 <= x0:
        x1 = x0 + 1.0
    if y1 <= y0:
        y1 = y0 + 1.0
    return (x0, y0, x1, y1)


def _ocr_extract(
    *,
    image_path: Path,
    doc_id: str,
    page_number: int,
) -> tuple[str, list[TextBlock], float]:
    reader = _get_easyocr_reader()
    rows = reader.readtext(str(image_path), detail=1) or []

    text_blocks: list[TextBlock] = []
    text_parts: list[str] = []
    confs: list[float] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, (list, tuple)) or len(row) < 3:
            continue
        text = str(row[1] or "").strip()
        if not text:
            continue
        conf = _safe_float(row[2], 0.0)
        confs.append(conf)
        text_parts.append(text)
        text_blocks.append(
            TextBlock(
                doc_id=doc_id,
                page_number=page_number,
                text=text,
                bbox=_bbox_from_easyocr(row[0]),
                reading_order=idx,
                content_hash=content_hash(text),
            )
        )
    merged = " ".join(text_parts).strip()
    mean_conf = (sum(confs) / len(confs)) if confs else 0.0
    return merged, text_blocks, max(0.0, min(1.0, mean_conf))


def _vlm_extract(
    *,
    image_path: Path,
    model: str,
) -> str:
    try:
        import requests
    except ModuleNotFoundError as exc:
        raise RuntimeError("requests is required for strategy_c VLM fallback") from exc

    image_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {
                "role": "user",
                "content": (
                    "You are extracting content from one document page. "
                    "Return STRICT JSON only, with no markdown and no commentary. "
                    "Schema: {\"plain_text\": string, \"bullets\": string[], \"tables\": any[], \"figures\": any[]}. "
                    "Rules: plain_text must contain only extracted readable text; "
                    "no labels like Title/Subtitle/Body Text; "
                    "no notes about images/background/layout; "
                    "bullets must be concise extracted list items; "
                    "tables and figures are optional arrays (use [] when none)."
                ),
                "images": [image_b64],
            }
        ],
    }
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    body = resp.json()
    message = body.get("message", {}) if isinstance(body, dict) else {}
    text = message.get("content", "") if isinstance(message, dict) else ""
    return str(text or "").strip()


def _strip_code_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return text


def _parse_vlm_json(raw_text: str) -> dict[str, Any]:
    fallback = {
        "plain_text": raw_text.strip(),
        "bullets": [],
        "tables": [],
        "figures": [],
    }
    candidate = _strip_code_fences(raw_text)
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return fallback
    if not isinstance(payload, dict):
        return fallback

    plain_text = payload.get("plain_text", "")
    bullets = payload.get("bullets", [])
    tables = payload.get("tables", [])
    figures = payload.get("figures", [])

    norm_plain_text = str(plain_text or "").strip()
    norm_bullets = [str(item).strip() for item in bullets if str(item or "").strip()] if isinstance(bullets, list) else []
    norm_tables = tables if isinstance(tables, list) else []
    norm_figures = figures if isinstance(figures, list) else []
    return {
        "plain_text": norm_plain_text,
        "bullets": norm_bullets,
        "tables": norm_tables,
        "figures": norm_figures,
    }


def _render_structured_text(*, plain_text: str, bullets: list[str]) -> str:
    parts: list[str] = []
    if plain_text:
        parts.append(plain_text)
    if bullets:
        parts.extend(f"- {item}" for item in bullets)
    return "\n".join(parts).strip()


def _figure_caption(figure_payload: Any) -> str | None:
    if isinstance(figure_payload, str):
        text = figure_payload.strip()
        return text or None
    if isinstance(figure_payload, dict):
        for key in ("caption", "title", "text", "description"):
            value = figure_payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _table_rows_from_payload(table_payload: Any) -> list[list[str]]:
    if isinstance(table_payload, list):
        rows: list[list[str]] = []
        for row in table_payload:
            if isinstance(row, list):
                rows.append([str(cell or "").strip() for cell in row])
        return rows

    if not isinstance(table_payload, dict):
        return []

    normalized_rows: list[list[str]] = []
    columns = table_payload.get("columns")
    if isinstance(columns, list):
        header = [str(cell or "").strip() for cell in columns]
        if any(header):
            normalized_rows.append(header)

    rows = table_payload.get("rows")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, list):
                normalized_rows.append([str(cell or "").strip() for cell in row])

    return normalized_rows


def _compute_confidence(
    *,
    char_count: int,
    ocr_mean_confidence: float,
    used_vlm: bool,
    ocr_min_chars: int,
) -> float:
    chars_factor = min(1.0, float(char_count) / max(float(ocr_min_chars), 1.0))
    if used_vlm:
        # VLM fallback generally improves text recovery when OCR is weak.
        score = 0.55 + (0.35 * chars_factor) + (0.10 * ocr_mean_confidence)
    else:
        score = (0.45 * chars_factor) + (0.55 * ocr_mean_confidence)
    return round(max(0.0, min(1.0, score)), 3)


def _is_easyocr_unavailable_error(exc: Exception) -> bool:
    text = str(exc).strip().lower()
    return "easyocr is required for strategy_c" in text


def _error_page(
    *,
    doc_id: str,
    page_number: int,
    processing_time_sec: float,
    message: str,
    char_count: int = 0,
    ocr_mean_confidence: float = 0.0,
    vlm_used: bool = False,
    vlm_wall_time_sec: float = 0.0,
    vlm_calls: int = 0,
) -> ExtractedPage:
    return ExtractedPage(
        doc_id=doc_id,
        page_number=page_number,
        status="error",
        text="",
        tables=[],
        metadata=ExtractionMetadata(
            strategy_used="strategy_c",
            confidence_score=0.0,
            processing_time_sec=processing_time_sec,
            cost_estimate_usd=0.0,
            vlm_used=vlm_used,
            vlm_wall_time_sec=round(max(0.0, vlm_wall_time_sec), 4),
            escalation_triggered=False,
            escalation_target=None,
            bbox_precision=None,
        ),
        signals={
            "char_count": int(char_count),
            "ocr_mean_confidence": round(max(0.0, min(1.0, ocr_mean_confidence)), 3),
            "used_vlm": 1 if vlm_used else 0,
            "vlm_wall_time_sec": round(max(0.0, vlm_wall_time_sec), 4),
            "vlm_calls": int(max(0, vlm_calls)),
            "table_count": 0,
            "image_area_ratio": 1.0,
            "char_density": 0.0,
        },
        text_blocks=[],
        table_blocks=[],
        figure_blocks=[],
        page_content_hash=content_hash(f"strategy_c:error:{page_number}:{message}"),
        error_message=message,
    )


def extract_pages_with_vision(
    pdf_path: Path,
    page_numbers: list[int],
    rules: dict,
) -> dict[int, ExtractedPage]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not page_numbers:
        return {}

    cfg = _strategy_c_cfg(rules)
    normalized_pages = sorted(set(int(p) for p in page_numbers))
    if len(normalized_pages) > cfg["max_pages_per_document"]:
        processed = normalized_pages[: cfg["max_pages_per_document"]]
        skipped = normalized_pages[cfg["max_pages_per_document"] :]
    else:
        processed = normalized_pages
        skipped = []

    # Validate page numbers against source PDF.
    src = fitz.open(pdf_path)
    try:
        page_count = src.page_count
    finally:
        src.close()
    invalid = [p for p in processed if p < 1 or p > page_count]
    if invalid:
        raise ValueError(f"Invalid page numbers for {pdf_path.name}: {invalid} (page_count={page_count})")

    doc_id = _compute_doc_id(pdf_path)
    out: dict[int, ExtractedPage] = {}
    doc_start = time.perf_counter()
    vlm_pages_used = 0

    with TemporaryDirectory(prefix="strategy_c_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        stop_due_to_runtime = False
        stop_page_number: int | None = None
        for page_number in processed:
            elapsed_doc = time.perf_counter() - doc_start
            # Runtime cap is document-scoped: fail this page, then stop Strategy C for remaining pages.
            if elapsed_doc >= cfg["max_total_runtime_seconds"]:
                stop_due_to_runtime = True
                stop_page_number = page_number
                out[page_number] = _error_page(
                    doc_id=doc_id,
                    page_number=page_number,
                    processing_time_sec=0.0,
                    message="budget_exceeded: max_total_runtime_seconds",
                )
                break

            page_start = time.perf_counter()
            vlm_wall_time_sec = 0.0
            page_vlm_calls = 0
            image_path = tmp_root / f"{doc_id}_p{page_number}.png"
            try:
                page_width, page_height = _write_single_page_image(
                    pdf_path,
                    page_number=page_number,
                    output_image=image_path,
                )
                ocr_error: Exception | None = None
                try:
                    ocr_text, ocr_blocks, ocr_mean_conf = _ocr_extract(
                        image_path=image_path,
                        doc_id=doc_id,
                        page_number=page_number,
                    )
                except Exception as exc:
                    ocr_error = exc
                    ocr_text, ocr_blocks, ocr_mean_conf = "", [], 0.0
                ocr_char_count = len(ocr_text)
                is_weak_ocr = (
                    ocr_char_count < cfg["ocr_min_chars_per_page"]
                    or ocr_mean_conf < cfg["ocr_min_mean_confidence"]
                )
                if ocr_error is not None:
                    if _is_easyocr_unavailable_error(ocr_error):
                        fallback_policy = cfg["failure_policy_ocr_unavailable"]
                    else:
                        fallback_policy = cfg["failure_policy_ocr_failure"]
                    if fallback_policy != "vlm_only":
                        raise RuntimeError(f"ocr_failed_without_fallback: {ocr_error}") from ocr_error
                    is_weak_ocr = True

                used_vlm = False
                final_text = ocr_text
                text_blocks = ocr_blocks
                tables_payload: list[dict] = []
                table_blocks: list[TableBlock] = []
                figure_blocks: list[FigureBlock] = []
                bbox_precision = "block_level"
                if is_weak_ocr:
                    # When VLM page budget is exhausted, this page is marked as a budget error.
                    if vlm_pages_used >= cfg["max_vlm_pages_per_document"]:
                        out[page_number] = _error_page(
                            doc_id=doc_id,
                            page_number=page_number,
                            processing_time_sec=time.perf_counter() - page_start,
                            message="budget_exceeded: max_vlm_pages_per_document",
                            char_count=ocr_char_count,
                            ocr_mean_confidence=ocr_mean_conf,
                        )
                        continue
                    elapsed_doc = time.perf_counter() - doc_start
                    if elapsed_doc >= cfg["max_total_runtime_seconds"]:
                        stop_due_to_runtime = True
                        stop_page_number = page_number
                        out[page_number] = _error_page(
                            doc_id=doc_id,
                            page_number=page_number,
                            processing_time_sec=time.perf_counter() - page_start,
                            message="budget_exceeded: max_total_runtime_seconds",
                            char_count=ocr_char_count,
                            ocr_mean_confidence=ocr_mean_conf,
                        )
                        break

                    try:
                        vlm_start = time.perf_counter()
                        vlm_text = _vlm_extract(image_path=image_path, model=cfg["vlm_model"])
                        vlm_wall_time_sec += time.perf_counter() - vlm_start
                        page_vlm_calls += 1
                    except Exception as vlm_exc:
                        page_vlm_calls += 1
                        vlm_wall_time_sec += max(0.0, time.perf_counter() - vlm_start)
                        if cfg["failure_policy_vlm_failure"] == "keep_ocr" and final_text:
                            vlm_text = ""
                        else:
                            raise RuntimeError(f"vlm_failed: {vlm_exc}") from vlm_exc
                    if vlm_text:
                        parsed = _parse_vlm_json(vlm_text)
                        final_text = _render_structured_text(
                            plain_text=parsed["plain_text"],
                            bullets=parsed["bullets"],
                        )
                        if not final_text:
                            final_text = str(vlm_text or "").strip()
                        text_blocks = [
                            TextBlock(
                                doc_id=doc_id,
                                page_number=page_number,
                                text=final_text,
                                bbox=(0.0, 0.0, page_width, page_height),
                                reading_order=0,
                                content_hash=content_hash(final_text),
                            )
                        ]
                        bbox_precision = "page_level"
                        tables_payload = [row for row in parsed["tables"] if isinstance(row, dict)]
                        table_blocks = [
                            TableBlock(
                                doc_id=doc_id,
                                page_number=page_number,
                                bbox=(0.0, 0.0, page_width, page_height),
                                content_hash=content_hash(json.dumps(tbl, sort_keys=True)),
                                table_index=idx,
                                rows=_table_rows_from_payload(tbl),
                            )
                            for idx, tbl in enumerate(tables_payload)
                        ]
                        figure_blocks = [
                            FigureBlock(
                                doc_id=doc_id,
                                page_number=page_number,
                                bbox=(0.0, 0.0, page_width, page_height),
                                content_hash=content_hash(
                                    json.dumps(fig, sort_keys=True, default=str)
                                ),
                                caption=_figure_caption(fig),
                            )
                            for fig in parsed["figures"]
                        ]
                    used_vlm = True
                    vlm_pages_used += 1

                char_count = len(final_text)
                if char_count == 0:
                    raise RuntimeError("empty_text_after_strategy_c_fallbacks")
                elapsed_page = time.perf_counter() - page_start
                confidence = _compute_confidence(
                    char_count=char_count,
                    ocr_mean_confidence=ocr_mean_conf,
                    used_vlm=used_vlm,
                    ocr_min_chars=cfg["ocr_min_chars_per_page"],
                )
                out[page_number] = ExtractedPage(
                    doc_id=doc_id,
                    page_number=page_number,
                    status="ok",
                    text=final_text,
                    tables=tables_payload,
                    metadata=ExtractionMetadata(
                        strategy_used="strategy_c",
                        confidence_score=confidence,
                        processing_time_sec=elapsed_page,
                        cost_estimate_usd=0.0,
                        vlm_used=used_vlm,
                        vlm_wall_time_sec=round(vlm_wall_time_sec, 4),
                        escalation_triggered=False,
                        escalation_target=None,
                        bbox_precision=bbox_precision,
                    ),
                    signals={
                        "char_count": char_count,
                        "ocr_mean_confidence": round(ocr_mean_conf, 3),
                        "ocr_block_count": len(ocr_blocks),
                        "used_vlm": 1 if used_vlm else 0,
                        "vlm_wall_time_sec": round(vlm_wall_time_sec, 4),
                        "vlm_calls": page_vlm_calls,
                        "table_count": len(table_blocks),
                        "image_area_ratio": 1.0,
                        "char_density": 0.0,
                    },
                    text_blocks=text_blocks,
                    table_blocks=table_blocks,
                    figure_blocks=figure_blocks,
                    page_content_hash=content_hash(final_text or f"strategy_c:page:{page_number}:empty"),
                )
            except Exception as exc:
                out[page_number] = _error_page(
                    doc_id=doc_id,
                    page_number=page_number,
                    processing_time_sec=time.perf_counter() - page_start,
                    message=f"strategy_c_failed: {exc}",
                    vlm_used=False,
                    vlm_wall_time_sec=vlm_wall_time_sec,
                    vlm_calls=page_vlm_calls,
                )
            finally:
                if image_path.exists():
                    image_path.unlink()

        if stop_due_to_runtime and stop_page_number is not None:
            for page_number in processed:
                if page_number <= stop_page_number:
                    continue
                if page_number not in out:
                    out[page_number] = _error_page(
                        doc_id=doc_id,
                        page_number=page_number,
                        processing_time_sec=0.0,
                        message="budget_exceeded: max_total_runtime_seconds",
                    )

        for page_number in skipped:
            out[page_number] = _error_page(
                doc_id=doc_id,
                page_number=page_number,
                processing_time_sec=0.0,
                message="budget_exceeded: max_pages_per_document",
            )

    return out
