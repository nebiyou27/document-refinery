"""Extraction ledger writer: one JSONL row per processed page."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def append_ledger_entry(
    doc_id: str,
    file_name: str,
    page_number: int,
    strategy_used: str,
    confidence: float,
    signals: dict[str, Any],
    cost_estimate: float,
    processing_time: float,
    escalated_to: str | None,
    ledger_root: Path | None = None,
) -> Path:
    if page_number < 1:
        raise ValueError("page_number must be >= 1")

    root = ledger_root or Path(".refinery/extraction_ledger")
    root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / f"{doc_id}.jsonl"

    row = {
        "doc_id": doc_id,
        "file_name": file_name,
        "page_number": page_number,
        "strategy_used": strategy_used,
        "confidence": round(confidence, 4),
        "signals": signals,
        "cost_estimate_usd": round(cost_estimate, 6),
        "processing_time_sec": round(processing_time, 4),
        "escalated_to": escalated_to,
    }
    with ledger_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return ledger_path
