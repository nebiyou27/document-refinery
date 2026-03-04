"""CLI entrypoint for Stage 2 extraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.extractor import run_extraction


def main() -> int:
    parser = argparse.ArgumentParser(description="Run extraction on a PDF.")
    parser.add_argument("pdf_path", type=Path, help="Path to PDF file")
    args = parser.parse_args()

    result = run_extraction(args.pdf_path)

    if result.status == "error":
        print(f"status=error doc_id={result.doc_id} message={result.error_message}")
        return 1

    total = len(result.pages)
    below_a = sum(1 for p in result.pages if p.metadata.escalation_triggered)
    pct = (below_a / total * 100.0) if total else 0.0
    print(f"doc_id={result.doc_id}")
    print(f"pages_processed={total}")
    print(f"pages_below_strategy_a_confidence={below_a}")
    print(f"planned_escalations_pct={pct:.2f}%")
    print(f"ledger=.refinery/extraction_ledger/{result.doc_id}.jsonl")
    print(f"extracted=.refinery/extracted/{result.doc_id}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

