"""CLI entrypoint for Stage 2 extraction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.extractor import run_extraction


def main() -> int:
    parser = argparse.ArgumentParser(description="Run extraction on a PDF.")
    parser.add_argument("pdf_path", type=Path, help="Path to PDF file")
    parser.add_argument(
        "--strategy",
        choices=["strategy_a", "strategy_b", "strategy_c"],
        default=None,
        help="Force extraction strategy (debug). Default runs A then page-level escalation.",
    )
    args = parser.parse_args()

    result = run_extraction(args.pdf_path, strategy=args.strategy)

    if result.status == "error":
        print(f"status=error doc_id={result.doc_id} message={result.error_message}")
        return 1

    total = len(result.pages)
    escalated = sum(1 for p in result.pages if p.metadata.escalation_triggered)
    final_strategy_a_pages = sum(1 for p in result.pages if p.metadata.strategy_used == "strategy_a")
    final_strategy_b_pages = sum(1 for p in result.pages if p.metadata.strategy_used == "strategy_b")
    final_strategy_c_pages = sum(1 for p in result.pages if p.metadata.strategy_used == "strategy_c")
    pct = (escalated / total * 100.0) if total else 0.0

    routing_path = Path(".refinery/extracted") / f"{result.doc_id}.routing.json"
    ledger_path = Path(".refinery/extraction_ledger") / f"{result.doc_id}.jsonl"
    suggested_strategy = "unknown"
    starting_strategy = "unknown"
    executed_strategy = result.metadata.strategy_used
    planned_b = executed_b = planned_c = executed_c = 0
    if routing_path.exists():
        payload = json.loads(routing_path.read_text(encoding="utf-8"))
        suggested_strategy = str(payload.get("document_level_strategy_suggestion", suggested_strategy))
        starting_strategy = str(payload.get("starting_strategy", starting_strategy))
        executed_strategy = str(payload.get("executed_strategy", executed_strategy))
        strategy_counts = payload.get("strategy_counts", {})
        if isinstance(strategy_counts, dict):
            planned_b = int(strategy_counts.get("planned_strategy_b", 0))
            executed_b = int(strategy_counts.get("executed_strategy_b", 0))
            planned_c = int(strategy_counts.get("planned_strategy_c", 0))
            executed_c = int(strategy_counts.get("executed_strategy_c", 0))

    executions_by_strategy = {"strategy_a": 0, "strategy_b": 0, "strategy_c": 0}
    if ledger_path.exists():
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            strategy_used = str(row.get("strategy_used", ""))
            if strategy_used in executions_by_strategy:
                executions_by_strategy[strategy_used] += 1

    print(f"doc_id={result.doc_id}")
    print(f"document_level_strategy_suggestion={suggested_strategy}")
    print(f"starting_strategy={starting_strategy}")
    print(f"executed_strategy={executed_strategy}")
    print(f"pages_processed={total}")
    print(f"pages_escalated={escalated}")
    print(f"final_pages_strategy_a={final_strategy_a_pages}")
    print(f"final_pages_strategy_b={final_strategy_b_pages}")
    print(f"final_pages_strategy_c={final_strategy_c_pages}")
    print(f"executions_strategy_a={executions_by_strategy['strategy_a']}")
    print(f"executions_strategy_b={executions_by_strategy['strategy_b']}")
    print(f"executions_strategy_c={executions_by_strategy['strategy_c']}")
    print(f"planned_strategy_b={planned_b} executed_strategy_b={executed_b}")
    print(f"planned_strategy_c={planned_c} executed_strategy_c={executed_c}")
    print(f"planned_escalations_pct={pct:.2f}%")
    print(f"ledger={ledger_path}")
    print(f"extracted=.refinery/extracted/{result.doc_id}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
