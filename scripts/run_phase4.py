"""CLI entrypoint for the integrated Phase 4 pipeline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.extractor import run_extraction
from src.agents.phase4_pipeline import Phase4Pipeline
from src.chunking import ChromaVectorStore, ChunkingConfig, ChunkingEngine, OllamaSummaryBackend, SummaryBackend, SummaryInput
from src.chunking.vector_store import EmbeddingBackend, VectorStoreError
from src.storage import FactTableSqliteWriter
from src.utils.hashing import canonicalize_text


class HeuristicSummaryBackend(SummaryBackend):
    """Deterministic local summary backend for CLI runs."""

    def summarize(self, summary_input: SummaryInput) -> str:
        words = canonicalize_text(summary_input.source_text).split()
        preview = " ".join(words[:24])
        return preview if len(words) <= 24 else f"{preview}..."


class HashEmbeddingBackend(EmbeddingBackend):
    """Deterministic local embedding backend using hashed token counts."""

    def __init__(self, dimensions: int = 32) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        vector = [0.0] * self.dimensions
        if not tokens:
            return vector
        for token in tokens:
            index = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dimensions
            vector[index] += 1.0
        scale = float(len(tokens))
        return [value / scale for value in vector]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the integrated Phase 4 pipeline on a PDF.")
    parser.add_argument("pdf_path", type=Path, help="Path to PDF file")
    parser.add_argument(
        "--strategy",
        choices=["strategy_a", "strategy_b", "strategy_c"],
        default=None,
        help="Optional extraction strategy override",
    )
    parser.add_argument(
        "--summary-backend",
        choices=["heuristic", "ollama"],
        default="heuristic",
        help="Summary backend for PageIndex nodes",
    )
    parser.add_argument(
        "--ollama-model",
        default="qwen3:1.7b",
        help="Ollama model to use when --summary-backend=ollama",
    )
    parser.add_argument(
        "--ollama-keep-alive",
        default="0s",
        help="Ollama keep_alive value when --summary-backend=ollama",
    )
    parser.add_argument(
        "--topic",
        action="append",
        dest="topics",
        default=[],
        help="Query topic to run through Phase 4. Repeat up to 3 times.",
    )
    parser.add_argument(
        "--claim",
        action="append",
        dest="claims",
        default=[],
        help="Claim to verify against the document. Repeat up to 3 times.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path(".refinery/chroma_phase4"),
        help="Chroma persist directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to debug_runs/<pdf_stem>_phase4",
    )
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Write Phase 4 JSON artifacts to disk",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-K retrieval results per query",
    )
    return parser.parse_args()


def build_summary_backend(args: argparse.Namespace) -> SummaryBackend:
    if args.summary_backend == "heuristic":
        return HeuristicSummaryBackend()
    return OllamaSummaryBackend(
        model=args.ollama_model,
        keep_alive=args.ollama_keep_alive,
    )


def derive_topics(tree, requested_topics: list[str]) -> list[str]:
    topics = [topic.strip() for topic in requested_topics if topic.strip()]
    if topics:
        return topics[:3]

    non_root_nodes = [node for node in sorted(tree.nodes, key=lambda current: current.order_index) if node.section_path]
    derived: list[str] = []
    for node in non_root_nodes:
        cleaned = re.sub(r"^\d+(?:\.\d+)*\s*", "", node.title).strip()
        if cleaned and cleaned not in derived:
            derived.append(cleaned)
        if len(derived) == 3:
            break

    fallback = ["overview", "results", "recommendations"]
    for topic in fallback:
        if len(derived) == 3:
            break
        if topic not in derived:
            derived.append(topic)
    return derived[:3]


def save_artifacts(*, output_dir: Path, pipeline_result) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "extracted_document.json").write_text(
        pipeline_result.extracted.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (output_dir / "ldus.json").write_text(
        json.dumps([ldu.model_dump() for ldu in pipeline_result.ldus], indent=2),
        encoding="utf-8",
    )
    (output_dir / "chunks.json").write_text(
        json.dumps([chunk.model_dump() for chunk in pipeline_result.chunks], indent=2),
        encoding="utf-8",
    )
    (output_dir / "page_index.json").write_text(
        pipeline_result.tree.model_dump_json(indent=2),
        encoding="utf-8",
    )
    (output_dir / "fact_table.json").write_text(
        pipeline_result.fact_table.model_dump_json(indent=2),
        encoding="utf-8",
    )
    FactTableSqliteWriter().write(
        fact_table=pipeline_result.fact_table,
        db_path=output_dir / "fact_table.sqlite",
    )
    phase4_report = {
        "doc_id": pipeline_result.extracted.doc_id,
        "query_runs": [
            {
                "query": run.query_result.query,
                "status": run.query_result.status,
                "route": run.query_result.route,
                "answer": run.query_result.answer,
                "failure_reason": run.query_result.failure_reason,
                "audit": asdict(run.audit_result),
                "provenance_chain": (
                    run.query_result.provenance_chain.model_dump(mode="json")
                    if run.query_result.provenance_chain
                    else None
                ),
            }
            for run in pipeline_result.query_runs
        ],
        "claim_verifications": [
            {
                "claim": verification.claim,
                "status": verification.status,
                "support_ratio": verification.support_ratio,
                "supporting_record_ids": list(verification.supporting_record_ids),
                "failure_reason": verification.failure_reason,
                "provenance_chain": (
                    verification.provenance_chain.model_dump(mode="json")
                    if verification.provenance_chain
                    else None
                ),
            }
            for verification in pipeline_result.claim_verifications
        ],
    }
    (output_dir / "phase4_report.json").write_text(
        json.dumps(phase4_report, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    pdf_path = args.pdf_path.resolve()

    if not pdf_path.exists():
        print(f"status=error message=missing_file path={pdf_path}")
        return 1
    if pdf_path.suffix.lower() != ".pdf":
        print(f"status=error message=expected_pdf path={pdf_path}")
        return 1

    try:
        extracted = run_extraction(pdf_path, strategy=args.strategy)
        if extracted.status == "error":
            print(f"status=error doc_id={extracted.doc_id} message={extracted.error_message}")
            return 1

        vector_store = ChromaVectorStore(
            embedding_backend=HashEmbeddingBackend(),
            collection_name=f"phase4_{extracted.doc_id}",
            persist_directory=args.persist_dir,
        )
        phase4 = Phase4Pipeline(
            vector_store=vector_store,
            summary_backend=build_summary_backend(args),
            chunking_engine=ChunkingEngine(config=ChunkingConfig()),
        )

        if args.topics:
            topics = args.topics[:3]
        else:
            bootstrap_result = phase4.run(extracted=extracted, queries=[], claims=[], top_k=args.top_k)
            topics = derive_topics(bootstrap_result.tree, [])

        pipeline_result = phase4.run(
            extracted=extracted,
            queries=topics,
            claims=args.claims[:3],
            top_k=args.top_k,
        )

        verified_queries = sum(1 for run in pipeline_result.query_runs if run.query_result.status == "verified")
        passed_audits = sum(1 for run in pipeline_result.query_runs if run.audit_result.status == "passed")
        verified_claims = sum(1 for verification in pipeline_result.claim_verifications if verification.status == "verified")

        print(f"status=ok doc_id={extracted.doc_id}")
        print(f"file={pdf_path}")
        print(f"pages={extracted.page_count}")
        print(f"ldus={len(pipeline_result.ldus)}")
        print(f"chunks={len(pipeline_result.chunks)}")
        print(f"facts={len(pipeline_result.fact_table.entries)}")
        print(f"queries={len(pipeline_result.query_runs)}")
        print(f"queries_verified={verified_queries}")
        print(f"audits_passed={passed_audits}")
        print(f"claims={len(pipeline_result.claim_verifications)}")
        print(f"claims_verified={verified_claims}")
        print(f"collection=phase4_{extracted.doc_id}")

        if args.save_artifacts:
            output_dir = args.output_dir or Path("debug_runs") / f"{pdf_path.stem}_phase4"
            save_artifacts(output_dir=output_dir, pipeline_result=pipeline_result)
            print(f"artifacts={output_dir}")

        return 0
    except (RuntimeError, ValueError, VectorStoreError) as exc:
        print(f"status=error message={exc}")
        return 1
    except Exception as exc:  # pragma: no cover - CLI fallback
        print(f"status=error message=unexpected:{exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
