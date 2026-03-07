"""Debug/demo runner for a single PDF through the current Phase 3 pipeline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.extractor import run_extraction
from src.agents.phase4_pipeline import Phase4Pipeline, Phase4QueryRun
from src.chunking import (
    ChromaVectorStore,
    ChunkingConfig,
    ChunkingEngine,
    OllamaEmbeddingBackend,
    OllamaSummaryBackend,
    SummaryBackend,
    SummaryInput,
    VectorStoreError,
)
from src.chunking.vector_store import EmbeddingBackend, VectorStoreMatch
from src.models.chunking import PageIndexNode
from src.utils.hashing import canonicalize_text


class HeuristicSummaryBackend(SummaryBackend):
    """Deterministic local summary backend for debug runs."""

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
    parser = argparse.ArgumentParser(description="Run a single PDF through the current Phase 3 pipeline.")
    parser.add_argument("pdf_path", type=Path, help="Path to one PDF file")
    parser.add_argument(
        "--strategy",
        choices=["strategy_a", "strategy_b", "strategy_c"],
        default=None,
        help="Optional extraction strategy override",
    )
    parser.add_argument(
        "--summary-backend",
        choices=["heuristic", "ollama"],
        default="ollama",
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
        "--embedding-backend",
        choices=["hash", "ollama"],
        default="ollama",
        help="Embedding backend for vector retrieval",
    )
    parser.add_argument(
        "--ollama-embedding-model",
        default="qwen3-embedding:0.6b",
        help="Ollama embedding model to use when --embedding-backend=ollama",
    )
    parser.add_argument(
        "--ollama-embedding-keep-alive",
        default="0s",
        help="Ollama keep_alive value when --embedding-backend=ollama",
    )
    parser.add_argument(
        "--topic",
        action="append",
        dest="topics",
        default=[],
        help="Sample topic to run through PageIndex and vector retrieval. Repeat up to 3 times.",
    )
    parser.add_argument(
        "--save-artifacts",
        action="store_true",
        help="Save intermediate JSON artifacts to a debug output folder",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional debug output directory. Defaults to .refinery/debug/<doc_id>",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path(".refinery/chroma_debug"),
        help="Chroma persist directory",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-K retrieval results to print",
    )
    return parser.parse_args()


def build_summary_backend(args: argparse.Namespace) -> SummaryBackend:
    if args.summary_backend == "heuristic":
        return HeuristicSummaryBackend()
    return OllamaSummaryBackend(
        model=args.ollama_model,
        keep_alive=args.ollama_keep_alive,
    )


def build_embedding_backend(args: argparse.Namespace) -> EmbeddingBackend:
    if args.embedding_backend == "hash":
        return HashEmbeddingBackend()
    return OllamaEmbeddingBackend(
        model=args.ollama_embedding_model,
        keep_alive=args.ollama_embedding_keep_alive,
    )


def build_topics(tree, requested_topics: list[str]) -> list[str]:
    topics = [topic.strip() for topic in requested_topics if topic.strip()]
    if topics:
        return topics[:3]

    non_root_nodes = [node for node in sorted(tree.nodes, key=lambda current: current.order_index) if node.section_path]
    derived_topics: list[str] = []
    for node in non_root_nodes:
        cleaned = re.sub(r"^\d+(?:\.\d+)*\s*", "", node.title).strip()
        if cleaned and cleaned not in derived_topics:
            derived_topics.append(cleaned)
        if len(derived_topics) == 3:
            break

    fallback = ["overview", "results", "recommendations"]
    for topic in fallback:
        if len(derived_topics) == 3:
            break
        if topic not in derived_topics:
            derived_topics.append(topic)
    return derived_topics[:3]


def print_tree(tree) -> None:
    print("\n== Sections / PageIndex Tree ==")
    nodes_by_parent: dict[str | None, list[PageIndexNode]] = {}
    for node in sorted(tree.nodes, key=lambda current: current.order_index):
        nodes_by_parent.setdefault(node.parent_id, []).append(node)

    def walk(parent_id: str | None, indent: int) -> None:
        for node in nodes_by_parent.get(parent_id, []):
            prefix = "  " * indent
            section_label = "ROOT" if not node.section_path else node.title
            print(
                f"{prefix}- {section_label} "
                f"[pages {node.start_page}-{node.end_page}] "
                f"ldus={len(node.ldu_ids)} children={len(node.child_ids)}"
            )
            walk(node.node_id, indent + 1)

    walk(None, 0)


def print_chunk_previews(chunks, limit: int = 8) -> None:
    print("\n== Chunk Previews ==")
    for chunk in chunks[:limit]:
        preview = canonicalize_text(chunk.text)
        preview = preview[:120] + ("..." if len(preview) > 120 else "")
        section = " > ".join(chunk.section_path) if chunk.section_path else "<root>"
        print(
            f"- seq={chunk.sequence_number} page={chunk.page_number} "
            f"section={section} hash={chunk.content_hash[:12]} preview={preview}"
        )


def print_node_summaries(tree) -> None:
    print("\n== Node Summaries ==")
    for node in sorted(tree.nodes, key=lambda current: current.order_index):
        if not node.section_path:
            continue
        summary = node.summary or "<none>"
        print(f"- {' > '.join(node.section_path)}: {summary}")


def print_query_results(
    *,
    query_run: Phase4QueryRun,
    audit_result: dict[str, Any] | None,
    baseline_matches: list[VectorStoreMatch],
) -> None:
    result = query_run.query_result
    print(f"\n== Query: {result.query} ==")
    print(f"Status: {result.status} route={result.route}")
    if result.answer:
        print(f"Answer: {result.answer}")
    if result.failure_reason:
        print(f"Failure: {result.failure_reason}")
    if audit_result:
        print(f"Audit: {audit_result['status']}")
        if audit_result.get("failure_reason"):
            print(f"Audit failure: {audit_result['failure_reason']}")
    print("PageIndex matches:")
    for match in result.page_index_matches:
        section = " > ".join(match.section_path)
        print(f"  - {section} score={match.score} pages={match.start_page}-{match.end_page}")

    print("Baseline vector retrieval:")
    for match in baseline_matches:
        section = match.metadata.get("section_path_str", "<unknown>")
        print(f"  - id={match.record_id} section={section} distance={match.distance}")

    print("Selected retrieval matches:")
    for match in result.retrieval_matches:
        section = match.metadata.get("section_path_str", "<unknown>")
        print(f"  - id={match.record_id} section={section} distance={match.distance}")
    if result.provenance_chain:
        print("ProvenanceChain:")
        for entry in result.provenance_chain.model_dump(mode="json")["entries"]:
            page_number = entry["provenance"]["page_number"]
            bbox = entry["provenance"]["bbox"]
            print(f"  - {entry['record_type']} {entry['record_id']} page={page_number} bbox={bbox}")


def save_artifacts(
    *,
    output_dir: Path,
    pipeline_result,
    query_outputs: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "extracted_document.json").write_text(
        pipeline_result.extracted.model_dump_json(indent=2), encoding="utf-8"
    )
    (output_dir / "ldus.json").write_text(
        json.dumps([ldu.model_dump() for ldu in pipeline_result.ldus], indent=2), encoding="utf-8"
    )
    (output_dir / "chunks.json").write_text(
        json.dumps([chunk.model_dump() for chunk in pipeline_result.chunks], indent=2),
        encoding="utf-8",
    )
    (output_dir / "page_index.json").write_text(pipeline_result.tree.model_dump_json(indent=2), encoding="utf-8")
    (output_dir / "retrieval_report.json").write_text(json.dumps(query_outputs, indent=2), encoding="utf-8")
    (output_dir / "fact_table.json").write_text(
        pipeline_result.fact_table.model_dump_json(indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    pdf_path = args.pdf_path.resolve()

    if not pdf_path.exists():
        print(f"ERROR: missing file: {pdf_path}", file=sys.stderr)
        return 1
    if pdf_path.suffix.lower() != ".pdf":
        print(f"ERROR: expected a PDF file, got: {pdf_path}", file=sys.stderr)
        return 1

    try:
        extracted = run_extraction(pdf_path, strategy=args.strategy)
        if extracted.status == "error":
            message = extracted.error_message or "unknown extraction error"
            print(f"ERROR: extraction failed for {pdf_path}: {message}", file=sys.stderr)
            return 1

        summary_backend = build_summary_backend(args)
        embedding_backend = build_embedding_backend(args)
        collection_name = f"phase3_debug_{extracted.doc_id}"
        vector_store = ChromaVectorStore(
            embedding_backend=embedding_backend,
            collection_name=collection_name,
            persist_directory=args.persist_dir,
        )
        phase4 = Phase4Pipeline(
            vector_store=vector_store,
            summary_backend=summary_backend,
            chunking_engine=ChunkingEngine(config=ChunkingConfig()),
        )
        if args.topics:
            pipeline_result = phase4.run(extracted=extracted, queries=args.topics[:3], top_k=args.top_k)
        else:
            bootstrap_result = phase4.run(extracted=extracted, queries=[], top_k=args.top_k)
            topics = build_topics(bootstrap_result.tree, [])
            pipeline_result = phase4.run(extracted=extracted, queries=topics, top_k=args.top_k)
        query_outputs: list[dict[str, Any]] = []

        print(f"doc_id={extracted.doc_id}")
        print(f"file={pdf_path}")
        print(f"pages={extracted.page_count}")
        print(f"ldus={len(pipeline_result.ldus)} chunks={len(pipeline_result.chunks)}")
        print(f"facts={len(pipeline_result.fact_table.entries)}")
        print(f"collection={collection_name}")

        print_tree(pipeline_result.tree)
        print_chunk_previews(list(pipeline_result.chunks))
        print_node_summaries(pipeline_result.tree)

        for query_run in pipeline_result.query_runs:
            result = query_run.query_result
            baseline_matches = vector_store.query(result.query, top_k=args.top_k)
            audit_result = asdict(query_run.audit_result)
            print_query_results(
                query_run=query_run,
                audit_result=audit_result,
                baseline_matches=baseline_matches,
            )
            query_outputs.append(
                {
                    "topic": result.query,
                    "status": result.status,
                    "route": result.route,
                    "answer": result.answer,
                    "failure_reason": result.failure_reason,
                    "audit": audit_result,
                    "pageindex_matches": [
                        {
                            "section_path": list(match.section_path),
                            "score": match.score,
                            "pages": [match.start_page, match.end_page],
                        }
                        for match in result.page_index_matches
                    ],
                    "baseline": [
                        {
                            "record_id": match.record_id,
                            "section_path": match.metadata.get("section_path", []),
                            "distance": match.distance,
                        }
                        for match in baseline_matches
                    ],
                    "selected_matches": [
                        {
                            "record_id": match.record_id,
                            "section_path": match.metadata.get("section_path", []),
                            "distance": match.distance,
                        }
                        for match in result.retrieval_matches
                    ],
                    "provenance_chain": (
                        result.provenance_chain.model_dump(mode="json") if result.provenance_chain else None
                    ),
                }
            )

        if args.save_artifacts:
            pdf_stem = pdf_path.stem.replace(" ", "_")
            output_dir = args.output_dir or Path("debug_runs") / pdf_stem
            save_artifacts(
                output_dir=output_dir,
                pipeline_result=pipeline_result,
                query_outputs=query_outputs,
            )
            print(f"\nSaved debug artifacts to {output_dir}")

        return 0
    except (RuntimeError, ValueError, VectorStoreError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover - debug script fallback
        print(f"UNEXPECTED ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
