"""Debug/demo runner for a single PDF through the current Phase 3 pipeline."""

from __future__ import annotations

import argparse
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
from src.chunking import (
    ChromaVectorStore,
    ChunkingConfig,
    ChunkingEngine,
    OllamaSummaryBackend,
    PageIndexBuilder,
    PageIndexQueryEngine,
    PageIndexSummarizer,
    ProvenanceChainBuilder,
    ProvenanceChainError,
    SummaryBackend,
    SummaryInput,
    VectorStoreError,
)
from src.chunking.page_index import PageIndexTree
from src.chunking.vector_store import EmbeddingBackend, VectorStoreMatch
from src.models.chunking import Chunk, LDU, PageIndexNode
from src.models.extracted_document import ExtractedDocument
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


def build_topics(tree: PageIndexTree, requested_topics: list[str]) -> list[str]:
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


def assisted_retrieve(
    *,
    topic: str,
    tree: PageIndexTree,
    query_engine: PageIndexQueryEngine,
    vector_store: ChromaVectorStore,
    top_k: int,
) -> tuple[list[Any], list[VectorStoreMatch]]:
    section_matches = query_engine.query(tree=tree, topic=topic, top_k=3)
    results: list[VectorStoreMatch] = []
    seen_record_ids: set[str] = set()

    for section_match in section_matches:
        candidates = vector_store.query(
            topic,
            top_k=top_k,
            section_path=section_match.section_path,
        )
        for candidate in candidates:
            if candidate.record_id in seen_record_ids:
                continue
            seen_record_ids.add(candidate.record_id)
            results.append(candidate)
            if len(results) >= top_k:
                break
        if len(results) >= top_k:
            break

    return section_matches, results


def print_tree(tree: PageIndexTree) -> None:
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


def print_chunk_previews(chunks: list[Chunk], limit: int = 8) -> None:
    print("\n== Chunk Previews ==")
    for chunk in chunks[:limit]:
        preview = canonicalize_text(chunk.text)
        preview = preview[:120] + ("..." if len(preview) > 120 else "")
        section = " > ".join(chunk.section_path) if chunk.section_path else "<root>"
        print(
            f"- seq={chunk.sequence_number} page={chunk.page_number} "
            f"section={section} hash={chunk.content_hash[:12]} preview={preview}"
        )


def print_node_summaries(tree: PageIndexTree) -> None:
    print("\n== Node Summaries ==")
    for node in sorted(tree.nodes, key=lambda current: current.order_index):
        if not node.section_path:
            continue
        summary = node.summary or "<none>"
        print(f"- {' > '.join(node.section_path)}: {summary}")


def print_query_results(
    *,
    topic: str,
    section_matches: list[Any],
    baseline_matches: list[VectorStoreMatch],
    assisted_matches: list[VectorStoreMatch],
    assisted_chain: dict[str, Any] | None,
) -> None:
    print(f"\n== Query: {topic} ==")
    print("PageIndex matches:")
    for match in section_matches:
        section = " > ".join(match.section_path)
        print(f"  - {section} score={match.score} pages={match.start_page}-{match.end_page}")

    print("Baseline vector retrieval:")
    for match in baseline_matches:
        section = match.metadata.get("section_path_str", "<unknown>")
        print(f"  - id={match.record_id} section={section} distance={match.distance}")

    print("PageIndex-assisted retrieval:")
    for match in assisted_matches:
        section = match.metadata.get("section_path_str", "<unknown>")
        print(f"  - id={match.record_id} section={section} distance={match.distance}")
    if assisted_chain:
        print("ProvenanceChain:")
        for entry in assisted_chain["entries"]:
            page_number = entry["provenance"]["page_number"]
            bbox = entry["provenance"]["bbox"]
            print(f"  - {entry['record_type']} {entry['record_id']} page={page_number} bbox={bbox}")


def save_artifacts(
    *,
    output_dir: Path,
    extracted: ExtractedDocument,
    ldus: list[LDU],
    chunks: list[Chunk],
    tree: PageIndexTree,
    query_outputs: list[dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "extracted_document.json").write_text(extracted.model_dump_json(indent=2), encoding="utf-8")
    (output_dir / "ldus.json").write_text(json.dumps([ldu.model_dump() for ldu in ldus], indent=2), encoding="utf-8")
    (output_dir / "chunks.json").write_text(
        json.dumps([chunk.model_dump() for chunk in chunks], indent=2),
        encoding="utf-8",
    )
    (output_dir / "page_index.json").write_text(tree.model_dump_json(indent=2), encoding="utf-8")
    (output_dir / "retrieval_report.json").write_text(json.dumps(query_outputs, indent=2), encoding="utf-8")


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

        engine = ChunkingEngine(config=ChunkingConfig())
        ldus = engine.build_ldus(extracted)
        chunks = engine.build_chunks(extracted, ldus=ldus)

        tree = PageIndexBuilder().build(doc_id=extracted.doc_id, ldus=ldus)
        summary_backend = build_summary_backend(args)
        summarized_tree = PageIndexSummarizer(summary_backend).summarize_tree(tree=tree, ldus=ldus)

        embedding_backend = HashEmbeddingBackend()
        collection_name = f"phase3_debug_{extracted.doc_id}"
        vector_store = ChromaVectorStore(
            embedding_backend=embedding_backend,
            collection_name=collection_name,
            persist_directory=args.persist_dir,
        )
        vector_store.ingest_ldus(ldus)
        vector_store.ingest_chunks(chunks)

        topics = build_topics(summarized_tree, args.topics)
        query_engine = PageIndexQueryEngine()
        provenance_builder = ProvenanceChainBuilder()
        query_outputs: list[dict[str, Any]] = []

        print(f"doc_id={extracted.doc_id}")
        print(f"file={pdf_path}")
        print(f"pages={extracted.page_count}")
        print(f"ldus={len(ldus)} chunks={len(chunks)}")
        print(f"collection={collection_name}")

        print_tree(summarized_tree)
        print_chunk_previews(chunks)
        print_node_summaries(summarized_tree)

        for topic in topics:
            section_matches, assisted_matches = assisted_retrieve(
                topic=topic,
                tree=summarized_tree,
                query_engine=query_engine,
                vector_store=vector_store,
                top_k=args.top_k,
            )
            baseline_matches = vector_store.query(topic, top_k=args.top_k)
            assisted_chain: dict[str, Any] | None = None
            try:
                assisted_chain = provenance_builder.build(assisted_matches, query=topic).model_dump(mode="json")
            except ProvenanceChainError:
                assisted_chain = None
            print_query_results(
                topic=topic,
                section_matches=section_matches,
                baseline_matches=baseline_matches,
                assisted_matches=assisted_matches,
                assisted_chain=assisted_chain,
            )
            query_outputs.append(
                {
                    "topic": topic,
                    "pageindex_matches": [
                        {
                            "section_path": list(match.section_path),
                            "score": match.score,
                            "pages": [match.start_page, match.end_page],
                        }
                        for match in section_matches
                    ],
                    "baseline": [
                        {
                            "record_id": match.record_id,
                            "section_path": match.metadata.get("section_path", []),
                            "distance": match.distance,
                        }
                        for match in baseline_matches
                    ],
                    "pageindex_assisted": [
                        {
                            "record_id": match.record_id,
                            "section_path": match.metadata.get("section_path", []),
                            "distance": match.distance,
                        }
                        for match in assisted_matches
                    ],
                    "provenance_chain": assisted_chain,
                }
            )

        if args.save_artifacts:
            pdf_stem = pdf_path.stem.replace(" ", "_")
            output_dir = args.output_dir or Path("debug_runs") / pdf_stem
            save_artifacts(
                output_dir=output_dir,
                extracted=extracted,
                ldus=ldus,
                chunks=chunks,
                tree=summarized_tree,
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
