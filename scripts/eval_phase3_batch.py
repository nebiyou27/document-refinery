"""Batch Phase 3 evaluation runner for the selected validation documents."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime
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
    LabeledRetrievalQuery,
    OllamaSummaryBackend,
    PageIndexBuilder,
    PageIndexQueryEngine,
    PageIndexSummarizer,
    RetrievalEvaluator,
    SummaryBackend,
    SummaryInput,
)
from src.chunking.page_index import PageIndexTree
from src.chunking.vector_store import EmbeddingBackend
from src.models.chunking import Chunk, LDU, PageIndexNode
from src.models.extracted_document import ExtractedDocument
from src.utils.hashing import canonicalize_text


LOGGER = logging.getLogger("phase3_batch_eval")


class HeuristicSummaryBackend(SummaryBackend):
    """Deterministic local summary backend for evaluation runs."""

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


@dataclass
class DocumentEvalResult:
    file_path: str
    doc_id: str | None
    success: bool
    failure_reason: str | None
    page_count: int
    ldu_count: int
    chunk_count: int
    pageindex_node_count: int
    summaries_generated: bool
    vector_ingestion_succeeded: bool
    retrieval_evaluation_succeeded: bool
    artifacts_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch Phase 3 evaluation over selected PDFs from a CSV.")
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path("phase0_selected_12.csv"),
        help="CSV listing the selected PDFs. Must include a 'file' column.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing the selected PDFs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Batch output root. Defaults to debug_runs/phase3_batch_eval/<timestamp>.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=Path(".refinery/chroma_debug"),
        help="Chroma persist directory for debug collections.",
    )
    parser.add_argument(
        "--summary-backend",
        choices=["heuristic", "ollama"],
        default="heuristic",
        help="Summary backend for PageIndex nodes.",
    )
    parser.add_argument(
        "--ollama-model",
        default="qwen3:1.7b",
        help="Ollama model to use when --summary-backend=ollama.",
    )
    parser.add_argument(
        "--ollama-keep-alive",
        default="0s",
        help="Ollama keep_alive value when --summary-backend=ollama.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Top-K vector retrieval depth for evaluation.",
    )
    parser.add_argument(
        "--section-top-k",
        type=int,
        default=3,
        help="Top-K PageIndex sections to traverse for assisted evaluation.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional limit for debugging.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )


def build_summary_backend(args: argparse.Namespace) -> SummaryBackend:
    if args.summary_backend == "heuristic":
        return HeuristicSummaryBackend()
    return OllamaSummaryBackend(
        model=args.ollama_model,
        keep_alive=args.ollama_keep_alive,
    )


def default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("debug_runs") / "phase3_batch_eval" / timestamp


def slugify(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    normalized = normalized.strip("._")
    return normalized or "document"


def load_selected_rows(csv_path: Path, max_docs: int | None) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if "file" not in (reader.fieldnames or []):
        raise ValueError(f"CSV {csv_path} must include a 'file' column")
    return rows[:max_docs] if max_docs is not None else rows


def resolve_pdf_path(file_name: str, data_dir: Path) -> Path:
    candidate = Path(file_name)
    if candidate.is_absolute():
        return candidate
    return data_dir / file_name


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def summarize_tree_stats(tree: PageIndexTree) -> dict[str, Any]:
    nodes = sorted(tree.nodes, key=lambda node: node.order_index)
    return {
        "doc_id": tree.doc_id,
        "root_id": tree.root_id,
        "node_count": len(nodes),
        "nodes": [node.model_dump() for node in nodes],
    }


def build_retrieval_queries(tree: PageIndexTree, chunks: list[Chunk], limit: int = 3) -> list[LabeledRetrievalQuery]:
    chunk_ids_by_section: dict[tuple[str, ...], list[str]] = {}
    for chunk in chunks:
        chunk_ids_by_section.setdefault(chunk.section_path, []).append(chunk.chunk_id or "")

    queries: list[LabeledRetrievalQuery] = []
    seen_topics: set[str] = set()
    for node in sorted(tree.nodes, key=lambda current: (current.depth, current.order_index)):
        if not node.section_path:
            continue
        relevant_ids = tuple(
            chunk_id
            for section_path, chunk_ids in chunk_ids_by_section.items()
            if section_path[: len(node.section_path)] == node.section_path
            for chunk_id in chunk_ids
            if chunk_id
        )
        if not relevant_ids:
            continue
        topic = build_query_topic(node)
        if not topic or topic in seen_topics:
            continue
        seen_topics.add(topic)
        queries.append(
            LabeledRetrievalQuery(
                query_id=f"q{len(queries) + 1}",
                topic=topic,
                relevant_record_ids=relevant_ids[:3],
            )
        )
        if len(queries) >= limit:
            break
    return queries


def build_query_topic(node: PageIndexNode) -> str:
    if node.summary:
        summary = canonicalize_text(node.summary)
        if summary:
            return summary
    title = re.sub(r"^\d+(?:\.\d+)*\s*", "", node.title).strip()
    return canonicalize_text(title)


def save_artifacts(
    *,
    output_dir: Path,
    extracted: ExtractedDocument,
    ldus: list[LDU],
    chunks: list[Chunk],
    tree: PageIndexTree,
    query_payload: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_text(output_dir / "extracted_document.json", extracted.model_dump_json(indent=2))
    save_json(output_dir / "ldus.json", [ldu.model_dump() for ldu in ldus])
    save_json(output_dir / "chunks.json", [chunk.model_dump() for chunk in chunks])
    save_json(output_dir / "page_index.json", summarize_tree_stats(tree))
    save_json(output_dir / "retrieval_evaluation.json", query_payload)


def process_document(
    *,
    pdf_path: Path,
    args: argparse.Namespace,
    summary_backend: SummaryBackend,
    batch_output_dir: Path,
) -> DocumentEvalResult:
    doc_folder = batch_output_dir / slugify(pdf_path.stem)
    doc_folder.mkdir(parents=True, exist_ok=True)

    result = DocumentEvalResult(
        file_path=str(pdf_path),
        doc_id=None,
        success=False,
        failure_reason=None,
        page_count=0,
        ldu_count=0,
        chunk_count=0,
        pageindex_node_count=0,
        summaries_generated=False,
        vector_ingestion_succeeded=False,
        retrieval_evaluation_succeeded=False,
        artifacts_dir=str(doc_folder),
    )

    try:
        LOGGER.info("Processing %s", pdf_path.name)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Missing PDF: {pdf_path}")

        extracted = run_extraction(pdf_path)
        result.doc_id = extracted.doc_id
        result.page_count = extracted.page_count
        save_text(doc_folder / "extracted_document.json", extracted.model_dump_json(indent=2))

        if extracted.status == "error":
            raise RuntimeError(extracted.error_message or "extraction failed")

        engine = ChunkingEngine(config=ChunkingConfig())
        ldus = engine.build_ldus(extracted)
        chunks = engine.build_chunks(extracted)
        result.ldu_count = len(ldus)
        result.chunk_count = len(chunks)

        tree = PageIndexBuilder().build(doc_id=extracted.doc_id, ldus=ldus)
        summarized_tree = PageIndexSummarizer(summary_backend).summarize_tree(tree=tree, ldus=ldus)
        result.pageindex_node_count = len(summarized_tree.nodes)
        result.summaries_generated = any(
            bool((node.summary or "").strip()) for node in summarized_tree.nodes if node.section_path
        )

        embedding_backend = HashEmbeddingBackend()
        collection_name = f"phase3_batch_eval_{extracted.doc_id}"
        vector_store = ChromaVectorStore(
            embedding_backend=embedding_backend,
            collection_name=collection_name,
            persist_directory=args.persist_dir,
        )
        vector_store.ingest_ldus(ldus)
        vector_store.ingest_chunks(chunks)
        result.vector_ingestion_succeeded = True

        queries = build_retrieval_queries(summarized_tree, chunks)
        LOGGER.info("Derived %s retrieval evaluation queries for %s", len(queries), pdf_path.name)
        if not queries:
            raise RuntimeError("no retrieval evaluation queries could be derived")

        evaluator = RetrievalEvaluator(vector_backend=vector_store, page_index_backend=PageIndexQueryEngine())
        baseline_report = evaluator.evaluate_baseline(queries, top_k=args.top_k, record_type="chunk")
        assisted_report = evaluator.evaluate_pageindex_assisted(
            summarized_tree,
            queries,
            section_top_k=args.section_top_k,
            top_k=args.top_k,
            record_type="chunk",
        )
        result.retrieval_evaluation_succeeded = True

        save_artifacts(
            output_dir=doc_folder,
            extracted=extracted,
            ldus=ldus,
            chunks=chunks,
            tree=summarized_tree,
            query_payload={
                "collection_name": collection_name,
                "queries": [asdict(query) for query in queries],
                "baseline": {
                    "metrics": asdict(baseline_report.metrics),
                    "per_query": [asdict(item) for item in baseline_report.per_query],
                },
                "pageindex_assisted": {
                    "metrics": asdict(assisted_report.metrics),
                    "per_query": [asdict(item) for item in assisted_report.per_query],
                },
            },
        )

        result.success = True
        save_json(doc_folder / "document_report.json", asdict(result))
        LOGGER.info(
            "Completed %s doc_id=%s pages=%s ldus=%s chunks=%s nodes=%s",
            pdf_path.name,
            result.doc_id,
            result.page_count,
            result.ldu_count,
            result.chunk_count,
            result.pageindex_node_count,
        )
        return result
    except Exception as exc:
        result.failure_reason = str(exc)
        save_json(
            doc_folder / "document_report.json",
            {
                **asdict(result),
                "error_type": exc.__class__.__name__,
            },
        )
        save_text(doc_folder / "failure.txt", f"{exc.__class__.__name__}: {exc}\n")
        LOGGER.exception("Failed %s", pdf_path.name)
        return result


def write_summary_reports(output_dir: Path, results: list[DocumentEvalResult]) -> None:
    payload = [asdict(result) for result in results]
    save_json(output_dir / "batch_report.json", payload)

    fieldnames = list(DocumentEvalResult.__dataclass_fields__.keys())
    with (output_dir / "batch_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in payload:
            writer.writerow(row)


def print_run_summary(results: list[DocumentEvalResult]) -> None:
    successes = sum(1 for result in results if result.success)
    failures = len(results) - successes
    failure_counter = Counter(
        result.failure_reason for result in results if result.failure_reason
    )
    common_failures = failure_counter.most_common(3)

    print("\nBatch summary")
    print(f"successes={successes}")
    print(f"failures={failures}")
    if common_failures:
        formatted = "; ".join(f"{reason} ({count})" for reason, count in common_failures)
    else:
        formatted = "none"
    print(f"common_failure_reasons={formatted}")


def main() -> int:
    args = parse_args()
    configure_logging(args.log_level)

    csv_path = args.csv_path.resolve()
    data_dir = args.data_dir.resolve()
    output_dir = (args.output_dir or default_output_dir()).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading selected document list from %s", csv_path)
    try:
        rows = load_selected_rows(csv_path, args.max_docs)
    except Exception as exc:
        LOGGER.error("Failed to read CSV %s: %s", csv_path, exc)
        return 1

    save_json(output_dir / "input_rows.json", rows)
    summary_backend = build_summary_backend(args)

    results: list[DocumentEvalResult] = []
    for index, row in enumerate(rows, start=1):
        file_name = (row.get("file") or "").strip()
        if not file_name:
            reason = "row missing file value"
            LOGGER.error("Skipping row %s: %s", index, reason)
            result = DocumentEvalResult(
                file_path="",
                doc_id=None,
                success=False,
                failure_reason=reason,
                page_count=0,
                ldu_count=0,
                chunk_count=0,
                pageindex_node_count=0,
                summaries_generated=False,
                vector_ingestion_succeeded=False,
                retrieval_evaluation_succeeded=False,
                artifacts_dir=str(output_dir / f"row_{index}"),
            )
            save_json(Path(result.artifacts_dir) / "document_report.json", asdict(result))
            results.append(result)
            continue

        pdf_path = resolve_pdf_path(file_name, data_dir)
        LOGGER.info("(%s/%s) %s", index, len(rows), pdf_path.name)
        results.append(
            process_document(
                pdf_path=pdf_path,
                args=args,
                summary_backend=summary_backend,
                batch_output_dir=output_dir,
            )
        )

    write_summary_reports(output_dir, results)
    print_run_summary(results)
    LOGGER.info("Wrote batch reports to %s", output_dir)
    return 0 if all(result.success for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
