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
    OllamaEmbeddingBackend,
    OllamaSummaryBackend,
    PageIndexBuilder,
    PageIndexQueryEngine,
    PageIndexSummarizer,
    RetrievalEvaluator,
    SummaryBackend,
    SummaryInput,
)
from src.chunking.page_index import PageIndexTree
from src.chunking.sections import SectionInferenceMode, SectionPathInferer
from src.document_classes import resolve_document_class
from src.chunking.vector_store import EmbeddingBackend
from src.models.document_profile import DocumentProfile
from src.models.chunking import Chunk, LDU, PageIndexNode
from src.models.extracted_document import ExtractedDocument
from src.utils.hashing import canonicalize_text


LOGGER = logging.getLogger("phase3_batch_eval")
NO_RETRIEVAL_QUERIES_REASON = "no retrieval evaluation queries could be derived"


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
    document_class: str | None
    extraction_quality: str
    retrieval_query_derivation_expected: bool
    section_inference_mode: str | None
    success: bool
    failure_reason: str | None
    page_count: int
    timed_out_page_count: int
    surviving_content_page_count: int
    ldu_count: int
    chunk_count: int
    pageindex_node_count: int
    non_root_pageindex_node_count: int
    summaries_generated: bool
    query_derivation_blocker: str | None
    vector_ingestion_succeeded: bool
    retrieval_evaluation_attempted: bool
    retrieval_evaluation_succeeded: bool
    retrieval_evaluation_failed: bool
    retrieval_evaluation_failure_reason: str | None
    retrieval_evaluation_skipped: bool
    retrieval_evaluation_skip_reason: str | None
    artifacts_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch Phase 3 evaluation over selected PDFs from a CSV or explicit PDF paths."
    )
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
        help="Directory containing CSV-selected PDFs.",
    )
    parser.add_argument(
        "--pdf",
        dest="pdf_paths",
        type=Path,
        action="append",
        default=None,
        help="Explicit PDF to evaluate. Repeat the flag to evaluate multiple PDFs in the given order.",
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
        default="ollama",
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
        "--embedding-backend",
        choices=["hash", "ollama"],
        default="ollama",
        help="Embedding backend for vector retrieval.",
    )
    parser.add_argument(
        "--ollama-embedding-model",
        default="qwen3-embedding:0.6b",
        help="Ollama embedding model to use when --embedding-backend=ollama.",
    )
    parser.add_argument(
        "--ollama-embedding-keep-alive",
        default="0s",
        help="Ollama keep_alive value when --embedding-backend=ollama.",
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


def build_embedding_backend(args: argparse.Namespace) -> EmbeddingBackend:
    if args.embedding_backend == "hash":
        return HashEmbeddingBackend()
    return OllamaEmbeddingBackend(
        model=args.ollama_embedding_model,
        keep_alive=args.ollama_embedding_keep_alive,
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


def resolve_explicit_pdf_paths(pdf_paths: list[Path]) -> list[Path]:
    resolved_paths = [path.resolve() for path in pdf_paths]
    missing_paths = [path for path in resolved_paths if not path.exists()]
    if missing_paths:
        missing_display = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing PDF(s): {missing_display}")
    return resolved_paths


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


def _contains_timeout(error_message: str | None) -> bool:
    if not error_message:
        return False
    normalized = error_message.lower()
    return "timed out" in normalized or "timeout=" in normalized


def _has_substantive_page_content(page: Any) -> bool:
    if getattr(page, "text_blocks", None):
        return True
    if getattr(page, "table_blocks", None):
        return True
    if getattr(page, "figure_blocks", None):
        return True
    if getattr(page, "tables", None):
        return True
    return bool(canonicalize_text(getattr(page, "text", "")))


def _count_surviving_blocks(extracted: ExtractedDocument) -> int:
    total = 0
    for page in extracted.pages:
        page_total = 0
        page_total += len(getattr(page, "text_blocks", []))
        page_total += len(getattr(page, "table_blocks", []))
        page_total += len(getattr(page, "figure_blocks", []))
        if page_total == 0 and getattr(page, "tables", None):
            page_total += len(page.tables)
        if page_total == 0 and canonicalize_text(getattr(page, "text", "")):
            page_total = 1
        total += page_total
    return total


def assess_extraction_quality(
    extracted: ExtractedDocument,
    tree: PageIndexTree | None = None,
) -> dict[str, Any]:
    timed_out_page_count = sum(1 for page in extracted.pages if _contains_timeout(page.error_message))
    surviving_content_page_count = sum(1 for page in extracted.pages if _has_substantive_page_content(page))
    surviving_block_count = _count_surviving_blocks(extracted)
    non_root_pageindex_node_count = 0
    if tree is not None:
        non_root_pageindex_node_count = sum(1 for node in tree.nodes if node.section_path)

    extraction_quality = "successful"
    blockers: list[str] = []

    if tree is not None and non_root_pageindex_node_count == 0:
        blockers.append("root_only_pageindex")
    if surviving_block_count <= 1:
        blockers.append("single_minimal_surviving_block")
    if timed_out_page_count > 0 and surviving_content_page_count <= 1:
        blockers.append("timeouts_left_minimal_surviving_content")

    if blockers:
        extraction_quality = "partial_minimal"

    return {
        "extraction_quality": extraction_quality,
        "timed_out_page_count": timed_out_page_count,
        "surviving_content_page_count": surviving_content_page_count,
        "non_root_pageindex_node_count": non_root_pageindex_node_count,
        "query_derivation_blocker": "; ".join(blockers) if blockers else None,
    }


def build_retrieval_queries(tree: PageIndexTree, chunks: list[Chunk], limit: int = 3) -> list[LabeledRetrievalQuery]:
    chunk_ids_by_section: dict[tuple[str, ...], list[str]] = {}
    chunk_texts_by_section: dict[tuple[str, ...], list[str]] = {}
    for chunk in chunks:
        chunk_ids_by_section.setdefault(chunk.section_path, []).append(chunk.chunk_id or "")
        chunk_texts_by_section.setdefault(chunk.section_path, []).append(getattr(chunk, "text", ""))

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
        relevant_chunk_texts = [
            text
            for section_path, texts in chunk_texts_by_section.items()
            if section_path[: len(node.section_path)] == node.section_path
            for text in texts
        ]
        topic = build_query_topic(node, chunk_texts=relevant_chunk_texts)
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


def build_query_topic(node: PageIndexNode, chunk_texts: list[str] | None = None) -> str:
    synthetic_table_match = re.match(r"^Page\s+\d+\s+Table\s+\d+(?::\s+(?P<label>.+))?$", node.title)
    if synthetic_table_match is not None:
        label = canonicalize_text(synthetic_table_match.group("label") or "")
        if _is_high_quality_synthetic_query_label(label):
            return label
        derived = _synthetic_table_query_topic_from_texts(chunk_texts or [])
        if derived:
            return derived
    if node.summary:
        summary = canonicalize_text(node.summary)
        if summary:
            return summary
    title = re.sub(r"^\d+(?:\.\d+)*\s*", "", node.title).strip()
    return canonicalize_text(title)


def _is_high_quality_synthetic_query_label(label: str) -> bool:
    normalized = canonicalize_text(label)
    if not normalized:
        return False
    if "..." in normalized:
        return False
    alpha_tokens = re.findall(r"[A-Za-z][A-Za-z0-9&/-]{2,}", normalized)
    if len(alpha_tokens) >= 2:
        return True
    grounded_keywords = (
        "budget",
        "expense",
        "expenses",
        "allocation",
        "amount",
        "amounts",
        "total",
        "totals",
        "category",
        "item",
        "procurement",
        "audit",
        "finding",
        "target",
        "actual",
        "revenue",
        "cost",
        "vendor",
        "payment",
    )
    lowered = normalized.casefold()
    return any(re.search(rf"\b{re.escape(keyword)}\b", lowered) for keyword in grounded_keywords)


def _synthetic_table_query_topic_from_texts(texts: list[str]) -> str:
    joined = canonicalize_text("\n".join(texts))
    if not joined:
        return ""
    labels = _meaningful_table_query_labels(joined)
    if labels:
        return " / ".join(labels[:2])
    keywords = _grounded_table_query_keywords(joined)
    if keywords:
        return " / ".join(keyword.title() for keyword in keywords[:2])
    if _looks_like_numeric_table(joined):
        return "Amounts / Totals"
    return ""


def _meaningful_table_query_labels(text: str) -> list[str]:
    for line in text.splitlines():
        normalized = canonicalize_text(line)
        if not normalized:
            continue
        labels: list[str] = []
        seen: set[str] = set()
        for cell in normalized.split("|"):
            tokens = [_normalize_query_token(token) for token in re.findall(r"[A-Za-z][A-Za-z0-9&/-]{1,}", cell)]
            tokens = [token for token in tokens if token]
            grounded_tokens = [token for token in tokens if _is_high_quality_synthetic_query_label(token)]
            token_group = " ".join(grounded_tokens[:3]).strip()
            if not _is_high_quality_synthetic_query_label(token_group):
                continue
            lowered = token_group.casefold()
            if lowered in seen:
                continue
            seen.add(lowered)
            labels.append(token_group)
        if labels:
            return labels
    return []


def _grounded_table_query_keywords(text: str) -> list[str]:
    grounded_keywords = (
        "budget",
        "expense",
        "expenses",
        "allocation",
        "allocations",
        "amount",
        "amounts",
        "total",
        "totals",
        "category",
        "categories",
        "item",
        "items",
        "procurement",
        "audit",
        "finding",
        "findings",
        "target",
        "actual",
        "revenue",
        "cost",
        "vendor",
        "payment",
        "payments",
        "usd",
    )
    lowered = text.casefold()
    keywords = [
        keyword
        for keyword in grounded_keywords
        if re.search(rf"\b{re.escape(keyword)}\b", lowered)
    ]
    if _looks_like_numeric_table(text) and any(keyword in keywords for keyword in ("target", "budget", "expense", "revenue", "cost", "vendor", "usd")):
        if "amounts" not in keywords:
            keywords.append("amounts")
    return keywords


def _normalize_query_token(token: str) -> str:
    cleaned = token.strip(" .,:;|/\\()[]{}+-_")
    if len(cleaned) < 2:
        return ""
    if sum(char.isalpha() for char in cleaned) < 2:
        return ""
    return cleaned


def _looks_like_numeric_table(text: str) -> bool:
    numeric_values = re.findall(r"\b\d[\d,]*(?:\.\d+)?\b", text)
    return len(numeric_values) >= 4


def load_document_profile(doc_id: str | None) -> DocumentProfile | None:
    if not doc_id:
        return None
    profile_path = ROOT / ".refinery" / "profiles" / f"{doc_id}.json"
    if not profile_path.exists():
        return None
    return DocumentProfile.model_validate_json(profile_path.read_text(encoding="utf-8"))


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
    selection_row: dict[str, str] | None,
    args: argparse.Namespace,
    summary_backend: SummaryBackend,
    batch_output_dir: Path,
) -> DocumentEvalResult:
    doc_folder = batch_output_dir / slugify(pdf_path.stem)
    doc_folder.mkdir(parents=True, exist_ok=True)

    result = DocumentEvalResult(
        file_path=str(pdf_path),
        doc_id=None,
        document_class=None,
        extraction_quality="failed",
        retrieval_query_derivation_expected=False,
        section_inference_mode=None,
        success=False,
        failure_reason=None,
        page_count=0,
        timed_out_page_count=0,
        surviving_content_page_count=0,
        ldu_count=0,
        chunk_count=0,
        pageindex_node_count=0,
        non_root_pageindex_node_count=0,
        summaries_generated=False,
        query_derivation_blocker=None,
        vector_ingestion_succeeded=False,
        retrieval_evaluation_attempted=False,
        retrieval_evaluation_succeeded=False,
        retrieval_evaluation_failed=False,
        retrieval_evaluation_failure_reason=None,
        retrieval_evaluation_skipped=False,
        retrieval_evaluation_skip_reason=None,
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

        extraction_stats = assess_extraction_quality(extracted)
        result.extraction_quality = extraction_stats["extraction_quality"]
        result.timed_out_page_count = extraction_stats["timed_out_page_count"]
        result.surviving_content_page_count = extraction_stats["surviving_content_page_count"]
        result.query_derivation_blocker = extraction_stats["query_derivation_blocker"]

        profile = load_document_profile(extracted.doc_id)
        document_policy = resolve_document_class(
            file_name=pdf_path.name,
            profile=profile,
            row=selection_row,
        )
        result.document_class = document_policy.document_class.value
        result.retrieval_query_derivation_expected = document_policy.retrieval_query_derivation_expected
        result.section_inference_mode = document_policy.section_inference_mode.value

        engine = ChunkingEngine(
            config=ChunkingConfig(),
            section_inferer=SectionPathInferer(
                mode=SectionInferenceMode(document_policy.section_inference_mode.value)
            ),
        )
        ldus = engine.build_ldus(extracted)
        chunks = engine.build_chunks(extracted, ldus=ldus)
        result.ldu_count = len(ldus)
        result.chunk_count = len(chunks)

        tree = PageIndexBuilder().build(doc_id=extracted.doc_id, ldus=ldus)
        summarized_tree = PageIndexSummarizer(summary_backend).summarize_tree(tree=tree, ldus=ldus)
        result.pageindex_node_count = len(summarized_tree.nodes)
        extraction_stats = assess_extraction_quality(extracted, summarized_tree)
        result.extraction_quality = extraction_stats["extraction_quality"]
        result.timed_out_page_count = extraction_stats["timed_out_page_count"]
        result.surviving_content_page_count = extraction_stats["surviving_content_page_count"]
        result.non_root_pageindex_node_count = extraction_stats["non_root_pageindex_node_count"]
        result.query_derivation_blocker = extraction_stats["query_derivation_blocker"]
        result.summaries_generated = any(
            bool((node.summary or "").strip()) for node in summarized_tree.nodes if node.section_path
        )

        embedding_backend = build_embedding_backend(args)
        collection_name = f"phase3_batch_eval_{extracted.doc_id}"
        vector_store = ChromaVectorStore(
            embedding_backend=embedding_backend,
            collection_name=collection_name,
            persist_directory=args.persist_dir,
        )
        vector_store.ingest_ldus(ldus)
        vector_store.ingest_chunks(chunks)
        result.vector_ingestion_succeeded = True

        result.retrieval_evaluation_attempted = True
        queries = build_retrieval_queries(summarized_tree, chunks)
        LOGGER.info("Derived %s retrieval evaluation queries for %s", len(queries), pdf_path.name)
        query_payload: dict[str, Any] = {
            "collection_name": collection_name,
            "document_class": result.document_class,
            "extraction_quality": result.extraction_quality,
            "timed_out_page_count": result.timed_out_page_count,
            "surviving_content_page_count": result.surviving_content_page_count,
            "non_root_pageindex_node_count": result.non_root_pageindex_node_count,
            "query_derivation_blocker": result.query_derivation_blocker,
            "retrieval_query_derivation_expected": result.retrieval_query_derivation_expected,
            "section_inference_mode": result.section_inference_mode,
            "queries": [asdict(query) for query in queries],
            "attempted": True,
            "succeeded": False,
            "failed": False,
            "failure_reason": None,
            "skipped": False,
            "skip_reason": None,
        }
        if not queries:
            if document_policy.zero_query_outcome == "failed":
                result.retrieval_evaluation_failed = True
                result.retrieval_evaluation_failure_reason = (
                    f"{NO_RETRIEVAL_QUERIES_REASON}; {document_policy.zero_query_reason}"
                )
                query_payload["failed"] = True
                query_payload["failure_reason"] = result.retrieval_evaluation_failure_reason
            else:
                result.retrieval_evaluation_skipped = True
                result.retrieval_evaluation_skip_reason = (
                    f"{NO_RETRIEVAL_QUERIES_REASON}; {document_policy.zero_query_reason}"
                )
                query_payload["skipped"] = True
                query_payload["skip_reason"] = result.retrieval_evaluation_skip_reason
        else:
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
            query_payload["succeeded"] = True
            query_payload["baseline"] = {
                "metrics": asdict(baseline_report.metrics),
                "per_query": [asdict(item) for item in baseline_report.per_query],
            }
            query_payload["pageindex_assisted"] = {
                "metrics": asdict(assisted_report.metrics),
                "per_query": [asdict(item) for item in assisted_report.per_query],
            }

        save_artifacts(
            output_dir=doc_folder,
            extracted=extracted,
            ldus=ldus,
            chunks=chunks,
            tree=summarized_tree,
            query_payload=query_payload,
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
        result.extraction_quality = "failed"
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
    quality_counter = Counter(result.extraction_quality for result in results)
    retrieval_skips = sum(1 for result in results if result.retrieval_evaluation_skipped)
    retrieval_failures = sum(1 for result in results if result.retrieval_evaluation_failed)
    failures = len(results) - successes
    failure_counter = Counter(
        result.failure_reason for result in results if result.failure_reason
    )
    common_failures = failure_counter.most_common(3)

    print("\nBatch summary")
    print(f"successes={successes}")
    print(f"failures={failures}")
    print(f"extraction_quality_successful={quality_counter.get('successful', 0)}")
    print(f"extraction_quality_partial_minimal={quality_counter.get('partial_minimal', 0)}")
    print(f"extraction_quality_failed={quality_counter.get('failed', 0)}")
    print(f"retrieval_evaluation_skips={retrieval_skips}")
    print(f"retrieval_evaluation_failures={retrieval_failures}")
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

    rows: list[dict[str, str]]
    pdf_paths: list[Path]
    if args.pdf_paths:
        try:
            pdf_paths = resolve_explicit_pdf_paths(args.pdf_paths)
        except Exception as exc:
            LOGGER.error("Failed to resolve explicit PDF paths: %s", exc)
            return 1
        rows = [{"file": str(path)} for path in pdf_paths]
        LOGGER.info("Using %s explicitly selected PDF(s)", len(pdf_paths))
    else:
        LOGGER.info("Loading selected document list from %s", csv_path)
        try:
            rows = load_selected_rows(csv_path, args.max_docs)
        except Exception as exc:
            LOGGER.error("Failed to read CSV %s: %s", csv_path, exc)
            return 1
        pdf_paths = [resolve_pdf_path((row.get("file") or "").strip(), data_dir) for row in rows]

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
                document_class=None,
                extraction_quality="failed",
                retrieval_query_derivation_expected=False,
                section_inference_mode=None,
                success=False,
                failure_reason=reason,
                page_count=0,
                timed_out_page_count=0,
                surviving_content_page_count=0,
                ldu_count=0,
                chunk_count=0,
                pageindex_node_count=0,
                non_root_pageindex_node_count=0,
                summaries_generated=False,
                query_derivation_blocker=None,
                vector_ingestion_succeeded=False,
                retrieval_evaluation_attempted=False,
                retrieval_evaluation_succeeded=False,
                retrieval_evaluation_failed=False,
                retrieval_evaluation_failure_reason=None,
                retrieval_evaluation_skipped=False,
                retrieval_evaluation_skip_reason=None,
                artifacts_dir=str(output_dir / f"row_{index}"),
            )
            save_json(Path(result.artifacts_dir) / "document_report.json", asdict(result))
            results.append(result)
            continue

        pdf_path = pdf_paths[index - 1]
        LOGGER.info("(%s/%s) %s", index, len(rows), pdf_path.name)
        results.append(
            process_document(
                pdf_path=pdf_path,
                selection_row=row,
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
