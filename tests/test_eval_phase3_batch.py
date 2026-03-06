from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import types

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

fake_extractor_module = types.ModuleType("src.agents.extractor")
fake_extractor_module.run_extraction = lambda pdf_path: None
sys.modules.setdefault("src.agents.extractor", fake_extractor_module)

from scripts import eval_phase3_batch as batch


@dataclass
class FakeExtracted:
    doc_id: str = "doc123"
    page_count: int = 2
    status: str = "success"
    error_message: str | None = None

    def model_dump_json(self, indent: int = 2) -> str:
        _ = indent
        return '{"doc_id":"doc123"}'


@dataclass
class FakeRecord:
    record_id: str

    def model_dump(self) -> dict[str, str]:
        return {"record_id": self.record_id}


@dataclass
class FakeNode:
    node_id: str
    title: str
    section_path: tuple[str, ...]
    summary: str
    order_index: int = 0

    def model_dump(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "title": self.title,
            "section_path": list(self.section_path),
            "summary": self.summary,
            "order_index": self.order_index,
        }


@dataclass
class FakeTree:
    doc_id: str
    root_id: str
    nodes: list[FakeNode]


@dataclass
class FakeMetrics:
    precision_at_3: float
    hit_rate: float


@dataclass
class FakePerQuery:
    query_id: str
    topic: str
    retrieved_record_ids: tuple[str, ...]
    relevant_record_ids: tuple[str, ...]


@dataclass
class FakeEvalReport:
    metrics: FakeMetrics
    per_query: list[FakePerQuery]


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        persist_dir=tmp_path / "persist",
        top_k=3,
        section_top_k=3,
    )


def _patch_pipeline(monkeypatch, tmp_path: Path, *, queries: list[object]) -> dict[str, object]:
    captured: dict[str, object] = {}
    tree = FakeTree(
        doc_id="doc123",
        root_id="root",
        nodes=[FakeNode(node_id="n1", title="Section", section_path=("1 Section",), summary="Section summary")],
    )

    monkeypatch.setattr(batch, "run_extraction", lambda pdf_path: FakeExtracted())

    class FakeEngine:
        def __init__(self, config, section_inferer=None) -> None:
            self.config = config
            self.section_inferer = section_inferer

        def build_ldus(self, extracted):
            _ = extracted
            return [FakeRecord("ldu-1")]

        def build_chunks(self, extracted):
            _ = extracted
            return [FakeRecord("chunk-1")]

    class FakeVectorStore:
        def __init__(self, *, embedding_backend, collection_name, persist_directory) -> None:
            _ = embedding_backend, collection_name, persist_directory

        def ingest_ldus(self, ldus) -> None:
            _ = ldus

        def ingest_chunks(self, chunks) -> None:
            _ = chunks

    monkeypatch.setattr(batch, "ChunkingEngine", FakeEngine)
    monkeypatch.setattr(batch, "PageIndexBuilder", lambda: SimpleNamespace(build=lambda doc_id, ldus: tree))
    monkeypatch.setattr(
        batch,
        "PageIndexSummarizer",
        lambda summary_backend: SimpleNamespace(summarize_tree=lambda tree, ldus: tree),
    )
    monkeypatch.setattr(batch, "ChromaVectorStore", FakeVectorStore)
    monkeypatch.setattr(batch, "build_retrieval_queries", lambda summarized_tree, chunks: queries)
    monkeypatch.setattr(batch, "save_artifacts", lambda **kwargs: captured.update(kwargs))
    return captured


def test_process_document_marks_missing_retrieval_queries_as_skipped(monkeypatch, tmp_path) -> None:
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    captured = _patch_pipeline(monkeypatch, tmp_path, queries=[])

    result = batch.process_document(
        pdf_path=pdf_path,
        selection_row={"document_class": "table_heavy_financial_administrative"},
        args=_args(tmp_path),
        summary_backend=object(),
        batch_output_dir=tmp_path / "out",
    )

    assert result.success is True
    assert result.failure_reason is None
    assert result.vector_ingestion_succeeded is True
    assert result.retrieval_evaluation_attempted is True
    assert result.retrieval_evaluation_succeeded is False
    assert result.retrieval_evaluation_failed is False
    assert result.retrieval_evaluation_skipped is True
    assert batch.NO_RETRIEVAL_QUERIES_REASON in (result.retrieval_evaluation_skip_reason or "")
    assert result.document_class == "table_heavy_financial_administrative"
    assert captured["query_payload"]["skipped"] is True
    assert batch.NO_RETRIEVAL_QUERIES_REASON in (captured["query_payload"]["skip_reason"] or "")


def test_process_document_marks_missing_retrieval_queries_as_failed_when_expected(monkeypatch, tmp_path) -> None:
    pdf_path = tmp_path / "narrative.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    captured = _patch_pipeline(monkeypatch, tmp_path, queries=[])

    result = batch.process_document(
        pdf_path=pdf_path,
        selection_row={"document_class": "narrative_report_like"},
        args=_args(tmp_path),
        summary_backend=object(),
        batch_output_dir=tmp_path / "out",
    )

    assert result.success is True
    assert result.retrieval_evaluation_attempted is True
    assert result.retrieval_evaluation_succeeded is False
    assert result.retrieval_evaluation_failed is True
    assert result.retrieval_evaluation_skipped is False
    assert batch.NO_RETRIEVAL_QUERIES_REASON in (result.retrieval_evaluation_failure_reason or "")
    assert captured["query_payload"]["failed"] is True
    assert batch.NO_RETRIEVAL_QUERIES_REASON in (captured["query_payload"]["failure_reason"] or "")


def test_process_document_preserves_real_processing_failures(monkeypatch, tmp_path) -> None:
    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(batch, "run_extraction", lambda pdf_path: (_ for _ in ()).throw(RuntimeError("boom")))

    result = batch.process_document(
        pdf_path=pdf_path,
        selection_row=None,
        args=_args(tmp_path),
        summary_backend=object(),
        batch_output_dir=tmp_path / "out",
    )

    failure_path = Path(result.artifacts_dir) / "failure.txt"
    assert result.success is False
    assert result.failure_reason == "boom"
    assert result.retrieval_evaluation_attempted is False
    assert result.retrieval_evaluation_skipped is False
    assert failure_path.exists()
    assert "RuntimeError: boom" in failure_path.read_text(encoding="utf-8")


def test_process_document_marks_retrieval_evaluation_success(monkeypatch, tmp_path) -> None:
    pdf_path = tmp_path / "success.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    query = batch.LabeledRetrievalQuery(
        query_id="q1",
        topic="results",
        relevant_record_ids=("chunk-1",),
    )
    captured = _patch_pipeline(monkeypatch, tmp_path, queries=[query])

    class FakeEvaluator:
        def __init__(self, *, vector_backend, page_index_backend) -> None:
            _ = vector_backend, page_index_backend

        def evaluate_baseline(self, queries, top_k, record_type):
            _ = queries, top_k, record_type
            return FakeEvalReport(
                metrics=FakeMetrics(precision_at_3=1.0, hit_rate=1.0),
                per_query=[
                    FakePerQuery(
                        query_id="q1",
                        topic="results",
                        retrieved_record_ids=("chunk-1",),
                        relevant_record_ids=("chunk-1",),
                    )
                ],
            )

        def evaluate_pageindex_assisted(self, tree, queries, section_top_k, top_k, record_type):
            _ = tree, queries, section_top_k, top_k, record_type
            return FakeEvalReport(
                metrics=FakeMetrics(precision_at_3=1.0, hit_rate=1.0),
                per_query=[
                    FakePerQuery(
                        query_id="q1",
                        topic="results",
                        retrieved_record_ids=("chunk-1",),
                        relevant_record_ids=("chunk-1",),
                    )
                ],
            )

    monkeypatch.setattr(batch, "RetrievalEvaluator", FakeEvaluator)
    monkeypatch.setattr(batch, "PageIndexQueryEngine", lambda: object())

    result = batch.process_document(
        pdf_path=pdf_path,
        selection_row={"document_class": "narrative_report_like"},
        args=_args(tmp_path),
        summary_backend=object(),
        batch_output_dir=tmp_path / "out",
    )

    assert result.success is True
    assert result.failure_reason is None
    assert result.retrieval_evaluation_attempted is True
    assert result.retrieval_evaluation_succeeded is True
    assert result.retrieval_evaluation_failed is False
    assert result.retrieval_evaluation_skipped is False
    assert result.retrieval_evaluation_skip_reason is None
    assert captured["query_payload"]["succeeded"] is True
    assert "baseline" in captured["query_payload"]
    assert "pageindex_assisted" in captured["query_payload"]


def test_print_run_summary_counts_skips_separately(capsys) -> None:
    results = [
        batch.DocumentEvalResult(
            file_path="a.pdf",
            doc_id="doc-a",
            document_class="table_heavy_financial_administrative",
            retrieval_query_derivation_expected=False,
            section_inference_mode="relaxed",
            success=True,
            failure_reason=None,
            page_count=1,
            ldu_count=1,
            chunk_count=1,
            pageindex_node_count=1,
            summaries_generated=True,
            vector_ingestion_succeeded=True,
            retrieval_evaluation_attempted=True,
            retrieval_evaluation_succeeded=False,
            retrieval_evaluation_failed=False,
            retrieval_evaluation_failure_reason=None,
            retrieval_evaluation_skipped=True,
            retrieval_evaluation_skip_reason=batch.NO_RETRIEVAL_QUERIES_REASON,
            artifacts_dir="out/a",
        ),
        batch.DocumentEvalResult(
            file_path="b.pdf",
            doc_id=None,
            document_class=None,
            retrieval_query_derivation_expected=False,
            section_inference_mode=None,
            success=False,
            failure_reason="boom",
            page_count=0,
            ldu_count=0,
            chunk_count=0,
            pageindex_node_count=0,
            summaries_generated=False,
            vector_ingestion_succeeded=False,
            retrieval_evaluation_attempted=False,
            retrieval_evaluation_succeeded=False,
            retrieval_evaluation_failed=False,
            retrieval_evaluation_failure_reason=None,
            retrieval_evaluation_skipped=False,
            retrieval_evaluation_skip_reason=None,
            artifacts_dir="out/b",
        ),
    ]

    batch.print_run_summary(results)
    output = capsys.readouterr().out

    assert "successes=1" in output
    assert "failures=1" in output
    assert "retrieval_evaluation_skips=1" in output
    assert "retrieval_evaluation_failures=0" in output
