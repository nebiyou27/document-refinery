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
    status: str = "ok"
    error_message: str | None = None
    pages: list[object] | None = None
    metadata: object | None = None

    def model_dump_json(self, indent: int = 2) -> str:
        _ = indent
        return '{"doc_id":"doc123"}'

    def __post_init__(self) -> None:
        if self.pages is None:
            self.pages = [
                FakePage(page_number=1, text="alpha", text_blocks=[FakeBlock("alpha")]),
                FakePage(page_number=2, text="beta", text_blocks=[FakeBlock("beta")]),
            ]


@dataclass
class FakeBlock:
    text: str = "content"


@dataclass
class FakePage:
    page_number: int
    status: str = "ok"
    text: str = ""
    tables: list[dict] | None = None
    text_blocks: list[object] | None = None
    table_blocks: list[object] | None = None
    figure_blocks: list[object] | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if self.tables is None:
            self.tables = []
        if self.text_blocks is None:
            self.text_blocks = []
        if self.table_blocks is None:
            self.table_blocks = []
        if self.figure_blocks is None:
            self.figure_blocks = []


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
    depth: int = 1
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


def _patch_pipeline(
    monkeypatch,
    tmp_path: Path,
    *,
    queries: list[object],
    extracted: FakeExtracted | None = None,
    tree: FakeTree | None = None,
) -> dict[str, object]:
    captured: dict[str, object] = {}
    tree = tree or FakeTree(
        doc_id="doc123",
        root_id="root",
        nodes=[FakeNode(node_id="n1", title="Section", section_path=("1 Section",), summary="Section summary")],
    )

    monkeypatch.setattr(batch, "run_extraction", lambda pdf_path: extracted or FakeExtracted())

    class FakeEngine:
        def __init__(self, config, section_inferer=None) -> None:
            self.config = config
            self.section_inferer = section_inferer

        def build_ldus(self, extracted):
            _ = extracted
            return [FakeRecord("ldu-1")]

        def build_chunks(self, extracted, ldus=None):
            _ = extracted, ldus
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
    assert result.extraction_quality == "successful"
    assert result.failure_reason is None
    assert result.timed_out_page_count == 0
    assert result.surviving_content_page_count == 2
    assert result.non_root_pageindex_node_count == 1
    assert result.query_derivation_blocker is None
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
    assert result.extraction_quality == "successful"
    assert result.retrieval_evaluation_attempted is True
    assert result.retrieval_evaluation_succeeded is False
    assert result.retrieval_evaluation_failed is True
    assert result.retrieval_evaluation_skipped is False
    assert batch.NO_RETRIEVAL_QUERIES_REASON in (result.retrieval_evaluation_failure_reason or "")
    assert captured["query_payload"]["failed"] is True
    assert batch.NO_RETRIEVAL_QUERIES_REASON in (captured["query_payload"]["failure_reason"] or "")


def test_process_document_classifies_true_extraction_failure_as_failed(monkeypatch, tmp_path) -> None:
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
    assert result.extraction_quality == "failed"
    assert result.failure_reason == "boom"
    assert result.timed_out_page_count == 0
    assert result.surviving_content_page_count == 0
    assert result.non_root_pageindex_node_count == 0
    assert result.query_derivation_blocker is None
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
    assert result.extraction_quality == "successful"
    assert result.failure_reason is None
    assert result.retrieval_evaluation_attempted is True
    assert result.retrieval_evaluation_succeeded is True
    assert result.retrieval_evaluation_failed is False
    assert result.retrieval_evaluation_skipped is False
    assert result.retrieval_evaluation_skip_reason is None
    assert captured["query_payload"]["succeeded"] is True
    assert "baseline" in captured["query_payload"]
    assert "pageindex_assisted" in captured["query_payload"]


def test_process_document_marks_minimal_degraded_extraction(monkeypatch, tmp_path) -> None:
    pdf_path = tmp_path / "degraded.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")
    extracted = FakeExtracted(
        page_count=3,
        pages=[
            FakePage(
                page_number=1,
                status="error",
                error_message="strategy_c_failed: vlm_failed(timeout=45.0s): Read timed out",
            ),
            FakePage(
                page_number=2,
                status="error",
                error_message="strategy_c_failed: HTTPConnectionPool: Read timed out",
            ),
            FakePage(page_number=3, text="only surviving text", text_blocks=[FakeBlock("only surviving text")]),
        ],
    )
    root_only_tree = FakeTree(
        doc_id="doc123",
        root_id="root",
        nodes=[FakeNode(node_id="root", title="ROOT", section_path=(), summary="", order_index=0)],
    )
    captured = _patch_pipeline(monkeypatch, tmp_path, queries=[], extracted=extracted, tree=root_only_tree)

    result = batch.process_document(
        pdf_path=pdf_path,
        selection_row={"document_class": "narrative_report_like"},
        args=_args(tmp_path),
        summary_backend=object(),
        batch_output_dir=tmp_path / "out",
    )

    assert result.success is True
    assert result.extraction_quality == "partial_minimal"
    assert result.timed_out_page_count == 2
    assert result.surviving_content_page_count == 1
    assert result.non_root_pageindex_node_count == 0
    assert result.query_derivation_blocker == (
        "root_only_pageindex; single_minimal_surviving_block; timeouts_left_minimal_surviving_content"
    )
    assert result.retrieval_evaluation_failed is True
    assert captured["query_payload"]["extraction_quality"] == "partial_minimal"
    assert captured["query_payload"]["query_derivation_blocker"] == result.query_derivation_blocker


def test_build_retrieval_queries_derives_queries_from_synthetic_table_sections() -> None:
    tree = FakeTree(
        doc_id="doc123",
        root_id="root",
        nodes=[
            FakeNode(node_id="root", title="ROOT", section_path=(), summary="", order_index=0),
            FakeNode(
                node_id="n1",
                title="Page 1 Table 1: Budget Allocation Summary",
                section_path=("Page 1 Table 1",),
                summary="| 92012. | 92012./c/ | 9/2013. | ...",
                order_index=1,
            ),
            FakeNode(
                node_id="n2",
                title="Page 2 Table 1: Vendor Procurement Amounts",
                section_path=("Page 2 Table 1",),
                summary="| 吊 | n/Target/ | nc/ | U-lo | ...",
                order_index=2,
            ),
        ],
    )
    chunks = [
        SimpleNamespace(chunk_id="chunk-1", section_path=("Page 1 Table 1",)),
        SimpleNamespace(chunk_id="chunk-2", section_path=("Page 2 Table 1",)),
    ]

    queries = batch.build_retrieval_queries(tree=tree, chunks=chunks, limit=3)

    assert [query.topic for query in queries] == [
        "Budget Allocation Summary",
        "Vendor Procurement Amounts",
    ]
    assert [query.relevant_record_ids for query in queries] == [
        ("chunk-1",),
        ("chunk-2",),
    ]


def test_build_query_topic_falls_back_to_summary_for_low_quality_synthetic_label() -> None:
    node = FakeNode(
        node_id="n1",
        title="Page 3 Table 1: 6C",
        section_path=("Page 3 Table 1",),
        summary="Budget and expense allocation totals for categories",
        order_index=1,
    )

    topic = batch.build_query_topic(node)

    assert topic == "Budget and expense allocation totals for categories"


def test_build_query_topic_derives_synthetic_table_topic_from_chunk_texts_when_label_is_weak() -> None:
    node = FakeNode(
        node_id="n1",
        title="Page 2 Table 1",
        section_path=("Page 2 Table 1",),
        summary="",
        order_index=1,
    )

    topic = batch.build_query_topic(
        node,
        chunk_texts=["Code | Budget | Expense | Total\nA1 | 100 | 90 | 10"],
    )

    assert topic == "Budget / Expense"


def test_build_query_topic_returns_generic_amounts_topic_for_numeric_table_without_grounded_labels() -> None:
    node = FakeNode(
        node_id="n1",
        title="Page 1 Table 1",
        section_path=("Page 1 Table 1",),
        summary="",
        order_index=1,
    )

    topic = batch.build_query_topic(
        node,
        chunk_texts=["001 | 2013 | 92012 | 43077\n7691 | 33585 | 23637 | 18100"],
    )

    assert topic == "Amounts / Totals"


def test_print_run_summary_counts_skips_separately(capsys) -> None:
    results = [
        batch.DocumentEvalResult(
            file_path="a.pdf",
            doc_id="doc-a",
            document_class="table_heavy_financial_administrative",
            extraction_quality="successful",
            retrieval_query_derivation_expected=False,
            section_inference_mode="relaxed",
            success=True,
            failure_reason=None,
            page_count=1,
            timed_out_page_count=0,
            surviving_content_page_count=1,
            ldu_count=1,
            chunk_count=1,
            pageindex_node_count=1,
            non_root_pageindex_node_count=1,
            summaries_generated=True,
            query_derivation_blocker=None,
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
            extraction_quality="failed",
            retrieval_query_derivation_expected=False,
            section_inference_mode=None,
            success=False,
            failure_reason="boom",
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
            artifacts_dir="out/b",
        ),
    ]

    batch.print_run_summary(results)
    output = capsys.readouterr().out

    assert "successes=1" in output
    assert "failures=1" in output
    assert "extraction_quality_successful=1" in output
    assert "extraction_quality_partial_minimal=0" in output
    assert "extraction_quality_failed=1" in output
    assert "retrieval_evaluation_skips=1" in output
    assert "retrieval_evaluation_failures=0" in output
