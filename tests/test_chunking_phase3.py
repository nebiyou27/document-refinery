from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chunking import (
    ChunkValidationError,
    ChunkValidator,
    ChunkingConfig,
    ChunkingEngine,
    OllamaSummaryBackend,
    PageIndexBuilder,
    PageIndexSummarizer,
    SummaryBackendError,
    SummaryInput,
)
from src.models.chunking import LDU, LDUKind
from src.models.extracted_document import (
    ExtractedDocument,
    ExtractedPage,
    ExtractionMetadata,
    FigureBlock,
    TableBlock,
    TextBlock,
)
from src.utils.hashing import ldu_content_hash


def _metadata() -> ExtractionMetadata:
    return ExtractionMetadata(
        strategy_used="strategy_b",
        confidence_score=0.95,
        processing_time_sec=0.01,
        cost_estimate_usd=0.0,
        escalation_triggered=False,
    )


def _document_from_pages(*pages: ExtractedPage) -> ExtractedDocument:
    return ExtractedDocument(
        doc_id="doc123",
        file_name="sample.pdf",
        file_path="data/sample.pdf",
        page_count=len(pages),
        metadata=_metadata(),
        pages=list(pages),
    )


def _ldu(
    *,
    text: str,
    page_number: int,
    source_block_order: int,
    section_path: tuple[str, ...],
) -> LDU:
    return LDU(
        doc_id="doc123",
        page_number=page_number,
        bbox=(0.0, float(source_block_order), 10.0, float(source_block_order + 1)),
        kind=LDUKind.text,
        text=text,
        section_path=section_path,
        source_block_order=source_block_order,
    )


class FakeSummaryBackend:
    def summarize(self, summary_input: SummaryInput) -> str:
        text = " ".join(summary_input.source_text.split())
        preview = text[:39]
        return f"{summary_input.title}: {preview}"


class MockOllamaClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def chat(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return self.response


class FailingOllamaClient:
    def chat(self, **kwargs: object) -> dict[str, object]:
        _ = kwargs
        raise RuntimeError("connection lost")


def test_ldu_content_hash_is_stable_and_page_agnostic() -> None:
    first = LDU(
        doc_id="doc123",
        page_number=1,
        bbox=(0.0, 0.0, 10.0, 10.0),
        kind=LDUKind.text,
        text="Revenue  grew\n\n  25% ",
        section_path=("Financial Highlights",),
    )
    second = LDU(
        doc_id="doc123",
        page_number=9,
        bbox=(5.0, 5.0, 15.0, 15.0),
        kind=LDUKind.text,
        text="Revenue grew 25%",
        section_path=("  financial   highlights ",),
    )

    assert first.content_hash == second.content_hash
    assert first.ldu_id != second.ldu_id
    assert first.content_hash == ldu_content_hash("Revenue grew 25%", ("Financial Highlights",))


def test_chunk_validator_rejects_table_ldu_without_header_row() -> None:
    validator = ChunkValidator()
    invalid_table = LDU(
        doc_id="doc123",
        page_number=1,
        bbox=(0.0, 0.0, 20.0, 20.0),
        kind=LDUKind.table,
        text="100 | 200",
        metadata={"row_count": 1, "header_row": []},
    )

    with pytest.raises(ChunkValidationError):
        validator.raise_for_issues(validator.validate_ldu(invalid_table))


def test_chunk_validator_rejects_figure_ldu_without_caption_metadata() -> None:
    validator = ChunkValidator()
    invalid_figure = LDU(
        doc_id="doc123",
        page_number=1,
        bbox=(0.0, 0.0, 20.0, 20.0),
        kind=LDUKind.figure,
        text="[figure]",
        metadata={"caption": ""},
    )

    with pytest.raises(ChunkValidationError):
        validator.raise_for_issues(validator.validate_ldu(invalid_figure))


def test_chunking_engine_emits_deterministic_chunks_with_explicit_rules() -> None:
    page = ExtractedPage(
        doc_id="doc123",
        page_number=1,
        metadata=_metadata(),
        signals={"char_count": 40, "char_density": 0.1, "image_area_ratio": 0.0, "table_count": 1},
        text_blocks=[
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="Executive summary",
                bbox=(0.0, 0.0, 50.0, 10.0),
                reading_order=0,
                content_hash="t1",
            ),
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="Revenue increased year over year.",
                bbox=(0.0, 12.0, 50.0, 22.0),
                reading_order=1,
                content_hash="t2",
            ),
        ],
        table_blocks=[
            TableBlock(
                doc_id="doc123",
                page_number=1,
                bbox=(0.0, 25.0, 50.0, 45.0),
                content_hash="tb1",
                table_index=0,
                rows=[["Year", "Revenue"], ["2025", "100"]],
            )
        ],
        figure_blocks=[
            FigureBlock(
                doc_id="doc123",
                page_number=1,
                bbox=(0.0, 46.0, 50.0, 66.0),
                content_hash="fg1",
                caption="Revenue trend chart",
            )
        ],
        page_content_hash="page1",
    )
    document = _document_from_pages(page)

    engine = ChunkingEngine(config=ChunkingConfig(max_chunk_chars=200))
    first_run = engine.build_chunks(document)
    second_run = engine.build_chunks(document)

    assert len(first_run) == 3
    assert [chunk.text for chunk in first_run] == [
        "Executive summary\n\nRevenue increased year over year.",
        "Year | Revenue\n2025 | 100",
        "Revenue trend chart",
    ]
    assert first_run[0].metadata["kinds"] == ["text", "text"]
    assert first_run[1].metadata["kinds"] == ["table"]
    assert first_run[2].metadata["kinds"] == ["figure"]
    assert [chunk.chunk_id for chunk in first_run] == [chunk.chunk_id for chunk in second_run]
    assert [chunk.content_hash for chunk in first_run] == [chunk.content_hash for chunk in second_run]


def test_section_path_inference_for_numbered_headings() -> None:
    page = ExtractedPage(
        doc_id="doc123",
        page_number=1,
        metadata=_metadata(),
        signals={"char_count": 60, "char_density": 0.1, "image_area_ratio": 0.0, "table_count": 0},
        text_blocks=[
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="3 Results",
                bbox=(0.0, 0.0, 60.0, 16.0),
                reading_order=0,
                content_hash="t1",
            ),
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="The retrieval pipeline improved materially.",
                bbox=(0.0, 20.0, 60.0, 32.0),
                reading_order=1,
                content_hash="t2",
            ),
        ],
        page_content_hash="page1",
    )

    ldus = ChunkingEngine().build_ldus(_document_from_pages(page))

    assert [ldu.section_path for ldu in ldus] == [("3 Results",), ("3 Results",)]


def test_section_path_inference_for_nested_numbered_headings() -> None:
    page = ExtractedPage(
        doc_id="doc123",
        page_number=1,
        metadata=_metadata(),
        signals={"char_count": 120, "char_density": 0.1, "image_area_ratio": 0.0, "table_count": 0},
        text_blocks=[
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="3 Results",
                bbox=(0.0, 0.0, 80.0, 16.0),
                reading_order=0,
                content_hash="t1",
            ),
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="3.2 Retrieval Precision",
                bbox=(0.0, 18.0, 80.0, 34.0),
                reading_order=1,
                content_hash="t2",
            ),
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="Precision improved by 8%.",
                bbox=(0.0, 38.0, 80.0, 50.0),
                reading_order=2,
                content_hash="t3",
            ),
        ],
        page_content_hash="page1",
    )

    ldus = ChunkingEngine().build_ldus(_document_from_pages(page))

    assert ldus[0].section_path == ("3 Results",)
    assert ldus[1].section_path == ("3 Results", "3.2 Retrieval Precision")
    assert ldus[2].section_path == ("3 Results", "3.2 Retrieval Precision")


def test_section_context_is_inherited_when_body_blocks_have_no_heading() -> None:
    page = ExtractedPage(
        doc_id="doc123",
        page_number=1,
        metadata=_metadata(),
        signals={"char_count": 140, "char_density": 0.1, "image_area_ratio": 0.0, "table_count": 1},
        text_blocks=[
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="2 Methods",
                bbox=(0.0, 0.0, 60.0, 16.0),
                reading_order=0,
                content_hash="t1",
            ),
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="We evaluated the pipeline on 10 benchmark queries.",
                bbox=(0.0, 20.0, 60.0, 32.0),
                reading_order=1,
                content_hash="t2",
            ),
        ],
        table_blocks=[
            TableBlock(
                doc_id="doc123",
                page_number=1,
                bbox=(0.0, 36.0, 60.0, 56.0),
                content_hash="tb1",
                table_index=0,
                rows=[["Metric", "Value"], ["Queries", "10"]],
            )
        ],
        page_content_hash="page1",
    )

    ldus = ChunkingEngine().build_ldus(_document_from_pages(page))

    assert [ldu.section_path for ldu in ldus] == [
        ("2 Methods",),
        ("2 Methods",),
        ("2 Methods",),
    ]


def test_section_boundary_changes_across_pages_affect_paths_hashes_and_chunks() -> None:
    first_page = ExtractedPage(
        doc_id="doc123",
        page_number=1,
        metadata=_metadata(),
        signals={"char_count": 100, "char_density": 0.1, "image_area_ratio": 0.0, "table_count": 0},
        text_blocks=[
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="1 Introduction",
                bbox=(0.0, 0.0, 70.0, 16.0),
                reading_order=0,
                content_hash="p1t1",
            ),
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="Shared body text",
                bbox=(0.0, 20.0, 70.0, 32.0),
                reading_order=1,
                content_hash="p1t2",
            ),
        ],
        page_content_hash="page1",
    )
    second_page = ExtractedPage(
        doc_id="doc123",
        page_number=2,
        metadata=_metadata(),
        signals={"char_count": 100, "char_density": 0.1, "image_area_ratio": 0.0, "table_count": 0},
        text_blocks=[
            TextBlock(
                doc_id="doc123",
                page_number=2,
                text="2 Discussion",
                bbox=(0.0, 0.0, 70.0, 16.0),
                reading_order=0,
                content_hash="p2t1",
            ),
            TextBlock(
                doc_id="doc123",
                page_number=2,
                text="Shared body text",
                bbox=(0.0, 20.0, 70.0, 32.0),
                reading_order=1,
                content_hash="p2t2",
            ),
        ],
        page_content_hash="page2",
    )

    engine = ChunkingEngine(config=ChunkingConfig(max_chunk_chars=200))
    document = _document_from_pages(first_page, second_page)

    ldus = engine.build_ldus(document)
    chunks = engine.build_chunks(document)

    assert ldus[1].section_path == ("1 Introduction",)
    assert ldus[3].section_path == ("2 Discussion",)
    assert ldus[1].content_hash != ldus[3].content_hash
    assert len(chunks) == 2
    assert chunks[0].section_path == ("1 Introduction",)
    assert chunks[1].section_path == ("2 Discussion",)


def test_page_index_builder_handles_flat_section_tree() -> None:
    ldus = [
        _ldu(text="Intro body", page_number=1, source_block_order=0, section_path=("1 Introduction",)),
        _ldu(text="Result body", page_number=2, source_block_order=0, section_path=("2 Results",)),
    ]

    tree = PageIndexBuilder().build(doc_id="doc123", ldus=ldus)
    nodes_by_title = {node.title: node for node in tree.nodes}
    root = next(node for node in tree.nodes if node.section_path == ())

    assert [nodes_by_title["1 Introduction"].parent_id, nodes_by_title["2 Results"].parent_id] == [root.node_id, root.node_id]
    assert root.child_ids == [nodes_by_title["1 Introduction"].node_id, nodes_by_title["2 Results"].node_id]
    assert nodes_by_title["1 Introduction"].ldu_ids == [ldus[0].ldu_id]
    assert nodes_by_title["2 Results"].ldu_ids == [ldus[1].ldu_id]


def test_page_index_builder_handles_nested_section_tree() -> None:
    ldus = [
        _ldu(text="Parent body", page_number=1, source_block_order=0, section_path=("3 Results",)),
        _ldu(
            text="Child body",
            page_number=1,
            source_block_order=1,
            section_path=("3 Results", "3.2 Retrieval Precision"),
        ),
    ]

    tree = PageIndexBuilder().build(doc_id="doc123", ldus=ldus)
    nodes_by_path = {node.section_path: node for node in tree.nodes}

    assert nodes_by_path[("3 Results",)].child_ids == [nodes_by_path[("3 Results", "3.2 Retrieval Precision")].node_id]
    assert nodes_by_path[("3 Results", "3.2 Retrieval Precision")].parent_id == nodes_by_path[("3 Results",)].node_id
    assert nodes_by_path[("3 Results", "3.2 Retrieval Precision")].ldu_ids == [ldus[1].ldu_id]


def test_page_index_builder_aggregates_repeated_sections_across_pages() -> None:
    ldus = [
        _ldu(text="Methods p1", page_number=1, source_block_order=0, section_path=("2 Methods",)),
        _ldu(text="Methods p2", page_number=2, source_block_order=0, section_path=("2 Methods",)),
    ]

    tree = PageIndexBuilder().build(doc_id="doc123", ldus=ldus)
    methods_node = next(node for node in tree.nodes if node.section_path == ("2 Methods",))

    assert methods_node.start_page == 1
    assert methods_node.end_page == 2
    assert methods_node.ldu_ids == [ldus[0].ldu_id, ldus[1].ldu_id]


def test_page_index_builder_constructs_parent_child_relationships_in_order() -> None:
    ldus = [
        _ldu(text="A", page_number=1, source_block_order=0, section_path=("1 Introduction",)),
        _ldu(text="B", page_number=1, source_block_order=1, section_path=("2 Results",)),
        _ldu(text="C", page_number=1, source_block_order=2, section_path=("2 Results", "2.1 Precision")),
    ]

    tree = PageIndexBuilder().build(doc_id="doc123", ldus=ldus)
    root = next(node for node in tree.nodes if node.section_path == ())
    intro = next(node for node in tree.nodes if node.section_path == ("1 Introduction",))
    results = next(node for node in tree.nodes if node.section_path == ("2 Results",))
    precision = next(node for node in tree.nodes if node.section_path == ("2 Results", "2.1 Precision"))

    assert root.child_ids == [intro.node_id, results.node_id]
    assert results.child_ids == [precision.node_id]
    assert intro.parent_id == root.node_id
    assert results.parent_id == root.node_id
    assert precision.parent_id == results.node_id


def test_page_index_builder_attaches_ldus_to_leaf_and_aggregates_page_ranges() -> None:
    ldus = [
        _ldu(text="Overview", page_number=1, source_block_order=0, section_path=("4 Discussion",)),
        _ldu(
            text="Subsection detail page 2",
            page_number=2,
            source_block_order=0,
            section_path=("4 Discussion", "4.1 Error Analysis"),
        ),
        _ldu(
            text="Subsection detail page 3",
            page_number=3,
            source_block_order=0,
            section_path=("4 Discussion", "4.1 Error Analysis"),
        ),
    ]

    tree = PageIndexBuilder().build(doc_id="doc123", ldus=ldus)
    discussion = next(node for node in tree.nodes if node.section_path == ("4 Discussion",))
    error_analysis = next(node for node in tree.nodes if node.section_path == ("4 Discussion", "4.1 Error Analysis"))
    root = next(node for node in tree.nodes if node.section_path == ())

    assert discussion.ldu_ids == [ldus[0].ldu_id]
    assert error_analysis.ldu_ids == [ldus[1].ldu_id, ldus[2].ldu_id]
    assert error_analysis.start_page == 2
    assert error_analysis.end_page == 3
    assert discussion.start_page == 1
    assert discussion.end_page == 3
    assert root.start_page == 1
    assert root.end_page == 3


def test_page_index_summarization_integrates_with_tree() -> None:
    ldus = [
        _ldu(text="Key findings show stable performance across cohorts.", page_number=1, source_block_order=0, section_path=("1 Overview",)),
    ]
    tree = PageIndexBuilder().build(doc_id="doc123", ldus=ldus)

    summarized = PageIndexSummarizer(FakeSummaryBackend()).summarize_tree(tree=tree, ldus=ldus)
    overview = next(node for node in summarized.nodes if node.section_path == ("1 Overview",))
    root = next(node for node in summarized.nodes if node.section_path == ())

    assert overview.summary == "1 Overview: Key findings show stable performance ac"
    assert root.summary is None


def test_page_index_summarization_uses_direct_ldus_for_leaf_nodes() -> None:
    ldus = [
        _ldu(text="Precision improved by 8 percent over baseline.", page_number=2, source_block_order=0, section_path=("3 Results", "3.2 Precision")),
    ]
    tree = PageIndexBuilder().build(doc_id="doc123", ldus=ldus)

    summarized = PageIndexSummarizer(FakeSummaryBackend()).summarize_tree(tree=tree, ldus=ldus)
    node = next(node for node in summarized.nodes if node.section_path == ("3 Results", "3.2 Precision"))

    assert node.summary == "3.2 Precision: Precision improved by 8 percent over ba"


def test_page_index_summarization_supports_parent_nodes_with_child_only_content() -> None:
    ldus = [
        _ldu(text="Child one content explains the first metric.", page_number=1, source_block_order=0, section_path=("2 Results", "2.1 Precision")),
        _ldu(text="Child two content explains the recall metric.", page_number=1, source_block_order=1, section_path=("2 Results", "2.2 Recall")),
    ]
    tree = PageIndexBuilder().build(doc_id="doc123", ldus=ldus)

    summarized = PageIndexSummarizer(FakeSummaryBackend()).summarize_tree(tree=tree, ldus=ldus)
    parent = next(node for node in summarized.nodes if node.section_path == ("2 Results",))
    precision = next(node for node in summarized.nodes if node.section_path == ("2 Results", "2.1 Precision"))
    recall = next(node for node in summarized.nodes if node.section_path == ("2 Results", "2.2 Recall"))

    assert precision.summary == "2.1 Precision: Child one content explains the first me"
    assert recall.summary == "2.2 Recall: Child two content explains the recall m"
    assert parent.summary == "2 Results: 2.1 Precision: Child one content explai"


def test_page_index_summarization_is_deterministic_with_fake_backend() -> None:
    ldus = [
        _ldu(text="Deterministic source text for summary generation.", page_number=1, source_block_order=0, section_path=("4 Discussion",)),
    ]
    tree = PageIndexBuilder().build(doc_id="doc123", ldus=ldus)
    summarizer = PageIndexSummarizer(FakeSummaryBackend())

    first = summarizer.summarize_tree(tree=tree, ldus=ldus)
    second = summarizer.summarize_tree(tree=tree, ldus=ldus)

    first_summaries = [node.summary for node in sorted(first.nodes, key=lambda node: node.order_index)]
    second_summaries = [node.summary for node in sorted(second.nodes, key=lambda node: node.order_index)]

    assert first_summaries == second_summaries


def test_ollama_backend_wires_through_page_index_summarizer() -> None:
    ldus = [
        _ldu(text="Revenue increased to 120 million Birr in 2025.", page_number=1, source_block_order=0, section_path=("5 Finance",)),
    ]
    tree = PageIndexBuilder().build(doc_id="doc123", ldus=ldus)
    client = MockOllamaClient(response={"message": {"content": "Revenue increased to 120 million Birr in 2025."}})
    backend = OllamaSummaryBackend(client=client)

    summarized = PageIndexSummarizer(backend).summarize_tree(tree=tree, ldus=ldus)
    finance = next(node for node in summarized.nodes if node.section_path == ("5 Finance",))

    assert finance.summary == "Revenue increased to 120 million Birr in 2025."
    assert len(client.calls) == 1
    assert client.calls[0]["model"] == "qwen3:1.7b"
    assert client.calls[0]["keep_alive"] == "0s"


def test_ollama_backend_uses_mocked_client_response_and_configuration() -> None:
    client = MockOllamaClient(response={"message": {"content": "Short factual summary."}})
    backend = OllamaSummaryBackend(client=client, model="qwen3:1.7b", keep_alive="30s")

    summary = backend.summarize(
        SummaryInput(
            node_id="node-1",
            title="3 Results",
            section_path=("3 Results",),
            source_text="Results improved on the validation split.",
        )
    )

    assert summary == "Short factual summary."
    assert client.calls[0]["model"] == "qwen3:1.7b"
    assert client.calls[0]["keep_alive"] == "30s"
    assert client.calls[0]["options"] == {"temperature": 0}
    messages = client.calls[0]["messages"]
    assert isinstance(messages, list)
    assert "retrieval summaries" in messages[0]["content"]
    assert "Section: 3 Results" in messages[1]["content"]


def test_ollama_backend_handles_failures_gracefully() -> None:
    ldus = [
        _ldu(text="System reliability remained stable.", page_number=1, source_block_order=0, section_path=("6 Operations",)),
    ]
    tree = PageIndexBuilder().build(doc_id="doc123", ldus=ldus)
    backend = OllamaSummaryBackend(client=FailingOllamaClient())

    with pytest.raises(SummaryBackendError):
        backend.summarize(
            SummaryInput(
                node_id="node-1",
                title="6 Operations",
                section_path=("6 Operations",),
                source_text="System reliability remained stable.",
            )
        )

    summarized = PageIndexSummarizer(backend).summarize_tree(tree=tree, ldus=ldus)
    operations = next(node for node in summarized.nodes if node.section_path == ("6 Operations",))
    assert operations.summary is None
