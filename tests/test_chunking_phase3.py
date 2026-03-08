from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chunking import (
    ChromaVectorStore,
    ChunkValidationError,
    ChunkValidator,
    ChunkingConfig,
    ChunkingEngine,
    EmbeddingBackend,
    LabeledRetrievalQuery,
    OllamaEmbeddingBackend,
    OllamaSummaryBackend,
    PageIndexBuilder,
    PageIndexMatch,
    PageIndexQueryEngine,
    PageIndexSummarizer,
    ProvenanceChainBuilder,
    ProvenanceChainError,
    RetrievalEvaluator,
    SummaryBackendError,
    SummaryInput,
    VectorStoreMatch,
)
from src.chunking.page_index import PageIndexTree
from src.chunking.sections import SectionInferenceMode, SectionPathInferer
from src.models.chunking import Chunk, LDU, LDUKind
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


def _node_by_path(tree: PageIndexTree, section_path: tuple[str, ...]):
    return next(node for node in tree.nodes if node.section_path == section_path)


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


class MockOllamaEmbeddingClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.calls: list[dict[str, object]] = []

    def embed(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        return self.response


class FailingOllamaEmbeddingClient:
    def embed(self, **kwargs: object) -> dict[str, object]:
        _ = kwargs
        raise RuntimeError("embedding service unavailable")


class FakeEmbeddingBackend(EmbeddingBackend):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            float(len(lowered)),
            float(lowered.count("results")),
            float(lowered.count("precision")),
            float(lowered.count("finance")),
        ]


class FakeChromaCollection:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}
        self.last_upsert: dict[str, object] | None = None
        self.upserts: list[dict[str, object]] = []
        self.last_query: dict[str, object] | None = None

    def upsert(
        self,
        *,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, object]],
    ) -> None:
        self.last_upsert = {
            "ids": ids,
            "documents": documents,
            "embeddings": embeddings,
            "metadatas": metadatas,
        }
        self.upserts.append(self.last_upsert)
        for index, record_id in enumerate(ids):
            self.records[record_id] = {
                "id": record_id,
                "document": documents[index],
                "embedding": embeddings[index],
                "metadata": metadatas[index],
            }

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, object] | None = None,
        include: list[str] | None = None,
    ) -> dict[str, object]:
        self.last_query = {
            "query_embeddings": query_embeddings,
            "n_results": n_results,
            "where": where,
            "include": include,
        }
        query_embedding = query_embeddings[0]
        filtered = [
            record for record in self.records.values() if self._matches_where(record["metadata"], where)
        ]
        ranked = sorted(
            filtered,
            key=lambda record: (
                self._distance(query_embedding, record["embedding"]),
                str(record["id"]),
            ),
        )[:n_results]
        return {
            "ids": [[record["id"] for record in ranked]],
            "documents": [[record["document"] for record in ranked]],
            "metadatas": [[record["metadata"] for record in ranked]],
            "distances": [[self._distance(query_embedding, record["embedding"]) for record in ranked]],
        }

    def _matches_where(self, metadata: object, where: dict[str, object] | None) -> bool:
        if where is None:
            return True
        if not isinstance(metadata, dict):
            return False
        if "$and" in where:
            clauses = where["$and"]
            if not isinstance(clauses, list):
                return False
            return all(self._matches_where(metadata, clause) for clause in clauses if isinstance(clause, dict))
        for key, value in where.items():
            if metadata.get(key) != value:
                return False
        return True

    def _distance(self, left: list[float], right: object) -> float:
        if not isinstance(right, list):
            return float("inf")
        return sum(abs(left[index] - float(right[index])) for index in range(min(len(left), len(right))))


def _require_last_upsert(collection: FakeChromaCollection) -> dict[str, object]:
    assert collection.last_upsert is not None
    return collection.last_upsert


def _require_last_query(collection: FakeChromaCollection) -> dict[str, object]:
    assert collection.last_query is not None
    return collection.last_query


class FakeVectorRetrievalBackend:
    def __init__(self, responses: dict[tuple[str, tuple[str, ...] | None], list[VectorStoreMatch]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def query(
        self,
        topic: str,
        *,
        top_k: int = 3,
        section_path: tuple[str, ...] | None = None,
        record_type: str | None = None,
    ) -> list[VectorStoreMatch]:
        self.calls.append(
            {
                "topic": topic,
                "top_k": top_k,
                "section_path": section_path,
                "record_type": record_type,
            }
        )
        return self.responses.get((topic, section_path), [])[:top_k]


class FakePageIndexTraversalBackend:
    def __init__(self, responses: dict[str, list[PageIndexMatch]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def query(self, tree: PageIndexTree, topic: str, top_k: int = 3) -> list[PageIndexMatch]:
        self.calls.append({"tree": tree.doc_id, "topic": topic, "top_k": top_k})
        return self.responses.get(topic, [])[:top_k]


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


def test_ldu_bbox_clamps_tiny_negative_float_noise_to_zero() -> None:
    ldu = LDU(
        doc_id="doc123",
        page_number=1,
        bbox=(-3.05e-05, 5.0, 10.0, 20.0),
        kind=LDUKind.text,
        text="Bounding boxes should survive float noise.",
        section_path=("1 Overview",),
    )

    assert ldu.bbox == (0.0, 5.0, 10.0, 20.0)


def test_ldu_bbox_rejects_materially_negative_coordinates() -> None:
    with pytest.raises(ValueError, match="bbox coordinates must be non-negative"):
        LDU(
            doc_id="doc123",
            page_number=1,
            bbox=(-0.01, 5.0, 10.0, 20.0),
            kind=LDUKind.text,
            text="This bbox should still fail validation.",
            section_path=("1 Overview",),
        )


def test_chunk_bbox_preserves_normal_valid_values() -> None:
    chunk = Chunk(
        doc_id="doc123",
        page_number=1,
        bbox=(1.5, 2.5, 10.0, 20.0),
        section_path=("1 Overview",),
        ldu_ids=["ldu-1"],
        text="Valid bbox values should remain unchanged.",
        sequence_number=0,
    )

    assert chunk.bbox == (1.5, 2.5, 10.0, 20.0)


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


def test_chunking_engine_populates_table_header_metadata_from_extracted_rows() -> None:
    page = ExtractedPage(
        doc_id="doc123",
        page_number=1,
        metadata=_metadata(),
        signals={"char_count": 40, "char_density": 0.1, "image_area_ratio": 0.0, "table_count": 1},
        table_blocks=[
            TableBlock(
                doc_id="doc123",
                page_number=1,
                bbox=(0.0, 0.0, 50.0, 20.0),
                content_hash="tb1",
                table_index=0,
                rows=[["", "Year", "Revenue"], ["2025", "100"]],
            )
        ],
        page_content_hash="page1",
    )

    ldus = ChunkingEngine().build_ldus(_document_from_pages(page))

    assert len(ldus) == 1
    assert ldus[0].kind == LDUKind.table
    assert ldus[0].metadata["header_row"] == ["Year", "Revenue"]
    assert ldus[0].metadata["row_count"] == 2


def test_chunking_engine_skips_whitespace_only_headerless_tables() -> None:
    page = ExtractedPage(
        doc_id="doc123",
        page_number=1,
        metadata=_metadata(),
        signals={"char_count": 0, "char_density": 0.0, "image_area_ratio": 0.0, "table_count": 1},
        table_blocks=[
            TableBlock(
                doc_id="doc123",
                page_number=1,
                bbox=(0.0, 0.0, 50.0, 20.0),
                content_hash="tb-empty",
                table_index=0,
                rows=[["", ""], ["", ""]],
            )
        ],
        page_content_hash="page1",
    )

    ldus = ChunkingEngine().build_ldus(_document_from_pages(page))

    assert ldus == []


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


def test_chunking_engine_skips_figure_blocks_without_caption() -> None:
    page = ExtractedPage(
        doc_id="doc123",
        page_number=1,
        metadata=_metadata(),
        signals={"char_count": 20, "char_density": 0.1, "image_area_ratio": 0.2, "table_count": 0},
        text_blocks=[
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="1 Overview",
                bbox=(0.0, 0.0, 50.0, 16.0),
                reading_order=0,
                content_hash="t1",
            )
        ],
        figure_blocks=[
            FigureBlock(
                doc_id="doc123",
                page_number=1,
                bbox=(0.0, 20.0, 50.0, 60.0),
                content_hash="fg1",
                caption=None,
            )
        ],
        page_content_hash="page1",
    )

    ldus = ChunkingEngine().build_ldus(_document_from_pages(page))

    assert len(ldus) == 1
    assert all(ldu.kind != LDUKind.figure for ldu in ldus)


def test_chunking_engine_keeps_captioned_figure_blocks() -> None:
    page = ExtractedPage(
        doc_id="doc123",
        page_number=1,
        metadata=_metadata(),
        signals={"char_count": 20, "char_density": 0.1, "image_area_ratio": 0.2, "table_count": 0},
        figure_blocks=[
            FigureBlock(
                doc_id="doc123",
                page_number=1,
                bbox=(0.0, 20.0, 50.0, 60.0),
                content_hash="fg1",
                caption="Revenue trend chart",
            )
        ],
        page_content_hash="page1",
    )

    ldus = ChunkingEngine().build_ldus(_document_from_pages(page))

    assert len(ldus) == 1
    assert ldus[0].kind == LDUKind.figure
    assert ldus[0].metadata["caption"] == "Revenue trend chart"


def test_chunking_engine_splits_oversized_text_ldu_into_multiple_chunks() -> None:
    long_text = " ".join(f"token{i}" for i in range(120))
    page = ExtractedPage(
        doc_id="doc123",
        page_number=1,
        metadata=_metadata(),
        signals={"char_count": len(long_text), "char_density": 0.1, "image_area_ratio": 0.0, "table_count": 0},
        text_blocks=[
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text=long_text,
                bbox=(0.0, 0.0, 50.0, 20.0),
                reading_order=0,
                content_hash="t-long",
            )
        ],
        page_content_hash="page1",
    )

    chunks = ChunkingEngine(config=ChunkingConfig(max_chunk_chars=120)).build_chunks(_document_from_pages(page))

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 120 for chunk in chunks)
    assert all(chunk.metadata["split_from_oversized_ldu"] is True for chunk in chunks)
    assert all(chunk.metadata["kinds"] == ["text"] for chunk in chunks)
    assert all(chunk.ldu_ids == [chunks[0].ldu_ids[0]] for chunk in chunks)


def test_chunking_engine_splits_oversized_table_ldu_into_multiple_chunks() -> None:
    rows = [["Header", "Value"]]
    rows.extend([[f"row-{index}", "x" * 40] for index in range(8)])
    page = ExtractedPage(
        doc_id="doc123",
        page_number=1,
        metadata=_metadata(),
        signals={"char_count": 0, "char_density": 0.0, "image_area_ratio": 0.0, "table_count": 1},
        table_blocks=[
            TableBlock(
                doc_id="doc123",
                page_number=1,
                bbox=(0.0, 0.0, 50.0, 20.0),
                content_hash="tb-big",
                table_index=0,
                rows=rows,
            )
        ],
        page_content_hash="page1",
    )

    chunks = ChunkingEngine(config=ChunkingConfig(max_chunk_chars=120)).build_chunks(_document_from_pages(page))

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 120 for chunk in chunks)
    assert all(chunk.metadata["split_from_oversized_ldu"] is True for chunk in chunks)
    assert all(chunk.metadata["kinds"] == ["table"] for chunk in chunks)


def test_chunking_engine_accounts_for_join_separator_in_chunk_limit() -> None:
    page = ExtractedPage(
        doc_id="doc123",
        page_number=1,
        metadata=_metadata(),
        signals={"char_count": 0, "char_density": 0.0, "image_area_ratio": 0.0, "table_count": 0},
        text_blocks=[
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="a" * 600,
                bbox=(0.0, 0.0, 50.0, 10.0),
                reading_order=0,
                content_hash="t1",
            ),
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="b" * 599,
                bbox=(0.0, 12.0, 50.0, 22.0),
                reading_order=1,
                content_hash="t2",
            ),
        ],
        page_content_hash="page1",
    )

    chunks = ChunkingEngine(config=ChunkingConfig(max_chunk_chars=1200)).build_chunks(_document_from_pages(page))

    assert len(chunks) == 2
    assert all(len(chunk.text) <= 1200 for chunk in chunks)


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


def test_section_path_inference_accepts_colon_after_numbered_heading() -> None:
    page = ExtractedPage(
        doc_id="doc123",
        page_number=1,
        metadata=_metadata(),
        signals={"char_count": 120, "char_density": 0.1, "image_area_ratio": 0.0, "table_count": 0},
        text_blocks=[
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="3: 12-Month Moving Average General Inflation",
                bbox=(0.0, 0.0, 120.0, 16.0),
                reading_order=0,
                content_hash="t1",
            ),
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="Inflation eased compared to the prior year.",
                bbox=(0.0, 20.0, 120.0, 32.0),
                reading_order=1,
                content_hash="t2",
            ),
        ],
        page_content_hash="page1",
    )

    ldus = ChunkingEngine().build_ldus(_document_from_pages(page))

    assert ldus[0].section_path == ("3 12-Month Moving Average General Inflation",)
    assert ldus[1].section_path == ("3 12-Month Moving Average General Inflation",)


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


def test_section_path_inference_suppresses_repeated_branding_across_pages() -> None:
    pages = []
    for page_number in (1, 2, 3):
        pages.append(
            ExtractedPage(
                doc_id="doc123",
                page_number=page_number,
                metadata=_metadata(),
                signals={"char_count": 80, "char_density": 0.1, "image_area_ratio": 0.0, "table_count": 0},
                text_blocks=[
                    TextBlock(
                        doc_id="doc123",
                        page_number=page_number,
                        text="ETHIOPIAN STATISTICAL SERVICE",
                        bbox=(0.0, 90.0, 80.0, 108.0),
                        reading_order=0,
                        content_hash=f"brand-{page_number}",
                    ),
                    TextBlock(
                        doc_id="doc123",
                        page_number=page_number,
                        text=f"Body paragraph on page {page_number} with enough text to be content.",
                        bbox=(0.0, 120.0, 120.0, 132.0),
                        reading_order=1,
                        content_hash=f"body-{page_number}",
                    ),
                ],
                page_content_hash=f"page{page_number}",
            )
        )

    ldus = ChunkingEngine().build_ldus(_document_from_pages(*pages))

    assert all(ldu.section_path == () for ldu in ldus)


def test_section_path_inference_suppresses_page_labels_and_contact_lines() -> None:
    page = ExtractedPage(
        doc_id="doc123",
        page_number=1,
        metadata=_metadata(),
        signals={"char_count": 120, "char_density": 0.1, "image_area_ratio": 0.0, "table_count": 0},
        text_blocks=[
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="Page 1",
                bbox=(0.0, 0.0, 40.0, 16.0),
                reading_order=0,
                content_hash="t1",
            ),
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="Telephone number +251 111 568464",
                bbox=(0.0, 20.0, 80.0, 36.0),
                reading_order=1,
                content_hash="t2",
            ),
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="Email: analyst@example.org",
                bbox=(0.0, 40.0, 80.0, 56.0),
                reading_order=2,
                content_hash="t3",
            ),
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="https://www.example.org/report",
                bbox=(0.0, 60.0, 80.0, 76.0),
                reading_order=3,
                content_hash="t4",
            ),
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="1 Overview",
                bbox=(0.0, 80.0, 80.0, 96.0),
                reading_order=4,
                content_hash="t5",
            ),
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="This section contains the actual body text for the report.",
                bbox=(0.0, 100.0, 120.0, 112.0),
                reading_order=5,
                content_hash="t6",
            ),
        ],
        page_content_hash="page1",
    )

    ldus = ChunkingEngine().build_ldus(_document_from_pages(page))

    assert ldus[0].section_path == ()
    assert ldus[1].section_path == ()
    assert ldus[2].section_path == ()
    assert ldus[3].section_path == ()
    assert ldus[4].section_path == ("1 Overview",)
    assert ldus[5].section_path == ("1 Overview",)


def test_section_path_inference_requires_body_after_unnumbered_heading() -> None:
    page = ExtractedPage(
        doc_id="doc123",
        page_number=1,
        metadata=_metadata(),
        signals={"char_count": 100, "char_density": 0.1, "image_area_ratio": 0.0, "table_count": 0},
        text_blocks=[
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="Content",
                bbox=(0.0, 0.0, 50.0, 18.0),
                reading_order=0,
                content_hash="t1",
            ),
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="Summary",
                bbox=(0.0, 26.0, 50.0, 44.0),
                reading_order=1,
                content_hash="t2",
            ),
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="Monthly Highlights",
                bbox=(0.0, 60.0, 60.0, 78.0),
                reading_order=2,
                content_hash="t3",
            ),
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="This paragraph expands on the monthly highlights with enough detail to count as body text.",
                bbox=(0.0, 90.0, 120.0, 102.0),
                reading_order=3,
                content_hash="t4",
            ),
        ],
        page_content_hash="page1",
    )

    ldus = ChunkingEngine().build_ldus(_document_from_pages(page))

    assert ldus[0].section_path == ()
    assert ldus[1].section_path == ()
    assert ldus[2].section_path == ("Monthly Highlights",)
    assert ldus[3].section_path == ("Monthly Highlights",)


def test_relaxed_section_path_inference_accepts_heading_followed_by_table() -> None:
    page = ExtractedPage(
        doc_id="doc123",
        page_number=1,
        metadata=_metadata(),
        signals={"char_count": 100, "char_density": 0.1, "image_area_ratio": 0.0, "table_count": 1},
        text_blocks=[
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="Revenue by Segment",
                bbox=(0.0, 0.0, 80.0, 18.0),
                reading_order=0,
                content_hash="t1",
            ),
        ],
        table_blocks=[
            TableBlock(
                doc_id="doc123",
                page_number=1,
                bbox=(0.0, 26.0, 120.0, 80.0),
                content_hash="tbl1",
                table_index=0,
                rows=[["Segment", "2025"], ["Cards", "120"]],
            ),
        ],
        page_content_hash="page1",
    )

    strict_ldus = ChunkingEngine().build_ldus(_document_from_pages(page))
    relaxed_ldus = ChunkingEngine(
        section_inferer=SectionPathInferer(mode=SectionInferenceMode.relaxed)
    ).build_ldus(_document_from_pages(page))

    assert strict_ldus[0].section_path == ()
    assert strict_ldus[1].section_path == ()
    assert relaxed_ldus[0].section_path == ("Revenue by Segment",)
    assert relaxed_ldus[1].section_path == ("Revenue by Segment",)


def test_relaxed_section_path_inference_accepts_heading_followed_by_list_item_from_recovered_text() -> None:
    page = ExtractedPage(
        doc_id="doc123",
        page_number=1,
        metadata=_metadata(),
        signals={"char_count": 100, "char_density": 0.1, "image_area_ratio": 0.0, "table_count": 0},
        text_blocks=[
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="Procurement Requirements",
                bbox=(0.0, 0.0, 100.0, 200.0),
                reading_order=0,
                content_hash="t1",
            ),
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="- 1. Bid security shall be submitted with the offer.",
                bbox=(0.0, 0.0, 100.0, 200.0),
                reading_order=1,
                content_hash="t2",
            ),
        ],
        page_content_hash="page1",
    )

    strict_ldus = ChunkingEngine().build_ldus(_document_from_pages(page))
    relaxed_ldus = ChunkingEngine(
        section_inferer=SectionPathInferer(mode=SectionInferenceMode.relaxed)
    ).build_ldus(_document_from_pages(page))

    assert strict_ldus[0].section_path == ()
    assert strict_ldus[1].section_path == ()
    assert relaxed_ldus[0].section_path == ("Procurement Requirements",)
    assert relaxed_ldus[1].section_path == ("Procurement Requirements",)


def test_relaxed_section_path_inference_suppresses_strategy_c_boilerplate_lines() -> None:
    page = ExtractedPage(
        doc_id="doc123",
        page_number=1,
        metadata=_metadata(),
        signals={"char_count": 100, "char_density": 0.1, "image_area_ratio": 0.0, "table_count": 0},
        text_blocks=[
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text=(
                    "The text extracted from this image does not conform to the provided schema "
                    "and is not suitable for plain text output."
                ),
                bbox=(0.0, 0.0, 100.0, 200.0),
                reading_order=0,
                content_hash="t1",
            ),
            TextBlock(
                doc_id="doc123",
                page_number=1,
                text="- 1. Bid security shall be submitted with the offer.",
                bbox=(0.0, 0.0, 100.0, 200.0),
                reading_order=1,
                content_hash="t2",
            ),
        ],
        page_content_hash="page1",
    )

    relaxed_ldus = ChunkingEngine(
        section_inferer=SectionPathInferer(mode=SectionInferenceMode.relaxed)
    ).build_ldus(_document_from_pages(page))

    assert relaxed_ldus[0].section_path == ()
    assert relaxed_ldus[1].section_path == ()


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


def test_page_index_builder_populates_entities_data_types_and_child_sections() -> None:
    ldus = [
        _ldu(
            text="CPI = 17.1% for Bread and Cereals in March EFY2017.",
            page_number=1,
            source_block_order=0,
            section_path=("2 Inflation",),
        ),
        LDU(
            doc_id="doc123",
            page_number=1,
            bbox=(0.0, 2.0, 10.0, 3.0),
            kind=LDUKind.table,
            text="Item | Weight\nBread and Cereals | 17.1",
            section_path=("2 Inflation",),
            source_block_order=1,
        ),
    ]

    tree = PageIndexBuilder().build(doc_id="doc123", ldus=ldus)
    root = _node_by_path(tree, ())
    inflation = _node_by_path(tree, ("2 Inflation",))

    assert root.child_sections == ["2 Inflation"]
    assert "CPI" in inflation.key_entities
    assert "17.1%" in inflation.key_entities
    assert "table" in inflation.data_types_present
    assert "equation" in inflation.data_types_present
    assert inflation.page_start == inflation.start_page
    assert inflation.page_end == inflation.end_page


def test_table_root_only_recovery_assigns_synthetic_section_paths_and_logs(caplog) -> None:
    page1 = ExtractedPage(
        doc_id="doc123",
        page_number=1,
        text="",
        metadata=_metadata(),
        signals={},
        table_blocks=[
            TableBlock(
                doc_id="doc123",
                page_number=1,
                bbox=(0.0, 0.0, 100.0, 120.0),
                content_hash="t1",
                table_index=0,
                rows=[
                    ["Code", "Budget", "Expense"],
                    ["A1", "100", "90"],
                ],
            )
        ],
        page_content_hash="page-1",
    )
    page2 = ExtractedPage(
        doc_id="doc123",
        page_number=2,
        text="",
        metadata=_metadata(),
        signals={},
        table_blocks=[
            TableBlock(
                doc_id="doc123",
                page_number=2,
                bbox=(0.0, 0.0, 100.0, 130.0),
                content_hash="t2",
                table_index=0,
                rows=[
                    ["Vendor", "Amount", "Date"],
                    ["ACME", "200", "2024-01-01"],
                ],
            )
        ],
        page_content_hash="page-2",
    )

    document = _document_from_pages(page1, page2)
    engine = ChunkingEngine(section_inferer=SectionPathInferer())

    with caplog.at_level("INFO"):
        ldus = engine.build_ldus(document)

    table_ldus = [ldu for ldu in ldus if ldu.kind == LDUKind.table]
    assert [ldu.section_path for ldu in table_ldus] == [
        ("Page 1 Table 1: Budget / Expense",),
        ("Page 2 Table 1: Vendor / Amount",),
    ]

    tree = PageIndexBuilder().build(doc_id="doc123", ldus=ldus)
    non_root_nodes = [node for node in tree.nodes if node.section_path]

    assert [node.section_path for node in non_root_nodes] == [
        ("Page 1 Table 1: Budget / Expense",),
        ("Page 2 Table 1: Vendor / Amount",),
    ]
    assert "Triggered synthetic table section recovery" in caplog.text


def test_table_root_only_recovery_uses_grounded_keyword_fallback_for_noisy_headers() -> None:
    page = ExtractedPage(
        doc_id="doc123",
        page_number=1,
        text="",
        metadata=_metadata(),
        signals={},
        table_blocks=[
            TableBlock(
                doc_id="doc123",
                page_number=1,
                bbox=(0.0, 80.0, 180.0, 180.0),
                content_hash="table-1",
                table_index=0,
                rows=[
                    ["001", "2013", "900"],
                    ["Budget", "Expense", "Total"],
                ],
            ),
            TableBlock(
                doc_id="doc123",
                page_number=1,
                bbox=(0.0, 220.0, 180.0, 320.0),
                content_hash="table-2",
                table_index=1,
                rows=[
                    ["Expense", "Amount", "Total"],
                    ["Travel", "40", "40"],
                ],
            ),
        ],
        page_content_hash="page-1",
    )

    ldus = ChunkingEngine(section_inferer=SectionPathInferer()).build_ldus(_document_from_pages(page))
    table_ldus = [ldu for ldu in ldus if ldu.kind == LDUKind.table]

    assert table_ldus[0].section_path == ("Page 1 Table 1: Budget / Expense",)
    assert table_ldus[1].section_path == ("Page 1 Table 2: Expense / Amount",)


def test_table_root_only_recovery_drops_low_quality_descriptor_when_no_grounded_signal_exists() -> None:
    page = ExtractedPage(
        doc_id="doc123",
        page_number=1,
        text="",
        metadata=_metadata(),
        signals={},
        table_blocks=[
            TableBlock(
                doc_id="doc123",
                page_number=1,
                bbox=(0.0, 80.0, 180.0, 180.0),
                content_hash="table-1",
                table_index=0,
                rows=[
                    ["6C", "001", "2013"],
                    ["A1", "100", "90"],
                ],
            ),
        ],
        page_content_hash="page-1",
    )
    page2 = ExtractedPage(
        doc_id="doc123",
        page_number=2,
        text="",
        metadata=_metadata(),
        signals={},
        table_blocks=[
            TableBlock(
                doc_id="doc123",
                page_number=2,
                bbox=(0.0, 80.0, 180.0, 180.0),
                content_hash="table-3",
                table_index=0,
                rows=[
                    ["7D", "002", "2014"],
                    ["A3", "120", "100"],
                ],
            )
        ],
        page_content_hash="page-2",
    )

    ldus = ChunkingEngine(section_inferer=SectionPathInferer()).build_ldus(_document_from_pages(page, page2))
    table_ldus = [ldu for ldu in ldus if ldu.kind == LDUKind.table]

    assert table_ldus[0].section_path == ("Page 1 Table 1",)
    assert table_ldus[-1].section_path == ("Page 2 Table 1",)


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


def test_ollama_embedding_backend_uses_mocked_client_response_and_configuration() -> None:
    client = MockOllamaEmbeddingClient(response={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})
    backend = OllamaEmbeddingBackend(client=client, model="qwen3-embedding:0.6b", keep_alive="15s")

    embeddings = backend.embed_documents(["Revenue improved.", "Results stabilized."])

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    assert client.calls[0]["model"] == "qwen3-embedding:0.6b"
    assert client.calls[0]["keep_alive"] == "15s"
    assert client.calls[0]["input"] == ["Revenue improved.", "Results stabilized."]


def test_ollama_embedding_backend_embeds_single_query() -> None:
    client = MockOllamaEmbeddingClient(response={"embeddings": [[0.5, 0.6, 0.7]]})
    backend = OllamaEmbeddingBackend(client=client)

    embedding = backend.embed_query("Financial risk controls")

    assert embedding == [0.5, 0.6, 0.7]
    assert client.calls[0]["model"] == "qwen3-embedding:0.6b"
    assert client.calls[0]["input"] == ["Financial risk controls"]


def test_ollama_embedding_backend_handles_failures_cleanly() -> None:
    backend = OllamaEmbeddingBackend(client=FailingOllamaEmbeddingClient())

    with pytest.raises(RuntimeError, match="Ollama embedding request failed"):
        backend.embed_query("System stability")


def test_page_index_query_selects_relevant_flat_sections() -> None:
    ldus = [
        _ldu(text="Revenue rose sharply.", page_number=1, source_block_order=0, section_path=("1 Overview",)),
        _ldu(text="Precision improved.", page_number=2, source_block_order=0, section_path=("2 Retrieval Precision",)),
        _ldu(text="Operational notes.", page_number=3, source_block_order=0, section_path=("3 Operations",)),
    ]
    tree = PageIndexBuilder().build(doc_id="doc123", ldus=ldus)
    _node_by_path(tree, ("1 Overview",)).summary = "High-level revenue trends."
    _node_by_path(tree, ("2 Retrieval Precision",)).summary = "Precision metrics and benchmark gains."
    _node_by_path(tree, ("3 Operations",)).summary = "Routine operational notes."

    results = PageIndexQueryEngine().query(tree=tree, topic="retrieval precision metrics")

    assert results[0].section_path == ("2 Retrieval Precision",)


def test_page_index_query_selects_relevant_nested_sections() -> None:
    ldus = [
        _ldu(text="Results overview.", page_number=1, source_block_order=0, section_path=("3 Results",)),
        _ldu(text="Precision rose from 0.82 to 0.90.", page_number=2, source_block_order=0, section_path=("3 Results", "3.2 Retrieval Precision")),
    ]
    tree = PageIndexBuilder().build(doc_id="doc123", ldus=ldus)
    _node_by_path(tree, ("3 Results",)).summary = "Overall evaluation results."
    _node_by_path(tree, ("3 Results", "3.2 Retrieval Precision")).summary = "Retrieval precision improved materially."

    results = PageIndexQueryEngine().query(tree=tree, topic="precision improvement")

    assert results[0].section_path == ("3 Results", "3.2 Retrieval Precision")


def test_page_index_query_ignores_synthetic_root_node() -> None:
    ldus = [
        _ldu(text="Overview text.", page_number=1, source_block_order=0, section_path=("1 Overview",)),
    ]
    tree = PageIndexBuilder().build(doc_id="doc123", ldus=ldus)
    root = _node_by_path(tree, ())
    root.summary = "Root should never be returned."
    _node_by_path(tree, ("1 Overview",)).summary = "Overview summary."

    results = PageIndexQueryEngine().query(tree=tree, topic="overview")

    assert all(match.node_id != tree.root_id for match in results)


def test_page_index_query_returns_top_three_in_deterministic_order() -> None:
    ldus = [
        _ldu(text="Finance data.", page_number=1, source_block_order=0, section_path=("1 Finance",)),
        _ldu(text="Finance risk analysis.", page_number=2, source_block_order=0, section_path=("2 Financial Risk",)),
        _ldu(text="Financial controls review.", page_number=3, source_block_order=0, section_path=("3 Controls",)),
        _ldu(text="Human resources update.", page_number=4, source_block_order=0, section_path=("4 HR",)),
    ]
    tree = PageIndexBuilder().build(doc_id="doc123", ldus=ldus)
    _node_by_path(tree, ("1 Finance",)).summary = "Finance metrics."
    _node_by_path(tree, ("2 Financial Risk",)).summary = "Financial risk exposure."
    _node_by_path(tree, ("3 Controls",)).summary = "Financial controls and compliance."
    _node_by_path(tree, ("4 HR",)).summary = "Workforce updates."

    results = PageIndexQueryEngine().query(tree=tree, topic="financial risk controls", top_k=3)

    assert [match.section_path for match in results] == [
        ("2 Financial Risk",),
        ("3 Controls",),
        ("1 Finance",),
    ]


def test_page_index_query_handles_missing_summaries() -> None:
    ldus = [
        _ldu(text="Methods section.", page_number=1, source_block_order=0, section_path=("1 Methods",)),
        _ldu(text="Results section.", page_number=2, source_block_order=0, section_path=("2 Results",)),
    ]
    tree = PageIndexBuilder().build(doc_id="doc123", ldus=ldus)

    results = PageIndexQueryEngine().query(tree=tree, topic="results")

    assert results[0].section_path == ("2 Results",)


def test_page_index_tree_traverse_returns_top_matches_for_topic() -> None:
    ldus = [
        _ldu(text="Revenue and margin overview.", page_number=1, source_block_order=0, section_path=("1 Overview",)),
        _ldu(text="Precision improved over baseline.", page_number=2, source_block_order=0, section_path=("2 Retrieval Precision",)),
    ]
    tree = PageIndexBuilder().build(doc_id="doc123", ldus=ldus)
    _node_by_path(tree, ("1 Overview",)).summary = "High-level performance summary."
    _node_by_path(tree, ("2 Retrieval Precision",)).summary = "Precision and recall metrics."

    results = tree.traverse(topic="precision metrics", top_k=1)

    assert len(results) == 1
    assert results[0].section_path == ("2 Retrieval Precision",)


def test_vector_store_ingests_ldus() -> None:
    collection = FakeChromaCollection()
    store = ChromaVectorStore(
        embedding_backend=FakeEmbeddingBackend(),
        collection=collection,
    )
    ldus = [
        _ldu(text="Results improved.", page_number=1, source_block_order=0, section_path=("2 Results",)),
        _ldu(text="Finance remained stable.", page_number=2, source_block_order=0, section_path=("5 Finance",)),
    ]

    store.ingest_ldus(ldus)

    last_upsert = _require_last_upsert(collection)
    assert last_upsert["ids"] == [ldus[0].ldu_id, ldus[1].ldu_id]
    assert last_upsert["documents"] == ["Results improved.", "Finance remained stable."]


def test_vector_store_preserves_chunk_metadata() -> None:
    collection = FakeChromaCollection()
    store = ChromaVectorStore(
        embedding_backend=FakeEmbeddingBackend(),
        collection=collection,
    )
    chunk = ChunkingEngine(config=ChunkingConfig(max_chunk_chars=200)).build_chunks(
        _document_from_pages(
            ExtractedPage(
                doc_id="doc123",
                page_number=1,
                metadata=_metadata(),
                signals={"char_count": 40, "char_density": 0.1, "image_area_ratio": 0.0, "table_count": 0},
                text_blocks=[
                    TextBlock(
                        doc_id="doc123",
                        page_number=1,
                        text="2 Results",
                        bbox=(0.0, 0.0, 50.0, 16.0),
                        reading_order=0,
                        content_hash="t1",
                    ),
                    TextBlock(
                        doc_id="doc123",
                        page_number=1,
                        text="Precision improved by 8 percent.",
                        bbox=(0.0, 20.0, 50.0, 32.0),
                        reading_order=1,
                        content_hash="t2",
                    ),
                ],
                page_content_hash="page1",
            )
        )
    )[0]

    store.ingest_chunks([chunk])

    metadata = _require_last_upsert(collection)["metadatas"][0]
    assert metadata["doc_id"] == "doc123"
    assert metadata["document_name"] == "sample.pdf"
    assert metadata["page_number"] == 1
    assert metadata["bbox"] == [0.0, 0.0, 50.0, 32.0]
    assert metadata["section_path"] == ["2 Results"]
    assert metadata["chunk_id"] == chunk.chunk_id
    assert metadata["ldu_ids"] == chunk.ldu_ids
    assert metadata["content_hash"] == chunk.content_hash
    assert metadata["strategy_used"] == "strategy_b"
    assert metadata["confidence_score"] == 0.95


def test_chunking_engine_propagates_page_provenance_metadata_into_ldus_and_chunks() -> None:
    document = _document_from_pages(
        ExtractedPage(
            doc_id="doc123",
            page_number=1,
            metadata=_metadata(),
            signals={"char_count": 40, "char_density": 0.1, "image_area_ratio": 0.0, "table_count": 0},
            text_blocks=[
                TextBlock(
                    doc_id="doc123",
                    page_number=1,
                    text="2 Results",
                    bbox=(0.0, 0.0, 50.0, 16.0),
                    reading_order=0,
                    content_hash="t1",
                ),
                TextBlock(
                    doc_id="doc123",
                    page_number=1,
                    text="Precision improved by 8 percent.",
                    bbox=(0.0, 20.0, 50.0, 32.0),
                    reading_order=1,
                    content_hash="t2",
                ),
            ],
            page_content_hash="page1",
        )
    )
    engine = ChunkingEngine(config=ChunkingConfig(max_chunk_chars=200))

    ldus = engine.build_ldus(document)
    chunks = engine.build_chunks(document, ldus=ldus)

    assert ldus[0].metadata["strategy_used"] == "strategy_b"
    assert ldus[0].metadata["confidence_score"] == 0.95
    assert ldus[0].metadata["document_name"] == "sample.pdf"
    assert chunks[0].metadata["strategy_used"] == "strategy_b"
    assert chunks[0].metadata["confidence_score"] == 0.95
    assert chunks[0].metadata["document_name"] == "sample.pdf"


def test_provenance_chain_builder_builds_grounded_entries_from_vector_matches() -> None:
    chunk = ChunkingEngine(config=ChunkingConfig(max_chunk_chars=200)).build_chunks(
        _document_from_pages(
            ExtractedPage(
                doc_id="doc123",
                page_number=1,
                metadata=_metadata(),
                signals={"char_count": 40, "char_density": 0.1, "image_area_ratio": 0.0, "table_count": 0},
                text_blocks=[
                    TextBlock(
                        doc_id="doc123",
                        page_number=1,
                        text="2 Results",
                        bbox=(0.0, 0.0, 50.0, 16.0),
                        reading_order=0,
                        content_hash="t1",
                    ),
                    TextBlock(
                        doc_id="doc123",
                        page_number=1,
                        text="Precision improved by 8 percent.",
                        bbox=(0.0, 20.0, 50.0, 32.0),
                        reading_order=1,
                        content_hash="t2",
                    ),
                ],
                page_content_hash="page1",
            )
        )
    )[0]
    match = VectorStoreMatch(
        record_id=chunk.chunk_id or "missing",
        text=chunk.text,
        metadata={
            "record_type": "chunk",
            "doc_id": chunk.doc_id,
            "document_name": "sample.pdf",
            "page_number": chunk.page_number,
            "bbox": list(chunk.bbox),
            "section_path": list(chunk.section_path),
            "content_hash": chunk.content_hash,
            "strategy_used": chunk.metadata["strategy_used"],
            "confidence_score": chunk.metadata["confidence_score"],
        },
        distance=0.125,
    )

    chain = ProvenanceChainBuilder().build([match], query="results precision")

    assert chain.query == "results precision"
    assert len(chain.entries) == 1
    assert chain.entries[0].record_id == chunk.chunk_id
    assert chain.entries[0].record_type == "chunk"
    assert chain.entries[0].section_path == ("2 Results",)
    assert chain.entries[0].provenance.document_name == "sample.pdf"
    assert chain.entries[0].provenance.strategy_used.value == "strategy_b"
    assert chain.entries[0].provenance.bbox == (0.0, 0.0, 50.0, 32.0)


def test_provenance_chain_builder_rejects_matches_without_bbox() -> None:
    match = VectorStoreMatch(
        record_id="chunk-1",
        text="Precision improved by 8 percent.",
        metadata={
            "record_type": "chunk",
            "doc_id": "doc123",
            "page_number": 1,
            "content_hash": "hash-1",
            "strategy_used": "strategy_b",
            "confidence_score": 0.95,
        },
        distance=0.1,
    )

    with pytest.raises(ProvenanceChainError, match="bbox is required"):
        ProvenanceChainBuilder().build([match], query="results")


def test_vector_store_omits_empty_section_path_metadata_for_root_records() -> None:
    collection = FakeChromaCollection()
    store = ChromaVectorStore(
        embedding_backend=FakeEmbeddingBackend(),
        collection=collection,
    )
    ldu = _ldu(text="Root-level content.", page_number=1, source_block_order=0, section_path=())

    store.ingest_ldus([ldu])

    metadata = _require_last_upsert(collection)["metadatas"][0]
    assert "section_path" not in metadata
    assert metadata["section_path_str"] == ""


def test_vector_store_batches_large_upserts() -> None:
    collection = FakeChromaCollection()
    store = ChromaVectorStore(
        embedding_backend=FakeEmbeddingBackend(),
        collection=collection,
        max_upsert_batch_size=2,
    )
    ldus = [
        _ldu(text=f"Record {index}", page_number=1, source_block_order=index, section_path=("1 Overview",))
        for index in range(5)
    ]

    store.ingest_ldus(ldus)

    assert [len(call["ids"]) for call in collection.upserts] == [2, 2, 1]
    assert len(collection.records) == 5


def test_vector_store_supports_filtered_retrieval_by_section_path() -> None:
    collection = FakeChromaCollection()
    store = ChromaVectorStore(
        embedding_backend=FakeEmbeddingBackend(),
        collection=collection,
    )
    ldus = [
        _ldu(text="Precision improved in results.", page_number=1, source_block_order=0, section_path=("2 Results",)),
        _ldu(text="Precision improved in methods.", page_number=2, source_block_order=0, section_path=("1 Methods",)),
    ]
    store.ingest_ldus(ldus)

    results = store.query(topic="precision", section_path=("2 Results",), record_type="ldu")

    assert [match.record_id for match in results] == [ldus[0].ldu_id]
    assert _require_last_query(collection)["where"] == {"$and": [{"section_path_str": "2 Results"}, {"record_type": "ldu"}]}


def test_vector_store_is_deterministic_with_fake_embedding_backend() -> None:
    collection = FakeChromaCollection()
    store = ChromaVectorStore(
        embedding_backend=FakeEmbeddingBackend(),
        collection=collection,
    )
    ldus = [
        _ldu(text="Finance risk overview.", page_number=1, source_block_order=0, section_path=("5 Finance",)),
        _ldu(text="Results precision overview.", page_number=2, source_block_order=0, section_path=("2 Results",)),
    ]
    store.ingest_ldus(ldus)

    first = store.query(topic="results precision")
    second = store.query(topic="results precision")

    assert [match.record_id for match in first] == [match.record_id for match in second]
    assert all(isinstance(match, VectorStoreMatch) for match in first)


def test_retrieval_evaluation_baseline_flow() -> None:
    vector_backend = FakeVectorRetrievalBackend(
        responses={
            ("precision query", None): [
                VectorStoreMatch(record_id="doc123-r1", text="Precision", metadata={}, distance=0.1),
                VectorStoreMatch(record_id="doc123-r9", text="Noise", metadata={}, distance=0.2),
                VectorStoreMatch(record_id="doc123-r8", text="Noise", metadata={}, distance=0.3),
            ]
        }
    )
    evaluator = RetrievalEvaluator(
        vector_backend=vector_backend,
        page_index_backend=FakePageIndexTraversalBackend(responses={}),
    )

    report = evaluator.evaluate_baseline(
        [LabeledRetrievalQuery(query_id="q1", topic="precision query", relevant_record_ids=("doc123-r1",))]
    )

    assert report.per_query[0].retrieved_record_ids == ("doc123-r1", "doc123-r9", "doc123-r8")
    assert report.metrics.precision_at_3 == pytest.approx(1.0 / 3.0)
    assert report.metrics.hit_rate == 1.0


def test_retrieval_evaluation_pageindex_assisted_flow() -> None:
    tree = PageIndexBuilder().build(
        doc_id="doc123",
        ldus=[
            _ldu(text="Methods", page_number=1, source_block_order=0, section_path=("1 Methods",)),
            _ldu(text="Results", page_number=2, source_block_order=0, section_path=("2 Results",)),
        ],
    )
    vector_backend = FakeVectorRetrievalBackend(
        responses={
            ("results query", ("2 Results",)): [
                VectorStoreMatch(record_id="doc123-r2", text="Results", metadata={}, distance=0.1),
                VectorStoreMatch(record_id="doc123-r3", text="More results", metadata={}, distance=0.2),
            ],
            ("results query", ("1 Methods",)): [
                VectorStoreMatch(record_id="doc123-r7", text="Methods", metadata={}, distance=0.1),
            ],
        }
    )
    page_index_backend = FakePageIndexTraversalBackend(
        responses={
            "results query": [
                PageIndexMatch(
                    node_id="node-results",
                    title="2 Results",
                    section_path=("2 Results",),
                    score=10,
                    start_page=2,
                    end_page=2,
                    summary="Results summary",
                ),
                PageIndexMatch(
                    node_id="node-methods",
                    title="1 Methods",
                    section_path=("1 Methods",),
                    score=5,
                    start_page=1,
                    end_page=1,
                    summary="Methods summary",
                ),
            ]
        }
    )
    evaluator = RetrievalEvaluator(vector_backend=vector_backend, page_index_backend=page_index_backend)

    report = evaluator.evaluate_pageindex_assisted(
        tree,
        [LabeledRetrievalQuery(query_id="q1", topic="results query", relevant_record_ids=("doc123-r2", "doc123-r3"))],
    )

    assert report.per_query[0].retrieved_record_ids == ("doc123-r2", "doc123-r3", "doc123-r7")
    assert report.metrics.precision_at_3 == pytest.approx(2.0 / 3.0)
    assert report.metrics.hit_rate == 1.0


def test_retrieval_evaluation_applies_section_filters_before_vector_search() -> None:
    tree = PageIndexBuilder().build(
        doc_id="doc123",
        ldus=[
            _ldu(text="Results", page_number=1, source_block_order=0, section_path=("2 Results",)),
            _ldu(text="Discussion", page_number=2, source_block_order=0, section_path=("3 Discussion",)),
        ],
    )
    vector_backend = FakeVectorRetrievalBackend(
        responses={
            ("topic", ("2 Results",)): [VectorStoreMatch(record_id="r1", text="A", metadata={}, distance=0.1)],
            ("topic", ("3 Discussion",)): [VectorStoreMatch(record_id="r2", text="B", metadata={}, distance=0.2)],
        }
    )
    page_index_backend = FakePageIndexTraversalBackend(
        responses={
            "topic": [
                PageIndexMatch("n1", "2 Results", ("2 Results",), 10, 1, 1, "summary"),
                PageIndexMatch("n2", "3 Discussion", ("3 Discussion",), 9, 2, 2, "summary"),
            ]
        }
    )
    evaluator = RetrievalEvaluator(vector_backend=vector_backend, page_index_backend=page_index_backend)

    evaluator.evaluate_pageindex_assisted(
        tree,
        [LabeledRetrievalQuery(query_id="q1", topic="topic", relevant_record_ids=("r1",))],
    )

    assert [call["section_path"] for call in vector_backend.calls] == [("2 Results",), ("3 Discussion",)]


def test_retrieval_evaluation_metrics_are_deterministic() -> None:
    vector_backend = FakeVectorRetrievalBackend(
        responses={
            ("q-results", None): [
                VectorStoreMatch(record_id="r1", text="A", metadata={}, distance=0.1),
                VectorStoreMatch(record_id="r2", text="B", metadata={}, distance=0.2),
                VectorStoreMatch(record_id="r3", text="C", metadata={}, distance=0.3),
            ],
            ("q-finance", None): [
                VectorStoreMatch(record_id="r9", text="X", metadata={}, distance=0.1),
                VectorStoreMatch(record_id="r8", text="Y", metadata={}, distance=0.2),
                VectorStoreMatch(record_id="r7", text="Z", metadata={}, distance=0.3),
            ],
        }
    )
    evaluator = RetrievalEvaluator(
        vector_backend=vector_backend,
        page_index_backend=FakePageIndexTraversalBackend(responses={}),
    )
    queries = [
        LabeledRetrievalQuery(query_id="q1", topic="q-results", relevant_record_ids=("r1", "r3")),
        LabeledRetrievalQuery(query_id="q2", topic="q-finance", relevant_record_ids=("r5",)),
    ]

    first = evaluator.evaluate_baseline(queries)
    second = evaluator.evaluate_baseline(queries)

    assert first.metrics == second.metrics
    assert first.metrics.precision_at_3 == pytest.approx((2.0 / 3.0 + 0.0) / 2.0)
    assert first.metrics.hit_rate == 0.5
