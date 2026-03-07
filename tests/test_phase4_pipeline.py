from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.phase4_pipeline import Phase4Pipeline
from src.chunking.page_index_summarizer import SummaryBackend, SummaryInput
from src.chunking.vector_store import ChromaVectorStore, EmbeddingBackend
from src.models.extracted_document import ExtractedDocument, ExtractedPage, ExtractionMetadata, TableBlock, TextBlock


class FakeSummaryBackend(SummaryBackend):
    def summarize(self, summary_input: SummaryInput) -> str:
        return f"{summary_input.title}: summary"


class FakeEmbeddingBackend(EmbeddingBackend):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        lowered = text.lower()
        return [float(len(lowered)), float(lowered.count("results")), float(lowered.count("revenue"))]


class FakeChromaCollection:
    def __init__(self) -> None:
        self.records: dict[str, dict[str, object]] = {}

    def upsert(self, *, ids, documents, embeddings, metadatas) -> None:
        for index, record_id in enumerate(ids):
            self.records[record_id] = {
                "id": record_id,
                "document": documents[index],
                "embedding": embeddings[index],
                "metadata": metadatas[index],
            }

    def query(self, *, query_embeddings, n_results, where=None, include=None) -> dict[str, object]:
        query_embedding = query_embeddings[0]
        filtered = [record for record in self.records.values() if self._matches_where(record["metadata"], where)]
        ranked = sorted(filtered, key=lambda record: self._distance(query_embedding, record["embedding"]))[:n_results]
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
            return all(self._matches_where(metadata, clause) for clause in clauses if isinstance(clause, dict))
        return all(metadata.get(key) == value for key, value in where.items())

    def _distance(self, left: list[float], right: object) -> float:
        if not isinstance(right, list):
            return float("inf")
        return sum(abs(left[index] - float(right[index])) for index in range(min(len(left), len(right))))


def _metadata() -> ExtractionMetadata:
    return ExtractionMetadata(
        strategy_used="strategy_b",
        confidence_score=0.95,
        processing_time_sec=0.01,
        cost_estimate_usd=0.0,
        escalation_triggered=False,
    )


def _document() -> ExtractedDocument:
    return ExtractedDocument(
        doc_id="doc123",
        file_name="finance.pdf",
        file_path="data/finance.pdf",
        page_count=1,
        metadata=_metadata(),
        pages=[
            ExtractedPage(
                doc_id="doc123",
                page_number=1,
                metadata=_metadata(),
                signals={"char_count": 40, "char_density": 0.1, "image_area_ratio": 0.0, "table_count": 1},
                text_blocks=[
                    TextBlock(
                        doc_id="doc123",
                        page_number=1,
                        text="2 Results",
                        bbox=(0.0, 0.0, 80.0, 16.0),
                        reading_order=0,
                        content_hash="t1",
                    ),
                    TextBlock(
                        doc_id="doc123",
                        page_number=1,
                        text="Revenue improved by 250.",
                        bbox=(0.0, 18.0, 100.0, 30.0),
                        reading_order=1,
                        content_hash="t2",
                    ),
                ],
                table_blocks=[
                    TableBlock(
                        doc_id="doc123",
                        page_number=1,
                        bbox=(0.0, 40.0, 200.0, 120.0),
                        content_hash="table1",
                        table_index=0,
                        rows=[
                            ["Category", "2024"],
                            ["Revenue", "250"],
                        ],
                    )
                ],
                page_content_hash="page1",
            )
        ],
    )


def test_phase4_pipeline_emits_queries_audit_and_fact_table() -> None:
    store = ChromaVectorStore(
        embedding_backend=FakeEmbeddingBackend(),
        collection=FakeChromaCollection(),
    )
    pipeline = Phase4Pipeline(
        vector_store=store,
        summary_backend=FakeSummaryBackend(),
    )

    result = pipeline.run(
        extracted=_document(),
        queries=["results revenue"],
        top_k=2,
    )

    assert len(result.ldus) >= 2
    assert len(result.chunks) >= 1
    assert len(result.query_runs) == 1
    assert result.query_runs[0].query_result.status == "verified"
    assert result.query_runs[0].audit_result.status == "passed"
    assert len(result.fact_table.entries) == 1
    assert result.fact_table.entries[0].numeric_value == 250.0
