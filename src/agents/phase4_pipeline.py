"""Integrated Phase 4 pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from src.agents.audit_mode import AuditMode, AuditResult
from src.agents.fact_table_extractor import FactTableExtractor
from src.agents.query_agent import QueryAgent, QueryAgentResult
from src.chunking.engine import ChunkingEngine
from src.chunking.page_index import PageIndexBuilder, PageIndexTree
from src.chunking.page_index_query import PageIndexQueryEngine
from src.chunking.page_index_summarizer import PageIndexSummarizer, SummaryBackend
from src.chunking.vector_store import ChromaVectorStore
from src.models import Chunk, ExtractedDocument, FactTable, LDU


@dataclass(frozen=True)
class Phase4QueryRun:
    """Combined query-agent and audit outputs for one topic."""

    query_result: QueryAgentResult
    audit_result: AuditResult


@dataclass(frozen=True)
class Phase4PipelineResult:
    """Integrated Phase 4 artifacts for one extracted document."""

    extracted: ExtractedDocument
    ldus: tuple[LDU, ...]
    chunks: tuple[Chunk, ...]
    tree: PageIndexTree
    fact_table: FactTable
    query_runs: tuple[Phase4QueryRun, ...]


class Phase4Pipeline:
    """Runs chunking, retrieval, audit, and fact extraction as one flow."""

    def __init__(
        self,
        *,
        vector_store: ChromaVectorStore,
        summary_backend: SummaryBackend,
        chunking_engine: ChunkingEngine | None = None,
        page_index_query: PageIndexQueryEngine | None = None,
        query_agent: QueryAgent | None = None,
        audit_mode: AuditMode | None = None,
        fact_table_extractor: FactTableExtractor | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.summary_backend = summary_backend
        self.chunking_engine = chunking_engine or ChunkingEngine()
        self.page_index_query = page_index_query or PageIndexQueryEngine()
        self.query_agent = query_agent or QueryAgent(
            page_index_backend=self.page_index_query,
            vector_backend=self.vector_store,
        )
        self.audit_mode = audit_mode or AuditMode()
        self.fact_table_extractor = fact_table_extractor or FactTableExtractor(
            chunking_engine=self.chunking_engine
        )

    def run(
        self,
        *,
        extracted: ExtractedDocument,
        queries: list[str],
        top_k: int = 3,
        section_top_k: int = 3,
    ) -> Phase4PipelineResult:
        ldus = tuple(self.chunking_engine.build_ldus(extracted))
        chunks = tuple(self.chunking_engine.build_chunks(extracted, ldus=list(ldus)))
        tree = PageIndexBuilder().build(doc_id=extracted.doc_id, ldus=list(ldus))
        summarized_tree = PageIndexSummarizer(self.summary_backend).summarize_tree(tree=tree, ldus=list(ldus))

        self.vector_store.ingest_ldus(list(ldus))
        self.vector_store.ingest_chunks(list(chunks))

        query_runs_list: list[Phase4QueryRun] = []
        for query in queries:
            query_result = self.query_agent.answer(
                tree=summarized_tree,
                query=query,
                top_k=top_k,
                section_top_k=section_top_k,
            )
            query_runs_list.append(
                Phase4QueryRun(
                    query_result=query_result,
                    audit_result=self.audit_mode.audit(query_result),
                )
            )
        query_runs = tuple(query_runs_list)

        fact_table = self.fact_table_extractor.extract(extracted)
        return Phase4PipelineResult(
            extracted=extracted,
            ldus=ldus,
            chunks=chunks,
            tree=summarized_tree,
            fact_table=fact_table,
            query_runs=query_runs,
        )
