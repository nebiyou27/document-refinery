from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.fact_table_extractor import FactTableExtractor
from src.agents.query_agent import QueryAgent
from src.agents.structured_fact_query import StructuredFactQueryBackend
from src.chunking.page_index import PageIndexTree
from src.chunking.page_index_query import PageIndexMatch
from src.chunking.vector_store import VectorStoreMatch
from src.models.chunking import PageIndexNode
from src.models.extracted_document import ExtractedDocument, ExtractedPage, ExtractionMetadata, TableBlock
from src.storage import FactTableSqliteWriter


class FakePageIndexBackend:
    def __init__(self, responses: dict[str, list[PageIndexMatch]]) -> None:
        self.responses = responses
        self.calls: list[dict[str, object]] = []

    def query(self, tree: PageIndexTree, topic: str, top_k: int = 3) -> list[PageIndexMatch]:
        self.calls.append({"tree": tree.doc_id, "topic": topic, "top_k": top_k})
        return self.responses.get(topic, [])[:top_k]


class FakeVectorBackend:
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


def _tree() -> PageIndexTree:
    return PageIndexTree(doc_id="doc123", root_id="root", nodes=[])


def _metadata() -> ExtractionMetadata:
    return ExtractionMetadata(
        strategy_used="strategy_b",
        confidence_score=0.96,
        processing_time_sec=0.01,
        cost_estimate_usd=0.0,
        escalation_triggered=False,
    )


def _document_with_table(rows: list[list[str]]) -> ExtractedDocument:
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
                signals={"char_count": 0, "char_density": 0.0, "image_area_ratio": 0.0, "table_count": 1},
                table_blocks=[
                    TableBlock(
                        doc_id="doc123",
                        page_number=1,
                        bbox=(0.0, 50.0, 200.0, 160.0),
                        content_hash="table-hash-1",
                        table_index=0,
                        rows=rows,
                    )
                ],
                page_content_hash="page-hash-1",
            )
        ],
    )


def _page_index_match(section_path: tuple[str, ...]) -> PageIndexMatch:
    return PageIndexMatch(
        node_id="node-1",
        title=section_path[-1],
        section_path=section_path,
        score=8,
        start_page=1,
        end_page=1,
        summary="Results summary",
    )


def _vector_match(record_id: str, text: str, *, section_path: tuple[str, ...]) -> VectorStoreMatch:
    return VectorStoreMatch(
        record_id=record_id,
        text=text,
        metadata={
            "record_type": "chunk",
            "doc_id": "doc123",
            "document_name": "sample.pdf",
            "page_number": 1,
            "bbox": [0.0, 10.0, 80.0, 22.0],
            "section_path": list(section_path),
            "content_hash": f"hash-{record_id}",
            "strategy_used": "strategy_b",
            "confidence_score": 0.93,
        },
        distance=0.1,
    )


def test_query_agent_returns_verified_answer_with_provenance_chain() -> None:
    page_index_backend = FakePageIndexBackend(
        responses={"results precision": [_page_index_match(("2 Results",))]}
    )
    vector_backend = FakeVectorBackend(
        responses={
            ("results precision", ("2 Results",)): [
                _vector_match("chunk-1", "Precision improved by 8 percent.", section_path=("2 Results",)),
                _vector_match("chunk-2", "Recall improved by 4 percent.", section_path=("2 Results",)),
            ]
        }
    )

    result = QueryAgent(page_index_backend=page_index_backend, vector_backend=vector_backend).answer(
        tree=_tree(),
        query="results precision",
        top_k=2,
    )

    assert result.status == "verified"
    assert result.route == "pageindex_assisted"
    assert result.answer == "Precision improved by 8 percent. Recall improved by 4 percent."
    assert result.provenance_chain is not None
    assert len(result.provenance_chain.entries) == 2
    assert result.provenance_chain.entries[0].provenance.document_name == "sample.pdf"
    assert vector_backend.calls[0]["section_path"] == ("2 Results",)


def test_query_agent_falls_back_to_baseline_vector_when_pageindex_assisted_is_empty() -> None:
    page_index_backend = FakePageIndexBackend(
        responses={"finance outlook": [_page_index_match(("5 Finance",))]}
    )
    vector_backend = FakeVectorBackend(
        responses={
            ("finance outlook", None): [
                _vector_match("chunk-9", "Finance remained stable through the quarter.", section_path=("5 Finance",))
            ]
        }
    )

    result = QueryAgent(page_index_backend=page_index_backend, vector_backend=vector_backend).answer(
        tree=_tree(),
        query="finance outlook",
    )

    assert result.status == "verified"
    assert result.route == "baseline_vector"
    assert result.answer == "Finance remained stable through the quarter."
    assert result.provenance_chain is not None
    assert len(vector_backend.calls) == 2
    assert vector_backend.calls[0]["section_path"] == ("5 Finance",)
    assert vector_backend.calls[1]["section_path"] is None


def test_query_agent_returns_unverifiable_when_no_evidence_is_found() -> None:
    result = QueryAgent(
        page_index_backend=FakePageIndexBackend(responses={}),
        vector_backend=FakeVectorBackend(responses={}),
    ).answer(
        tree=_tree(),
        query="missing topic",
    )

    assert result.status == "unverifiable"
    assert result.answer is None
    assert result.provenance_chain is None
    assert result.failure_reason == "No supporting evidence retrieved"


def test_query_agent_returns_unverifiable_when_provenance_cannot_be_built() -> None:
    page_index_backend = FakePageIndexBackend(
        responses={"results": [_page_index_match(("2 Results",))]}
    )
    vector_backend = FakeVectorBackend(
        responses={
            ("results", ("2 Results",)): [
                VectorStoreMatch(
                    record_id="chunk-1",
                    text="Precision improved by 8 percent.",
                    metadata={
                        "record_type": "chunk",
                        "doc_id": "doc123",
                        "page_number": 1,
                        "content_hash": "hash-1",
                        "strategy_used": "strategy_b",
                        "confidence_score": 0.93,
                    },
                    distance=0.1,
                )
            ]
        }
    )

    result = QueryAgent(page_index_backend=page_index_backend, vector_backend=vector_backend).answer(
        tree=_tree(),
        query="results",
    )

    assert result.status == "unverifiable"
    assert result.answer is None
    assert result.provenance_chain is None
    assert "bbox is required" in (result.failure_reason or "")


def test_query_agent_answers_fact_query_from_structured_sqlite(tmp_path: Path) -> None:
    document = _document_with_table(
        [
            ["Category", "30June2022 Birr'ooo", "30June2021 As Restated Birr'ooo"],
            ["Cashand cashequivalentsat the endoftheyear", "28,191,157", "15,194,080"],
        ]
    )
    db_path = tmp_path / "fact_table.sqlite"
    FactTableSqliteWriter().write(
        fact_table=FactTableExtractor().extract(document),
        db_path=db_path,
    )
    vector_backend = FakeVectorBackend(responses={})
    result = QueryAgent(
        page_index_backend=FakePageIndexBackend(responses={}),
        vector_backend=vector_backend,
        structured_query_backend=StructuredFactQueryBackend(db_path=db_path),
    ).answer(
        tree=_tree(),
        query="What were cash and cash equivalents at the end of the year?",
    )

    assert result.status == "verified"
    assert result.route == "structured_query"
    assert result.answer is not None
    assert "28,191,157" in result.answer
    assert "15,194,080" in result.answer
    assert result.provenance_chain is not None
    assert len(result.provenance_chain.entries) == 2
    assert result.provenance_chain.entries[0].provenance.page_number == 1
    assert vector_backend.calls == []


def test_query_agent_prefers_parts_list_snippet_for_parts_query() -> None:
    page_index_backend = FakePageIndexBackend(
        responses={"What are the parts of the final research?": [_page_index_match(("6 PARTS OF THE RESEARCH PROPOSAL",))]}
    )
    vector_backend = FakeVectorBackend(
        responses={
            ("What are the parts of the final research?", ("6 PARTS OF THE RESEARCH PROPOSAL",)): [
                _vector_match(
                    "chunk-wrong",
                    (
                        "Now that you have established that your question remains unanswered, "
                        "your final task in this section is to argue why it is worth answering."
                    ),
                    section_path=("6 PARTS OF THE RESEARCH PROPOSAL",),
                ),
            ],
            ("the final research should contain the following", ("6 PARTS OF THE RESEARCH PROPOSAL",)): [
                _vector_match(
                    "chunk-right",
                    (
                        "Research Methodology details... Parts of the Final Research "
                        "Your final research should contain the following: Title page; "
                        "Signed Approval Sheet; Abstract; Chapter 1 -- Research Problem."
                    ),
                    section_path=("6 PARTS OF THE RESEARCH PROPOSAL",),
                ),
            ]
        }
    )

    result = QueryAgent(page_index_backend=page_index_backend, vector_backend=vector_backend).answer(
        tree=_tree(),
        query="What are the parts of the final research?",
        top_k=2,
    )

    assert result.status == "verified"
    assert result.answer is not None
    assert result.answer.startswith("Parts of the Final Research")
    assert "Title page" in result.answer
    assert any(call["topic"] == "the final research should contain the following" for call in vector_backend.calls)


def test_query_agent_parts_query_lexical_fallback_prefers_final_research_summary() -> None:
    page_index_backend = FakePageIndexBackend(
        responses={
            "What are the parts of the final research?": [
                PageIndexMatch(
                    node_id="node-generic",
                    title="6 PARTS OF THE RESEARCH PROPOSAL",
                    section_path=("6 PARTS OF THE RESEARCH PROPOSAL",),
                    score=10,
                    start_page=5,
                    end_page=8,
                    summary="Parts of the research proposal with broad guidance.",
                ),
                PageIndexMatch(
                    node_id="node-target",
                    title="6 PARTS OF THE RESEARCH PROPOSAL",
                    section_path=("6 PARTS OF THE RESEARCH PROPOSAL",),
                    score=9,
                    start_page=8,
                    end_page=8,
                    summary=(
                        "Parts of the Final Research: Your final research should contain "
                        "the following items including Title page and Abstract."
                    ),
                ),
            ]
        }
    )

    tree = PageIndexTree(
        doc_id="doc123",
        root_id="root",
        nodes=[
            PageIndexNode(
                node_id="root",
                title="ROOT",
                section_path=(),
                parent_id=None,
                depth=0,
                start_page=1,
                end_page=8,
                bbox=(0.0, 0.0, 10.0, 10.0),
                child_ids=["node-generic", "node-target"],
                order_index=0,
            ),
            PageIndexNode(
                node_id="node-generic",
                title="6 PARTS OF THE RESEARCH PROPOSAL",
                section_path=("6 PARTS OF THE RESEARCH PROPOSAL",),
                parent_id="root",
                depth=1,
                start_page=5,
                end_page=8,
                bbox=(0.0, 0.0, 10.0, 10.0),
                order_index=1,
            ),
            PageIndexNode(
                node_id="node-target",
                title="6 PARTS OF THE RESEARCH PROPOSAL",
                section_path=("6 PARTS OF THE RESEARCH PROPOSAL",),
                parent_id="root",
                depth=1,
                start_page=8,
                end_page=8,
                bbox=(0.0, 0.0, 10.0, 10.0),
                order_index=2,
            ),
        ],
    )

    result = QueryAgent(
        page_index_backend=page_index_backend,
        vector_backend=FakeVectorBackend(responses={}),
    ).answer(
        tree=tree,
        query="What are the parts of the final research?",
        top_k=2,
    )

    assert result.status == "verified"
    assert result.route == "pageindex_lexical"
    assert result.answer is not None
    assert "Parts of the Final Research" in result.answer


def test_query_agent_numeric_query_uses_hybrid_variants_and_reranks_table_like_chunk() -> None:
    page_index_backend = FakePageIndexBackend(
        responses={"combined ratio 2018": [_page_index_match(("3 Financial Highlights",))]}
    )
    vector_backend = FakeVectorBackend(
        responses={
            ("combined ratio 2018", ("3 Financial Highlights",)): [
                _vector_match(
                    "chunk-narrative",
                    "The global reinsurance sector saw changing demand conditions in 2017 and 2018.",
                    section_path=("3 Global Reinsurance Sector",),
                )
            ],
            ("combined ratio 2018 financial highlights", ("3 Financial Highlights",)): [
                _vector_match(
                    "chunk-table",
                    "Financial highlights table: Combined ratio 2018 | 96.4%; 2017 | 98.1%",
                    section_path=("3 Financial Highlights",),
                )
            ],
        }
    )

    result = QueryAgent(page_index_backend=page_index_backend, vector_backend=vector_backend).answer(
        tree=_tree(),
        query="combined ratio 2018",
        top_k=1,
    )

    assert result.status == "verified"
    assert result.answer is not None
    assert "Combined ratio 2018" in result.answer
    assert any(call["topic"] == "combined ratio 2018 financial highlights" for call in vector_backend.calls)


def test_query_agent_section_query_penalizes_front_matter_chunk() -> None:
    page_index_backend = FakePageIndexBackend(
        responses={"research questions": [_page_index_match(("1 Introduction",))]}
    )
    vector_backend = FakeVectorBackend(
        responses={
            ("research questions", ("1 Introduction",)): [
                _vector_match(
                    "chunk-front",
                    "Title page and author student ID details for the submission.",
                    section_path=("Title Page",),
                ),
                _vector_match(
                    "chunk-core",
                    "Research Questions: This study addresses two research questions on retention.",
                    section_path=("1.2 Research Questions",),
                ),
            ]
        }
    )

    result = QueryAgent(page_index_backend=page_index_backend, vector_backend=vector_backend).answer(
        tree=_tree(),
        query="research questions",
        top_k=1,
    )

    assert result.status == "verified"
    assert result.answer is not None
    assert result.answer.startswith("Research Questions")


def test_query_agent_definition_query_prefers_definition_over_author_list() -> None:
    page_index_backend = FakePageIndexBackend(
        responses={"definition of leverage": [_page_index_match(("2 Concepts",))]}
    )
    vector_backend = FakeVectorBackend(
        responses={
            ("definition of leverage", ("2 Concepts",)): [
                _vector_match(
                    "chunk-front",
                    "Author names and student ID list from the title page.",
                    section_path=("Title Page",),
                ),
                _vector_match(
                    "chunk-def",
                    "Leverage is the use of debt to amplify potential returns.",
                    section_path=("2 Concepts",),
                ),
            ]
        }
    )

    result = QueryAgent(page_index_backend=page_index_backend, vector_backend=vector_backend).answer(
        tree=_tree(),
        query="definition of leverage",
        top_k=1,
    )

    assert result.status == "verified"
    assert result.answer is not None
    assert result.answer.startswith("Leverage is")
