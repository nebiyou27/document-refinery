from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.query_agent import QueryAgent
from src.chunking.page_index import PageIndexTree
from src.chunking.page_index_query import PageIndexMatch
from src.chunking.vector_store import VectorStoreMatch


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
