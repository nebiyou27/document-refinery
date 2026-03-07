from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.audit_mode import AuditMode
from src.agents.query_agent import QueryAgentResult
from src.models import ExtractionStrategy, ProvenanceChain, ProvenanceChainEntry, ProvenanceRef


def _entry(record_id: str, snippet: str) -> ProvenanceChainEntry:
    return ProvenanceChainEntry(
        record_id=record_id,
        record_type="chunk",
        section_path=("2 Results",),
        snippet=snippet,
        provenance=ProvenanceRef(
            document_name="sample.pdf",
            doc_id="doc123",
            page_number=1,
            bbox=(0.0, 10.0, 80.0, 22.0),
            content_hash=f"hash-{record_id}",
            strategy_used=ExtractionStrategy.strategy_b,
            confidence_score=0.94,
        ),
    )


def _verified_result(answer: str, snippets: list[str]) -> QueryAgentResult:
    entries = tuple(_entry(f"chunk-{index}", snippet) for index, snippet in enumerate(snippets, start=1))
    return QueryAgentResult(
        query="results precision",
        status="verified",
        answer=answer,
        provenance_chain=ProvenanceChain(entries=entries, query="results precision"),
        retrieval_matches=(),
        page_index_matches=(),
        route="pageindex_assisted",
    )


def test_audit_mode_passes_when_claims_are_supported_by_provenance() -> None:
    result = _verified_result(
        answer="Precision improved by 8 percent. Recall improved by 4 percent.",
        snippets=[
            "Precision improved by 8 percent.",
            "Recall improved by 4 percent.",
        ],
    )

    audit = AuditMode().audit(result)

    assert audit.status == "passed"
    assert len(audit.findings) == 2
    assert all(finding.supported for finding in audit.findings)


def test_audit_mode_fails_when_answer_claim_is_not_supported() -> None:
    result = _verified_result(
        answer="Precision improved by 8 percent. Revenue doubled year over year.",
        snippets=[
            "Precision improved by 8 percent.",
            "Recall improved by 4 percent.",
        ],
    )

    audit = AuditMode().audit(result)

    assert audit.status == "failed"
    assert audit.failure_reason == "One or more claims lack provenance support"
    assert audit.findings[0].supported is True
    assert audit.findings[1].supported is False


def test_audit_mode_returns_unverifiable_when_query_result_is_not_verified() -> None:
    result = QueryAgentResult(
        query="results precision",
        status="unverifiable",
        answer=None,
        provenance_chain=None,
        retrieval_matches=(),
        page_index_matches=(),
        route="baseline_vector",
        failure_reason="No supporting evidence retrieved",
    )

    audit = AuditMode().audit(result)

    assert audit.status == "unverifiable"
    assert audit.findings == ()
    assert audit.failure_reason == "Query result is not verified and cannot be audited"
