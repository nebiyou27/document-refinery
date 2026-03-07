"""Deterministic Phase 4 audit mode for answer-to-provenance verification."""

from __future__ import annotations

from dataclasses import dataclass
import re

from src.agents.query_agent import QueryAgentResult
from src.models import ProvenanceChainEntry
from src.utils.hashing import canonicalize_text


@dataclass(frozen=True)
class AuditFinding:
    """Single audited claim with its support decision."""

    claim: str
    supported: bool
    support_ratio: float
    supporting_record_ids: tuple[str, ...]


@dataclass(frozen=True)
class AuditResult:
    """Overall audit outcome for a query-agent response."""

    status: str
    findings: tuple[AuditFinding, ...]
    failure_reason: str | None = None


class AuditMode:
    """Verifies that answer claims are grounded in provenance snippets."""

    def __init__(self, *, minimum_support_ratio: float = 0.6) -> None:
        self.minimum_support_ratio = minimum_support_ratio

    def audit(self, result: QueryAgentResult) -> AuditResult:
        if result.status != "verified" or result.answer is None or result.provenance_chain is None:
            return AuditResult(
                status="unverifiable",
                findings=(),
                failure_reason="Query result is not verified and cannot be audited",
            )

        claims = self._split_claims(result.answer)
        if not claims:
            return AuditResult(
                status="unverifiable",
                findings=(),
                failure_reason="Answer did not contain auditable claims",
            )

        findings = tuple(self._audit_claim(claim, result.provenance_chain.entries) for claim in claims)
        overall_status = "passed" if all(finding.supported for finding in findings) else "failed"
        return AuditResult(
            status=overall_status,
            findings=findings,
            failure_reason=None if overall_status == "passed" else "One or more claims lack provenance support",
        )

    def _audit_claim(
        self,
        claim: str,
        entries: tuple[ProvenanceChainEntry, ...],
    ) -> AuditFinding:
        claim_tokens = self._tokens(claim)
        best_ratio = 0.0
        supporting_record_ids: list[str] = []

        for entry in entries:
            support_ratio = self._support_ratio(claim_tokens, self._tokens(entry.snippet))
            if support_ratio <= 0:
                continue
            if support_ratio > best_ratio:
                best_ratio = support_ratio
                supporting_record_ids = [entry.record_id]
            elif support_ratio == best_ratio:
                supporting_record_ids.append(entry.record_id)

        return AuditFinding(
            claim=claim,
            supported=best_ratio >= self.minimum_support_ratio,
            support_ratio=best_ratio,
            supporting_record_ids=tuple(dict.fromkeys(supporting_record_ids)),
        )

    def _split_claims(self, answer: str) -> list[str]:
        normalized = canonicalize_text(answer)
        if not normalized:
            return []
        parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
        return parts or [normalized]

    def _tokens(self, text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def _support_ratio(self, claim_tokens: set[str], snippet_tokens: set[str]) -> float:
        if not claim_tokens:
            return 0.0
        overlap = claim_tokens & snippet_tokens
        return len(overlap) / len(claim_tokens)
