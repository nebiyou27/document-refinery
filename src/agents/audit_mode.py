"""Deterministic Phase 4 audit mode for answer-to-provenance verification."""

from __future__ import annotations

from dataclasses import dataclass
import re

from src.agents.query_agent import QueryAgentResult
from src.models import ProvenanceChain, ProvenanceChainEntry
from src.utils.hashing import canonicalize_text

STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "at",
        "by",
        "for",
        "in",
        "is",
        "of",
        "on",
        "or",
        "report",
        "state",
        "stated",
        "states",
        "that",
        "the",
        "to",
        "was",
        "were",
        "with",
    }
)


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


@dataclass(frozen=True)
class ClaimVerificationResult:
    """Outcome of verifying a single claim against retrieved provenance."""

    claim: str
    status: str
    provenance_chain: ProvenanceChain | None
    support_ratio: float
    supporting_record_ids: tuple[str, ...]
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

    def verify_claim(self, claim: str, result: QueryAgentResult) -> ClaimVerificationResult:
        normalized_claim = canonicalize_text(claim)
        if not normalized_claim:
            return ClaimVerificationResult(
                claim=claim,
                status="unverifiable",
                provenance_chain=None,
                support_ratio=0.0,
                supporting_record_ids=(),
                failure_reason="Claim is empty after normalization",
            )

        if result.status != "verified" or result.provenance_chain is None:
            return ClaimVerificationResult(
                claim=claim,
                status="unverifiable",
                provenance_chain=None,
                support_ratio=0.0,
                supporting_record_ids=(),
                failure_reason="No verified evidence available for claim",
            )

        finding = self._audit_claim(normalized_claim, result.provenance_chain.entries)
        if not finding.supported:
            return ClaimVerificationResult(
                claim=claim,
                status="unverifiable",
                provenance_chain=None,
                support_ratio=finding.support_ratio,
                supporting_record_ids=finding.supporting_record_ids,
                failure_reason="Claim not found in retrieved evidence",
            )

        supported_entries = tuple(
            entry for entry in result.provenance_chain.entries if entry.record_id in finding.supporting_record_ids
        )
        provenance_chain = ProvenanceChain(entries=supported_entries, query=result.query)
        return ClaimVerificationResult(
            claim=claim,
            status="verified",
            provenance_chain=provenance_chain,
            support_ratio=finding.support_ratio,
            supporting_record_ids=finding.supporting_record_ids,
            failure_reason=None,
        )

    def _audit_claim(
        self,
        claim: str,
        entries: tuple[ProvenanceChainEntry, ...],
    ) -> AuditFinding:
        claim_text = self._normalize_for_matching(claim)
        claim_tokens = self._content_tokens(claim_text)
        claim_numbers = self._numbers(claim_text)
        best_ratio = 0.0
        supporting_record_ids: list[str] = []
        matched_by_numeric_rule = False

        for entry in entries:
            snippet_text = self._normalize_for_matching(entry.snippet)
            snippet_tokens = self._content_tokens(snippet_text)
            support_ratio = self._support_ratio(claim_tokens, snippet_tokens)
            numeric_match = self._has_exact_numeric_support(claim_numbers, snippet_text)
            entity_overlap = self._entity_overlap_ratio(claim_tokens, snippet_tokens)
            supported = support_ratio >= self.minimum_support_ratio or (
                numeric_match and entity_overlap >= 0.5
            )
            effective_ratio = (
                max(support_ratio, entity_overlap, self.minimum_support_ratio) if supported else support_ratio
            )
            if support_ratio <= 0:
                if not supported:
                    continue
            if effective_ratio > best_ratio:
                best_ratio = effective_ratio
                supporting_record_ids = [entry.record_id]
                matched_by_numeric_rule = supported and support_ratio < self.minimum_support_ratio
            elif effective_ratio == best_ratio:
                supporting_record_ids.append(entry.record_id)
                matched_by_numeric_rule = matched_by_numeric_rule or (
                    supported and support_ratio < self.minimum_support_ratio
                )

        return AuditFinding(
            claim=claim,
            supported=best_ratio >= self.minimum_support_ratio or matched_by_numeric_rule,
            support_ratio=best_ratio,
            supporting_record_ids=tuple(dict.fromkeys(supporting_record_ids)),
        )

    def _split_claims(self, answer: str) -> list[str]:
        normalized = canonicalize_text(answer)
        if not normalized:
            return []
        parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip()]
        return parts or [normalized]

    def _normalize_for_matching(self, text: str) -> str:
        normalized = canonicalize_text(text).lower()
        normalized = re.sub(r"(?<=\d),(?=\d)", "", normalized)
        normalized = normalized.replace("_", " ")
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        return canonicalize_text(normalized)

    def _tokens(self, text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", self._normalize_for_matching(text)))

    def _content_tokens(self, text: str) -> set[str]:
        tokens = {token for token in re.findall(r"[a-z0-9]+", text) if token not in STOPWORDS}
        return tokens or set(re.findall(r"[a-z0-9]+", text))

    def _support_ratio(self, claim_tokens: set[str], snippet_tokens: set[str]) -> float:
        if not claim_tokens:
            return 0.0
        matched = {token for token in claim_tokens if self._token_matches(token, snippet_tokens)}
        return len(matched) / len(claim_tokens)

    def _entity_overlap_ratio(self, claim_tokens: set[str], snippet_tokens: set[str]) -> float:
        entity_tokens = {token for token in claim_tokens if not token.isdigit()}
        if not entity_tokens:
            return 0.0
        matched = {token for token in entity_tokens if self._token_matches(token, snippet_tokens)}
        return len(matched) / len(entity_tokens)

    def _token_matches(self, token: str, snippet_tokens: set[str]) -> bool:
        if token in snippet_tokens:
            return True
        if token.isdigit() or len(token) < 4:
            return False
        return any(token in snippet_token for snippet_token in snippet_tokens if snippet_token.isalpha())

    def _numbers(self, text: str) -> set[str]:
        return set(re.findall(r"\d+", text))

    def _has_exact_numeric_support(self, claim_numbers: set[str], snippet_text: str) -> bool:
        if not claim_numbers:
            return False
        snippet_numbers = self._numbers(snippet_text)
        return any(number in snippet_numbers for number in claim_numbers)
