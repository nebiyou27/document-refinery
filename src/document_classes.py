"""Deterministic document-family classification and Phase 3 policy rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Mapping

from src.models.document_profile import DocumentProfile, DomainHint, LayoutComplexity


class DocumentClass(str, Enum):
    narrative_report_like = "narrative_report_like"
    statistical_bulletin = "statistical_bulletin"
    table_heavy_financial_administrative = "table_heavy_financial_administrative"
    audit_procurement_structured = "audit_procurement_structured"


class SectionInferenceMode(str, Enum):
    strict = "strict"
    relaxed = "relaxed"


@dataclass(frozen=True)
class DocumentClassPolicy:
    document_class: DocumentClass
    retrieval_query_derivation_expected: bool
    zero_query_outcome: str
    zero_query_reason: str
    section_inference_mode: SectionInferenceMode


DOCUMENT_CLASS_POLICIES: dict[DocumentClass, DocumentClassPolicy] = {
    DocumentClass.narrative_report_like: DocumentClassPolicy(
        document_class=DocumentClass.narrative_report_like,
        retrieval_query_derivation_expected=True,
        zero_query_outcome="failed",
        zero_query_reason="document class expects narrative/section retrieval queries",
        section_inference_mode=SectionInferenceMode.strict,
    ),
    DocumentClass.statistical_bulletin: DocumentClassPolicy(
        document_class=DocumentClass.statistical_bulletin,
        retrieval_query_derivation_expected=True,
        zero_query_outcome="failed",
        zero_query_reason="document class expects recurring bulletin sections suitable for retrieval queries",
        section_inference_mode=SectionInferenceMode.strict,
    ),
    DocumentClass.table_heavy_financial_administrative: DocumentClassPolicy(
        document_class=DocumentClass.table_heavy_financial_administrative,
        retrieval_query_derivation_expected=False,
        zero_query_outcome="skipped",
        zero_query_reason="document class is table-led; missing derived section queries is not scored as a retrieval failure",
        section_inference_mode=SectionInferenceMode.relaxed,
    ),
    DocumentClass.audit_procurement_structured: DocumentClassPolicy(
        document_class=DocumentClass.audit_procurement_structured,
        retrieval_query_derivation_expected=True,
        zero_query_outcome="failed",
        zero_query_reason="document class is structured and should expose retrievable sections",
        section_inference_mode=SectionInferenceMode.relaxed,
    ),
}


_AUDIT_PROCUREMENT_PATTERNS = (
    "audit",
    "auditor",
    "procurement",
    "tender",
    "bid",
    "bidding",
    "lot",
    "rfx",
)
_STATISTICAL_BULLETIN_PATTERNS = (
    "bulletin",
    "statistical",
    "statistics",
    "monthly",
    "quarterly",
    "digest",
    "indicator",
    "cpi",
    "ppi",
)
_FINANCIAL_ADMIN_PATTERNS = (
    "annual report",
    "financial",
    "budget",
    "expense",
    "revenue",
    "statement",
    "accounts",
    "fiscal",
    "balance sheet",
)


def resolve_document_class(
    *,
    file_name: str,
    profile: DocumentProfile | None = None,
    row: Mapping[str, str] | None = None,
) -> DocumentClassPolicy:
    override = _parse_override((row or {}).get("document_class"))
    if override is not None:
        return DOCUMENT_CLASS_POLICIES[override]

    normalized_name = _normalize_name(file_name)
    if _contains_any(normalized_name, _AUDIT_PROCUREMENT_PATTERNS):
        return DOCUMENT_CLASS_POLICIES[DocumentClass.audit_procurement_structured]
    if _contains_any(normalized_name, _STATISTICAL_BULLETIN_PATTERNS):
        return DOCUMENT_CLASS_POLICIES[DocumentClass.statistical_bulletin]

    if profile is not None:
        if profile.layout_complexity == LayoutComplexity.table_heavy:
            return DOCUMENT_CLASS_POLICIES[DocumentClass.table_heavy_financial_administrative]
        if profile.domain_hint == DomainHint.financial:
            return DOCUMENT_CLASS_POLICIES[DocumentClass.table_heavy_financial_administrative]

    if _contains_any(normalized_name, _FINANCIAL_ADMIN_PATTERNS):
        return DOCUMENT_CLASS_POLICIES[DocumentClass.table_heavy_financial_administrative]
    return DOCUMENT_CLASS_POLICIES[DocumentClass.narrative_report_like]


def _parse_override(raw_value: str | None) -> DocumentClass | None:
    if raw_value is None:
        return None
    normalized = raw_value.strip().lower()
    if not normalized:
        return None
    normalized = normalized.replace("-", "_").replace(" ", "_")
    aliases = {
        "narrative": DocumentClass.narrative_report_like,
        "narrative_report_like": DocumentClass.narrative_report_like,
        "report": DocumentClass.narrative_report_like,
        "statistical_bulletin": DocumentClass.statistical_bulletin,
        "bulletin": DocumentClass.statistical_bulletin,
        "table_heavy_financial_administrative": DocumentClass.table_heavy_financial_administrative,
        "table_heavy_financial_admin": DocumentClass.table_heavy_financial_administrative,
        "financial_admin": DocumentClass.table_heavy_financial_administrative,
        "audit_procurement_structured": DocumentClass.audit_procurement_structured,
        "audit_procurement": DocumentClass.audit_procurement_structured,
        "audit": DocumentClass.audit_procurement_structured,
        "procurement": DocumentClass.audit_procurement_structured,
    }
    return aliases.get(normalized)


def _normalize_name(file_name: str) -> str:
    stem = file_name.rsplit(".", 1)[0]
    return re.sub(r"[_\-]+", " ", stem).casefold()


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)
