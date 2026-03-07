"""Phase 4 provenance assembly helpers for retrieval outputs."""

from __future__ import annotations

from typing import Any, Iterable

from src.chunking.vector_store import VectorStoreMatch
from src.models import ExtractionStrategy, ProvenanceChain, ProvenanceChainEntry, ProvenanceRef
from src.utils.hashing import canonicalize_text


class ProvenanceChainError(ValueError):
    """Raised when retrieval output cannot be converted into grounded provenance."""


class ProvenanceChainBuilder:
    """Builds validated provenance chains from retrieval matches."""

    def build(
        self,
        matches: Iterable[VectorStoreMatch],
        *,
        query: str | None = None,
    ) -> ProvenanceChain:
        entries = tuple(self._entry_from_match(match) for match in matches)
        if not entries:
            raise ProvenanceChainError("ProvenanceChain requires at least one retrieval match")
        return ProvenanceChain(entries=entries, query=query)

    def _entry_from_match(self, match: VectorStoreMatch) -> ProvenanceChainEntry:
        metadata = match.metadata
        record_id = self._require_string(match.record_id, "record_id")
        record_type = self._require_string(metadata.get("record_type"), "record_type")
        doc_id = self._require_string(metadata.get("doc_id"), "doc_id")
        content_hash = self._require_string(metadata.get("content_hash"), "content_hash")
        strategy_raw = self._require_string(metadata.get("strategy_used"), "strategy_used")
        confidence_score = self._require_float(metadata.get("confidence_score"), "confidence_score")
        page_number = self._require_int(metadata.get("page_number"), "page_number")
        bbox = self._require_bbox(metadata.get("bbox"))
        section_path = self._coerce_section_path(metadata.get("section_path"))
        document_name = self._coerce_document_name(metadata.get("document_name"), doc_id=doc_id)
        snippet = canonicalize_text(match.text)
        if not snippet:
            raise ProvenanceChainError("snippet cannot be empty")

        try:
            strategy_used = ExtractionStrategy(strategy_raw)
        except ValueError as exc:
            raise ProvenanceChainError(f"unsupported strategy_used: {strategy_raw}") from exc

        return ProvenanceChainEntry(
            record_id=record_id,
            record_type=record_type,
            section_path=section_path,
            snippet=snippet,
            distance=match.distance,
            provenance=ProvenanceRef(
                document_name=document_name,
                doc_id=doc_id,
                page_number=page_number,
                bbox=bbox,
                content_hash=content_hash,
                strategy_used=strategy_used,
                confidence_score=confidence_score,
            ),
        )

    def _require_string(self, value: Any, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ProvenanceChainError(f"{field_name} is required for provenance assembly")
        return value

    def _require_int(self, value: Any, field_name: str) -> int:
        if not isinstance(value, int) or value < 1:
            raise ProvenanceChainError(f"{field_name} must be a positive integer")
        return value

    def _require_float(self, value: Any, field_name: str) -> float:
        if not isinstance(value, (float, int)):
            raise ProvenanceChainError(f"{field_name} is required for provenance assembly")
        return float(value)

    def _require_bbox(self, value: Any) -> tuple[float, float, float, float]:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ProvenanceChainError("bbox is required for provenance assembly")
        try:
            x0, y0, x1, y1 = (float(coordinate) for coordinate in value)
        except (TypeError, ValueError) as exc:
            raise ProvenanceChainError("bbox must contain numeric coordinates") from exc
        return (x0, y0, x1, y1)

    def _coerce_section_path(self, value: Any) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, list):
            raise ProvenanceChainError("section_path must be a list when present")
        if not all(isinstance(part, str) for part in value):
            raise ProvenanceChainError("section_path values must be strings")
        return tuple(value)

    def _coerce_document_name(self, value: Any, *, doc_id: str) -> str:
        if isinstance(value, str) and value.strip():
            return value
        return doc_id
