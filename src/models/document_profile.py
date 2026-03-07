"""
src/models/document_profile.py
================================
Core schemas for Stage 1 — Triage Agent.

All downstream stages depend on these contracts.
Do not change field names without updating triage.py,
extractor.py, and all three strategy files.
"""

import hashlib
from enum import Enum
from pydantic import BaseModel, Field, field_validator


# ================================================================
# Enums — allowed values for each classification dimension
# ================================================================

class OriginType(str, Enum):
    """
    How the document was created.
    Determines which extraction strategy is viable.
    """
    native_digital = "native_digital"   # born-digital, clean text layer
    mixed          = "mixed"            # text + images, partial structure
    scanned_image  = "scanned_image"    # no text layer — OCR required
    form_fillable  = "form_fillable"    # interactive form fields
    error          = "error"            # unreadable / corrupted — Invariant I-11


class LayoutComplexity(str, Enum):
    """
    Document layout structure.
    Drives the choice between Strategy A (simple) and B (complex).
    """
    single_column = "single_column"
    multi_column  = "multi_column"
    table_heavy   = "table_heavy"
    figure_heavy  = "figure_heavy"
    mixed         = "mixed"
    unknown       = "unknown"           # fallback for unclassifiable layouts


class DomainHint(str, Enum):
    """
    Document subject domain.
    Used to select domain-specific extraction prompts and chunking rules.
    """
    financial = "financial"
    legal     = "legal"
    technical = "technical"
    medical   = "medical"
    general   = "general"


class ExtractionStrategy(str, Enum):
    """
    Which extraction strategy the router should apply.
    A = cheapest, C = most expensive.
    """
    strategy_a = "strategy_a"   # fast text — pdfplumber
    strategy_b = "strategy_b"   # layout-aware — Docling
    strategy_c = "strategy_c"   # vision — EasyOCR + Gemini Flash
    none       = "none"         # used only for ERROR profiles


# ================================================================
# PageSignal — per-page measurements from triage
# ================================================================

class PageSignal(BaseModel):
    """
    Signals measured on a single page during triage.
    Used by the escalation guard to make page-level routing decisions.
    """
    page_number:       int
    char_count:        int
    char_density:      float    # chars / page_area (pts²)
    image_area_ratio:  float    # image_area / page_area
    table_count:       int      # tables detected on this page
    x_jump_ratio:      float    # proxy for multi-column layout
    confidence:        float    # 0.0–1.0 confidence for fast text extraction
    assigned_strategy: ExtractionStrategy  # page-level strategy after confidence check


# ================================================================
# DocumentProfile — output of Stage 1 Triage Agent
# ================================================================

class DocumentProfile(BaseModel):
    """
    Complete profile of a document produced by the Triage Agent.
    Every downstream stage reads from this — treat it as immutable
    once produced.

    Error case (Invariant I-11):
        If the PDF is unreadable, origin_type = OriginType.error
        and error_message is populated. All other fields are defaults.
        The pipeline must never silently skip an unreadable document.
    """
    doc_id:              str
    file_path:           str
    file_name:           str                    # basename only, for display
    origin_type:         OriginType
    layout_complexity:   LayoutComplexity
    domain_hint:         DomainHint
    extraction_strategy: ExtractionStrategy
    estimated_cost_usd:  float
    page_count:          int
    per_page_signals:    list[PageSignal] = []
    error_message:       str | None = None      # populated only for error profiles
    is_ood:              bool = False           # True if out-of-distribution signals


# ================================================================
# ProvenanceRef — atomic citation unit
# ================================================================

class ProvenanceRef(BaseModel):
    """
    Every extracted fact must carry a ProvenanceRef.
    No exceptions — see Invariant I-1.

    Used by:
        - ChunkValidator to verify LDU provenance
        - QueryAgent to assemble ProvenanceChain
        - AuditMode to verify claims against source
    """
    document_name:    str
    doc_id:           str
    page_number:      int
    bbox:             tuple[float, float, float, float]  # (x0, y0, x1, y1)
    content_hash:     str       # SHA-256 of the extracted content
    strategy_used:    ExtractionStrategy
    confidence_score: float

    # ── Invariant I-9: bbox must be geometrically valid ──────
    @field_validator("bbox")
    @classmethod
    def bbox_must_be_valid(cls, v):
        x0, y0, x1, y1 = v
        assert x0 >= 0,  f"bbox x0 ({x0}) must be >= 0"
        assert y0 >= 0,  f"bbox y0 ({y0}) must be >= 0"
        assert x0 < x1,  f"bbox x0 ({x0}) must be < x1 ({x1})"
        assert y0 < y1,  f"bbox y0 ({y0}) must be < y1 ({y1})"
        return v

    # ── Invariant I-8: content_hash must be non-empty ────────
    @field_validator("content_hash")
    @classmethod
    def hash_must_be_present(cls, v):
        assert len(v) > 0, "content_hash cannot be empty"
        return v

    @staticmethod
    def make_hash(content: str) -> str:
        """
        Always use this to generate content_hash.
        Deterministic: same content always produces same hash.
        """
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


class ProvenanceChainEntry(BaseModel):
    """Single ordered evidence item used to ground a query response."""

    record_id: str = Field(min_length=1)
    record_type: str = Field(min_length=1)
    section_path: tuple[str, ...] = ()
    snippet: str = Field(min_length=1)
    distance: float | None = None
    provenance: ProvenanceRef


class ProvenanceChain(BaseModel):
    """Ordered evidence chain backing a response or retrieval trace."""

    entries: tuple[ProvenanceChainEntry, ...]
    query: str | None = None

    @field_validator("entries")
    @classmethod
    def entries_must_be_present(cls, value: tuple[ProvenanceChainEntry, ...]) -> tuple[ProvenanceChainEntry, ...]:
        assert len(value) > 0, "ProvenanceChain must include at least one entry"
        return value


# ================================================================
# Factory helpers
# ================================================================

def make_error_profile(file_path: str, error_message: str) -> DocumentProfile:
    """
    Invariant I-11: unreadable PDF must emit ERROR profile — never None.

    Usage:
        try:
            profile = triage(path)
        except RuntimeError as e:
            profile = make_error_profile(path, str(e))
    """
    from pathlib import Path
    p = Path(file_path)
    return DocumentProfile(
        doc_id              = hashlib.md5(file_path.encode()).hexdigest()[:12],
        file_path           = file_path,
        file_name           = p.name,
        origin_type         = OriginType.error,
        layout_complexity   = LayoutComplexity.unknown,
        domain_hint         = DomainHint.general,
        extraction_strategy = ExtractionStrategy.none,
        estimated_cost_usd  = 0.0,
        page_count          = 0,
        per_page_signals    = [],
        error_message       = error_message,
    )
