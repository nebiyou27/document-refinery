"""Model package exports."""

from .document_profile import (
    DocumentProfile,
    DomainHint,
    ExtractionStrategy,
    LayoutComplexity,
    OriginType,
    PageSignal,
    ProvenanceRef,
    make_error_profile,
)
from .chunking import Chunk, LDU, LDUKind, PageIndexNode, ValidationIssue, ValidationSeverity
from .extracted_document import (
    ExtractedDocument,
    ExtractedPage,
    ExtractionMetadata,
    FigureBlock,
    TableBlock,
    TextBlock,
)

__all__ = [
    "DocumentProfile",
    "DomainHint",
    "ExtractionStrategy",
    "LayoutComplexity",
    "OriginType",
    "PageSignal",
    "ProvenanceRef",
    "make_error_profile",
    "Chunk",
    "LDU",
    "LDUKind",
    "PageIndexNode",
    "ValidationIssue",
    "ValidationSeverity",
    "ExtractedDocument",
    "ExtractedPage",
    "ExtractionMetadata",
    "FigureBlock",
    "TableBlock",
    "TextBlock",
]
