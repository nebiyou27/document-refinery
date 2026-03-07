"""Normalized numeric fact-table models for finance/table-heavy documents."""

from __future__ import annotations

from pydantic import BaseModel, Field

from .document_profile import ProvenanceRef


class FactTableEntry(BaseModel):
    """Single numeric fact extracted from a table cell with provenance."""

    fact_id: str = Field(min_length=1)
    table_id: str = Field(min_length=1)
    doc_id: str = Field(min_length=1)
    document_name: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    section_path: tuple[str, ...] = ()
    row_label: str = Field(min_length=1)
    column_label: str = Field(min_length=1)
    raw_value: str = Field(min_length=1)
    numeric_value: float
    unit: str | None = None
    provenance: ProvenanceRef


class FactTable(BaseModel):
    """Collection of normalized numeric table facts for one document."""

    doc_id: str = Field(min_length=1)
    entries: tuple[FactTableEntry, ...]
