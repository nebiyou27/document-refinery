"""Stage 3 chunking and page-index data models."""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from src.utils.hashing import canonicalize_text, ldu_content_hash


_BBOX_ZERO_TOLERANCE = 1e-4


def normalize_bbox(value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Clamp tiny negative float noise to zero without accepting real negatives."""

    normalized = tuple(0.0 if -_BBOX_ZERO_TOLERANCE < coordinate < 0.0 else coordinate for coordinate in value)
    x0, y0, x1, y1 = normalized
    if x0 < 0 or y0 < 0:
        raise ValueError("bbox coordinates must be non-negative")
    if x0 >= x1 or y0 >= y1:
        raise ValueError("bbox must satisfy x0 < x1 and y0 < y1")
    return normalized


class LDUKind(str, Enum):
    """Logical document unit kinds emitted by the chunking engine."""

    text = "text"
    table = "table"
    figure = "figure"


class ValidationSeverity(str, Enum):
    """Severity levels for chunk validation results."""

    warning = "warning"
    error = "error"


class ValidationIssue(BaseModel):
    """Structured validation issue emitted by the chunk validator."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    severity: ValidationSeverity = ValidationSeverity.error
    rule_id: str = Field(min_length=1)
    ldu_id: str | None = None
    chunk_id: str | None = None


class LDU(BaseModel):
    """Atomic semantic unit with provenance and deterministic content hashing."""

    ldu_id: str | None = None
    doc_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    bbox: tuple[float, float, float, float]
    kind: LDUKind
    text: str
    section_path: tuple[str, ...] = Field(default_factory=tuple)
    content_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    source_block_order: int = Field(default=0, ge=0)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        return normalize_bbox(value)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        normalized = canonicalize_text(value)
        if not normalized:
            raise ValueError("text cannot be empty after canonicalization")
        return value

    @model_validator(mode="after")
    def populate_stable_fields(self) -> "LDU":
        expected_hash = ldu_content_hash(text=self.text, section_path=self.section_path)
        if self.content_hash is None:
            self.content_hash = expected_hash
        elif self.content_hash != expected_hash:
            raise ValueError("content_hash does not match canonicalized text and section_path")

        if self.ldu_id is None:
            identity = "|".join(
                [
                    self.doc_id,
                    str(self.page_number),
                    str(self.source_block_order),
                    self.kind.value,
                    ",".join(f"{value:.4f}" for value in self.bbox),
                    self.content_hash,
                ]
            )
            self.ldu_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return self


class Chunk(BaseModel):
    """Validated grouping of one or more LDUs for indexing and retrieval."""

    chunk_id: str | None = None
    doc_id: str = Field(min_length=1)
    page_number: int = Field(ge=1)
    bbox: tuple[float, float, float, float]
    section_path: tuple[str, ...] = Field(default_factory=tuple)
    ldu_ids: list[str] = Field(default_factory=list)
    text: str
    content_hash: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    sequence_number: int = Field(default=0, ge=0)

    @field_validator("bbox")
    @classmethod
    def validate_bbox(cls, value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        return normalize_bbox(value)

    @model_validator(mode="after")
    def populate_hashes(self) -> "Chunk":
        if not self.ldu_ids:
            raise ValueError("ldu_ids cannot be empty")

        expected_hash = ldu_content_hash(text=self.text, section_path=self.section_path)
        if self.content_hash is None:
            self.content_hash = expected_hash
        elif self.content_hash != expected_hash:
            raise ValueError("content_hash does not match canonicalized text and section_path")

        if self.chunk_id is None:
            identity = "|".join(
                [
                    self.doc_id,
                    str(self.page_number),
                    str(self.sequence_number),
                    self.content_hash,
                    ",".join(self.ldu_ids),
                ]
            )
            self.chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return self


class PageIndexNode(BaseModel):
    """Navigation tree node for Stage 4 page-index construction."""

    node_id: str
    title: str = Field(min_length=1)
    section_path: tuple[str, ...] = Field(default_factory=tuple)
    parent_id: str | None = None
    depth: int = Field(ge=0)
    start_page: int = Field(ge=1)
    end_page: int = Field(ge=1)
    bbox: tuple[float, float, float, float] | None = None
    child_ids: list[str] = Field(default_factory=list)
    ldu_ids: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)
    order_index: int = Field(default=0, ge=0)
    summary: str | None = None
