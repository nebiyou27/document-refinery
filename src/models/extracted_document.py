"""Normalized extraction output models for all strategies."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class BlockType(str, Enum):
    text = "text"
    table = "table"
    figure = "figure"


class ExtractionMetadata(BaseModel):
    strategy_used: str
    confidence_score: float = Field(ge=0.0, le=1.0)
    processing_time_sec: float = Field(ge=0.0)
    cost_estimate_usd: float = Field(ge=0.0)
    escalation_triggered: bool = False
    escalation_target: str | None = None


class TextBlock(BaseModel):
    doc_id: str
    page_number: int = Field(ge=1)
    block_type: BlockType = BlockType.text
    text: str
    bbox: tuple[float, float, float, float]
    reading_order: int = Field(ge=0)
    content_hash: str = Field(min_length=1)


class TableBlock(BaseModel):
    doc_id: str
    page_number: int = Field(ge=1)
    block_type: BlockType = BlockType.table
    bbox: tuple[float, float, float, float]
    content_hash: str = Field(min_length=1)
    table_index: int = Field(ge=0)
    rows: list[list[str]] = Field(default_factory=list)


class FigureBlock(BaseModel):
    doc_id: str
    page_number: int = Field(ge=1)
    block_type: BlockType = BlockType.figure
    bbox: tuple[float, float, float, float]
    content_hash: str = Field(min_length=1)
    caption: str | None = None


class ExtractedPage(BaseModel):
    doc_id: str
    page_number: int = Field(ge=1)
    status: Literal["ok", "error"] = "ok"
    metadata: ExtractionMetadata
    signals: dict[str, float | int]
    text_blocks: list[TextBlock] = Field(default_factory=list)
    table_blocks: list[TableBlock] = Field(default_factory=list)
    figure_blocks: list[FigureBlock] = Field(default_factory=list)
    page_content_hash: str = Field(min_length=1)
    error_message: str | None = None

    @model_validator(mode="after")
    def validate_error_state(self) -> "ExtractedPage":
        if self.status == "error" and not self.error_message:
            raise ValueError("error_message is required when status='error'")
        return self


class ExtractedDocument(BaseModel):
    doc_id: str
    file_name: str
    file_path: str
    page_count: int = Field(ge=0)
    status: Literal["ok", "error"] = "ok"
    metadata: ExtractionMetadata
    pages: list[ExtractedPage] = Field(default_factory=list)
    error_message: str | None = None
