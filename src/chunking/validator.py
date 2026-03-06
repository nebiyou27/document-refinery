"""Explicit, testable validation rules for Stage 3 chunking."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.models.chunking import Chunk, LDU, LDUKind, ValidationIssue, ValidationSeverity


class ChunkValidationError(ValueError):
    """Raised when chunking invariants fail."""

    def __init__(self, issues: list[ValidationIssue]) -> None:
        self.issues = issues
        joined = "; ".join(f"{issue.rule_id}:{issue.code}:{issue.message}" for issue in issues)
        super().__init__(joined)


class ChunkingRules(BaseModel):
    """Rule switches used by the validator and engine."""

    require_provenance: bool = True
    require_table_header: bool = True
    require_figure_caption_metadata: bool = True
    standalone_kinds: tuple[LDUKind, ...] = (LDUKind.table, LDUKind.figure)
    split_on_section_change: bool = True
    allow_multi_page_chunks: bool = False
    max_chunk_chars: int = Field(default=1200, ge=1)


class ChunkValidator:
    """Enforces documented Stage 3 invariants before chunks are emitted."""

    def __init__(self, rules: ChunkingRules | None = None) -> None:
        self.rules = rules or ChunkingRules()

    def validate_ldu(self, ldu: LDU) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []

        if self.rules.require_provenance:
            if ldu.page_number < 1 or not ldu.content_hash:
                issues.append(
                    ValidationIssue(
                        rule_id="I-1",
                        code="provenance_missing",
                        message="LDU must include page_number and content_hash",
                        ldu_id=ldu.ldu_id,
                    )
                )

        if self.rules.require_table_header and ldu.kind == LDUKind.table:
            header_row = ldu.metadata.get("header_row")
            if not isinstance(header_row, list) or not header_row or not all(str(cell).strip() for cell in header_row):
                issues.append(
                    ValidationIssue(
                        rule_id="I-5",
                        code="table_header_missing",
                        message="Table LDUs must carry a non-empty header_row in metadata",
                        ldu_id=ldu.ldu_id,
                    )
                )

        if self.rules.require_figure_caption_metadata and ldu.kind == LDUKind.figure:
            caption = ldu.metadata.get("caption")
            if not isinstance(caption, str) or not caption.strip():
                issues.append(
                    ValidationIssue(
                        rule_id="I-6",
                        code="figure_caption_missing",
                        message="Figure LDUs must store caption text in metadata",
                        ldu_id=ldu.ldu_id,
                    )
                )

        return issues

    def validate_chunk(self, chunk: Chunk, ldus: list[LDU]) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if self.rules.require_provenance:
            if chunk.page_number < 1 or not chunk.content_hash:
                issues.append(
                    ValidationIssue(
                        rule_id="I-1",
                        code="chunk_provenance_missing",
                        message="Chunk must include page_number and content_hash",
                        chunk_id=chunk.chunk_id,
                    )
                )

        if not self.rules.allow_multi_page_chunks and any(ldu.page_number != chunk.page_number for ldu in ldus):
            issues.append(
                ValidationIssue(
                    rule_id="chunk_rule",
                    code="multi_page_chunk_disallowed",
                    message="Chunk contains LDUs from multiple pages",
                    chunk_id=chunk.chunk_id,
                    severity=ValidationSeverity.error,
                )
            )

        if len(chunk.text) > self.rules.max_chunk_chars:
            issues.append(
                ValidationIssue(
                    rule_id="chunk_rule",
                    code="chunk_too_large",
                    message=f"Chunk text length exceeds max_chunk_chars={self.rules.max_chunk_chars}",
                    chunk_id=chunk.chunk_id,
                    severity=ValidationSeverity.error,
                )
            )

        return issues

    def raise_for_issues(self, issues: list[ValidationIssue]) -> None:
        errors = [issue for issue in issues if issue.severity == ValidationSeverity.error]
        if errors:
            raise ChunkValidationError(errors)
