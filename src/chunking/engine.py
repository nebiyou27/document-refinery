"""Deterministic Stage 3 chunking scaffold."""

from __future__ import annotations

from pydantic import BaseModel, Field

from src.chunking.validator import ChunkValidator, ChunkingRules
from src.models.chunking import Chunk, LDU, LDUKind
from src.models.extracted_document import ExtractedDocument, FigureBlock, TableBlock, TextBlock


class ChunkingConfig(BaseModel):
    """Deterministic chunking parameters."""

    max_chunk_chars: int = Field(default=1200, ge=1)
    standalone_kinds: tuple[LDUKind, ...] = (LDUKind.table, LDUKind.figure)
    split_on_section_change: bool = True
    allow_multi_page_chunks: bool = False


class ChunkingEngine:
    """Scaffold that turns extracted blocks into validated LDUs and chunks."""

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        validator: ChunkValidator | None = None,
    ) -> None:
        self.config = config or ChunkingConfig()
        self.validator = validator or ChunkValidator(
            ChunkingRules(
                standalone_kinds=self.config.standalone_kinds,
                split_on_section_change=self.config.split_on_section_change,
                allow_multi_page_chunks=self.config.allow_multi_page_chunks,
                max_chunk_chars=self.config.max_chunk_chars,
            )
        )

    def build_ldus(self, document: ExtractedDocument) -> list[LDU]:
        ldus: list[LDU] = []
        for page in sorted(document.pages, key=lambda current: current.page_number):
            for text_block in sorted(page.text_blocks, key=lambda block: block.reading_order):
                ldu = self._ldu_from_text_block(document.doc_id, text_block)
                self.validator.raise_for_issues(self.validator.validate_ldu(ldu))
                ldus.append(ldu)

            for table_block in sorted(page.table_blocks, key=lambda block: block.table_index):
                ldu = self._ldu_from_table_block(document.doc_id, table_block)
                self.validator.raise_for_issues(self.validator.validate_ldu(ldu))
                ldus.append(ldu)

            for figure_index, figure_block in enumerate(page.figure_blocks):
                ldu = self._ldu_from_figure_block(document.doc_id, figure_block, figure_index)
                self.validator.raise_for_issues(self.validator.validate_ldu(ldu))
                ldus.append(ldu)

        return ldus

    def build_chunks(self, document: ExtractedDocument) -> list[Chunk]:
        ldus = self.build_ldus(document)
        chunks: list[Chunk] = []
        buffer: list[LDU] = []
        sequence_number = 0

        for ldu in ldus:
            if self._should_emit_buffer(buffer, next_ldu=ldu):
                chunks.append(self._make_chunk(document.doc_id, buffer, sequence_number))
                sequence_number += 1
                buffer = []

            if ldu.kind in self.config.standalone_kinds:
                if buffer:
                    chunks.append(self._make_chunk(document.doc_id, buffer, sequence_number))
                    sequence_number += 1
                    buffer = []
                chunks.append(self._make_chunk(document.doc_id, [ldu], sequence_number))
                sequence_number += 1
                continue

            buffer.append(ldu)

        if buffer:
            chunks.append(self._make_chunk(document.doc_id, buffer, sequence_number))

        return chunks

    def _should_emit_buffer(self, buffer: list[LDU], next_ldu: LDU) -> bool:
        if not buffer:
            return False

        current_page = buffer[0].page_number
        current_section = buffer[0].section_path
        current_length = len(self._join_text(buffer))

        if not self.config.allow_multi_page_chunks and next_ldu.page_number != current_page:
            return True
        if self.config.split_on_section_change and next_ldu.section_path != current_section:
            return True
        if current_length + len(next_ldu.text) > self.config.max_chunk_chars:
            return True
        return False

    def _make_chunk(self, doc_id: str, ldus: list[LDU], sequence_number: int) -> Chunk:
        text = self._join_text(ldus)
        bbox = self._merge_bbox(ldus)
        chunk = Chunk(
            doc_id=doc_id,
            page_number=ldus[0].page_number,
            bbox=bbox,
            section_path=ldus[0].section_path,
            ldu_ids=[ldu.ldu_id for ldu in ldus if ldu.ldu_id is not None],
            text=text,
            metadata={"kinds": [ldu.kind.value for ldu in ldus]},
            sequence_number=sequence_number,
        )
        self.validator.raise_for_issues(self.validator.validate_chunk(chunk, ldus))
        return chunk

    def _ldu_from_text_block(self, doc_id: str, block: TextBlock) -> LDU:
        return LDU(
            doc_id=doc_id,
            page_number=block.page_number,
            bbox=block.bbox,
            kind=LDUKind.text,
            text=block.text,
            metadata={"block_type": block.block_type.value},
            source_block_order=block.reading_order,
        )

    def _ldu_from_table_block(self, doc_id: str, block: TableBlock) -> LDU:
        header_row = block.rows[0] if block.rows else []
        table_text = "\n".join(" | ".join(cell for cell in row) for row in block.rows)
        return LDU(
            doc_id=doc_id,
            page_number=block.page_number,
            bbox=block.bbox,
            kind=LDUKind.table,
            text=table_text or "[table]",
            metadata={
                "block_type": block.block_type.value,
                "header_row": header_row,
                "row_count": len(block.rows),
            },
            source_block_order=10_000 + block.table_index,
        )

    def _ldu_from_figure_block(self, doc_id: str, block: FigureBlock, figure_index: int) -> LDU:
        caption = block.caption or ""
        return LDU(
            doc_id=doc_id,
            page_number=block.page_number,
            bbox=block.bbox,
            kind=LDUKind.figure,
            text=caption or "[figure]",
            metadata={
                "block_type": block.block_type.value,
                "caption": caption,
            },
            source_block_order=20_000 + figure_index,
        )

    def _join_text(self, ldus: list[LDU]) -> str:
        return "\n\n".join(ldu.text for ldu in ldus)

    def _merge_bbox(self, ldus: list[LDU]) -> tuple[float, float, float, float]:
        x0 = min(ldu.bbox[0] for ldu in ldus)
        y0 = min(ldu.bbox[1] for ldu in ldus)
        x1 = max(ldu.bbox[2] for ldu in ldus)
        y1 = max(ldu.bbox[3] for ldu in ldus)
        return (x0, y0, x1, y1)
