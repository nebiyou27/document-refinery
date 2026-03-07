"""Deterministic Stage 3 chunking scaffold."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from src.chunking.sections import SectionCandidate, SectionPathInferer
from src.chunking.validator import ChunkValidator, ChunkingRules
from src.models.chunking import Chunk, LDU, LDUKind
from src.models.extracted_document import ExtractedDocument, FigureBlock, TableBlock, TextBlock
from src.utils.hashing import canonicalize_text


@dataclass(frozen=True)
class OrderedUnit:
    """Internal ordered extracted unit enriched with a section candidate."""

    candidate: SectionCandidate
    block: TextBlock | TableBlock | FigureBlock
    kind: LDUKind
    source_block_order: int


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
        section_inferer: SectionPathInferer | None = None,
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
        self.section_inferer = section_inferer or SectionPathInferer()

    def build_ldus(self, document: ExtractedDocument) -> list[LDU]:
        ordered_units = self._collect_ordered_units(document)
        section_paths = self.section_inferer.infer_paths(
            [unit.candidate for unit in ordered_units]
        )
        ldus: list[LDU] = []
        for unit in ordered_units:
            candidate = unit.candidate
            section_path = section_paths.get(candidate.candidate_id, ())
            ldu = self._ldu_from_candidate(
                doc_id=document.doc_id,
                unit=unit,
                section_path=section_path,
            )
            self.validator.raise_for_issues(self.validator.validate_ldu(ldu))
            ldus.append(ldu)

        return ldus

    def build_chunks(self, document: ExtractedDocument, ldus: list[LDU] | None = None) -> list[Chunk]:
        if ldus is None:
            ldus = self.build_ldus(document)
        chunks: list[Chunk] = []
        buffer: list[LDU] = []
        sequence_number = 0

        for ldu in ldus:
            if self._ldu_exceeds_chunk_limit(ldu):
                if buffer:
                    chunks.append(self._make_chunk(document.doc_id, buffer, sequence_number))
                    sequence_number += 1
                    buffer = []
                oversized_chunks = self._split_oversized_ldu(document.doc_id, ldu, sequence_number)
                chunks.extend(oversized_chunks)
                sequence_number += len(oversized_chunks)
                continue

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

    def _ldu_exceeds_chunk_limit(self, ldu: LDU) -> bool:
        return len(ldu.text) > self.config.max_chunk_chars

    def _should_emit_buffer(self, buffer: list[LDU], next_ldu: LDU) -> bool:
        if not buffer:
            return False

        current_page = buffer[0].page_number
        current_section = buffer[0].section_path
        current_length = len(self._join_text(buffer))
        separator_length = 2 if buffer else 0

        if not self.config.allow_multi_page_chunks and next_ldu.page_number != current_page:
            return True
        if self.config.split_on_section_change and next_ldu.section_path != current_section:
            return True
        if current_length + separator_length + len(next_ldu.text) > self.config.max_chunk_chars:
            return True
        return False

    def _make_chunk(self, doc_id: str, ldus: list[LDU], sequence_number: int) -> Chunk:
        text = self._join_text(ldus)
        bbox = self._merge_bbox(ldus)
        metadata = {"kinds": [ldu.kind.value for ldu in ldus]}
        chunk = Chunk(
            doc_id=doc_id,
            page_number=ldus[0].page_number,
            bbox=bbox,
            section_path=ldus[0].section_path,
            ldu_ids=[ldu.ldu_id for ldu in ldus if ldu.ldu_id is not None],
            text=text,
            metadata=metadata,
            sequence_number=sequence_number,
        )
        self.validator.raise_for_issues(self.validator.validate_chunk(chunk, ldus))
        return chunk

    def _split_oversized_ldu(self, doc_id: str, ldu: LDU, sequence_number: int) -> list[Chunk]:
        pieces = self._split_text_to_max_chars(ldu.text)
        chunks: list[Chunk] = []
        for offset, piece in enumerate(pieces):
            metadata = {
                "kinds": [ldu.kind.value],
                "split_from_oversized_ldu": True,
                "split_part_index": offset,
                "split_part_count": len(pieces),
            }
            chunk = Chunk(
                doc_id=doc_id,
                page_number=ldu.page_number,
                bbox=ldu.bbox,
                section_path=ldu.section_path,
                ldu_ids=[ldu.ldu_id] if ldu.ldu_id is not None else [],
                text=piece,
                metadata=metadata,
                sequence_number=sequence_number + offset,
            )
            self.validator.raise_for_issues(self.validator.validate_chunk(chunk, [ldu]))
            chunks.append(chunk)
        return chunks

    def _split_text_to_max_chars(self, text: str) -> list[str]:
        max_chars = self.config.max_chunk_chars
        if len(text) <= max_chars:
            return [text]

        lines = text.splitlines()
        if not lines:
            return [text[index : index + max_chars] for index in range(0, len(text), max_chars)]

        pieces: list[str] = []
        current: list[str] = []

        for line in lines:
            line_segments = self._split_long_line(line, max_chars)
            for segment in line_segments:
                candidate_lines = current + [segment]
                candidate_text = "\n".join(candidate_lines)
                if current and len(candidate_text) > max_chars:
                    pieces.append("\n".join(current))
                    current = [segment]
                else:
                    current = candidate_lines

        if current:
            pieces.append("\n".join(current))

        return [piece for piece in pieces if piece]

    def _split_long_line(self, line: str, max_chars: int) -> list[str]:
        if len(line) <= max_chars:
            return [line]

        segments: list[str] = []
        remaining = line
        while len(remaining) > max_chars:
            split_at = remaining.rfind(" ", 0, max_chars + 1)
            if split_at <= 0:
                split_at = max_chars
            segment = remaining[:split_at].rstrip()
            if not segment:
                segment = remaining[:max_chars]
                split_at = max_chars
            segments.append(segment)
            remaining = remaining[split_at:].lstrip()
        if remaining:
            segments.append(remaining)
        return segments

    def _collect_ordered_units(self, document: ExtractedDocument) -> list[OrderedUnit]:
        ordered_units: list[OrderedUnit] = []
        for page in sorted(document.pages, key=lambda current: current.page_number):
            for text_block in sorted(page.text_blocks, key=lambda block: block.reading_order):
                ordered_units.append(
                    OrderedUnit(
                        candidate=SectionCandidate(
                            candidate_id=f"text:{page.page_number}:{text_block.reading_order}",
                            kind=LDUKind.text,
                            page_number=text_block.page_number,
                            source_block_order=text_block.reading_order,
                            text=text_block.text,
                            bbox=text_block.bbox,
                        ),
                        block=text_block,
                        kind=LDUKind.text,
                        source_block_order=text_block.reading_order,
                    )
                )

            for table_block in sorted(page.table_blocks, key=lambda block: block.table_index):
                table_text = "\n".join(" | ".join(cell for cell in row) for row in table_block.rows) or "[table]"
                if not canonicalize_text(table_text):
                    continue
                if not self._table_header_row(table_block):
                    continue
                ordered_units.append(
                    OrderedUnit(
                        candidate=SectionCandidate(
                            candidate_id=f"table:{page.page_number}:{table_block.table_index}",
                            kind=LDUKind.table,
                            page_number=table_block.page_number,
                            source_block_order=10_000 + table_block.table_index,
                            text=table_text,
                            bbox=table_block.bbox,
                        ),
                        block=table_block,
                        kind=LDUKind.table,
                        source_block_order=10_000 + table_block.table_index,
                    )
                )

            for figure_index, figure_block in enumerate(page.figure_blocks):
                if not (figure_block.caption or "").strip():
                    continue
                ordered_units.append(
                    OrderedUnit(
                        candidate=SectionCandidate(
                            candidate_id=f"figure:{page.page_number}:{figure_index}",
                            kind=LDUKind.figure,
                            page_number=figure_block.page_number,
                            source_block_order=20_000 + figure_index,
                            text=figure_block.caption or "[figure]",
                            bbox=figure_block.bbox,
                        ),
                        block=figure_block,
                        kind=LDUKind.figure,
                        source_block_order=20_000 + figure_index,
                    )
                )

        ordered_units.sort(
            key=lambda unit: (unit.candidate.page_number, unit.source_block_order)
        )
        return ordered_units

    def _ldu_from_candidate(
        self,
        doc_id: str,
        unit: OrderedUnit,
        section_path: tuple[str, ...],
    ) -> LDU:
        kind = unit.kind
        if kind == LDUKind.text:
            return self._ldu_from_text_block(
                doc_id=doc_id,
                block=unit.block,
                section_path=section_path,
            )
        if kind == LDUKind.table:
            return self._ldu_from_table_block(
                doc_id=doc_id,
                block=unit.block,
                section_path=section_path,
            )
        if kind == LDUKind.figure:
            return self._ldu_from_figure_block(
                doc_id=doc_id,
                block=unit.block,
                section_path=section_path,
                source_block_order=unit.source_block_order,
            )
        raise ValueError(f"Unsupported candidate kind: {kind}")

    def _ldu_from_text_block(
        self,
        doc_id: str,
        block: TextBlock | TableBlock | FigureBlock,
        section_path: tuple[str, ...],
    ) -> LDU:
        if not isinstance(block, TextBlock):
            raise TypeError("Expected TextBlock")
        return LDU(
            doc_id=doc_id,
            page_number=block.page_number,
            bbox=block.bbox,
            kind=LDUKind.text,
            text=block.text,
            section_path=section_path,
            metadata={"block_type": block.block_type.value},
            source_block_order=block.reading_order,
        )

    def _ldu_from_table_block(
        self,
        doc_id: str,
        block: TextBlock | TableBlock | FigureBlock,
        section_path: tuple[str, ...],
    ) -> LDU:
        if not isinstance(block, TableBlock):
            raise TypeError("Expected TableBlock")
        header_row = self._table_header_row(block)
        table_text = "\n".join(" | ".join(cell for cell in row) for row in block.rows)
        normalized_table_text = canonicalize_text(table_text)
        return LDU(
            doc_id=doc_id,
            page_number=block.page_number,
            bbox=block.bbox,
            kind=LDUKind.table,
            text=table_text if normalized_table_text else "[table]",
            section_path=section_path,
            metadata={
                "block_type": block.block_type.value,
                "header_row": header_row,
                "row_count": len(block.rows),
            },
            source_block_order=10_000 + block.table_index,
        )

    def _table_header_row(self, block: TableBlock) -> list[str]:
        if not block.rows:
            return []
        return [str(cell).strip() for cell in block.rows[0] if str(cell).strip()]

    def _ldu_from_figure_block(
        self,
        doc_id: str,
        block: TextBlock | TableBlock | FigureBlock,
        section_path: tuple[str, ...],
        source_block_order: int,
    ) -> LDU:
        if not isinstance(block, FigureBlock):
            raise TypeError("Expected FigureBlock")
        caption = block.caption or ""
        return LDU(
            doc_id=doc_id,
            page_number=block.page_number,
            bbox=block.bbox,
            kind=LDUKind.figure,
            text=caption or "[figure]",
            section_path=section_path,
            metadata={
                "block_type": block.block_type.value,
                "caption": caption,
            },
            source_block_order=source_block_order,
        )

    def _join_text(self, ldus: list[LDU]) -> str:
        return "\n\n".join(ldu.text for ldu in ldus)

    def _merge_bbox(self, ldus: list[LDU]) -> tuple[float, float, float, float]:
        x0 = min(ldu.bbox[0] for ldu in ldus)
        y0 = min(ldu.bbox[1] for ldu in ldus)
        x1 = max(ldu.bbox[2] for ldu in ldus)
        y1 = max(ldu.bbox[3] for ldu in ldus)
        return (x0, y0, x1, y1)
