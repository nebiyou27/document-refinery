"""Deterministic FactTable extraction for numeric table-heavy documents."""

from __future__ import annotations

import hashlib
import re

from src.chunking.engine import ChunkingEngine
from src.models import ExtractedDocument, ExtractionStrategy, FactTable, FactTableEntry, LDUKind, ProvenanceRef
from src.models.extracted_document import TableBlock
from src.utils.hashing import canonicalize_text


class FactTableExtractor:
    """Builds normalized numeric facts from extracted table blocks."""

    def __init__(self, *, chunking_engine: ChunkingEngine | None = None) -> None:
        self.chunking_engine = chunking_engine or ChunkingEngine()

    def extract(self, document: ExtractedDocument) -> FactTable:
        table_ldus = {
            (ldu.page_number, ldu.source_block_order - 10_000): ldu
            for ldu in self.chunking_engine.build_ldus(document)
            if ldu.kind is LDUKind.table and ldu.source_block_order >= 10_000
        }
        entries: list[FactTableEntry] = []

        for page in document.pages:
            for table_block in page.table_blocks:
                entries.extend(
                    self._entries_for_table_block(
                        document=document,
                        table_block=table_block,
                        table_ldu=table_ldus.get((table_block.page_number, table_block.table_index)),
                    )
                )

        return FactTable(doc_id=document.doc_id, entries=tuple(entries))

    def _entries_for_table_block(
        self,
        *,
        document: ExtractedDocument,
        table_block: TableBlock,
        table_ldu,
    ) -> list[FactTableEntry]:
        rows = table_block.rows
        if len(rows) < 2:
            return []

        header = [canonicalize_text(str(cell)) for cell in rows[0]]
        if len(header) < 2:
            return []

        section_path = table_ldu.section_path if table_ldu is not None else ()
        document_name = document.file_name
        strategy_raw = (
            table_ldu.metadata.get("strategy_used")
            if table_ldu is not None
            else table_block.block_type.value
        )
        confidence_score = (
            float(table_ldu.metadata.get("confidence_score", document.metadata.confidence_score))
            if table_ldu is not None
            else float(document.metadata.confidence_score)
        )
        strategy_used = self._coerce_strategy(strategy_raw, fallback=document.metadata.strategy_used)
        content_hash = table_ldu.content_hash if table_ldu is not None else table_block.content_hash
        table_id = self._table_id(document.doc_id, table_block)
        entries: list[FactTableEntry] = []

        for row_index, row in enumerate(rows[1:], start=1):
            normalized_row = [canonicalize_text(str(cell)) for cell in row]
            if not normalized_row:
                continue
            row_label = normalized_row[0] if normalized_row and normalized_row[0] else f"row_{row_index}"
            for column_index, raw_value in enumerate(normalized_row[1:], start=1):
                if column_index >= len(header):
                    continue
                numeric_value = self._parse_numeric_value(raw_value)
                if numeric_value is None:
                    continue
                column_label = header[column_index]
                unit = self._infer_unit(column_label=column_label, row_label=row_label, raw_value=raw_value)
                provenance = ProvenanceRef(
                    document_name=document_name,
                    doc_id=document.doc_id,
                    page_number=table_block.page_number,
                    bbox=table_block.bbox,
                    content_hash=content_hash,
                    strategy_used=strategy_used,
                    confidence_score=confidence_score,
                )
                fact_id = self._fact_id(
                    doc_id=document.doc_id,
                    table_id=table_id,
                    row_label=row_label,
                    column_label=column_label,
                    raw_value=raw_value,
                )
                entries.append(
                    FactTableEntry(
                        fact_id=fact_id,
                        table_id=table_id,
                        doc_id=document.doc_id,
                        document_name=document_name,
                        page_number=table_block.page_number,
                        section_path=section_path,
                        row_label=row_label,
                        column_label=column_label,
                        raw_value=raw_value,
                        numeric_value=numeric_value,
                        unit=unit,
                        provenance=provenance,
                    )
                )
        return entries

    def _parse_numeric_value(self, raw_value: str) -> float | None:
        value = raw_value.strip()
        if not value:
            return None
        negative = value.startswith("(") and value.endswith(")")
        cleaned = value.strip("()").replace(",", "").replace("$", "").replace("€", "").replace("£", "")
        cleaned = re.sub(r"\s+", "", cleaned)
        is_percent = cleaned.endswith("%")
        if is_percent:
            cleaned = cleaned[:-1]
        if not re.fullmatch(r"[-+]?\d*\.?\d+", cleaned):
            return None
        number = float(cleaned)
        if negative:
            number = -number
        return number

    def _infer_unit(self, *, column_label: str, row_label: str, raw_value: str) -> str | None:
        lowered_context = " ".join([column_label.lower(), row_label.lower(), raw_value.lower()])
        if "%" in raw_value or "percent" in lowered_context or "percentage" in lowered_context:
            return "percent"
        if "$" in raw_value:
            return "usd"
        if "€" in raw_value:
            return "eur"
        if "£" in raw_value:
            return "gbp"
        return None

    def _table_id(self, doc_id: str, table_block: TableBlock) -> str:
        return hashlib.sha256(
            f"{doc_id}|{table_block.page_number}|{table_block.table_index}|{table_block.content_hash}".encode("utf-8")
        ).hexdigest()

    def _fact_id(self, *, doc_id: str, table_id: str, row_label: str, column_label: str, raw_value: str) -> str:
        payload = "|".join([doc_id, table_id, row_label, column_label, raw_value])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _coerce_strategy(self, strategy_raw: str, *, fallback: str) -> ExtractionStrategy:
        candidate = strategy_raw or fallback
        try:
            return ExtractionStrategy(candidate)
        except ValueError:
            return ExtractionStrategy(fallback)
