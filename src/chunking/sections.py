"""Deterministic section-path inference for Stage 3."""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from enum import Enum

from src.models.chunking import LDUKind
from src.utils.hashing import canonicalize_text

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SectionCandidate:
    """Ordered extracted unit used for section-path inference."""

    candidate_id: str
    kind: LDUKind
    page_number: int
    source_block_order: int
    text: str
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class HeadingMatch:
    """Normalized heading classification result."""

    title: str
    level: int


class SectionInferenceMode(str, Enum):
    strict = "strict"
    relaxed = "relaxed"


class SectionPathInferer:
    """Infers deterministic hierarchical section paths from extracted blocks."""

    _NUMBERED_HEADING_RE = re.compile(
        r"^(?P<number>\d+(?:\.\d+)*)(?:[.):])?\s+(?P<title>[A-Za-z0-9][^\n]{0,120})$"
    )
    _LIST_ITEM_RE = re.compile(
        r"^(?:[-*•]\s*)?(?:(?:\d+(?:\.\d+)*|[A-Za-z])(?:[.)])?)\s+\S"
    )
    _PAGE_LABEL_RE = re.compile(r"^page\s+\d+\s*$", re.IGNORECASE)
    _EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b", re.IGNORECASE)
    _URL_RE = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
    _PHONE_RE = re.compile(r"(?:\+\d[\d\s().-]{6,}|\b\d{3,}[\d\s().-]{4,}\b)")
    _CONTACT_RE = re.compile(
        r"\b(?:telephone|tel|phone|mobile|fax|email|e-mail|postal address|p\.?o\.?\s*box|website|web site|enquiries)\b",
        re.IGNORECASE,
    )
    _LOW_VALUE_BOILERPLATE_RE = re.compile(
        r"(?:does\s+not\s+conform\s+to\s+the\s+provided\s+schema|"
        r"unable\s+to\s+provide\s+a\s+structured\s+json\s+output|"
        r"not\s+suitable\s+for\s+plain\s+text\s+output|"
        r"following\s+the\s+specified\s+format|"
        r"includes\s+several\s+visual\s+elements|"
        r"text\s+is\s+in\s+an\s+undefined\s+language)",
        re.IGNORECASE,
    )
    _ISSUE_LABEL_RE = re.compile(r"^issue\s+no\.?\s+\S+", re.IGNORECASE)
    _TABLE_TITLE_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9&/-]{2,}")
    _MIN_REPEAT_PAGES = 3
    _TABLE_CONTEXT_KEYWORDS = (
        "budget",
        "expense",
        "expenses",
        "allocation",
        "allocations",
        "amount",
        "amounts",
        "total",
        "totals",
        "category",
        "categories",
        "item",
        "items",
        "procurement",
        "audit",
        "finding",
        "findings",
        "target",
        "actual",
        "revenue",
        "cost",
        "payment",
        "payments",
        "vendor",
    )
    _TABLE_TITLE_KEYWORDS = (
        "budget",
        "expense",
        "expenses",
        "allocation",
        "allocations",
        "amount",
        "amounts",
        "total",
        "totals",
        "category",
        "categories",
        "item",
        "items",
        "procurement",
        "audit",
        "finding",
        "findings",
        "revenue",
        "cost",
        "vendor",
        "payment",
        "payments",
    )

    def __init__(self, mode: SectionInferenceMode = SectionInferenceMode.strict) -> None:
        self.mode = mode

    def infer_paths(self, candidates: list[SectionCandidate]) -> dict[str, tuple[str, ...]]:
        paths_by_id: dict[str, tuple[str, ...]] = {}
        current_path: list[str] = []
        repeated_pages = self._repeated_page_counts(candidates)

        for index, candidate in enumerate(candidates):
            heading = self._classify_heading(
                candidate=candidate,
                candidates=candidates,
                index=index,
                repeated_pages=repeated_pages,
            )
            if heading is not None:
                current_path = current_path[: max(heading.level - 1, 0)]
                current_path.append(heading.title)

            paths_by_id[candidate.candidate_id] = tuple(current_path)

        recovered_paths = self._recover_root_only_table_paths(
            candidates=candidates,
            paths_by_id=paths_by_id,
            repeated_pages=repeated_pages,
        )
        if recovered_paths is not None:
            return recovered_paths
        return paths_by_id

    def _recover_root_only_table_paths(
        self,
        *,
        candidates: list[SectionCandidate],
        paths_by_id: dict[str, tuple[str, ...]],
        repeated_pages: dict[str, int],
    ) -> dict[str, tuple[str, ...]] | None:
        if any(path for path in paths_by_id.values()):
            return None

        table_candidates = [candidate for candidate in candidates if candidate.kind is LDUKind.table]
        if not table_candidates:
            return None
        if self._has_unresolved_table_heading_candidate(candidates, repeated_pages):
            return None

        table_pages = {candidate.page_number for candidate in table_candidates}
        table_area = sum(self._bbox_area(candidate.bbox) for candidate in table_candidates)
        total_area = sum(self._bbox_area(candidate.bbox) for candidate in candidates)
        table_area_ratio = (table_area / total_area) if total_area > 0.0 else 0.0
        should_recover = (
            len(table_candidates) >= 2
            or len(table_pages) >= 2
            or table_area_ratio >= 0.6
        )
        if not should_recover:
            return None

        recovered_paths = dict(paths_by_id)
        page_table_counts: dict[int, int] = {}
        for candidate in candidates:
            if candidate.kind is not LDUKind.table:
                continue
            page_table_counts[candidate.page_number] = page_table_counts.get(candidate.page_number, 0) + 1
            recovered_paths[candidate.candidate_id] = (
                self._synthetic_table_section_title(
                    candidate=candidate,
                    ordinal=page_table_counts[candidate.page_number],
                    candidates=candidates,
                    repeated_pages=repeated_pages,
                ),
            )

        LOGGER.info(
            "Triggered synthetic table section recovery for root-only candidate set: "
            "table_candidates=%s table_pages=%s table_area_ratio=%.2f",
            len(table_candidates),
            sorted(table_pages),
            table_area_ratio,
        )
        return recovered_paths

    def _has_unresolved_table_heading_candidate(
        self,
        candidates: list[SectionCandidate],
        repeated_pages: dict[str, int],
    ) -> bool:
        for index, candidate in enumerate(candidates):
            if candidate.kind is not LDUKind.text:
                continue
            normalized_text = canonicalize_text(candidate.text)
            if not normalized_text:
                continue
            if self._should_suppress_styled_heading(
                text=normalized_text,
                candidate=candidate,
                repeated_pages=repeated_pages,
            ):
                continue
            if len(normalized_text) > 80 or len(normalized_text.split()) > 10:
                continue
            if normalized_text.endswith((".", "?", "!", ";", ":")):
                continue
            next_candidate = candidates[index + 1] if index + 1 < len(candidates) else None
            if next_candidate is None or next_candidate.page_number != candidate.page_number:
                continue
            if next_candidate.kind is not LDUKind.table:
                continue

            bbox_height = candidate.bbox[3] - candidate.bbox[1]
            visually_prominent = bbox_height >= 14.0 or normalized_text.isupper()
            if visually_prominent or self._title_case_ratio(normalized_text) >= 0.6:
                return True
        return False

    def _synthetic_table_section_title(
        self,
        *,
        candidate: SectionCandidate,
        ordinal: int,
        candidates: list[SectionCandidate],
        repeated_pages: dict[str, int],
    ) -> str:
        contextual_title = self._nearby_table_context_title(
            candidate=candidate,
            candidates=candidates,
            repeated_pages=repeated_pages,
        )
        header_title = self._table_header_title(candidate.text)
        keyword_title = self._table_keyword_title(candidate.text)
        base = f"Page {candidate.page_number} Table {ordinal}"
        descriptor = next(
            (
                option
                for option in (contextual_title, header_title, keyword_title)
                if self._is_high_quality_table_title_descriptor(option)
            ),
            "",
        )
        if not descriptor:
            return base
        return f"{base}: {descriptor}"

    def _nearby_table_context_title(
        self,
        *,
        candidate: SectionCandidate,
        candidates: list[SectionCandidate],
        repeated_pages: dict[str, int],
    ) -> str:
        best_text = ""
        best_gap: float | None = None
        for other in candidates:
            if other.kind is not LDUKind.text:
                continue
            if other.page_number != candidate.page_number:
                continue
            if other.bbox[3] > candidate.bbox[1]:
                continue
            normalized = canonicalize_text(other.text)
            if not normalized:
                continue
            if self._should_suppress_styled_heading(
                text=normalized,
                candidate=other,
                repeated_pages=repeated_pages,
            ):
                continue
            if len(normalized) > 80 or len(normalized.split()) > 10:
                continue
            gap = candidate.bbox[1] - other.bbox[3]
            if gap < 0.0 or gap > 120.0:
                continue
            bbox_height = other.bbox[3] - other.bbox[1]
            visually_prominent = bbox_height >= 14.0 or normalized.isupper()
            if not visually_prominent and self._title_case_ratio(normalized) < 0.6:
                continue
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_text = normalized
        return best_text

    def _table_header_title(self, text: str) -> str:
        first_line = next(
            (line.strip() for line in text.splitlines() if canonicalize_text(line)),
            "",
        )
        if not first_line:
            return ""
        labels = self._meaningful_table_labels(first_line.split("|"))
        if not labels:
            return ""
        return " / ".join(labels[:2])[:80]

    def _table_keyword_title(self, text: str) -> str:
        normalized = canonicalize_text(text).casefold()
        if not normalized:
            return ""
        matched = [
            keyword
            for keyword in self._TABLE_CONTEXT_KEYWORDS
            if re.search(rf"\b{re.escape(keyword)}\b", normalized)
        ]
        if not matched:
            return ""
        labels = [keyword.title() for keyword in matched[:2]]
        return " / ".join(labels)

    def _is_high_quality_table_title_descriptor(self, text: str) -> bool:
        normalized = canonicalize_text(text)
        if not normalized:
            return False
        if "..." in normalized:
            return False
        alpha_tokens = [
            token
            for token in self._TABLE_TITLE_TOKEN_RE.findall(normalized)
            if any(char.isalpha() for char in token)
        ]
        if len(alpha_tokens) >= 2:
            return True
        return any(
            re.search(rf"\b{re.escape(keyword)}\b", normalized.casefold())
            for keyword in self._TABLE_TITLE_KEYWORDS
        )

    def _meaningful_table_labels(self, cells: list[str]) -> list[str]:
        labels: list[str] = []
        seen: set[str] = set()
        for raw_cell in cells:
            normalized = canonicalize_text(raw_cell)
            if not normalized:
                continue
            tokens = self._TABLE_TITLE_TOKEN_RE.findall(normalized)
            filtered_tokens = [
                token
                for token in tokens
                if any(char.isalpha() for char in token)
                and sum(char.isalpha() for char in token) >= 3
            ]
            if not filtered_tokens:
                continue
            label = " ".join(filtered_tokens[:3]).strip()
            if not label:
                continue
            if not self._is_high_quality_table_title_descriptor(label):
                continue
            lowered = label.casefold()
            if lowered in seen:
                continue
            seen.add(lowered)
            labels.append(label)
        return labels

    def _bbox_area(self, bbox: tuple[float, float, float, float]) -> float:
        return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])

    def _classify_heading(
        self,
        candidate: SectionCandidate,
        candidates: list[SectionCandidate],
        index: int,
        repeated_pages: dict[str, int],
    ) -> HeadingMatch | None:
        if candidate.kind is not LDUKind.text:
            return None

        normalized_text = canonicalize_text(candidate.text)
        if not normalized_text:
            return None

        numbered = self._match_numbered_heading(normalized_text)
        if numbered is not None:
            return numbered

        if self._should_suppress_styled_heading(
            text=normalized_text,
            candidate=candidate,
            repeated_pages=repeated_pages,
        ):
            return None

        if self._looks_like_styled_heading(normalized_text, candidate, candidates, index):
            return HeadingMatch(title=normalized_text, level=1)

        return None

    def _repeated_page_counts(self, candidates: list[SectionCandidate]) -> dict[str, int]:
        pages_by_text: dict[str, set[int]] = {}
        for candidate in candidates:
            if candidate.kind is not LDUKind.text:
                continue
            normalized_text = canonicalize_text(candidate.text)
            if not normalized_text:
                continue
            pages_by_text.setdefault(normalized_text.casefold(), set()).add(candidate.page_number)
        return {
            text: len(page_numbers)
            for text, page_numbers in pages_by_text.items()
        }

    def _match_numbered_heading(self, text: str) -> HeadingMatch | None:
        match = self._NUMBERED_HEADING_RE.match(text)
        if match is None:
            return None

        number = match.group("number")
        title = canonicalize_text(f"{number} {match.group('title')}")
        level = len(number.split("."))
        return HeadingMatch(title=title, level=level)

    def _looks_like_styled_heading(
        self,
        text: str,
        candidate: SectionCandidate,
        candidates: list[SectionCandidate],
        index: int,
    ) -> bool:
        if len(text) > 80:
            return False
        if text.endswith((".", "?", "!", ";", ":")):
            return False
        if len(text.split()) > 10:
            return False

        bbox_height = candidate.bbox[3] - candidate.bbox[1]
        visually_prominent = bbox_height >= 14.0 or text.isupper()

        title_case_ratio = self._title_case_ratio(text)
        if not visually_prominent and title_case_ratio < 0.6:
            return False

        next_candidate = candidates[index + 1] if index + 1 < len(candidates) else None
        if next_candidate is None:
            return False
        if next_candidate.page_number != candidate.page_number:
            return False

        next_text = canonicalize_text(next_candidate.text)
        if (
            self.mode == SectionInferenceMode.relaxed
            and self._LIST_ITEM_RE.match(next_text)
            and any(char.isalpha() for char in text)
        ):
            return True

        vertical_gap = next_candidate.bbox[1] - candidate.bbox[3]
        if vertical_gap < 4.0:
            return False

        if self.mode == SectionInferenceMode.relaxed and next_candidate.kind in {
            LDUKind.table,
            LDUKind.figure,
        }:
            return True

        return self._looks_like_body_text(next_text)

    def _should_suppress_styled_heading(
        self,
        *,
        text: str,
        candidate: SectionCandidate,
        repeated_pages: dict[str, int],
    ) -> bool:
        lowered = text.casefold()
        repeated_on_pages = repeated_pages.get(lowered, 0)
        if repeated_on_pages >= self._MIN_REPEAT_PAGES:
            return True
        if self._PAGE_LABEL_RE.match(text):
            return True
        if self._ISSUE_LABEL_RE.match(text):
            return True
        if self._EMAIL_RE.search(text) or self._URL_RE.search(text):
            return True
        if self._LOW_VALUE_BOILERPLATE_RE.search(text):
            return True
        if self._CONTACT_RE.search(text):
            return True
        if self._PHONE_RE.search(text) and len(text.split()) <= 8:
            return True
        return False

    def _looks_like_body_text(self, text: str) -> bool:
        if not text:
            return False
        if self._PAGE_LABEL_RE.match(text):
            return False
        if self._EMAIL_RE.search(text) or self._URL_RE.search(text) or self._CONTACT_RE.search(text):
            return False
        if len(text) >= 40:
            return True
        if len(text.split()) >= 8:
            return True
        if text.endswith((".", "?", "!", ";")):
            return True
        if self.mode == SectionInferenceMode.relaxed and len(text) >= 20:
            return True
        if self.mode == SectionInferenceMode.relaxed and len(text.split()) >= 4:
            return True
        return False

    def _title_case_ratio(self, text: str) -> float:
        alpha_words = [word for word in re.split(r"\s+", text) if any(char.isalpha() for char in word)]
        if not alpha_words:
            return 0.0

        title_cased = [word for word in alpha_words if word[:1].isupper()]
        return len(title_cased) / len(alpha_words)
