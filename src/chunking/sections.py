"""Deterministic section-path inference for Stage 3."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re

from src.models.chunking import LDUKind
from src.utils.hashing import canonicalize_text


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
        r"^(?P<number>\d+(?:\.\d+)*)(?:[.)])?\s+(?P<title>[A-Za-z][^\n]{0,120})$"
    )
    _PAGE_LABEL_RE = re.compile(r"^page\s+\d+\s*$", re.IGNORECASE)
    _EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w+\b", re.IGNORECASE)
    _URL_RE = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
    _PHONE_RE = re.compile(r"(?:\+\d[\d\s().-]{6,}|\b\d{3,}[\d\s().-]{4,}\b)")
    _CONTACT_RE = re.compile(
        r"\b(?:telephone|tel|phone|mobile|fax|email|e-mail|postal address|p\.?o\.?\s*box|website|web site|enquiries)\b",
        re.IGNORECASE,
    )
    _ISSUE_LABEL_RE = re.compile(r"^issue\s+no\.?\s+\S+", re.IGNORECASE)
    _MIN_REPEAT_PAGES = 3

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

        return paths_by_id

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

        vertical_gap = next_candidate.bbox[1] - candidate.bbox[3]
        if vertical_gap < 4.0:
            return False

        if self.mode == SectionInferenceMode.relaxed and next_candidate.kind in {
            LDUKind.table,
            LDUKind.figure,
        }:
            return True

        next_text = canonicalize_text(next_candidate.text)
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
