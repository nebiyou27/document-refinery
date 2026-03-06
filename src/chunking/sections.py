"""Deterministic section-path inference for Stage 3."""

from __future__ import annotations

from dataclasses import dataclass
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


class SectionPathInferer:
    """Infers deterministic hierarchical section paths from extracted blocks."""

    _NUMBERED_HEADING_RE = re.compile(
        r"^(?P<number>\d+(?:\.\d+)*)(?:[.)])?\s+(?P<title>[A-Za-z][^\n]{0,120})$"
    )

    def infer_paths(self, candidates: list[SectionCandidate]) -> dict[str, tuple[str, ...]]:
        paths_by_id: dict[str, tuple[str, ...]] = {}
        current_path: list[str] = []

        for index, candidate in enumerate(candidates):
            heading = self._classify_heading(candidate, candidates, index)
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
    ) -> HeadingMatch | None:
        if candidate.kind is not LDUKind.text:
            return None

        normalized_text = canonicalize_text(candidate.text)
        if not normalized_text:
            return None

        numbered = self._match_numbered_heading(normalized_text)
        if numbered is not None:
            return numbered

        if self._looks_like_styled_heading(normalized_text, candidate, candidates, index):
            return HeadingMatch(title=normalized_text, level=1)

        return None

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
        if bbox_height >= 14.0:
            return True

        if text.isupper():
            return True

        title_case_ratio = self._title_case_ratio(text)
        if title_case_ratio < 0.6:
            return False

        next_candidate = candidates[index + 1] if index + 1 < len(candidates) else None
        if next_candidate is None:
            return False

        vertical_gap = next_candidate.bbox[1] - candidate.bbox[3]
        return vertical_gap >= 4.0 or next_candidate.page_number != candidate.page_number

    def _title_case_ratio(self, text: str) -> float:
        alpha_words = [word for word in re.split(r"\s+", text) if any(char.isalpha() for char in word)]
        if not alpha_words:
            return 0.0

        title_cased = [word for word in alpha_words if word[:1].isupper()]
        return len(title_cased) / len(alpha_words)
