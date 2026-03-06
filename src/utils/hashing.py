"""Deterministic hashing helpers for extraction provenance."""

from __future__ import annotations

import hashlib
import re
from typing import Iterable


def content_hash(text: str) -> str:
    """Return a stable SHA-256 hash for extracted content."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def canonicalize_text(text: str) -> str:
    """Normalize whitespace so equivalent text yields the same hash."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def canonicalize_section_path(section_path: Iterable[str]) -> tuple[str, ...]:
    """Normalize section path segments for stable Stage 3 hashing."""
    normalized_segments: list[str] = []
    for segment in section_path:
        normalized = re.sub(r"\s+", " ", segment).strip().casefold()
        if normalized:
            normalized_segments.append(normalized)
    return tuple(normalized_segments)


def ldu_content_hash(text: str, section_path: Iterable[str]) -> str:
    """Hash canonicalized LDU text plus section path, excluding page number."""
    normalized_text = canonicalize_text(text)
    normalized_path = " > ".join(canonicalize_section_path(section_path))
    payload = "\n".join([normalized_path, normalized_text])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
