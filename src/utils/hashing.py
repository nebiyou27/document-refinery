"""Deterministic hashing helpers for extraction provenance."""

from __future__ import annotations

import hashlib


def content_hash(text: str) -> str:
    """Return a stable SHA-256 hash for extracted content."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

