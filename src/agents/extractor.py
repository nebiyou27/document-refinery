"""Stage 2 extraction router (Strategy A implemented, B/C stubbed)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.models.document_profile import DocumentProfile, ExtractionStrategy, OriginType
from src.models.extracted_document import ExtractedDocument, ExtractionMetadata
from src.strategies.strategy_a import extract_with_pdfplumber
from src.utils.hashing import content_hash
from src.utils.ledger import append_ledger_entry

RULES_PATH = Path("rubric/extraction_rules.yaml")
PROFILES_DIR = Path(".refinery/profiles")
EXTRACTED_DIR = Path(".refinery/extracted")
LEDGER_DIR = Path(".refinery/extraction_ledger")


def load_rules(rules_path: Path = RULES_PATH) -> dict:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required to load rubric/extraction_rules.yaml") from exc
    with rules_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Invalid extraction_rules.yaml: root must be a mapping")

    required_paths = [
        ("strategy_routing", "strategy_a", "confidence_gates", "min_char_density"),
        ("strategy_routing", "strategy_a", "confidence_gates", "max_image_area_ratio"),
        ("strategy_routing", "strategy_a", "confidence_gates", "min_confidence_score"),
        ("strategy_routing", "strategy_a", "escalation_target"),
    ]
    for path in required_paths:
        node = data
        traversed: list[str] = []
        for part in path:
            traversed.append(part)
            if not isinstance(node, dict) or part not in node:
                joined = ".".join(path)
                at = ".".join(traversed[:-1]) or "<root>"
                raise KeyError(f"Missing config key '{joined}' (stopped at '{at}')")
            node = node[part]
    return data


def choose_escalation_target_for_page(page_signals: dict, rules: dict) -> str:
    strategy_b_default = "strategy_b"
    routing = rules.get("strategy_routing")
    if not isinstance(routing, dict):
        return strategy_b_default

    strategy_a = routing.get("strategy_a")
    if not isinstance(strategy_a, dict):
        return strategy_b_default

    escalation = strategy_a.get("escalation")
    if not isinstance(escalation, dict):
        return strategy_b_default

    default_target = escalation.get("default_target", strategy_b_default)
    default_target_str = (
        str(default_target).strip() if str(default_target).strip() else strategy_b_default
    )

    to_c_if = escalation.get("to_c_if")
    if not isinstance(to_c_if, dict):
        return default_target_str

    image_area_ratio_gte = float(to_c_if.get("image_area_ratio_gte", 0.85))
    char_density_lte = float(to_c_if.get("char_density_lte", 0.001))
    char_count_lte = float(to_c_if.get("char_count_lte", 80))

    image_area_ratio = float(page_signals.get("image_area_ratio", 0.0))
    char_density = float(page_signals.get("char_density", 0.0))
    char_count = float(page_signals.get("char_count", 0.0))

    scanned_like = (
        image_area_ratio >= image_area_ratio_gte
        and char_density <= char_density_lte
        and char_count <= char_count_lte
    )
    if scanned_like:
        return "strategy_c"
    return default_target_str


def _compute_doc_id(pdf_path: Path) -> str:
    return hashlib.md5(pdf_path.read_bytes()).hexdigest()[:12]


def _load_or_compute_profile(pdf_path: Path) -> DocumentProfile:
    from src.agents.triage import triage

    doc_id = _compute_doc_id(pdf_path)
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    profile_path = PROFILES_DIR / f"{doc_id}.json"
    if profile_path.exists():
        return DocumentProfile.model_validate_json(profile_path.read_text(encoding="utf-8"))

    profile = triage(str(pdf_path))
    profile_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return profile


def _error_document(pdf_path: Path, message: str) -> ExtractedDocument:
    return ExtractedDocument(
        doc_id=_compute_doc_id(pdf_path) if pdf_path.exists() else content_hash(str(pdf_path))[:12],
        file_name=pdf_path.name,
        file_path=str(pdf_path),
        page_count=0,
        status="error",
        metadata=ExtractionMetadata(
            strategy_used="none",
            confidence_score=0.0,
            processing_time_sec=0.0,
            cost_estimate_usd=0.0,
            escalation_triggered=False,
            escalation_target=None,
        ),
        pages=[],
        error_message=message,
    )


def run_extraction(pdf_path: Path) -> ExtractedDocument:
    if not pdf_path.exists():
        return _error_document(pdf_path, f"File not found: {pdf_path}")

    rules = load_rules()
    profile = _load_or_compute_profile(pdf_path)
    if profile.origin_type == OriginType.error:
        return _error_document(pdf_path, profile.error_message or "Unreadable PDF")

    # Skeleton routing decision; execution stays on A for this phase.
    initial_strategy = profile.extraction_strategy
    if initial_strategy not in {
        ExtractionStrategy.strategy_a,
        ExtractionStrategy.strategy_b,
        ExtractionStrategy.strategy_c,
    }:
        return _error_document(pdf_path, f"Unsupported extraction strategy: {initial_strategy}")

    extracted = extract_with_pdfplumber(pdf_path=pdf_path, rules=rules)
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    ledger_path = LEDGER_DIR / f"{extracted.doc_id}.jsonl"
    if ledger_path.exists():
        ledger_path.unlink()

    a_min_conf = float(
        rules["strategy_routing"]["strategy_a"]["confidence_gates"]["min_confidence_score"]
    )
    pages_below_a = 0
    page_escalation_targets: dict[str, int] = {}
    for page in extracted.pages:
        escalated_to = None
        should_escalate = page.status == "error" or page.metadata.confidence_score < a_min_conf
        if should_escalate:
            pages_below_a += 1
            target = choose_escalation_target_for_page(page.signals, rules)
            escalated_to = target
            page.metadata.escalation_triggered = True
            page.metadata.escalation_target = target
            page_escalation_targets[target] = page_escalation_targets.get(target, 0) + 1

        append_ledger_entry(
            doc_id=extracted.doc_id,
            file_name=extracted.file_name,
            page_number=page.page_number,
            strategy_used=page.metadata.strategy_used,
            confidence=page.metadata.confidence_score,
            signals=page.signals,
            cost_estimate=page.metadata.cost_estimate_usd,
            processing_time=page.metadata.processing_time_sec,
            escalated_to=escalated_to,
        )

    extracted.metadata.escalation_triggered = pages_below_a > 0
    if pages_below_a == 0:
        extracted.metadata.escalation_target = None
    elif len(page_escalation_targets) == 1:
        extracted.metadata.escalation_target = next(iter(page_escalation_targets))
    else:
        extracted.metadata.escalation_target = "mixed"
    extracted.metadata.strategy_used = "strategy_a"
    escalation_pct = (pages_below_a / max(len(extracted.pages), 1)) * 100.0

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EXTRACTED_DIR / f"{extracted.doc_id}.json"
    output_path.write_text(extracted.model_dump_json(indent=2), encoding="utf-8")

    # Keep trace of advisory strategy from triage for future B/C plug-in logic.
    advisory_path = EXTRACTED_DIR / f"{extracted.doc_id}.routing.json"
    advisory_path.write_text(
        json.dumps(
            {
                "doc_id": extracted.doc_id,
                "document_level_strategy_suggestion": initial_strategy.value,
                "executed_strategy": "strategy_a",
                "page_escalations_planned": {
                    "count": pages_below_a,
                    "total_pages": len(extracted.pages),
                    "percent": round(escalation_pct, 2),
                    "target": extracted.metadata.escalation_target,
                    "targets": page_escalation_targets,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return extracted
