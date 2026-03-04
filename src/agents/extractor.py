"""Stage 2 extraction router (Strategy A + Strategy B, Strategy C advisory)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.models.document_profile import DocumentProfile, ExtractionStrategy, OriginType
from src.models.extracted_document import ExtractedDocument, ExtractedPage, ExtractionMetadata
from src.strategies.strategy_a import extract_with_pdfplumber
from src.strategies.strategy_b import extract_pages_with_docling
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
        ("strategy_routing", "strategy_b", "confidence_gates", "min_confidence_score"),
        ("strategy_routing", "strategy_b", "escalation_target"),
        ("strategy_routing", "strategy_b", "escalation_scope"),
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


def _log_page_ledger(doc: ExtractedDocument, page: ExtractedPage, escalated_to: str | None) -> None:
    append_ledger_entry(
        doc_id=doc.doc_id,
        file_name=doc.file_name,
        page_number=page.page_number,
        strategy_used=page.metadata.strategy_used,
        confidence=page.metadata.confidence_score,
        signals=page.signals,
        cost_estimate=page.metadata.cost_estimate_usd,
        processing_time=page.metadata.processing_time_sec,
        escalated_to=escalated_to,
    )


def _update_document_metadata(doc: ExtractedDocument) -> dict[str, int]:
    page_escalation_targets: dict[str, int] = {}
    for page in doc.pages:
        if page.metadata.escalation_triggered and page.metadata.escalation_target:
            target = page.metadata.escalation_target
            page_escalation_targets[target] = page_escalation_targets.get(target, 0) + 1

    doc.metadata.escalation_triggered = any(p.metadata.escalation_triggered for p in doc.pages)
    if not page_escalation_targets:
        doc.metadata.escalation_target = None
    elif len(page_escalation_targets) == 1:
        doc.metadata.escalation_target = next(iter(page_escalation_targets))
    else:
        doc.metadata.escalation_target = "mixed"

    avg_conf = sum(p.metadata.confidence_score for p in doc.pages) / max(len(doc.pages), 1)
    doc.metadata.confidence_score = round(avg_conf, 3)
    return page_escalation_targets


def run_extraction(pdf_path: Path, strategy: str | None = None) -> ExtractedDocument:
    if not pdf_path.exists():
        return _error_document(pdf_path, f"File not found: {pdf_path}")

    rules = load_rules()
    profile = _load_or_compute_profile(pdf_path)
    if profile.origin_type == OriginType.error:
        return _error_document(pdf_path, profile.error_message or "Unreadable PDF")

    # Honor explicit override when debugging.
    forced = (strategy or "").strip().lower() or None

    # Skeleton routing decision; execute A by default and escalate pages to B.
    initial_strategy = profile.extraction_strategy
    if initial_strategy not in {
        ExtractionStrategy.strategy_a,
        ExtractionStrategy.strategy_b,
        ExtractionStrategy.strategy_c,
    }:
        return _error_document(pdf_path, f"Unsupported extraction strategy: {initial_strategy}")

    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    ledger_doc_id = _compute_doc_id(pdf_path)
    ledger_path = LEDGER_DIR / f"{ledger_doc_id}.jsonl"
    if ledger_path.exists():
        ledger_path.unlink()

    if forced == "strategy_b":
        all_pages = list(range(1, profile.page_count + 1))
        pages_b = extract_pages_with_docling(
            pdf_path=pdf_path,
            page_numbers=all_pages,
            rules=rules,
            batch_size=5,
        )
        extracted_b = extract_with_pdfplumber(pdf_path=pdf_path, rules=rules)
        for page_number in all_pages:
            extracted_b.pages[page_number - 1] = pages_b[page_number]
        b_min_conf = float(
            rules["strategy_routing"]["strategy_b"]["confidence_gates"]["min_confidence_score"]
        )
        for page in extracted_b.pages:
            if page.status == "error" or page.metadata.confidence_score < b_min_conf:
                page.metadata.escalation_triggered = True
                page.metadata.escalation_target = "strategy_c"
            else:
                page.metadata.escalation_triggered = False
                page.metadata.escalation_target = None
        for page in extracted_b.pages:
            _log_page_ledger(extracted_b, page, page.metadata.escalation_target)
        extracted_b.metadata.strategy_used = "strategy_b"
        page_escalation_targets = _update_document_metadata(extracted_b)
        executed_strategy = "strategy_b"
        pages_b_executed = len(extracted_b.pages)
        extracted = extracted_b
    else:
        extracted = extract_with_pdfplumber(pdf_path=pdf_path, rules=rules)
        a_min_conf = float(
            rules["strategy_routing"]["strategy_a"]["confidence_gates"]["min_confidence_score"]
        )
        b_min_conf = float(
            rules["strategy_routing"]["strategy_b"]["confidence_gates"]["min_confidence_score"]
        )

        pages_to_strategy_b: set[int] = set()
        for page in extracted.pages:
            should_escalate = page.status == "error" or page.metadata.confidence_score < a_min_conf
            escalated_to = None
            if should_escalate:
                target = choose_escalation_target_for_page(page.signals, rules)
                escalated_to = target
                page.metadata.escalation_triggered = True
                page.metadata.escalation_target = target
                if target == "strategy_b":
                    pages_to_strategy_b.add(page.page_number)
            _log_page_ledger(extracted, page, escalated_to)

        pages_b_executed = 0
        if pages_to_strategy_b:
            b_by_page = extract_pages_with_docling(
                pdf_path=pdf_path,
                page_numbers=sorted(pages_to_strategy_b),
                rules=rules,
                batch_size=5,
            )

            for page_no in sorted(pages_to_strategy_b):
                b_page = b_by_page.get(page_no)
                if b_page is None:
                    continue

                escalated_to = None
                if b_page.status == "error" or b_page.metadata.confidence_score < b_min_conf:
                    escalated_to = "strategy_c"
                    b_page.metadata.escalation_triggered = True
                    b_page.metadata.escalation_target = "strategy_c"
                else:
                    b_page.metadata.escalation_triggered = False
                    b_page.metadata.escalation_target = None

                extracted.pages[page_no - 1] = b_page
                _log_page_ledger(extracted, b_page, escalated_to)
                pages_b_executed += 1

        page_escalation_targets = _update_document_metadata(extracted)
        extracted.metadata.strategy_used = "strategy_a+strategy_b" if pages_b_executed else "strategy_a"
        executed_strategy = extracted.metadata.strategy_used

    pages_below_strategy_a = sum(
        1
        for p in extracted.pages
        if p.metadata.strategy_used == "strategy_a" and p.metadata.escalation_triggered
    )
    escalation_pct = (sum(page_escalation_targets.values()) / max(len(extracted.pages), 1)) * 100.0

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
                "executed_strategy": executed_strategy,
                "strategy_b_executed": pages_b_executed > 0,
                "strategy_b_pages_executed": pages_b_executed,
                "page_escalations_planned": {
                    "count": sum(page_escalation_targets.values()),
                    "total_pages": len(extracted.pages),
                    "percent": round(escalation_pct, 2),
                    "target": extracted.metadata.escalation_target,
                    "targets": page_escalation_targets,
                    "pages_below_strategy_a_confidence": pages_below_strategy_a,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return extracted
