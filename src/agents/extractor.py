"""Stage 2 extraction router (Strategy A + Strategy B + Strategy C)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.models.document_profile import DocumentProfile, ExtractionStrategy, OriginType
from src.models.extracted_document import ExtractedDocument, ExtractedPage, ExtractionMetadata
from src.strategies.strategy_a import extract_with_pdfplumber
from src.strategies.strategy_b import extract_pages_with_docling
from src.strategies.strategy_c import extract_pages_with_vision
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
        ("strategy_routing", "strategy_a", "escalation", "default_target"),
        ("strategy_routing", "strategy_a", "escalation", "to_c_if", "image_area_ratio_gte"),
        ("strategy_routing", "strategy_a", "escalation", "to_c_if", "char_density_lte"),
        ("strategy_routing", "strategy_a", "escalation", "to_c_if", "char_count_lte"),
        ("strategy_routing", "strategy_b", "confidence_gates", "min_confidence_score"),
        ("strategy_routing", "strategy_b", "escalation_target"),
        ("strategy_routing", "strategy_b", "escalation_scope"),
        ("strategy_routing", "strategy_c", "tools", "primary_ocr"),
        ("strategy_routing", "strategy_c", "tools", "fallback_vlm", "provider"),
        ("strategy_routing", "strategy_c", "tools", "fallback_vlm", "model"),
        (
            "strategy_routing",
            "strategy_c",
            "execution_policy",
            "escalate_to_vlm_on_low_ocr_confidence",
        ),
        ("strategy_routing", "strategy_c", "execution_policy", "ocr_min_chars_per_page"),
        ("strategy_routing", "strategy_c", "execution_policy", "ocr_min_mean_confidence"),
        ("strategy_routing", "strategy_c", "budget_guard", "cost_per_page_estimate_usd"),
        ("strategy_routing", "strategy_c", "budget_guard", "max_pages_per_document"),
        ("strategy_routing", "strategy_c", "budget_guard", "max_vlm_pages_per_document"),
        ("strategy_routing", "strategy_c", "budget_guard", "max_total_runtime_seconds"),
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
    escalation = rules["strategy_routing"]["strategy_a"]["escalation"]
    default_target = str(escalation["default_target"]).strip()
    to_c_if = escalation["to_c_if"]
    image_area_ratio_gte = float(to_c_if["image_area_ratio_gte"])
    char_density_lte = float(to_c_if["char_density_lte"])
    char_count_lte = float(to_c_if["char_count_lte"])

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
    return default_target


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
            cost_model="local_compute",
            budget_enforced=True,
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


def _record_page_execution(
    *,
    doc: ExtractedDocument,
    page: ExtractedPage,
    escalated_to: str | None,
    executions_by_strategy: dict[str, int],
) -> None:
    _log_page_ledger(doc, page, escalated_to)
    strategy_name = page.metadata.strategy_used
    executions_by_strategy[strategy_name] = executions_by_strategy.get(strategy_name, 0) + 1


def _missing_page(
    *,
    doc_id: str,
    page_number: int,
    strategy_used: str,
    message: str,
) -> ExtractedPage:
    return ExtractedPage(
        doc_id=doc_id,
        page_number=page_number,
        status="error",
        text="",
        tables=[],
        metadata=ExtractionMetadata(
            strategy_used=strategy_used,
            confidence_score=0.0,
            processing_time_sec=0.0,
            cost_estimate_usd=0.0,
            escalation_triggered=False,
            escalation_target=None,
        ),
        signals={
            "char_count": 0,
            "char_density": 0.0,
            "image_area_ratio": 0.0,
            "table_count": 0,
        },
        text_blocks=[],
        table_blocks=[],
        figure_blocks=[],
        page_content_hash=content_hash(f"{strategy_used}:{page_number}:{message}"),
        error_message=message,
    )


def _assemble_document(
    *,
    pdf_path: Path,
    profile: DocumentProfile,
    pages_by_number: dict[int, ExtractedPage],
    default_strategy: str,
) -> ExtractedDocument:
    pages: list[ExtractedPage] = []
    for page_number in range(1, profile.page_count + 1):
        page = pages_by_number.get(page_number)
        if page is None:
            page = _missing_page(
                doc_id=profile.doc_id,
                page_number=page_number,
                strategy_used=default_strategy,
                message=f"{default_strategy}_missing_page_output",
            )
        pages.append(page)

    avg_conf = sum(p.metadata.confidence_score for p in pages) / max(len(pages), 1)
    processing_time_sec = sum(p.metadata.processing_time_sec for p in pages)
    cost_estimate_usd = sum(p.metadata.cost_estimate_usd for p in pages)
    vlm_pages_used = sum(1 for p in pages if p.metadata.vlm_used or int(p.signals.get("used_vlm", 0)) == 1)
    vlm_calls = sum(int(p.signals.get("vlm_calls", 1 if int(p.signals.get("used_vlm", 0)) == 1 else 0)) for p in pages)
    vlm_seconds_total = sum(float(p.metadata.vlm_wall_time_sec or p.signals.get("vlm_wall_time_sec", 0.0)) for p in pages)
    all_error = all(p.status == "error" for p in pages) if pages else True

    return ExtractedDocument(
        doc_id=profile.doc_id,
        file_name=pdf_path.name,
        file_path=str(pdf_path),
        page_count=profile.page_count,
        status="error" if all_error else "ok",
        metadata=ExtractionMetadata(
            strategy_used=default_strategy,
            confidence_score=round(avg_conf, 3),
            processing_time_sec=processing_time_sec,
            cost_estimate_usd=round(cost_estimate_usd, 6),
            cost_model="local_compute",
            budget_enforced=True,
            vlm_pages_used=vlm_pages_used,
            vlm_calls=vlm_calls,
            vlm_seconds_total=round(vlm_seconds_total, 4),
            escalation_triggered=False,
            escalation_target=None,
        ),
        pages=pages,
        error_message=f"{default_strategy}_failed" if all_error else None,
    )


def _final_strategy_from_pages(pages: list[ExtractedPage]) -> str:
    final_strategies = {page.metadata.strategy_used for page in pages}
    if final_strategies == {"strategy_a"}:
        return "strategy_a"
    if final_strategies == {"strategy_b"}:
        return "strategy_b"
    if final_strategies == {"strategy_c"}:
        return "strategy_c"
    return "+".join(sorted(final_strategies))


def _final_pages_by_strategy(pages: list[ExtractedPage]) -> dict[str, int]:
    return {
        "strategy_a": sum(1 for p in pages if p.metadata.strategy_used == "strategy_a"),
        "strategy_b": sum(1 for p in pages if p.metadata.strategy_used == "strategy_b"),
        "strategy_c": sum(1 for p in pages if p.metadata.strategy_used == "strategy_c"),
    }


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
    doc.metadata.cost_model = "local_compute"
    doc.metadata.budget_enforced = True
    doc.metadata.vlm_pages_used = sum(
        1 for p in doc.pages if p.metadata.vlm_used or int(p.signals.get("used_vlm", 0)) == 1
    )
    doc.metadata.vlm_calls = sum(
        int(p.signals.get("vlm_calls", 1 if int(p.signals.get("used_vlm", 0)) == 1 else 0))
        for p in doc.pages
    )
    doc.metadata.vlm_seconds_total = round(
        sum(float(p.metadata.vlm_wall_time_sec or p.signals.get("vlm_wall_time_sec", 0.0)) for p in doc.pages),
        4,
    )
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
    initial_strategy = profile.extraction_strategy
    if initial_strategy not in {
        ExtractionStrategy.strategy_a,
        ExtractionStrategy.strategy_b,
        ExtractionStrategy.strategy_c,
    }:
        return _error_document(pdf_path, f"Unsupported extraction strategy: {initial_strategy}")
    start_strategy = initial_strategy.value
    if forced is not None:
        if forced not in {"strategy_a", "strategy_b", "strategy_c"}:
            return _error_document(pdf_path, f"Unsupported forced strategy override: {forced}")
        start_strategy = forced

    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    ledger_doc_id = _compute_doc_id(pdf_path)
    ledger_path = LEDGER_DIR / f"{ledger_doc_id}.jsonl"
    if ledger_path.exists():
        ledger_path.unlink()

    pages_b_executed = 0
    pages_c_executed = 0
    planned_strategy_b = 0
    planned_strategy_c = 0
    executions_by_strategy: dict[str, int] = {"strategy_a": 0, "strategy_b": 0, "strategy_c": 0}

    if start_strategy == "strategy_b":
        all_pages = list(range(1, profile.page_count + 1))
        planned_strategy_b = len(all_pages)
        pages_b = extract_pages_with_docling(
            pdf_path=pdf_path,
            page_numbers=all_pages,
            rules=rules,
            batch_size=5,
        )
        extracted_b = _assemble_document(
            pdf_path=pdf_path,
            profile=profile,
            pages_by_number=pages_b,
            default_strategy="strategy_b",
        )
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
        pages_to_strategy_c: set[int] = set()
        for page in extracted_b.pages:
            _record_page_execution(
                doc=extracted_b,
                page=page,
                escalated_to=page.metadata.escalation_target,
                executions_by_strategy=executions_by_strategy,
            )
            if page.metadata.escalation_target == "strategy_c":
                pages_to_strategy_c.add(page.page_number)

        planned_strategy_c = len(pages_to_strategy_c)
        if pages_to_strategy_c:
            pages_c = extract_pages_with_vision(
                pdf_path=pdf_path,
                page_numbers=sorted(pages_to_strategy_c),
                rules=rules,
            )
            for page_number in sorted(pages_to_strategy_c):
                c_page = pages_c.get(page_number)
                if c_page is None:
                    continue
                extracted_b.pages[page_number - 1] = c_page
                _record_page_execution(
                    doc=extracted_b,
                    page=c_page,
                    escalated_to=None,
                    executions_by_strategy=executions_by_strategy,
                )
                pages_c_executed += 1

        extracted_b.metadata.strategy_used = _final_strategy_from_pages(extracted_b.pages)
        page_escalation_targets = _update_document_metadata(extracted_b)
        executed_strategy = extracted_b.metadata.strategy_used
        pages_b_executed = len(all_pages)
        extracted = extracted_b
    elif start_strategy == "strategy_c":
        all_pages = list(range(1, profile.page_count + 1))
        planned_strategy_c = len(all_pages)
        pages_c = extract_pages_with_vision(
            pdf_path=pdf_path,
            page_numbers=all_pages,
            rules=rules,
        )
        extracted = _assemble_document(
            pdf_path=pdf_path,
            profile=profile,
            pages_by_number=pages_c,
            default_strategy="strategy_c",
        )
        for page_number in all_pages:
            c_page = extracted.pages[page_number - 1]
            c_page.metadata.escalation_triggered = False
            c_page.metadata.escalation_target = None
            _record_page_execution(
                doc=extracted,
                page=c_page,
                escalated_to=None,
                executions_by_strategy=executions_by_strategy,
            )
            pages_c_executed += 1

        extracted.metadata.strategy_used = _final_strategy_from_pages(extracted.pages)
        page_escalation_targets = _update_document_metadata(extracted)
        executed_strategy = extracted.metadata.strategy_used
    else:
        extracted = extract_with_pdfplumber(pdf_path=pdf_path, rules=rules)
        a_min_conf = float(
            rules["strategy_routing"]["strategy_a"]["confidence_gates"]["min_confidence_score"]
        )
        b_min_conf = float(
            rules["strategy_routing"]["strategy_b"]["confidence_gates"]["min_confidence_score"]
        )

        pages_to_strategy_b: set[int] = set()
        pages_planned_strategy_c_from_a: set[int] = set()
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
                elif target == "strategy_c":
                    pages_planned_strategy_c_from_a.add(page.page_number)
            _record_page_execution(
                doc=extracted,
                page=page,
                escalated_to=escalated_to,
                executions_by_strategy=executions_by_strategy,
            )

        pages_b_executed = 0
        pages_executed_strategy_b: set[int] = set()
        pages_to_strategy_c_from_b: set[int] = set()
        planned_strategy_b = len(pages_to_strategy_b)
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
                _record_page_execution(
                    doc=extracted,
                    page=b_page,
                    escalated_to=escalated_to,
                    executions_by_strategy=executions_by_strategy,
                )
                pages_b_executed += 1
                pages_executed_strategy_b.add(page_no)
                if escalated_to == "strategy_c":
                    pages_to_strategy_c_from_b.add(page_no)

        pages_to_strategy_c = sorted(
            set(pages_planned_strategy_c_from_a).union(pages_to_strategy_c_from_b)
        )
        planned_strategy_c = len(pages_to_strategy_c)
        pages_c_executed = 0
        if pages_to_strategy_c:
            c_by_page = extract_pages_with_vision(
                pdf_path=pdf_path,
                page_numbers=pages_to_strategy_c,
                rules=rules,
            )
            for page_no in pages_to_strategy_c:
                c_page = c_by_page.get(page_no)
                if c_page is None:
                    continue
                extracted.pages[page_no - 1] = c_page
                _record_page_execution(
                    doc=extracted,
                    page=c_page,
                    escalated_to=None,
                    executions_by_strategy=executions_by_strategy,
                )
                pages_c_executed += 1

        page_escalation_targets = _update_document_metadata(extracted)
        extracted.metadata.strategy_used = _final_strategy_from_pages(extracted.pages)
        executed_strategy = extracted.metadata.strategy_used
        pages_b_executed = len(pages_executed_strategy_b)

    pages_below_strategy_a = sum(
        1
        for p in extracted.pages
        if p.metadata.strategy_used == "strategy_a" and p.metadata.escalation_triggered
    )
    escalation_pct = (sum(page_escalation_targets.values()) / max(len(extracted.pages), 1)) * 100.0

    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = EXTRACTED_DIR / f"{extracted.doc_id}.json"
    output_path.write_text(extracted.model_dump_json(indent=2), encoding="utf-8")

    strategy_counts = {
        "final_pages_by_strategy": _final_pages_by_strategy(extracted.pages),
        "executions_by_strategy": {
            "strategy_a": executions_by_strategy.get("strategy_a", 0),
            "strategy_b": executions_by_strategy.get("strategy_b", 0),
            "strategy_c": executions_by_strategy.get("strategy_c", 0),
        },
        "planned_strategy_b": planned_strategy_b,
        "planned_strategy_c": planned_strategy_c,
        "executed_strategy_b": pages_b_executed,
        "executed_strategy_c": pages_c_executed,
    }

    # Keep trace of advisory strategy and actual start strategy for auditability.
    advisory_path = EXTRACTED_DIR / f"{extracted.doc_id}.routing.json"
    advisory_path.write_text(
        json.dumps(
            {
                "doc_id": extracted.doc_id,
                "document_level_strategy_suggestion": initial_strategy.value,
                "starting_strategy": start_strategy,
                "executed_strategy": executed_strategy,
                "strategy_b_executed": pages_b_executed > 0,
                "strategy_b_pages_executed": pages_b_executed,
                "strategy_counts": strategy_counts,
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
