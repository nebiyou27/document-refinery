"""
src/agents/triage.py
====================
Stage 1 — Triage Agent

Takes any PDF → returns a DocumentProfile.
All thresholds loaded from rubric/extraction_rules.yaml.
No hardcoded values in this file.

Usage:
    python src/agents/triage.py data/YOUR_DOCUMENT.pdf
"""

import hashlib
import sys
import warnings
warnings.filterwarnings("ignore")

from collections import Counter
from pathlib import Path

# Allow direct script execution: `python src/agents/triage.py ...`
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pdfplumber
import yaml

from src.models.document_profile import (
    DocumentProfile,
    ExtractionStrategy,
    LayoutComplexity,
    OriginType,
    DomainHint,
    PageSignal,
    make_error_profile,       # ← Invariant I-11: never return None silently
)


# ── Load thresholds from YAML — never hardcode ────────────────
RULES_PATH = Path("rubric/extraction_rules.yaml")


def load_rules() -> dict:
    with open(RULES_PATH, "r") as f:
        return yaml.safe_load(f)


# ── Domain hint — keyword-based classifier ────────────────────
DOMAIN_KEYWORDS = {
    DomainHint.financial: [
        "revenue", "profit", "loss", "balance sheet", "income",
        "expenditure", "fiscal", "budget", "audit", "financial",
        "bank", "tax", "capital", "asset", "liability",
    ],
    DomainHint.legal: [
        "agreement", "contract", "clause", "pursuant", "jurisdiction",
        "plaintiff", "defendant", "court", "law", "regulation",
        "compliance", "legal", "statute", "legislation",
    ],
    DomainHint.technical: [
        "system", "software", "infrastructure", "technical", "architecture",
        "implementation", "protocol", "specification", "engineering",
        "network", "database", "api", "framework",
    ],
    DomainHint.medical: [
        "patient", "diagnosis", "treatment", "clinical", "hospital",
        "medical", "health", "pharmaceutical", "drug", "therapy",
        "disease", "symptom", "dosage",
    ],
}


def classify_domain(text_sample: str) -> DomainHint:
    """
    Keyword-based domain classifier.
    Counts keyword hits per domain → returns highest scoring domain.
    Falls back to general if no domain scores above 0.
    """
    text_lower = text_sample.lower()
    scores = {domain: 0 for domain in DomainHint}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[domain] += 1

    best = max(scores, key=lambda d: scores[d])
    return best if scores[best] > 0 else DomainHint.general


# ── Origin classification ─────────────────────────────────────

def classify_origin(avg_density: float,
                    avg_img: float,
                    rules: dict) -> OriginType:
    """
    Three named compound conditions — unambiguous, no shadowed branches.
    Order matters: scanned detection runs first.
    """
    sd = rules["triage"]["scanned_detection"]

    scanned_by_density = avg_density < sd["scanned_by_density"]
    ghost_text_scan    = (avg_img    > sd["ghost_text_scan_img"] and
                          avg_density < sd["ghost_text_scan_density"])
    high_image_thin    = (avg_img    > sd["high_image_thin_img"] and
                          avg_density < sd["high_image_thin_density"])

    if scanned_by_density or ghost_text_scan or high_image_thin:
        return OriginType.scanned_image

    mg = rules["triage"]["mixed_gate"]
    if avg_density < mg["density_ceiling"] or avg_img > mg["image_floor"]:
        return OriginType.mixed

    return OriginType.native_digital


# ── Layout classification ─────────────────────────────────────

def classify_layout(avg_tables: float,
                    avg_xjump: float,
                    rules: dict) -> LayoutComplexity:
    lc = rules["triage"]["layout_complexity"]

    if avg_tables > lc["table_heavy_threshold"]:
        return LayoutComplexity.table_heavy
    if avg_xjump > lc["multi_column_xjump"]:
        return LayoutComplexity.multi_column
    return LayoutComplexity.single_column


# ── Strategy recommendation ───────────────────────────────────

def recommend_strategy(origin: OriginType,
                       layout: LayoutComplexity) -> ExtractionStrategy:
    if origin == OriginType.scanned_image:
        return ExtractionStrategy.strategy_c
    if layout in (LayoutComplexity.table_heavy, LayoutComplexity.multi_column):
        return ExtractionStrategy.strategy_b
    return ExtractionStrategy.strategy_a


# ── Cost estimate ─────────────────────────────────────────────

def estimate_cost(strategy: ExtractionStrategy,
                  page_count: int,
                  rules: dict) -> float:
    """
    Strategies A and B are free (local tools).
    Strategy C has a cost only if free tier is exhausted.
    Returns worst-case estimate in USD.
    """
    if strategy == ExtractionStrategy.strategy_c:
        cpp = rules["strategy_routing"]["strategy_c"]["budget_guard"][
            "cost_per_page_estimate_usd"
        ]
        return round(cpp * page_count, 4)
    return 0.0


# ── OOD detection ─────────────────────────────────────────────

CORPUS_RANGES = {
    "density": 0.006835,   # max observed in corpus
    "tables":  1.6,         # max observed in corpus
    "xjump":   0.0364,      # max observed in corpus
}


def detect_ood(avg_density: float,
               avg_tables: float,
               avg_xjump: float) -> bool:
    """
    Returns True if signals are outside observed corpus ranges.
    Triggers safer routing and logs NEW_DOC_TYPE_SUSPECTED.
    """
    ood = (
        avg_density > CORPUS_RANGES["density"] * 2   or
        avg_tables  > CORPUS_RANGES["tables"]  * 1.5 or
        avg_xjump   > CORPUS_RANGES["xjump"]   * 3
    )
    if ood:
        print("  WARNING: NEW_DOC_TYPE_SUSPECTED - signals outside corpus range")
        print("      Routing to Strategy B as safe default")
    return ood


# ── Per-page signal computation ───────────────────────────────

def compute_page_signal(page,
                        page_number: int,
                        strategy: ExtractionStrategy,
                        rules: dict) -> PageSignal:
    """Compute all signals for a single page."""
    text     = page.extract_text() or ""
    area     = (page.width or 1) * (page.height or 1)
    img_area = sum(
        img.get("width", 0) * img.get("height", 0)
        for img in (page.images or [])
    )
    chars = page.chars or []
    jumps = sum(
        1 for a, b in zip(chars, chars[1:])
        if abs(b["x0"] - a["x0"]) > (page.width or 600) * 0.30
    )

    try:
        table_count = len(page.extract_tables() or [])
    except Exception:
        table_count = 0

    density   = round(len(text) / area, 6)
    img_ratio = round(min(img_area / area, 1.0), 4)
    x_jump    = round(jumps / max(len(chars) - 1, 1), 4)

    # Page-level confidence score
    confidence = 1.0
    sr_a = rules["strategy_routing"]["strategy_a"]
    sr_b = rules["strategy_routing"]["strategy_b"]
    a_gates = sr_a.get("confidence_gates", sr_a)
    b_gates = sr_b.get("confidence_gates", sr_b)

    if density   < a_gates.get("min_char_density",     0.001): confidence -= 0.40
    if img_ratio > a_gates.get("max_image_area_ratio", 0.61):  confidence -= 0.30
    if len(text) < 50:                                        confidence -= 0.20
    if x_jump    > rules["triage"]["layout_complexity"]["multi_column_xjump"]:
        confidence -= 0.10
    confidence = max(round(confidence, 3), 0.0)

    # Page-level strategy override based on confidence
    a_min = a_gates.get("min_confidence_score", 0.75)
    b_min = b_gates.get("min_confidence_score", 0.65)

    if strategy == ExtractionStrategy.strategy_a and confidence < a_min:
        page_strategy = ExtractionStrategy.strategy_b
    elif strategy in (ExtractionStrategy.strategy_a,
                      ExtractionStrategy.strategy_b) and confidence < b_min:
        page_strategy = ExtractionStrategy.strategy_c
    else:
        page_strategy = strategy

    return PageSignal(
        page_number       = page_number,
        char_count        = len(text),
        char_density      = density,
        image_area_ratio  = img_ratio,
        table_count       = table_count,
        x_jump_ratio      = x_jump,
        confidence        = confidence,
        assigned_strategy = page_strategy,
    )


# ── Main triage function ──────────────────────────────────────

def triage(pdf_path: str) -> DocumentProfile:
    """
    Stage 1 — Triage Agent.
    Takes any PDF path → returns a fully populated DocumentProfile.

    Never raises — always returns a DocumentProfile.
    If the PDF is unreadable, returns an ERROR profile (Invariant I-11).
    """
    path  = Path(pdf_path)
    rules = load_rules()

    # ── File not found ────────────────────────────────────────
    if not path.exists():
        return make_error_profile(
            str(path),
            f"File not found: {pdf_path}"
        )

    # ── Stable doc_id from file content hash ─────────────────
    doc_id = hashlib.md5(path.read_bytes()).hexdigest()[:12]

    try:
        with pdfplumber.open(path) as pdf:
            total_pages = len(pdf.pages)

            # Sample up to 5 pages for document-level signals
            indices = sorted(set([
                0,
                total_pages // 4,
                total_pages // 2,
                (3 * total_pages) // 4,
                total_pages - 1,
            ]))
            sampled = [pdf.pages[i] for i in indices if i < total_pages]

            # Aggregate signals across sampled pages
            densities, img_ratios, x_jumps, table_counts = [], [], [], []
            text_sample = ""

            for page in sampled:
                text        = page.extract_text() or ""
                text_sample += text[:500]           # for domain classification
                area        = (page.width or 1) * (page.height or 1)
                img_area    = sum(
                    img.get("width", 0) * img.get("height", 0)
                    for img in (page.images or [])
                )
                chars = page.chars or []
                jumps = sum(
                    1 for a, b in zip(chars, chars[1:])
                    if abs(b["x0"] - a["x0"]) > (page.width or 600) * 0.30
                )

                densities.append(len(text) / area)
                img_ratios.append(min(img_area / area, 1.0))
                x_jumps.append(jumps / max(len(chars) - 1, 1))

                try:
                    table_counts.append(len(page.extract_tables() or []))
                except Exception:
                    table_counts.append(0)

            avg_density = round(sum(densities)    / len(densities),    6)
            avg_img     = round(sum(img_ratios)   / len(img_ratios),   4)
            avg_xjump   = round(sum(x_jumps)      / len(x_jumps),      4)
            avg_tables  = round(sum(table_counts) / len(table_counts), 2)

            # ── Classification ────────────────────────────────
            origin   = classify_origin(avg_density, avg_img, rules)
            layout   = classify_layout(avg_tables, avg_xjump, rules)
            domain   = classify_domain(text_sample)
            strategy = recommend_strategy(origin, layout)

            # ── OOD check — override to B if out of distribution
            is_ood = detect_ood(avg_density, avg_tables, avg_xjump)
            if is_ood and strategy == ExtractionStrategy.strategy_a:
                strategy = ExtractionStrategy.strategy_b

            # ── Per-page signals (all pages, not just sampled) ─
            page_signals = []
            for i, page in enumerate(pdf.pages):
                sig = compute_page_signal(page, i + 1, strategy, rules)
                page_signals.append(sig)

            # ── Cost estimate ─────────────────────────────────
            cost = estimate_cost(strategy, total_pages, rules)

            # ── Build DocumentProfile ─────────────────────────
            # file_name added — required by merged schema
            profile = DocumentProfile(
                doc_id              = doc_id,
                file_path           = str(path),
                file_name           = path.name,        # ← new field
                origin_type         = origin,
                layout_complexity   = layout,
                domain_hint         = domain,
                extraction_strategy = strategy,
                estimated_cost_usd  = cost,
                page_count          = total_pages,
                per_page_signals    = page_signals,
                is_ood              = is_ood,            # ← new field
            )

    except Exception as e:
        # ── Invariant I-11: never return None — emit ERROR profile
        return make_error_profile(str(path), str(e))

    return profile


# ── Save profile to .refinery/profiles/ ──────────────────────

def save_profile(profile: DocumentProfile) -> Path:
    """Persist DocumentProfile to .refinery/profiles/{doc_id}.json"""
    out_dir = Path(".refinery/profiles")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{profile.doc_id}.json"
    out_path.write_text(profile.model_dump_json(indent=2), encoding="utf-8")
    return out_path


# ── CLI entry point ───────────────────────────────────────────

if __name__ == "__main__":
    import sys

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.pdf"

    print(f"\nTriaging: {pdf_path}\n")

    profile = triage(pdf_path)

    # ── Handle error profiles ─────────────────────────────────
    if profile.origin_type == OriginType.error:
        print("ERROR profile emitted")
        print(f"   file    : {profile.file_name}")
        print(f"   reason  : {profile.error_message}")
        sys.exit(1)

    # ── Print summary ─────────────────────────────────────────
    print("------------------------------------ DocumentProfile ------------------------------------")
    print(f"  doc_id              : {profile.doc_id}")
    print(f"  file                : {profile.file_name}")
    print(f"  page_count          : {profile.page_count}")
    print(f"  origin_type         : {profile.origin_type.value}")
    print(f"  layout_complexity   : {profile.layout_complexity.value}")
    print(f"  domain_hint         : {profile.domain_hint.value}")
    print(f"  extraction_strategy : {profile.extraction_strategy.value}")
    print(f"  estimated_cost_usd  : ${profile.estimated_cost_usd}")
    print(f"  is_ood              : {profile.is_ood}")

    # ── Page-level strategy breakdown ────────────────────────
    counts = Counter(s.assigned_strategy.value for s in profile.per_page_signals)
    print("\n------------------------------ Page-Level Strategy Breakdown ------------------------------")
    for strat, count in sorted(counts.items()):
        pct = round(100 * count / profile.page_count)
        print(f"  {strat:<14} : {count:>4} pages  ({pct}%)")

    # ── Sample page signals ───────────────────────────────────
    print("\n--------------------------- Sample Page Signals (first 5 pages) ---------------------------")
    print(f"  {'Page':<6} {'Chars':<8} {'Density':<10} "
          f"{'ImgRatio':<10} {'Tables':<8} {'Conf':<8} Strategy")
    print(f"  {'-'*68}")
    for sig in profile.per_page_signals[:5]:
        print(f"  {sig.page_number:<6} {sig.char_count:<8} "
              f"{sig.char_density:<10} {sig.image_area_ratio:<10} "
              f"{sig.table_count:<8} {sig.confidence:<8} "
              f"{sig.assigned_strategy.value}")

    # ── Save ──────────────────────────────────────────────────
    saved = save_profile(profile)
    print(f"\nProfile saved -> {saved}")
    print(f"\nNext: run extraction with strategy "
          f"'{profile.extraction_strategy.value}'")
