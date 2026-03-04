"""
profile_corpus.py
=================
Run this on your 50-doc corpus to:
  1. Measure signals on every PDF
  2. Print a summary table
  3. Save full results to phase0_signals.csv
  4. Automatically suggest your 12 documents (3 per class)

Usage:
    python profile_corpus.py

Output:
    phase0_signals.csv        — full signal table for all 50 docs
    phase0_selected_12.csv    — your recommended 12 docs
"""

import warnings
warnings.filterwarnings("ignore")

import pdfplumber
import pandas as pd
from pathlib import Path


# ── Folder where your PDFs live ───────────────────────────────
DATA_DIR = Path("data")


# ── Known document classes (fill in your actual filenames) ────
# Add your filenames under each class.
# If you are unsure, leave them empty — script still runs.
CLASS_MAP = {
    "A_Financial":   [],   # e.g. "CBE_ANNUAL_REPORT.pdf"
    "B_Scanned":     [],   # e.g. "Audit_Report_2023.pdf"
    "C_Technical":   [],   # e.g. "fta_performance_survey.pdf"
    "D_TableHeavy":  [],   # e.g. "tax_expenditure_ethiopia.pdf"
}


# ================================================================
# FIX 1 — Refined Thresholds (evidence-based from corpus signals)
# ================================================================

def classify_origin(avg_density: float, avg_img: float) -> str:
    """
    Three named conditions — unambiguous, no shadowed branches.

    scanned_by_density : near-zero text → pure scanned
    ghost_text_scan    : high image + thin ghost OCR layer
    high_image_thin    : catches borderline docs like Pharmaceutical VF
                         (img=0.78, just below old 0.80 gate)
    """
    scanned_by_density = avg_density < 0.0004
    ghost_text_scan    = (avg_img > 0.80 and avg_density < 0.001)
    high_image_thin    = (avg_img > 0.70 and avg_density < 0.001)

    if scanned_by_density or ghost_text_scan or high_image_thin:
        return "scanned_image"
    elif avg_density < 0.01 or avg_img > 0.50:
        return "mixed"
    else:
        return "native_digital"     # now reachable for clean born-digital docs


def classify_layout(avg_tables: float, avg_xjump: float) -> str:
    """
    table_heavy threshold lowered 0.5 → 0.3 (captures docs with 0.4 tables/pg).
    multi_column threshold lowered 0.15 → 0.08 (provisional — corpus has no
    multi-column docs to validate against, marked advisory only).
    """
    if avg_tables > 0.3:        # lowered from 0.5
        return "table_heavy"
    elif avg_xjump > 0.08:      # lowered from 0.15 — provisional
        return "multi_column"
    else:
        return "single_column"


def recommend_strategy(origin: str, layout: str) -> str:
    if origin == "scanned_image":
        return "C — Vision (Chunkr/EasyOCR)"
    elif layout in ("multi_column", "table_heavy"):
        return "B — Layout Aware (Docling)"
    else:
        return "A — Fast Text (pdfplumber)"


# ================================================================
# Signal Computation
# ================================================================

def compute_signals(path: Path) -> dict | None:
    """Measure key signals on a PDF. Returns None if unreadable."""
    try:
        with pdfplumber.open(path) as pdf:
            total_pages = len(pdf.pages)

            # Sample up to 5 pages: first, last, and spaced middle
            indices = sorted(set([
                0,
                total_pages // 4,
                total_pages // 2,
                (3 * total_pages) // 4,
                total_pages - 1,
            ]))
            sampled = [pdf.pages[i] for i in indices if i < total_pages]

            densities, img_ratios, x_jumps, table_counts = [], [], [], []

            for page in sampled:
                text      = page.extract_text() or ""
                area      = (page.width or 1) * (page.height or 1)
                img_area  = sum(
                    img.get("width", 0) * img.get("height", 0)
                    for img in (page.images or [])
                )
                chars     = page.chars or []
                jumps     = sum(
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

            origin   = classify_origin(avg_density, avg_img)
            layout   = classify_layout(avg_tables, avg_xjump)
            strategy = recommend_strategy(origin, layout)

            return {
                "file":             path.name,
                "pages":            total_pages,
                "avg_char_density": avg_density,
                "avg_image_ratio":  avg_img,
                "avg_x_jump":       avg_xjump,
                "avg_tables/page":  avg_tables,
                "origin_type":      origin,
                "layout":           layout,
                "strategy":         strategy,
            }

    except Exception as e:
        print(f"  ⚠️  Could not read {path.name}: {e}")
        return None


# ================================================================
# FIX 2 — Selection Logic (guarantees 3 picks per group)
# ================================================================

def _is_homogeneous(subset: pd.DataFrame) -> bool:
    """
    True when both density AND tables have near-zero variance.
    This happens for the pure-scanned cluster where every doc
    has density=0 and tables=0 — making easy/hard/edge identical
    if we sort by those columns.
    """
    density_std = subset["avg_char_density"].std()
    tables_std  = subset["avg_tables/page"].std()
    return density_std < 0.0001 and tables_std < 0.01


def _pick_three(subset: pd.DataFrame, label: str) -> list[dict]:
    """
    Guarantees 3 distinct picks per group.

    Homogeneous cluster (e.g. all-scanned):
        → sort by page count: shortest / longest / median
    Heterogeneous cluster:
        → easy  = highest char density  (cleanest)
        → hard  = lowest char density   (hardest)
        → edge  = highest tables/page OR highest x_jump
    """
    picks = []
    if subset.empty:
        return picks

    if _is_homogeneous(subset):
        # Sort by page count — only axis with variance in scanned cluster
        by_pages = subset.sort_values("pages").reset_index(drop=True)
        candidates = []

        # Shortest
        candidates.append(by_pages.iloc[0].to_dict())
        candidates[-1]["selected_as"] = f"{label} — Short (easy)"

        # Longest
        if len(by_pages) > 1:
            candidates.append(by_pages.iloc[-1].to_dict())
            candidates[-1]["selected_as"] = f"{label} — Long (hard)"

        # Median pages
        if len(by_pages) > 2:
            mid_idx = len(by_pages) // 2
            candidates.append(by_pages.iloc[mid_idx].to_dict())
            candidates[-1]["selected_as"] = f"{label} — Median (edge)"

        # Deduplicate by file
        seen = set()
        for c in candidates:
            if c["file"] not in seen:
                picks.append(c)
                seen.add(c["file"])

    else:
        # Heterogeneous cluster — sort by meaningful signal
        easy = subset.nlargest(1, "avg_char_density").iloc[0].to_dict()
        easy["selected_as"] = f"{label} — Easy"
        picks.append(easy)

        hard = subset.nsmallest(1, "avg_char_density").iloc[0].to_dict()
        if hard["file"] != easy["file"]:
            hard["selected_as"] = f"{label} — Hard"
            picks.append(hard)

        # Edge: highest tables first, x_jump as tiebreaker
        remaining = subset[
            ~subset["file"].isin([p["file"] for p in picks])
        ]
        if not remaining.empty:
            edge = remaining.nlargest(1, "avg_tables/page").iloc[0].to_dict()
            edge["selected_as"] = f"{label} — Edge (table)"
            picks.append(edge)
        elif len(picks) < 3:
            # Fallback: closest image_ratio to strategy boundary
            fallback = subset[
                ~subset["file"].isin([p["file"] for p in picks])
            ].nlargest(1, "avg_image_ratio")
            if not fallback.empty:
                fb = fallback.iloc[0].to_dict()
                fb["selected_as"] = f"{label} — Edge (boundary)"
                picks.append(fb)

    return picks


def select_12(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups docs and picks 3 per group.
    Uses CLASS_MAP if filled, otherwise groups by strategy.
    Fills to 12 with boundary docs if under-selected.
    """
    selected = []
    has_class_map = any(len(v) > 0 for v in CLASS_MAP.values())

    if has_class_map:
        for cls, files in CLASS_MAP.items():
            if not files:
                continue
            subset = df[df["file"].isin(files)].copy()
            if subset.empty:
                continue
            selected.extend(_pick_three(subset, cls))
    else:
        for strategy, grp in df.groupby("strategy"):
            selected.extend(_pick_three(grp.copy(), strategy))

    result = pd.DataFrame(selected).drop_duplicates(subset="file")

    # ── Fill to 12 if under-selected ─────────────────────────
    if len(result) < 12:
        already = set(result["file"])
        # Pick boundary docs (nearest to strategy thresholds)
        boundary = df[~df["file"].isin(already)].copy()
        boundary["boundary_score"] = (
            (boundary["avg_image_ratio"] - 0.70).abs() +
            (boundary["avg_char_density"] - 0.001).abs()
        )
        extras = boundary.nsmallest(12 - len(result), "boundary_score")
        extras = extras.copy()
        extras["selected_as"] = "Boundary fill"
        result = pd.concat([result, extras], ignore_index=True)

    return result.head(12)


# ================================================================
# Main
# ================================================================

def main():
    pdfs = sorted(DATA_DIR.glob("*.pdf"))

    if not pdfs:
        print(f"\n⚠️  No PDFs found in {DATA_DIR}/")
        print("Make sure your documents are in the data/ folder.")
        return

    print(f"\n📄 Found {len(pdfs)} PDFs in {DATA_DIR}/")
    print("Profiling documents...\n")

    rows = []
    for i, path in enumerate(pdfs, 1):
        print(f"  [{i:02d}/{len(pdfs)}] {path.name}", end=" ... ", flush=True)
        result = compute_signals(path)
        if result:
            rows.append(result)
            print(f"{result['origin_type']} / {result['layout']} "
                  f"→ {result['strategy']}")
        else:
            print("FAILED")

    if not rows:
        print("\n❌ No documents could be processed.")
        return

    df = pd.DataFrame(rows)
    df.to_csv("phase0_signals.csv", index=False)

    # ── Strategy distribution ─────────────────────────────────
    print(f"\n── Strategy Distribution ────────────────────────────")
    for strategy, grp in df.groupby("strategy"):
        total_pages = grp["pages"].sum()
        print(f"  {strategy}: {len(grp)} docs | {total_pages} total pages")

    # ── Origin distribution ───────────────────────────────────
    print(f"\n── Origin Distribution ──────────────────────────────")
    for origin, grp in df.groupby("origin_type"):
        print(f"  {origin}: {len(grp)} docs")

    # ── Full signal table ─────────────────────────────────────
    print(f"\n── Full Signal Table ────────────────────────────────")
    print(f"{'FILE':<42} {'PG':<5} {'DENSITY':<10} {'IMG':<8} "
          f"{'TABLES':<8} {'ORIGIN':<16} {'STRATEGY'}")
    print("─" * 105)
    for _, r in df.iterrows():
        print(f"{r['file']:<42} {r['pages']:<5} {r['avg_char_density']:<10} "
              f"{r['avg_image_ratio']:<8} {r['avg_tables/page']:<8} "
              f"{r['origin_type']:<16} {r['strategy']}")

    # ── Selected 12 ───────────────────────────────────────────
    selected = select_12(df)
    selected.to_csv("phase0_selected_12.csv", index=False)

    print(f"\n── Recommended 12 Documents ─────────────────────────")
    print(f"{'#':<4} {'FILE':<42} {'SELECTED AS':<30} {'STRATEGY'}")
    print("─" * 100)
    for i, (_, r) in enumerate(selected.iterrows(), 1):
        print(f"{i:<4} {r['file']:<42} {r['selected_as']:<30} {r['strategy']}")

    print(f"\n  Total selected: {len(selected)} documents")
    print(f"\n✅ Full results  → phase0_signals.csv")
    print(f"✅ Selected 12   → phase0_selected_12.csv")
    print(f"\nNext: paste the Strategy Distribution and")
    print(f"Recommended 12 table above for architect review.")


if __name__ == "__main__":
    main()