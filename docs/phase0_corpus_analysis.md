# Phase 0 Corpus Analysis — Principal FDE Review

> **Reviewer:** Principal Forward Deployed Engineer  
> **Date:** 2026-03-03  
> **Corpus:** 50 PDFs · 8 signals per document  
> **Data Sources:** `phase0_signals.csv`, `phase0_selected_12.csv`, `profile_corpus.py`

---

## 1. Distribution Patterns

### 1.1 Strategy Distribution

| Strategy | Count | % of Corpus | Total Pages | Avg Pages |
|----------|-------|-------------|-------------|-----------|
| **A — Fast Text** | 6 | 12% | 594 | 99 |
| **B — Layout Aware** | 22 | 44% | 1,391 | 63 |
| **C — Vision/OCR** | 22 | 44% | 810 | 37 |

> [!WARNING]
> Strategy A captures only **12% of documents** but **35% of total pages** (594/2,795). The corpus is overwhelmingly routed to the two expensive strategies (88% of docs, 65% of pages).

### 1.2 Origin Type Distribution

| Origin Type | Count | % |
|-------------|-------|---|
| `scanned_image` | 20 | 40% |
| `mixed` | 30 | 60% |
| `native_digital` | 0 | **0%** |

> [!CAUTION]
> **Zero native-digital documents exist in the corpus.** Every document with a text layer also trips the `avg_density < 0.03 OR avg_img > 0.50` mixed-origin gate. The `native_digital` label is unreachable under current thresholds for this corpus — the density ceiling of 0.03 is likely too high, or the corpus genuinely contains no clean-born-digital PDFs.

### 1.3 Layout Distribution

| Layout | Count | Documents |
|--------|-------|-----------|
| `single_column` | 28 | 20 scanned + 6 mixed + 2 borderline |
| `table_heavy` | 22 | All mixed-origin |
| `multi_column` | 0 | None detected |

No document crosses the `avg_x_jump > 0.15` multi-column threshold. The maximum observed `avg_x_jump` is **0.0364** (EthSwitch-10th), which is ~4× below the threshold. Either the corpus genuinely has no multi-column layouts, or the x-jump metric is too coarse to detect them.

---

## 2. Cluster Identification

The data forms **two sharp clusters** and a **transition band**:

### Cluster 1: Pure Scanned (20 docs)
- `avg_char_density = 0.0`, `avg_image_ratio = 1.0` (or ≥ 0.92), `avg_x_jump = 0.0`, `avg_tables/page = 0.0`
- Perfectly separable. No signal ambiguity.
- All correctly routed to Strategy C.

### Cluster 2: Digital-Mixed with Tables (22 docs)
- `avg_char_density ∈ [0.00068, 0.00684]`, `avg_image_ratio ∈ [0.005, 0.7812]`
- `avg_tables/page ∈ [0.4, 1.6]`
- All routed to Strategy B via the `table_heavy` layout gate.
- This cluster shows high internal variance in `avg_image_ratio` (σ ≈ 0.23).

### Cluster 3: Digital-Mixed without Tables (6 docs)
- `avg_char_density ∈ [0.00118, 0.00439]`, `avg_tables/page ∈ [0.0, 0.4]`
- Routed to Strategy A.
- Smallest cluster. Represents the cleanest extractable content.

### Transition Band (2 critical borderlines)

| Document | Density | Image Ratio | Tables/pg | Routed | Concern |
|----------|---------|-------------|-----------|--------|---------|
| `Audit Report - 2023.pdf` | **0.00005** | **0.8032** | 0.0 | C ✓ | Density barely above 0 — a few stray chars from a thin OCR layer. Correct routing. |
| `Pharmaceutical...VF.pdf` | **0.00068** | **0.7812** | 1.0 | B | Image ratio 0.78 is just below the 0.80 threshold. With 1 table/page it routes to B via table_heavy, but at 78% images, this doc likely needs C for most pages. **Potential misclassification.** |

---

## 3. Threshold Validation

### 3.1 Current Thresholds (from `profile_corpus.py`)

```python
# Scanned gate
if avg_density < 0.01 and avg_img > 0.80:       # → scanned_image
# Mixed gate
elif avg_density < 0.03 or avg_img > 0.50:       # → mixed
# Layout gates
if avg_tables > 0.5:                              # → table_heavy
elif avg_xjump > 0.15:                            # → multi_column
```

### 3.2 Validation Against Data

| Threshold | Value | Valid? | Evidence |
|-----------|-------|--------|----------|
| `density < 0.01` for scanned | ✅ Yes | All scanned docs have density ≤ 0.00005. All mixed docs have density ≥ 0.00068. Gap: **13.6×** between closest pair. Clean separation. |
| `img_ratio > 0.80` for scanned | ⚠️ Borderline | `Audit Report - 2023.pdf` slips in at 0.8032 (just 0.4% above threshold). `Pharmaceutical...VF.pdf` at 0.7812 is excluded. The boundary is **tight**. |
| `density < 0.03` for mixed | ✅ But vacuous | The max density in the corpus is 0.00684. Every document with text satisfies `density < 0.03`. This gate never rejects anything into `native_digital`. |
| `img_ratio > 0.50` for mixed | ✅ Functional | This correctly catches docs like `Annual_Report_JUNE-2019.pdf` (img=0.69) and `ETS_Annual_Report_2024_2025.pdf` (img=0.61) that have moderate image content. |
| `avg_tables > 0.5` for table_heavy | ⚠️ Aggressive | Three documents sit at exactly 0.4 tables/page (just below 0.5) and are routed to Strategy A. One of them (`CBE Annual Report 2012-13.pdf`, 20 pages) has tables but is treated as simple text. Lowering to **0.3** would capture these. |
| `avg_xjump > 0.15` for multi_column | ❓ Untestable | Max observed x_jump is 0.0364. No doc approaches 0.15. Threshold cannot be validated with this corpus. |

---

## 4. Outlier and Misclassification Analysis

### 4.1 Confirmed Outliers

| Document | Signal Anomaly | Risk |
|----------|---------------|------|
| `Audit Report - 2023.pdf` | 95 pages, density=0.00005, img=0.8032 | Has a ghost text layer (5e-05 density). Profiler correctly routes to C, but this density is non-zero — Strategy A would produce near-empty garbage. **Correctly handled.** |
| `EthSwitch-Annual-Report-2019.pdf` | 70 pages, density=0.0, img=0.9886 | Image ratio < 1.0 but density = 0.0. The 1.14% non-image area might be page borders/artifacts. **Safe.** |
| `Ethswitch-Annual-report-2020.2021_.pdf` | 91 pages, density=0.0, img=0.9244 | Same pattern as above. 7.6% non-image area could be whitespace/margins. **Safe.** |

### 4.2 Potential Misclassifications

| Document | Current Route | Concern | Suggested Route |
|----------|--------------|---------|-----------------|
| `Pharmaceutical...VF.pdf` | **B** | `img_ratio=0.7812`, just below 0.80 C threshold. At 78% images, most pages are likely scanned. Table detection may fail on image-heavy pages. | **C** (or B→C escalation) |
| `Annual_Report_JUNE-2019.pdf` | **B** | `img_ratio=0.6925`, density=0.00186. High image content may degrade Docling's table extraction. | Monitor for B→C page-level escalation |
| `CBE Annual Report 2012-13.pdf` | **A** | Only 20 pages with 0.4 tables/page and 0.1033 image ratio. Currently routed A, but it has tables. | Consider **B** if table fidelity matters |

---

## 5. Strategy Distribution Evaluation

### 5.1 Is Strategy B Over-Triggering?

**Yes, mildly.** 22/50 docs (44%) route to B. The primary driver is the `table_heavy` gate at `avg_tables > 0.5`.

Breakdown of the 22 B-routed documents:
- **7 Consumer Price Index reports** with avg_tables=1.4 → legitimate, table-critical
- **7 Annual Reports** with avg_tables ∈ [0.6, 1.6] → legitimate, but many have high image ratios (some > 0.4) that suggest pages where B won't outperform A
- **3 documents** with avg_tables ∈ [0.6, 0.8] and moderate complexity → legitimate
- **5 remaining** → mixed legitimacy

The **cost-weighted concern**: B's 22 docs × 63 avg pages × 4× cost = **5,544 cost units**, vs. the same pages at A = 1,386 units. If even 30% of B pages are simple text pages misrouted due to document-level routing, that's **~1,250 wasted cost units**.

> [!IMPORTANT]
> **Recommendation:** Keep document-level B routing but add a page-level pre-check: pages within a B-routed doc that have `table_count=0 AND img_ratio < 0.20` should be processed at Strategy A first, only escalating to B if structure_confidence < T_struct.

### 5.2 Is Strategy C Too Aggressive or Too Conservative?

**Conservative by count (22/50 docs), but correct for this corpus.** The pure-scanned cluster is unambiguous. The one concern is the `Pharmaceutical...VF.pdf` borderline doc that should probably be C but routes to B.

However, **C is expensive**: 22 docs × 37 avg pages × 12× cost = **9,768 cost units**. At the corpus level, C consumes **59% of total cost** for 40% of documents. This is acceptable only because these documents genuinely have zero text layer.

### 5.3 Cost Optimization Opportunity

| Scenario | Cost Units | Savings vs Current |
|----------|------------|-------------------|
| Current routing | ~16,698 | baseline |
| Page-level A-first within B docs (est. 30% pages eligible) | ~15,448 | **~7.5%** |
| Reclassify `Pharmaceutical...VF.pdf` to C | +120 (net) | quality gain, small cost increase |

---

## 6. Representativeness of `phase0_selected_12.csv`

### 6.1 Critical Finding: Only 8 Documents Selected, Not 12

The CSV contains **only 8 rows**. The `select_12()` function groups by strategy and picks easy/hard/edge per group. With 3 strategies × 3 picks = 9 max, minus deduplication → **8 unique docs**.

**Root cause in** [profile_corpus.py](file:///d:/TRP-1/Week-3/document-refinery/profile_corpus.py#L167-L189): The `_pick_three` function's edge case selection (highest `avg_tables/page`) often collides with the easy case (highest `avg_char_density`) for the C — Vision group, since all scanned docs have density=0 and tables=0, making easy/hard/edge indistinguishable.

### 6.2 Coverage Gaps

| Dimension | Covered? | Gap |
|-----------|----------|-----|
| Pure scanned, short (3 pg) | ✅ `2013-E.C...pdf` | — |
| Pure scanned, long (95 pg) | ✅ `Audit Report - 2023.pdf` | — |
| Pure scanned, medium (30-70 pg) | ❌ | Missing. `CBE Annual Report 2010-11.pdf` (62 pg) or `EthSwitch-Annual-Report-2019.pdf` (70 pg) should be included. |
| B with high image ratio | ❌ | `Annual_Report_JUNE-2019.pdf` (img=0.69) not selected — tests B's limits on image-heavy content. |
| B with low tables | ❌ | `Company_Profile_2024_25.pdf` (tables=0.6, 24 pg) — smallest B doc, tests minimum table complexity. |
| A with borderline tables | ❌ | `CBE Annual Report 2012-13.pdf` (tables=0.4) — tests whether A handles light tables adequately. |
| CPI-style narrow table format | ❌ | 7 nearly-identical CPI PDFs exist but none are selected. At least one should validate B's table extraction on tabular-dominant, short docs. |
| C — Vision Edge case | ❌ | The C group has no edge case selected (see §6.1). `Ethswitch-Annual-report-2020.2021_.pdf` (img=0.9244, not 1.0) would test boundary detection. |

### 6.3 Selection Logic Improvements

1. **Fix `_pick_three` for homogeneous clusters:** When density/tables have zero variance (all-scanned group), fall back to **page count** as the sorting axis — pick shortest, longest, and median.
2. **Add cross-strategy edge picks:** Select 1-2 docs that sit near strategy boundaries (e.g., `Pharmaceutical...VF.pdf` at img=0.78).
3. **Ensure at least 1 CPI report:** The 7 CPI docs form a distinct sub-type (short, table-dense, low image) not represented.
4. **Target 12 minimum:** The function returns `head(12)` but generates < 12. Add fallback random sampling to fill remaining slots from under-represented clusters.

---

## 7. Refined Escalation Thresholds

### 7.1 Proposed Threshold Table

| Parameter | Current | Proposed | Justification |
|-----------|---------|----------|---------------|
| **Scanned: `density <`** | 0.01 | **0.0004** | Midpoint of the observed gap. Scanned cluster max = 0.00005, mixed cluster min = 0.00068. Midpoint: `(0.00005 + 0.00068) / 2 ≈ 0.000365`, rounded to 0.0004. This keeps separation symmetric: 8× margin above scanned, 1.7× margin below mixed. Setting at 0.001 (previous draft) would encroach into the mixed cluster. |
| **Scanned: compound gate** | `density < 0.01 AND img > 0.80` | **`density < 0.0004 OR (img_ratio > 0.80 AND density < 0.001)`** | Raising img_ratio to 0.85 would exclude `Audit Report - 2023.pdf` (img=0.8032), which is genuinely scanned with a ghost text layer (density=0.00005). The compound condition handles this correctly: it routes via the density arm (`0.00005 < 0.0004`), while the image arm catches any scanned doc with density in [0.0004, 0.001) and img > 0.80. |
| **Mixed: `density <`** | 0.03 | **0.01** | Current 0.03 is vacuous (max corpus density = 0.00684). Lowering to 0.01 remains above all observed values but leaves room for future native-digital docs (expected density > 0.01). |
| **Mixed: `img_ratio >`** | 0.50 | **0.50** | No change. Correctly captures the transition band. |
| **Table-heavy: `avg_tables >`** | 0.5 | **0.3** | Lowering by 0.2. Captures 3 docs at avg_tables=0.4 that contain real tables. These docs (`CBE 2012-13`, `CBE 2016-17`, `CBE 2017-18`, `CBE 2018-19`, `CBE 2023-24`, `ETS 2024-25`) have genuine tabular content that Strategy A may flatten. |
| **Multi-column: `avg_xjump >`** | 0.15 | **0.08** (provisional) | Multi-column routing cannot be validated in this corpus — max observed xjump is 0.0364, ~4× below either threshold. Threshold lowered to 0.08 as a more conservative advisory gate but **marked provisional**: requires validation against a new corpus containing known multi-column documents (e.g., newspapers, academic papers) before being used as a hard-route trigger. |
| **NEW — High-image B→C gate:** | — | `img_ratio > 0.70 AND density < 0.001` | Catches `Pharmaceutical...VF.pdf`-type docs that are effectively scanned but have enough table structure to initially route to B. Forces C when text layer is too thin to support B's layout analysis. |
| **NEW — Page budget for C:** | 10 pages | **15 pages** | With 37 avg pages in C docs, 10 is reached by 27% of a doc. Increasing to 15 (40%) before doc-level C trigger reduces premature whole-doc reprocessing. |

### 7.2 Impact Estimate

| Metric | Current | After Refinement |
|--------|---------|-----------------|
| Strategy A docs | 6 (12%) | 3–4 (6–8%) — some shift to B with lower table threshold |
| Strategy B docs | 22 (44%) | 24–25 (48–50%) |
| Strategy C docs | 22 (44%) | 21–23 (42–46%) |
| Misclassification risk | 2–3 docs | ≤ 1 doc |
| Estimated cost units | ~16,698 | ~17,200 (+3%) — quality gain at marginal cost. **This 3% cost increase reduces misclassification risk by ~50%** (from 2–3 docs to ≤1 doc), yielding a favorable cost-quality tradeoff curve. |

---

## 8. Risks for Unseen Document Types

| Risk | Gap in Current Corpus | Potential Impact |
|------|----------------------|------------------|
| **Native-digital PDFs** (born-digital with no images) | Zero instances. Density > 0.01, img < 0.05. | Strategy A path is under-tested. May encounter unexpected font encoding or embedded-form issues. |
| **True multi-column layouts** (newspapers, academic papers) | Zero instances above xjump=0.04. | Column bleed in Strategy A is completely untested. The 0.15 threshold is theoretical. |
| **Mixed document with partial scans** (e.g., pg 1-10 digital, pg 11-50 scanned) | Profile aggregates mask per-page variance. A doc with 50% scanned pages could average 0.50 image ratio and route to B globally. | Page-level escalation is critical — document-level routing will fail silently for these. |
| **Non-Latin scripts** (Amharic, Ge'ez) | Some Ethiopian docs may contain Amharic. EasyOCR support for Amharic is limited. | Strategy C may produce garbage for non-Latin scanned text. Needs language-detection signal in Phase 0. |
| **Fillable PDF forms** | Zero instances detected. | Form fields create text layers that look like native-digital but have unusual spatial patterns. Strategy A may extract field labels without values. |
| **Password-protected / corrupted PDFs** | Not represented. `compute_signals` silently returns `None` for unreadable files. | Risk of silent data loss. Should emit explicit `ERROR` profile, not skip. |

> [!CAUTION]
> **Document-level averaging masks page-level heterogeneity.** A document with 50% scanned pages and 50% digital pages may average to `img_ratio ≈ 0.5` and `density ≈ 0.002`, incorrectly routing to Strategy B globally. The per-page signal variance is invisible at the document level. Therefore, **document-level routing must be advisory only; page-level signals must drive final escalation decisions.** Any production deployment that relies solely on document-level thresholds will silently misroute partially-scanned documents.

---

## 9. Architect's Verdict

The corpus exhibits a **clean bimodal split** — 40% pure-scanned and 60% mixed-digital with embedded tables — which validates the three-strategy architecture. However, three structural weaknesses demand immediate attention: **(1)** the `native_digital` origin class is **unreachable** under current thresholds and untestable with this corpus, meaning Strategy A's fast path is validated on only 6 documents (12% of corpus, zero of which are truly born-digital); **(2)** the selection logic produced **8 documents, not 12**, failing to cover medium-length scanned docs, high-image B-candidates, and the CPI table sub-type — this must be fixed before Phase 1 extraction begins; **(3)** the `Pharmaceutical...VF.pdf` borderline case (img=0.78, density=0.00068) exposes a **0.02-wide decision gap** between Strategy B and C routing where documents will be misclassified in production. The refined thresholds above close this gap at a marginal 3% cost increase that reduces misclassification risk by ~50%. The biggest unseen risk is **partially-scanned documents** where document-level averaging masks per-page variance — **page-level escalation is not optional, it is load-bearing for correctness.**
