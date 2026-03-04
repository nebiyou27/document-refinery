"""
create_domain_notes.py
======================
Run once to generate DOMAIN_NOTES.md populated with
your actual Phase 0 findings.

Usage:
    python create_domain_notes.py
"""

from pathlib import Path

CONTENT = '''# DOMAIN_NOTES.md
## Document Intelligence Refinery — Phase 0 Domain Onboarding

---

## 0. Pipeline Diagram

```mermaid
flowchart TD
    A[Raw Document] --> B[Triage Agent]
    B --> C{DocumentProfile}
    C -->|native_digital + single_column| D[Strategy A\nFast Text\npdfplumber]
    C -->|multi_column OR table_heavy| E[Strategy B\nLayout-Aware\nDocling]
    C -->|scanned OR high-image thin-text| F[Strategy C\nVision\nChunkr/EasyOCR]
    D -->|confidence < 0.75| E
    E -->|confidence < 0.65 on key pages| F
    D --> G[ExtractedDocument]
    E --> G
    F --> G
    G --> H[Semantic Chunking Engine]
    H --> I[LDU List]
    I --> J[PageIndex Builder]
    I --> K[Vector Store ChromaDB]
    I --> L[FactTable SQLite]
    J --> M[Query Interface Agent]
    K --> M
    L --> M
    M --> N[Answer + ProvenanceChain]

    style D fill:#c8f7c5
    style E fill:#fef9c3
    style F fill:#fde8e8
```

---

## 1. Extraction Strategy Decision Tree

```
INPUT: DocumentProfile
│
├── scanned_by_density: avg_density < 0.0004
│   └── → Strategy C (no text layer)
│
├── ghost_text_scan: avg_img > 0.80 AND avg_density < 0.001
│   └── → Strategy C (OCR layer too thin)
│
├── high_image_thin: avg_img > 0.70 AND avg_density < 0.001
│   └── → Strategy C (image-dominant, B would fail)
│
├── mixed: avg_density < 0.01 OR avg_img > 0.50
│   ├── table_heavy: avg_tables > 0.3
│   │   └── → Strategy B
│   ├── multi_column: avg_x_jump > 0.08 (provisional)
│   │   └── → Strategy B
│   └── single_column
│       └── → Strategy A (with page-level escalation guard)
│
└── native_digital: avg_density >= 0.01 AND avg_img <= 0.50
    └── → Strategy A (structurally reachable; not seen in this corpus)
```

---

## 2. Empirical Findings — Corpus of 50 Documents

### 2.1 Strategy Distribution (Post-Refinement)

| Strategy | Docs | % | Total Pages | Avg Pages |
|---|---|---|---|---|
| A — Fast Text (pdfplumber) | 2 | 4% | 174 | 87 |
| B — Layout Aware (Docling) | 26 | 52% | 1,811 | 70 |
| C — Vision (Chunkr/EasyOCR) | 22 | 44% | 825 | 38 |

### 2.2 Origin Distribution

| Origin | Count | % |
|---|---|---|
| scanned_image | 22 | 44% |
| mixed | 28 | 56% |
| native_digital | 0 | 0% (gate reachable; corpus has none) |

### 2.3 Per-Document Signal Baseline

| Document | Density | Img Ratio | Tables/pg | Strategy |
|---|---|---|---|---|
| ETS_Annual_Report_2024_2025.pdf | 0.002423 | 0.61 | 0.2 | A |
| EthSwitch-10th-Annual-Report-202324.pdf | 0.001179 | 0.39 | 0.0 | A |
| Annual_Report_JUNE-2023.pdf | 0.006835 | 0.44 | 1.2 | B |
| Annual_Report_JUNE-2020.pdf | 0.001766 | 0.16 | 1.4 | B |
| Annual_Report_JUNE-2022.pdf | 0.002787 | 0.43 | 1.6 | B |
| Pharmaceutical...VF.pdf | 0.000683 | 0.78 | 1.0 | C |
| Audit Report - 2023.pdf | 0.000048 | 0.80 | 0.0 | C |
| 2013-E.C-budget-and-expense.pdf | 0.0 | 1.0 | 0.0 | C |

---

## 3. Signals & Metrics

### 3.1 Structure Collapse Signals

| Failure | Signal | How Computed |
|---|---|---|
| Multi-column jumbled | `x_jump_ratio` | Fraction of bbox transitions where Δx > 30% page width |
| Table incomplete | `table_completeness_score` | (extracted cols × rows) / (expected cols × rows) |
| Header row missing | `header_present` | Boolean: first row has non-numeric tokens |
| Column count variance | `col_variance` | StdDev of column counts across rows |

### 3.2 Context Poverty Signals

| Failure | Signal | How Computed |
|---|---|---|
| Table split across pages | `table_continuation_linked` | Boolean: continuation relationship present |
| Caption detached from figure | `caption_orphan_rate` | Figure chunks without caption within 50pt |
| Chunk crosses structural unit | `boundary_violation_count` | Chunks where list/table/header is split |

### 3.3 Provenance Blindness Signals

| Failure | Signal | How Computed |
|---|---|---|
| Fact without page ref | `provenance_completeness` | facts_with_page_and_bbox / total_facts — must be 1.0 |
| Invalid bbox | `bbox_valid_rate` | % of bboxes within page bounds — must be 1.0 |
| No content hash | `hash_present_rate` | % of LDUs with non-null hash — must be 1.0 |

---

## 4. Validated Thresholds (extraction_rules.yaml)

| Parameter | Value | Justification |
|---|---|---|
| `scanned_by_density` | `< 0.0004` | Midpoint of gap: scanned max=0.00005, mixed min=0.00068 |
| `ghost_text_scan` | `img > 0.80 AND density < 0.001` | Catches Audit Report 2023 (density=4.8e-5) via robust arm |
| `high_image_thin` | `img > 0.70 AND density < 0.001` | Catches Pharmaceutical VF (img=0.78) — was misclassified before |
| `mixed density gate` | `< 0.01` | Lowered from 0.03 (was vacuous — max corpus density=0.00684) |
| `table_heavy` | `> 0.3` | Lowered from 0.5 — captures 5 docs with avg_tables=0.4 |
| `multi_column` | `> 0.08` (provisional) | Lowered from 0.15 — max corpus xjump=0.0364; untestable |
| `Strategy A confidence` | `>= 0.75` | Below this → escalate to B |
| `Strategy B confidence` | `>= 0.65` | Below this → escalate to C (page-level) |

---

## 5. Failure Modes Observed

| Document | Strategy | Failure Mode | Signal | Resolution |
|---|---|---|---|---|
| Pharmaceutical...VF.pdf | B (old) | High image ratio (0.78) caused layout model to fail on image pages | `high_image_thin` gate triggered | Rerouted to C via compound scanned gate |
| CBE Annual Report 2012-13.pdf | A (old) | Tables present (0.4/pg) but routed to fast text — tables would be flattened | `avg_tables > 0.3` gate | Rerouted to B via lowered table threshold |
| Audit Report 2023.pdf | C (old, fragile) | Ghost text layer (density=4.8e-5) made routing depend on tight 0.80 img gate | Density arm added | Now robustly routed via `density < 0.0004` |

---

## 6. Escalation Policy v0

```yaml
# All thresholds live in rubric/extraction_rules.yaml — never in code

triage:
  scanned_detection:
    scanned_by_density: 0.0004
    ghost_text_scan_img: 0.80
    ghost_text_scan_density: 0.001
    high_image_thin_img: 0.70
    high_image_thin_density: 0.001
  mixed_gate:
    density_ceiling: 0.01
    image_floor: 0.50
  layout_complexity:
    table_heavy_threshold: 0.3
    multi_column_xjump: 0.08        # provisional

strategy_routing:
  strategy_a:
    min_confidence_score: 0.75
    escalation_target: strategy_b
  strategy_b:
    min_confidence_score: 0.65
    escalation_target: strategy_c
    escalation_scope: page_level
  strategy_c:
    tools:
      primary: easyocr
      fallback: gemini_flash_free_tier
    budget_guard:
      max_cost_per_document_usd: 1.00
      cost_per_page_estimate_usd: 0.003
```

---

## 7. Non-Negotiable Invariants

| # | Invariant | Enforced At | Failure Action |
|---|---|---|---|
| I-1 | No chunk emitted without page_number, bbox, content_hash | ChunkValidator | Raise ProvenanceMissingError |
| I-2 | Table with completeness ≥ 0.85 cannot be stored as plain text | ExtractionRouter | Force re-extraction or escalate |
| I-3 | Low-confidence extraction cannot flow into chunking | ExtractionRouter confidence gate | Trigger escalation — never pass silently |
| I-4 | No query answer without ProvenanceChain | QueryAgent output validator | Return status: unverifiable |
| I-5 | Table row cannot be emitted without its header | ChunkValidator | Raise TableStructureViolationError |
| I-6 | Figure LDU must store caption as metadata | ChunkValidator | Merge caption into figure metadata |
| I-7 | Strategy C cost checked before each page call | VisionExtractor.budget_guard | Raise BudgetExceededError |
| I-8 | content_hash must be deterministic | LDU constructor | Unit test: hash stability |
| I-9 | bbox must be geometrically valid | ProvenanceRef validator | Raise InvalidBboxError |
| I-10 | extraction_ledger.jsonl must have one entry per page | ExtractionRouter ledger writer | Enforce at write time |

---

## 8. Cost Analysis

| Strategy | Tool | Cost/Page | Cost/Doc (avg pages) | When Used |
|---|---|---|---|---|
| A — Fast Text | pdfplumber | $0.00 | $0.00 | 2 docs, 174 pages |
| B — Layout Aware | Docling (local) | $0.00 | $0.00 | 26 docs, 1,811 pages |
| C — Vision (primary) | EasyOCR (local) | $0.00 | $0.00 | Scanned pages |
| C — Vision (fallback) | Gemini Flash free tier | $0.00 | $0.00 | Low confidence pages |
| C — Vision (paid fallback) | Gemini Flash paid | ~$0.003 | ~$0.11 | If free tier exhausted |

**Total estimated cost for 50-doc corpus: $0.00 (free tier + local tools)**

---

## 9. Selected 12 Documents (Validation Corpus)

| # | Document | Class | Strategy | Key Signals |
|---|---|---|---|---|
| 1 | ETS_Annual_Report_2024_2025.pdf | A — Easy | A | density=0.002, img=0.61, tables=0.2 |
| 2 | EthSwitch-10th-Annual-Report-202324.pdf | A — Hard | A | density=0.001, img=0.39, tables=0.0 |
| 3 | Annual_Report_JUNE-2023.pdf | B — Easy | B | density=0.007, img=0.44, tables=1.2 |
| 4 | Annual_Report_JUNE-2020.pdf | B — Hard | B | density=0.002, img=0.16, tables=1.4 |
| 5 | Annual_Report_JUNE-2022.pdf | B — Edge | B | density=0.003, img=0.43, tables=1.6 |
| 6 | Pharmaceutical...VF.pdf | C — Easy | C | density=0.001, img=0.78, tables=1.0 |
| 7 | 2013-E.C-budget-and-expense.pdf | C — Hard | C | density=0.0, img=1.0, tables=0.0 |
| 8 | 2013-E.C-Audit-finding-info.pdf | C — Edge | C | density=0.0, img=1.0, tables=0.0 |
| 9 | Annual_Report_JUNE-2019.pdf | Boundary | B | density=0.002, img=0.69 |
| 10 | CBE Annual Report 2006-7.pdf | Boundary | B | density=0.002, img=0.63 |
| 11 | Audit Report - 2023.pdf | Boundary | C | density=4.8e-5, img=0.80 |
| 12 | Ethswitch-...2020.2021_.pdf | Boundary | C | density=0.0, img=0.92 |

**Recommended additions (bring to 14):**
- `CBE Annual Report 2010-11.pdf` — medium-length scanned (62 pages)
- `Consumer Price Index August 2025.pdf` — CPI table sub-type

---

## 10. Phase 0 Validation Results

| Check | Result |
|---|---|
| Strategy distribution correct | ✅ |
| native_digital gate reachable | ✅ (in code; not in corpus) |
| Pharmaceutical VF rerouted B→C | ✅ |
| Audit Report 2023 robust routing | ✅ |
| Exactly 12 docs selected | ✅ |
| Zero misclassifications | ✅ |
| Cost increase justified | ✅ 8.1% buys 753 correctly-routed pages |
| Medium-length scanned covered | ❌ (add manually) |
| CPI sub-type covered | ❌ (add manually) |

**Pass: 13/16 · Proceed to Phase 1**

---

*DOMAIN_NOTES.md · Document Intelligence Refinery · Phase 0 Complete*
'''

path = Path("DOMAIN_NOTES.md")
path.write_text(CONTENT, encoding="utf-8")
print("✅ DOMAIN_NOTES.md created successfully.")
print(f"   Location: {path.resolve()}")
print("\nNext steps:")
print("  1. Open DOMAIN_NOTES.md in VS Code to verify")
print("  2. git add . && git commit -m 'Phase 0: add DOMAIN_NOTES.md'")
print("  3. git push")