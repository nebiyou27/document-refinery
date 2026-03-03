# Phase 0: Document Intelligence Refinery — Architecture & Reasoning Framework

> **Version:** 0.1-draft · **Date:** 2026-03-03 · **Status:** Phase 0 Design  
> **Scope:** Architectural blueprint for a confidence-gated, escalation-driven extraction pipeline processing 4 enterprise document classes across ~50 PDFs.

---

## Table of Contents

1. [Failure-Mode Detection Framework](#1-failure-mode-detection-framework)
2. [Cost-Quality Tradeoff Model](#2-cost-quality-tradeoff-model)
3. [Formal Escalation Policy](#3-formal-escalation-policy)
4. [Schema Contract Design](#4-schema-contract-design)
5. [What Must Be Impossible](#5-what-must-be-impossible)
6. [48-Hour Phase 0 Execution Plan](#6-48-hour-phase-0-execution-plan)

---

## 1. Phase 0 Measurable Signals

Phase 0 signals are **computable now**, before any extraction pipeline exists. They use only the raw PDF and a fast-text pass (PyMuPDF/pdfplumber). These signals drive the profiler and populate the escalation decision tree.

### 1.1 Per-Page Signals

| Signal | ID | Computation | What It Detects |
|--------|----|-------------|------------------|
| **Extractable character count** | `char_count` | `len(page.get_text())` via PyMuPDF | Zero → scanned/image page (Context Poverty) |
| **Text-layer presence** | `has_text_layer` | `char_count > 0` | False → must route to Strategy C |
| **Table-line heuristic** | `table_line_count` | Count horizontal + vertical ruling lines via `page.find_tables()` or line-detection geometry | >0 → page has tabular structure (Structure Collapse risk if extraction flattens) |
| **Font-size variance** | `font_size_std` | Std-dev of font sizes across all spans on the page | High variance → heading hierarchy present (Structure Collapse risk if lost) |
| **Page dimensions** | `page_w`, `page_h` | `page.rect.width`, `page.rect.height` in points | Non-standard sizes flag unusual layouts |
| **Image-to-page area ratio** | `image_area_ratio` | `sum(img_bbox_area) / page_area` for all embedded images | >0.5 → image-heavy page, likely needs OCR |

### 1.2 Per-Document Aggregates

| Signal | ID | Computation | Decision Gate |
|--------|----|-------------|---------------|
| **Scanned-page fraction** | `scanned_fraction` | `pages_with_char_count_0 / total_pages` | ≥0.80 → classify as scanned, skip Strategy A |
| **Table-page fraction** | `table_page_fraction` | `pages_with_table_lines / total_pages` | >0.50 → tabular-dominant document |
| **Mean char density** | `mean_char_density` | `mean(char_count / page_area)` across text-layer pages | Low density + text layer → possible multi-column or sparse layout |
| **Total pages** | `total_pages` | Page count | Cost estimation input |
| **File size** | `file_size_bytes` | Raw file size | Anomaly detection (e.g., 200-page doc at 50KB → likely broken) |
| **SHA-256 hash** | `source_file_hash` | `sha256(file_bytes)` | Deterministic `document_id` derivation |

### 1.3 Failure-Mode Mapping

| Failure Mode | Phase 0 Detection Signal | Trigger |
|-------------|--------------------------|----------|
| **Structure Collapse** | `table_line_count > 0 AND char_count > 0` | Page has tables + text layer → Strategy A may flatten; profile it for B escalation |
| **Context Poverty** | `char_count == 0 AND image_area_ratio > 0.1` | Non-blank page with no text → content is trapped in images |
| **Provenance Blindness** | Not measurable in Phase 0 | By design: provenance is an extraction-time invariant (§5), not a profiling signal |

---

## 2. Cost-Quality Tradeoff Model

### 2.1 Relative Cost Model (Per Page)

All costs are normalized against Strategy A as the 1.0× baseline.

| Strategy | Method | Relative Cost | Typical Latency/Page | Quality Ceiling | When It Excels |
|----------|--------|--------------|----------------------|-----------------|----------------|
| **A: Fast Text** | pdfplumber / PyMuPDF | **1.0×** | ~50ms | High for native digital PDFs | Clean digital PDFs with embedded text layers |
| **B: Layout-Aware** | Docling or MinerU | **3–5×** | ~200–500ms | High for complex layouts | Multi-column, deep table nesting, header/footer stripping |
| **C: Vision/OCR** | VLM (e.g. Qwen-VL) or EasyOCR | **8–15×** | ~1–3s | Highest for scanned/image docs | Scanned pages, mixed image+text, degraded quality |

**Cost Drivers Breakdown:**

| Cost Component | Strategy A | Strategy B | Strategy C |
|----------------|-----------|-----------|-----------|
| CPU compute | Low | Medium | High (GPU preferred) |
| Memory | ~50MB | ~200MB | ~2GB+ (model weights) |
| API cost (if cloud) | $0 | $0–0.005/page | $0.01–0.05/page |
| Implementation complexity | Low | Medium | High |

### 2.2 Decision Thresholds for Escalation

Escalation decisions are made per page, not per document, to minimize cost.

```
Page arrives → Run Strategy A
  IF structure_confidence ≥ T_struct AND context_confidence ≥ T_context:
    ACCEPT page result from A
  ELSE:
    RUN Strategy B on this page
    IF structure_confidence ≥ T_struct AND context_confidence ≥ T_context:
      ACCEPT page result from B
    ELSE:
      RUN Strategy C on this page
      ACCEPT page result from C (terminal strategy)
      FLAG page for human review if confidence still below thresholds
```

**Default threshold recommendations** (tunable per document class):

| Threshold | Default Value | Rationale |
|-----------|--------------|-----------|
| `T_struct` | 0.80 | Below this, table/layout data is unreliable for downstream consumers |
| `T_context` | 0.75 | Below this, semantic completeness is insufficient for entity extraction |
| `T_provenance` | 1.00 | Non-negotiable — every LDU must have provenance |
| `T_escalation_budget` | 0.30 | Max fraction of pages in a document that can escalate before document-level escalation triggers |

### 2.3 When NOT to Escalate

Escalation is wasteful or counterproductive in these cases:

1. **High CDR + Low TGCS on a known-narrative page.** If the page is prose-heavy (document class profile says `expected_tables = 0` for this page range), a low TGCS is irrelevant. **Don't escalate for table structure on a page with no tables.**

2. **Strategy A yields `structure_confidence ≥ 0.95`.** The marginal quality gain from B or C is negligible. The 3–15× cost multiplier is pure waste.

3. **Page is a cover page / table of contents / blank separator.** These pages have low information density by design. Escalation produces no incremental value. **Detect via CDR < 0.10 AND page_number ∈ known_boilerplate_ranges.**

4. **Budget ceiling reached.** If the per-document escalation budget (`T_escalation_budget`) is exhausted, escalate the entire document to Strategy B/C instead of escalating additional individual pages. This prevents the pathological case where 80% of pages escalate page-by-page (costing more than a blanket re-extraction).

5. **Document class is "fully scanned" and known a priori.** If the document profile classifier confidently tags a document as fully scanned (no text layer), skip A entirely and start at C. Don't pay for a guaranteed-to-fail Strategy A run.

---

## 3. Formal Escalation Policy (YAML Configuration)

```yaml
# escalation_policy.yaml
# ----------------------------------------------------------
# Configuration-driven escalation thresholds.
# All thresholds are overridable per document_class.
# ----------------------------------------------------------

version: "1.0"

global_defaults:
  # --- Confidence thresholds (0.0 – 1.0) ---
  structure_confidence_threshold: 0.80
  context_confidence_threshold: 0.75
  provenance_confidence_threshold: 1.00   # Hard invariant — never lower this

  # --- Cost controls ---
  max_page_escalation_fraction: 0.30      # If >30% pages escalate A→B, switch entire doc to B
  max_strategy_c_pages: 10                # Max pages sent to Strategy C before doc-level C trigger
  skip_strategy_a_if_scanned: true        # Bypass fast-text for known-scanned docs

  # --- Signal weights (structure) ---
  structure_weights:
    table_grid_consistency: 0.35
    column_bleed_ratio: 0.20
    heading_hierarchy_depth: 0.15
    line_order_monotonicity: 0.30

  # --- Signal weights (context) ---
  context_weights:
    character_density_ratio: 0.30
    sentence_completion_rate: 0.30
    entity_yield_ratio: 0.25
    cross_reference_resolution: 0.15

  # --- Boilerplate detection ---
  boilerplate_detection:
    min_char_density_for_content: 0.10
    known_boilerplate_page_types: ["cover", "toc", "blank", "separator"]

strategies:
  strategy_a:
    name: "Fast Text Extraction"
    tools: ["pdfplumber", "pymupdf"]
    cost_multiplier: 1.0
    timeout_per_page_ms: 5000

  strategy_b:
    name: "Layout-Aware Extraction"
    tools: ["docling", "mineru"]
    cost_multiplier: 4.0
    timeout_per_page_ms: 15000

  strategy_c:
    name: "Vision/OCR Extraction"
    tools: ["qwen_vl", "easyocr"]
    cost_multiplier: 12.0
    timeout_per_page_ms: 30000
    gpu_required: true

# --- Page-Level Escalation Rules ---
page_escalation_rules:
  - name: "structure_failure"
    condition: "structure_confidence < structure_confidence_threshold"
    action: "escalate_to_next_strategy"
    description: "Page layout extraction is unreliable"

  - name: "context_failure"
    condition: "context_confidence < context_confidence_threshold"
    action: "escalate_to_next_strategy"
    description: "Semantic content is incomplete"

  - name: "provenance_hard_failure"
    condition: "provenance_confidence < provenance_confidence_threshold"
    action: "escalate_to_next_strategy"
    severity: "critical"
    description: "LDU lacks spatial provenance — non-negotiable escalation"

  - name: "zero_text_extraction"
    condition: "character_density_ratio == 0.0 AND page_is_not_blank"
    action: "escalate_directly_to_strategy_c"
    description: "No text extracted from non-blank page → image-only, skip B"

# --- Document-Level Escalation Rules ---
document_escalation_rules:
  - name: "excessive_page_escalation"
    condition: "escalated_page_count / total_pages > max_page_escalation_fraction"
    action: "reprocess_entire_document_at_next_strategy"
    description: "Too many pages failing individually — cheaper to reprocess whole doc"

  - name: "scanned_document_bypass"
    condition: "document_profile.is_scanned == true AND skip_strategy_a_if_scanned"
    action: "start_at_strategy_c"
    description: "Known scanned doc skips text extraction entirely"

  - name: "strategy_c_page_cap"
    condition: "strategy_c_page_count > max_strategy_c_pages"
    action: "reprocess_entire_document_at_strategy_c"
    description: "Cap exceeded — reprocess entire doc at C to amortize overhead"

# --- Document Class Overrides ---
document_class_overrides:
  annual_financial_report:
    structure_confidence_threshold: 0.85   # Stricter — tables are critical
    structure_weights:
      table_grid_consistency: 0.45         # Tables matter more here
      column_bleed_ratio: 0.20
      heading_hierarchy_depth: 0.10
      line_order_monotonicity: 0.25

  scanned_legal_report:
    skip_strategy_a_if_scanned: true
    context_confidence_threshold: 0.70     # Lower bar — OCR is inherently noisier

  mixed_technical_report:
    structure_confidence_threshold: 0.80
    context_weights:
      cross_reference_resolution: 0.25     # Technical docs have heavy cross-refs

  fiscal_table_report:
    structure_confidence_threshold: 0.90   # Very strict — numerical accuracy is paramount
    structure_weights:
      table_grid_consistency: 0.50
      column_bleed_ratio: 0.15
      heading_hierarchy_depth: 0.05
      line_order_monotonicity: 0.30
```

---

## 4. Schema Contract Design

### 4.1 Core Entities

#### `DocumentProfile`

The classifier output that determines routing strategy before extraction begins.

```
DocumentProfile:
  document_id          : UUID                    # Deterministic, derived from file content hash
  source_file_hash     : SHA-256                 # Hash of original PDF bytes
  filename             : string
  total_pages          : int                     # ≥ 1
  document_class       : enum                    # annual_financial_report | scanned_legal_report
                                                 # | mixed_technical_report | fiscal_table_report
  classification_confidence : float [0.0, 1.0]
  is_scanned           : bool                    # True if ≥80% of pages have no text layer
  has_text_layer        : bool                    # True if any page has extractable text
  dominant_layout       : enum                    # single_column | multi_column | tabular | mixed
  expected_table_density : float [0.0, 1.0]      # Fraction of pages expected to contain tables
  language              : ISO-639-1               # Primary language detected
  created_at            : ISO-8601 timestamp
  metadata              : map<string, string>     # Extensible key-value pairs for new domains
```

**Invariants:**
- `total_pages ≥ 1`
- `document_id` is deterministic: same file always produces the same ID
- `source_file_hash` is computed before any processing begins and is immutable

---

#### `ExtractedDocument`

The top-level output of the pipeline for a single PDF.

```
ExtractedDocument:
  document_id            : UUID                  # Must match DocumentProfile.document_id
  profile                : DocumentProfile       # Embedded, not a foreign key
  extraction_timestamp   : ISO-8601 timestamp
  pipeline_version       : SemVer string         # e.g., "1.2.3"
  pages                  : list<PageResult>      # Ordered, 1-indexed
  aggregate_confidence   : ConfidenceVector
  escalation_summary     : EscalationSummary
  ldus                   : list<LDU>             # All LDUs across all pages, ordered
  warnings               : list<QualityWarning>
```

```
PageResult:
  page_number            : int                   # 1-indexed
  strategy_used          : enum [A, B, C]
  escalation_path        : list<enum [A, B, C]>  # e.g., [A, B] means A failed, B succeeded
  structure_confidence   : float [0.0, 1.0]
  context_confidence     : float [0.0, 1.0]
  provenance_confidence  : float [0.0, 1.0]
  ldu_refs               : list<UUID>            # References to LDUs on this page
  processing_time_ms     : int
```

```
ConfidenceVector:
  structure              : float [0.0, 1.0]      # Document-level aggregate (mean of pages)
  context                : float [0.0, 1.0]
  provenance             : float [0.0, 1.0]
  overall                : float [0.0, 1.0]      # Weighted combination
```

```
EscalationSummary:
  total_pages            : int
  pages_at_strategy_a    : int
  pages_at_strategy_b    : int
  pages_at_strategy_c    : int
  pages_flagged_for_review : int
  document_level_escalation : bool
  total_cost_units       : float                 # Sum of (pages × strategy cost_multiplier)
```

**Invariants:**
- `len(pages) == profile.total_pages`
- Every `page_number` in `pages` is unique and in `[1, total_pages]`
- `sum(pages_at_a + pages_at_b + pages_at_c) == total_pages`
- `document_id` matches across `ExtractedDocument` and its `DocumentProfile`

---

#### `LDU` (Logical Document Unit)

The atomic unit of extracted content. Every piece of information in the system is an LDU.

```
LDU:
  ldu_id                 : UUID
  document_id            : UUID                  # Parent document
  ldu_type               : enum                  # paragraph | table | heading | list_item
                                                 # | footnote | figure_caption | key_value_pair
  content                : string                # Extracted text content (UTF-8, normalized)
  content_hash           : SHA-256               # Hash of normalized content
  provenance             : ProvenanceRef         # REQUIRED — never null
  semantic_metadata      : map<string, any>      # Type-specific structured data
  ordering_index         : int                   # Global reading order within document
  parent_ldu_id          : UUID | null           # For hierarchical nesting (e.g., cell → table)
  children_ldu_ids       : list<UUID>            # Inverse of parent relationship
  confidence             : float [0.0, 1.0]      # Extraction confidence for this unit
  strategy_source        : enum [A, B, C]        # Which strategy produced this LDU
```

`semantic_metadata` examples by `ldu_type`:

| ldu_type | semantic_metadata keys |
|----------|----------------------|
| `table` | `row_count`, `col_count`, `has_header_row`, `merged_cells` |
| `heading` | `level` (1–6), `numbering` (e.g., "3.2.1") |
| `key_value_pair` | `key`, `value`, `unit` |
| `footnote` | `reference_marker`, `referenced_ldu_id` |

**Invariants:**
- `provenance` is **never null** — this is the single most important invariant in the system
- `content` is non-empty (whitespace-only content is rejected)
- `content_hash == sha256(normalize(content))` — always recomputable
- `ordering_index` values are unique within a document and form a contiguous sequence
- If `parent_ldu_id` is set, then `ldu_id ∈ parent.children_ldu_ids` (bidirectional consistency)

---

#### `ProvenanceRef`

The spatial anchor tying every LDU back to its source location.

```
ProvenanceRef:
  page_number            : int                   # 1-indexed, required
  bounding_box           : BoundingBox           # Required
  source_file_hash       : SHA-256               # Hash of original PDF — enables verification
  extraction_method      : enum [A, B, C]        # Strategy that produced this provenance
  raw_content_hash       : SHA-256               # Hash of raw bytes at this bbox in source
  confidence             : float [0.0, 1.0]      # Confidence of the spatial anchor itself
```

```
BoundingBox:
  x0                     : float                 # Left edge (points from page origin)
  y0                     : float                 # Top edge
  x1                     : float                 # Right edge
  y1                     : float                 # Bottom edge
  coordinate_system      : enum                  # pdf_points | pixels
  page_width             : float                 # For normalization
  page_height            : float                 # For normalization
```

**Invariants:**
- `x0 < x1 AND y0 < y1` (non-degenerate box)
- `x1 <= page_width AND y1 <= page_height` (within page bounds)
- `page_number ≥ 1`
- `source_file_hash` matches `DocumentProfile.source_file_hash`
- `raw_content_hash` is independently verifiable by re-extracting from the source PDF

---

### 4.2 Non-Negotiable Schema Invariants (Summary)

| # | Invariant | Enforcement Point |
|---|-----------|-------------------|
| 1 | Every LDU has a non-null `ProvenanceRef` | Schema validation + runtime assertion |
| 2 | Every `ProvenanceRef` has a valid `BoundingBox` | Schema validation |
| 3 | `document_id` is deterministic from file content | ID generation function |
| 4 | `content_hash` matches `sha256(normalize(content))` | Post-extraction validation |
| 5 | `source_file_hash` is immutable and set before processing | Pipeline entry point |
| 6 | `ordering_index` is unique and contiguous per document | Post-extraction validation |
| 7 | Parent-child LDU relationships are bidirectionally consistent | Post-extraction validation |
| 8 | `page_number` ∈ `[1, total_pages]` for all refs | Schema validation |
| 9 | `BoundingBox` is non-degenerate and within page bounds | Schema validation |
| 10 | `strategy_used` on each page matches the strategy that *actually produced the accepted output* | Pipeline orchestrator |

---

## 5. What Must Be Impossible

These are **architectural invariants** — conditions the system must structurally prevent, not merely discourage.

### 5.1 No Unprovenanced Facts

> **It must be impossible to persist an LDU without a valid ProvenanceRef.**

**Enforcement:** The `LDU` constructor (or factory/builder) requires `ProvenanceRef` as a non-optional argument. The persistence layer (database writer, serializer) rejects any LDU where `provenance is null` or `bounding_box` is degenerate. This is not a "best practice" — it is a schema-level constraint that makes invalid states unrepresentable.

**Risk if violated:** Downstream consumers cannot verify, audit, or debug any extracted fact. Regulatory compliance (SOX, GDPR data lineage) becomes impossible.

### 5.2 No Silent Quality Degradation

> **It must be impossible for a page to pass through the pipeline without a computed confidence score.**

**Enforcement:** The pipeline orchestrator computes `structure_confidence`, `context_confidence`, and `provenance_confidence` for every page *before* accepting the page result. The `PageResult` object cannot be constructed with null confidence fields. If confidence computation itself fails, the page is flagged as `ERROR` — it never silently enters the "accepted" pool.

**Risk if violated:** Bad extractions silently pollute the corpus. Consumers trust data that should have been escalated or flagged.

### 5.3 No Undocumented Escalation Decisions

> **It must be impossible for a page to be processed by Strategy B or C without an `escalation_path` entry explaining why.**

**Enforcement:** The escalation engine logs every decision with the triggering rule name, the computed confidence values, and the threshold that was breached. `PageResult.escalation_path` is populated by the escalation engine, not by the strategy executor. The path is append-only.

**Risk if violated:** Cost analysis becomes impossible. You cannot answer "why did we spend 12× on page 47?" without this trail.

### 5.4 No Configuration Drift Between Environments

> **It must be impossible to run the pipeline without an explicit, versioned escalation policy file.**

**Enforcement:** The pipeline entrypoint requires `--policy-file` as a mandatory argument. There is no hardcoded default policy. The policy file's SHA-256 hash is recorded in `ExtractedDocument.pipeline_version` metadata. Two runs with different policies produce different lineage records.

**Risk if violated:** Production uses different thresholds than staging. Extraction quality silently diverges across environments.

### 5.5 No Orphaned LDUs

> **It must be impossible for an LDU to exist outside the context of an `ExtractedDocument`.**

**Enforcement:** LDUs are created exclusively within a document extraction transaction. There is no public API to create standalone LDUs. The persistence layer enforces referential integrity: `ldu.document_id` must reference an existing `ExtractedDocument`.

**Risk if violated:** Orphaned LDUs create phantom data that cannot be traced, versioned, or invalidated.

### 5.6 No Non-Deterministic Document IDs

> **It must be impossible for the same PDF file to produce different `document_id` values across runs.**

**Enforcement:** `document_id = UUID5(namespace=REFINERY_NS, name=sha256(file_bytes))`. No timestamps, no random components. Re-processing the same file produces the same ID.

**Risk if violated:** Deduplication fails. The same document creates duplicate records. Downstream joins break.

### 5.7 No Untested Document Class Onboarding

> **It must be impossible to register a new document class without providing a validation fixture of ≥5 annotated pages.**

**Enforcement:** The document class registry requires a `fixtures/` directory with ground-truth annotations. The CI pipeline runs the full extraction + confidence computation on these fixtures as a gate. No fixture → no registration.

**Risk if violated:** A new domain enters production with unknown quality characteristics. Silent failures accumulate.

---

## 6. Phase 0 Execution Plan

Phase 0 is **profiling only** — no extraction pipeline, no schema implementation. The goal is to produce evidence artifacts that justify every downstream design decision.

### Step 1: Corpus Ingestion & Fingerprinting

| Action | Output | Location |
|--------|--------|----------|
| Drop 50 PDFs into `data/` | Raw corpus | `data/*.pdf` |
| Run corpus profiler | Per-document profile JSON | `.refinery/profiles/{document_id}.json` |
| | Page-level extraction ledger | `.refinery/extraction_ledger/ledger.jsonl` |
| | Summary CSV | `.refinery/corpus_summary.csv` |

The profiler computes all §1 signals using PyMuPDF (Strategy A tooling). No escalation, no extraction — just measurement.

### Step 2: Evidence-Driven Classification

Using the profiler outputs, manually or heuristically assign each document to a class:

| Heuristic | → Class |
|-----------|---------|
| `scanned_fraction ≥ 0.80` | `scanned_legal_report` |
| `table_page_fraction > 0.50 AND scanned_fraction < 0.20` | `fiscal_table_report` |
| `table_page_fraction ∈ [0.10, 0.50] AND font_size_std > 3.0` | `mixed_technical_report` |
| Remaining digital docs with low table density | `annual_financial_report` |

Record classification + confidence in each profile JSON. Override manually where heuristics fail.

### Step 3: Escalation Decision Tree (from Evidence)

The profiler output directly populates row-level routing decisions:

```
For each page in each document:
  ├─ char_count == 0 AND image_area_ratio > 0.1
  │   → ROUTE: Strategy C (OCR).  Reason: no text layer, non-blank.
  ├─ char_count > 0 AND table_line_count > 0
  │   → FLAG: Structure Collapse risk.  Candidate for Strategy B.
  ├─ char_count > 0 AND table_line_count == 0 AND font_size_std < 1.0
  │   → ROUTE: Strategy A (fast text).  Reason: simple prose, no tables.
  └─ char_count > 0 AND image_area_ratio > 0.5
      → FLAG: Mixed content.  Candidate for Strategy B or C.

For each document:
  ├─ scanned_fraction ≥ 0.80
  │   → ROUTE: Entire doc to Strategy C.  Skip A entirely.
  ├─ flagged_pages / total_pages > 0.30
  │   → ROUTE: Entire doc to Strategy B.  Page-by-page escalation too expensive.
  └─ Otherwise
      → ROUTE: Strategy A with page-level escalation.
```

### Step 4: Phase 0 Deliverables Checklist

| # | Deliverable | Format | Proves |
|---|-------------|--------|--------|
| 1 | Per-document profiles | `.refinery/profiles/*.json` | Every PDF is fingerprinted with measurable signals |
| 2 | Page-level ledger | `.refinery/extraction_ledger/ledger.jsonl` | Every page has signals; no page is unmeasured |
| 3 | Corpus summary | `.refinery/corpus_summary.csv` | Corpus-wide statistics for threshold calibration |
| 4 | Classification assignments | Embedded in profile JSONs | Each doc is assigned a class with justification |
| 5 | Escalation decision tree | This document (§6.3) | Routing logic is evidence-based, not assumed |
| 6 | Risk flags | `warnings` field in profiles | Documents/pages that need manual review are identified |

### Phase 0 Exit Criteria

- [ ] All 50 PDFs have profile JSONs in `.refinery/profiles/`
- [ ] Ledger JSONL has one entry per page across all documents
- [ ] Summary CSV is loadable and has one row per document
- [ ] Every document has a `document_class` assignment
- [ ] At least one document per class is identified
- [ ] Scanned documents are correctly flagged (`scanned_fraction ≥ 0.80`)
- [ ] Table-heavy documents are correctly flagged (`table_page_fraction > 0.50`)
