# Document Intelligence Refinery

> A production-grade, multi-stage agentic pipeline that ingests heterogeneous enterprise documents and emits structured, queryable, spatially-indexed knowledge.

Built for the **TRP-1 Week 3 Challenge** — Forward Deployed Engineer Program.

---

## What It Does

Most enterprise data is locked inside PDFs — scanned reports, financial statements, legal documents, slide decks. Traditional tools either destroy the structure or hallucinate when given raw text dumps.

This pipeline solves that by:

- **Classifying** every document before touching it (scanned vs digital vs mixed)
- **Extracting** content using the right tool for each page — not one-size-fits-all
- **Chunking** into semantically coherent units that preserve tables, captions, and reading order
- **Indexing** with a smart navigation tree so retrieval finds the right section first
- **Answering** natural language questions with exact page + bounding box citations

---

## Pipeline Architecture

Detailed Phase 2 diagram: [Architecture Overview](docs/architecture_overview.md).


```
Raw Document
     │
     ▼
┌─────────────────┐
│  Stage 1        │  Triage Agent
│  Document       │  Classifies origin, layout, domain
│  Profiler       │  Outputs: DocumentProfile
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stage 2        │  Multi-Strategy Extraction
│  Extraction     │  Strategy A → B → C (escalating)
│  Router         │  Outputs: ExtractedDocument
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stage 3        │  Semantic Chunking Engine
│  Chunking       │  Preserves tables, captions, lists
│  Engine         │  Outputs: List[LDU]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stage 4        │  PageIndex Builder
│  Navigation     │  Smart table of contents
│  Index          │  Outputs: PageIndex tree
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Stage 5        │  Query Interface Agent
│  Query          │  Answers with page + bbox citations
│  Agent          │  Outputs: Answer + ProvenanceChain
└─────────────────┘
```

---

## Extraction Strategies

The system uses three strategies and escalates automatically:

| Strategy | Tool | Cost | When Used |
|---|---|---|---|
| **A — Fast Text** | pdfplumber | Free | Clean digital, single-column |
| **B — Layout Aware** | Docling | Free (local) | Multi-column, table-heavy, mixed |
| **C — Vision** | EasyOCR + Gemini Flash | Free | Scanned docs, low-confidence pages |

**Key principle:** Document-level routing is a best guess. Page-level escalation is the correctness guarantee.

---

## Setup

### Requirements

- Python 3.10+
- Git
- 4GB+ RAM (for Docling local model)

### 1. Clone the repository

```bash
git clone https://github.com/nebiyou27/document-refinery.git
cd document-refinery
```

### 2. Create and activate virtual environment

```bash
# Create
python -m venv venv

# Activate — Mac/Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

### 4. Set up API keys

Copy the template and fill in your keys:

```bash
# Open .env and add your keys
```

```env
GOOGLE_API_KEY=your_gemini_key_here       # free at aistudio.google.com
CHUNKR_API_KEY=your_chunkr_key_here       # free at chunkr.ai
ANTHROPIC_API_KEY=your_anthropic_key_here # optional
OPENROUTER_API_KEY=your_openrouter_key_here # optional
```

### 5. Add your documents

```bash
# Drop your PDF files into the data/ folder
cp /path/to/your/documents/*.pdf data/
```

### 6. Verify setup

```bash
python verify_installs.py
```

Expected output:
```
✅ pdfplumber working
✅ pymupdf working
✅ pydantic working
✅ spacy working
🎉 Setup complete. Ready to build.
```

---

## Usage

### Phase 0 — Profile your corpus

Run this first on any new document set to measure signals and select your validation sample:

```bash
python profile_corpus.py
```

Output:
- `phase0_signals.csv` — signals for every document
- `phase0_selected_12.csv` — recommended 12 documents for validation

### Run escalation guard on a single document

```bash
python src/agents/escalation_guard.py data/YOUR_DOCUMENT.pdf
```

This shows per-page signals, confidence scores, and strategy decisions:

```
Page  Chars   Density    ImgRatio   Tables  Confidence  Strategy
1     1823    0.09123    0.12000    0       0.90        strategy_a
2     0       0.00001    0.94000    0       0.30        strategy_c
3     654     0.03200    0.48000    3       0.65        strategy_b

── Summary ──────────────────────────
Strategy A:  1 pages  $0.00
Strategy B:  1 pages  $0.00
Strategy C:  1 pages  ~$0.003
```

### Run Phase 4 on a single document

```bash
venv\Scripts\python.exe scripts/run_phase4.py data/2013-E.C-Assigned-regular-budget-and-expense.pdf --save-artifacts
```

Optional query and claim verification:

```bash
venv\Scripts\python.exe scripts/run_phase4.py data/2013-E.C-Assigned-regular-budget-and-expense.pdf \
  --topic "budget totals" \
  --claim "The report states revenue was $4.2B in Q3" \
  --save-artifacts
```

Phase 4 emits:
- retrieval-backed answers with `ProvenanceChain`
- Audit Mode results for generated answers
- direct claim verification results: either citation-backed `verified` or `unverifiable`
- `fact_table.json` with numeric facts and provenance for table-heavy documents

### Run tests

```bash
pytest tests/ -v
```

---

## Project Structure

```text
document-refinery/
|-- src/
|   |-- agents/
|   |-- models/
|   |-- strategies/
|   `-- utils/
|-- data/                              # Source PDF corpus
|-- tests/
|-- rubric/
|   `-- extraction_rules.yaml          # Extraction/routing rules
|-- .refinery/
|   |-- chunks/                        # Generated chunk outputs
|   |-- extraction_ledger/             # Per-page strategy decisions
|   |-- pageindex/                     # Generated navigation indices
|   `-- profiles/                      # Document profile outputs
|-- docs/
|   |-- architecture_overview.md
|   |-- phase0_corpus_analysis.md
|   `-- phase0_post_refinement_validation.md
|-- create_domain_notes.py             # Domain notes generation script
|-- profile_corpus.py                  # Corpus profiling script
|-- DOMAIN_NOTES.md
|-- .gitignore
`-- README.md
```
---

## Configuration

All routing thresholds live in `rubric/extraction_rules.yaml`.
Change values there — never hardcode them in Python files.

For local OCR smoke tests via `scripts/run_tesseract_ocr.py`, keep `--psm` explicit and optional.
Recommended default: `--psm 3`.
Optional alternative: `--psm 6` for some dense, block-like pages.
For the table-heavy Amharic page `data/2013-E.C-Assigned-regular-budget-and-expense.pdf` page 1 with `lang=amh+eng` and `preprocess=threshold`, observed mean confidence was higher with `--psm 3` than with `--psm 6`.

Key thresholds (empirically validated on 50-document corpus):

```yaml
triage:
  scanned_detection:
    scanned_by_density:      0.0004   # below → pure scanned
    ghost_text_scan_img:     0.80     # high image + thin text
    high_image_thin_img:     0.70     # borderline scanned

  layout_complexity:
    table_heavy_threshold:   0.3      # tables/page above → Strategy B
    multi_column_xjump:      0.08     # provisional

strategy_routing:
  strategy_a:
    min_confidence_score:    0.75     # below → escalate to B
  strategy_b:
    min_confidence_score:    0.65     # below → escalate to C
    escalation_scope:        page_level
```

---

## Non-Negotiable Invariants

The system enforces these hard constraints. Violations raise exceptions — never silent failures:

| # | Rule |
|---|---|
| I-1 | No chunk emitted without `page_number` + `bbox` + `content_hash` |
| I-2 | Tables with completeness ≥ 0.85 cannot be stored as plain text |
| I-3 | Low-confidence extraction cannot flow into chunking without escalation |
| I-4 | No query answer returned without a `ProvenanceChain` |
| I-5 | Table row cannot be emitted without its header row |
| I-9 | `bbox` must be geometrically valid — all values within page bounds |
| I-11 | Corrupted PDF emits ERROR profile — never returns `None` silently |

---

## Document Classes Supported

| Class | Type | Challenge | Example |
|---|---|---|---|
| A | Native digital PDF | Multi-column, embedded tables | CBE Annual Report |
| B | Scanned image PDF | No text layer — pure OCR | DBE Audit Report |
| C | Mixed PDF | Narrative + tables + figures | FTA Assessment Report |
| D | Table-heavy PDF | Multi-year fiscal data tables | Tax Expenditure Report |

---

## Cost

| Scenario | Cost |
|---|---|
| Full 50-doc corpus (local tools only) | **$0.00** |
| Per document average | **$0.00** |
| If Gemini free tier exhausted | ~$0.003/page (Strategy C only) |

All extraction tools (pdfplumber, Docling, EasyOCR) run locally.
Gemini Flash is only called as a fallback for low-confidence scanned pages,
within a free tier of 1,500 calls/day.

---

## Phase Status

| Phase | Description | Status |
|---|---|---|
| **Phase 0** | Domain onboarding, corpus profiling, threshold validation |
| **Phase 1** | Triage Agent — document classifier |
| **Phase 2** | Multi-strategy extraction + escalation router |
| **Phase 3** | Semantic chunking engine + PageIndex builder |
| **Phase 4** | Query agent + provenance layer + audit mode + fact table |

---

## Phase 4 Behavior

The integrated Phase 4 entrypoint is `scripts/run_phase4.py`.

It runs:
- chunking and PageIndex summarization
- vector retrieval and query answering
- provenance assembly
- Audit Mode
- FactTable extraction for numeric tables

Audit Mode supports two checks:
- Answer audit: verifies that a generated answer is grounded in retrieved provenance snippets
- Claim verification: given a claim such as `The report states revenue was $4.2B in Q3`, the system must either return `verified` with source citations or `unverifiable`

When `--save-artifacts` is enabled, the output directory includes:
- `extracted_document.json`
- `ldus.json`
- `chunks.json`
- `page_index.json`
- `fact_table.json`
- `phase4_report.json`

---

## References

- [MinerU](https://github.com/opendatalab/MinerU) — PDF extraction pipeline
- [Docling](https://github.com/DS4SD/docling) — Enterprise document understanding
- [Chunkr](https://github.com/lumina-ai-inc/chunkr) — RAG-optimized chunking
- [PageIndex](https://github.com/VectifyAI/PageIndex) — Document navigation index
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) — Free local OCR
- [ChromaDB](https://github.com/chroma-core/chroma) — Local vector store

---

## Author

**Nebiyou** · TRP-1 FDE Program · Week 3

