# Document Refinery

A multi-stage pipeline for PDF intelligence:
1. Profile and classify documents.
2. Extract page content with adaptive strategies.
3. Build semantic chunks and a PageIndex tree.
4. Run retrieval-backed Q&A with provenance and audit checks.

Built for TRP-1 Week 3.

## Challenge Alignment
- End-to-end agentic document pipeline: implemented.
- Reproducible CLI flow: implemented via `scripts/` entrypoints.
- Dockerfile (recommended): implemented in the project root.

## Architecture
- Stage 1: Triage/profile (`src/agents/triage.py`, `profile_corpus.py`)
- Stage 2: Extraction router (`src/agents/extractor.py`)
  - Strategy A: `pdfplumber` fast text extraction
  - Strategy B: Docling layout-aware extraction
  - Strategy C: OCR + VLM fallback for hard pages
- Stage 3: Chunking + PageIndex (`src/chunking/`)
- Stage 4: Query + provenance + audit + fact table (`src/agents/phase4_pipeline.py`)

Reference design notes: [docs/architecture_overview.md](docs/architecture_overview.md)

## Repository Layout
```text
document-refinery/
|-- src/
|   |-- agents/
|   |-- chunking/
|   |-- models/
|   |-- ocr/
|   |-- storage/
|   |-- strategies/
|   `-- utils/
|-- scripts/
|   |-- run_extract.py
|   |-- run_phase4.py
|   |-- run_tesseract_ocr.py
|   |-- eval_phase3_batch.py
|   `-- test_one_pdf.py
|-- tests/
|-- docs/
|-- rubric/
|   `-- extraction_rules.yaml
|-- data/
|-- .refinery/
|-- debug_runs/
|-- requirements.txt
|-- Dockerfile
`-- README.md
```

## Requirements
- Python 3.10+
- Git
- Optional but recommended for default Phase 4 mode:
  - Ollama running locally (`http://localhost:11434`)
  - Models such as `qwen3:1.7b` and `qwen3-embedding:0.6b`
- Optional for OCR smoke tests outside Docker:
  - Tesseract OCR

## Setup (Local)
```bash
git clone <your-repo-url>
cd document-refinery
python -m venv venv
```

Activate environment:
- Windows PowerShell:
```powershell
venv\Scripts\Activate.ps1
```
- macOS/Linux:
```bash
source venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Profile corpus (Phase 0)
```bash
python profile_corpus.py
```
Outputs:
- `phase0_signals.csv`
- `phase0_selected_12.csv`

### 2. Run extraction (Stage 2)
```bash
python scripts/run_extract.py data/YOUR_DOCUMENT.pdf
```
Optional strategy override:
```bash
python scripts/run_extract.py data/YOUR_DOCUMENT.pdf --strategy strategy_b
```

Artifacts are written under `.refinery/` (including extraction ledger and extracted JSON).

### 3. Run integrated Phase 4 pipeline
Default mode uses Ollama-backed summarization and embeddings:
```bash
python scripts/run_phase4.py data/YOUR_DOCUMENT.pdf --save-artifacts
```

If you do not want Ollama for a local deterministic smoke run:
```bash
python scripts/run_phase4.py data/YOUR_DOCUMENT.pdf --summary-backend heuristic --embedding-backend hash --save-artifacts
```

Add retrieval topics and claims:
```bash
python scripts/run_phase4.py data/YOUR_DOCUMENT.pdf --topic "budget totals" --topic "year over year change" --claim "Revenue increased in Q3" --save-artifacts
```

Phase 4 artifact directory (default):
- `debug_runs/<pdf_stem>_phase4/`
- files include:
  - `extracted_document.json`
  - `ldus.json`
  - `chunks.json`
  - `page_index.json`
  - `fact_table.json`
  - `fact_table.sqlite`
  - `phase4_report.json`

### 4. OCR smoke test utility
```bash
python scripts/run_tesseract_ocr.py data/YOUR_DOCUMENT.pdf --first-page 1 --last-page 1 --psm 3
```

### 5. Batch Phase 3 evaluation
```bash
python scripts/eval_phase3_batch.py --csv-path phase0_selected_12.csv --data-dir data
```

## Testing
```bash
pytest tests/ -v
```

## Configuration
Routing and escalation thresholds are configured in:
- `rubric/extraction_rules.yaml`

Change values in config, not in code.

## Docker (Recommended)
Build image:
```bash
docker build -t document-refinery:latest .
```

Run Phase 4 (Linux/macOS shell):
```bash
docker run --rm -v "$(pwd)/data:/app/data" -v "$(pwd)/debug_runs:/app/debug_runs" document-refinery:latest python scripts/run_phase4.py data/YOUR_DOCUMENT.pdf --summary-backend heuristic --embedding-backend hash --save-artifacts
```

Run Phase 4 (PowerShell):
```powershell
docker run --rm -v "${PWD}/data:/app/data" -v "${PWD}/debug_runs:/app/debug_runs" document-refinery:latest python scripts/run_phase4.py data/YOUR_DOCUMENT.pdf --summary-backend heuristic --embedding-backend hash --save-artifacts
```

Run tests in container:
```bash
docker run --rm document-refinery:latest pytest tests/ -v
```

## Notes
- `scripts/run_phase4.py` defaults to Ollama backends. If Ollama is not running, use `--summary-backend heuristic --embedding-backend hash`.
- Strategy C depends on OCR/VLM fallback paths for low-confidence pages.
- Generated runtime artifacts are expected under `.refinery/` and `debug_runs/`.
