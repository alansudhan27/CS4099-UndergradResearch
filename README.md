# Gen AI Technical Debt Annotation Pipeline

An automated pipeline for analysing technical debt introduced by generative AI tools (ChatGPT, GitHub Copilot, Claude, etc.) in real-world codebases. The system samples developer comments that reference AI tools, fetches surrounding code context from GitHub, and uses an LLM annotator to classify the type, severity, and developer awareness of any technical debt introduced.

---

## What It Does

1. **Prepare** — Stratified-samples ~500 records from the dataset, fetches GitHub context (file content, commit history, diffs), computes cyclomatic complexity metrics, and finds non-AI "baseline" commits for statistical comparison.
2. **Annotate** — Sends each prepared record to an LLM (Groq cloud or a local Ollama model), which classifies the technical debt across 7 categories and rates code quality, AI suggestion quality, and developer awareness.
3. **Report** — Produces a single Markdown report with statistical comparisons (Wilcoxon signed-rank, sign test), per-category breakdowns, and a detailed section for every annotated record showing the real code diff, before/after complexity metrics, and AI analysis.

---

## Codebase Overview

```
annotate_tech_debt/
├── config.py           — All constants, paths, API keys, model names
├── models.py           — Dataclasses: TechDebtAnnotation, AnnotatedRecord, ComplexityMetrics, etc.
├── data_selector.py    — Stratified sampling with proportional allocation and reserve pools
├── github_context.py   — GitHub REST API calls: file content, commit history, baseline fetching
├── diff_parser.py      — Unified diff parsing into before/after code blocks
├── complexity.py       — Cyclomatic complexity via lizard
├── groq_annotator.py   — LLM prompt construction and annotation (Groq + Ollama backends)
├── pattern_analyzer.py — Statistical analysis: Wilcoxon, sign test, per-category aggregations
├── report_generator.py — Markdown report generation
└── main.py             — CLI entry point: --stage prepare | annotate
```

**Dataset:** `combined_whole_dataset_per_comment.json` — each record is a developer comment referencing an AI tool, with the associated commit diff and metadata.

**Outputs** (all written to `pipeline_output/`):

| File | Stage | Description |
|------|-------|-------------|
| `sampled_dataset.json` | prepare | Sampled records in original schema (reusable dataset) |
| `prepared_dataset.json` | prepare | Enriched records with GitHub context, baselines, complexity |
| `prepared_dataset.csv` | prepare | Flat CSV for R/pandas analysis |
| `preparation_summary.md` | prepare | Sampling stats, API usage, data quality report |
| `annotation_report.md` | annotate | Full analysis report with statistical tests and per-record details |

---

## Prerequisites

- Python 3.11+
- A Groq API key (free tier works) — or a locally running Ollama instance
- A GitHub personal access token (for the GitHub REST API)
- The dataset file: `combined_whole_dataset_per_comment.json` in the project root

---

## Setup

### 1. Clone / navigate to the project

```bash
cd CS4099-UndgraduateResearch
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Create a `.env` file in the project root (if it doesn't already exist):

```
GROQ_API_KEY=your_groq_api_key_here
GITHUB_TOKEN=your_github_token_here
```

- **Groq API key:** Sign up at [console.groq.com](https://console.groq.com) → API Keys → Create key. Free tier provides sufficient quota.
- **GitHub token:** Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new token. No special scopes are needed — a token with no extra permissions is fine (it just raises the rate limit from 60 to 5,000 requests/hour).

To use a local Ollama model instead of Groq for annotation, you can also set:

```
OLLAMA_BASE_URL=http://localhost:11434/v1   # default, only needed to override
OLLAMA_MODEL=deepseek-r1:70b               # or any model installed in Ollama
```

---

## Running the Pipeline

The pipeline has two stages that must be run in order.

### Stage 1 — Prepare

Samples the dataset, fetches GitHub context for each record, and produces the enriched dataset used for annotation.

```bash
python -m annotate_tech_debt --stage prepare
```

**What happens:**

1. Loads `combined_whole_dataset_per_comment.json`
2. Runs stratified sampling: ~500 primary records (proportional by AI-use category, minimum 25 per category) + 40 exploratory records from uncategorised comments
3. For each record, fetches from GitHub:
   - Full file content at the commit SHA
   - Last 50 commits touching that file
   - The patch for up to 15 candidate baseline (non-AI) commits
4. Computes cyclomatic complexity (before/after the AI commit) using lizard
5. Selects the best 2 baseline commits per record by patch-size similarity
6. Records with zero baselines are rejected and replaced from a reserve pool
7. Saves all four output files to `pipeline_output/`

**Expected runtime:** 45–90 minutes for ~500 records (dominated by GitHub API calls; the free-tier rate limit of 5,000/hour is the bottleneck).

**Progress is saved after every record.** If the run is interrupted, resume it with:

```bash
python -m annotate_tech_debt --stage prepare --resume
```

The `--resume` flag reads `pipeline_output/.prepare_progress.json` and skips already-fetched records.

---

### Stage 2 — Annotate

Reads the prepared dataset and sends each record to an LLM for technical debt classification.

#### Using Groq (default)

```bash
python -m annotate_tech_debt --stage annotate
```

The default rate limit is 30 requests/minute (Groq free tier). To use a higher limit if your account allows:

```bash
python -m annotate_tech_debt --stage annotate --rpm 60
```

**Expected runtime:** ~17 minutes at 30 rpm for 500 records.

#### Using a local Ollama model

Make sure Ollama is running locally with your chosen model pulled:

```bash
ollama pull deepseek-r1:70b
ollama serve   # if not already running
```

Then run:

```bash
python -m annotate_tech_debt --stage annotate --backend ollama
```

To use a different Ollama model:

```bash
OLLAMA_MODEL=llama3.3:405b python -m annotate_tech_debt --stage annotate --backend ollama
```

**Expected runtime:** Varies by hardware and model size. No rate limiting is applied for local models.

---

### Resuming and Recovering

Annotation progress is saved after every record. If the run is interrupted:

```bash
python -m annotate_tech_debt --stage annotate --resume
```

To continue even if some records keep failing:

```bash
python -m annotate_tech_debt --stage annotate --resume --skip-failed
```

To re-annotate specific records by ID (useful for spot-checking or reprocessing failures):

```bash
python -m annotate_tech_debt --stage annotate --ids record_id_1 record_id_2
```

---

## Full CLI Reference

```
python -m annotate_tech_debt --stage <prepare|annotate> [options]

Options:
  --stage prepare|annotate   Required. Which stage to run.
  --resume                   Resume from last checkpoint (both stages).
  --rpm N                    Groq requests per minute (annotate only, default: 30).
  --skip-failed              Generate report even if some annotations failed.
  --ids ID [ID ...]          Re-annotate specific record IDs only.
  --backend groq|ollama      LLM backend to use (annotate only, default: groq).
```

---

## Typical Full Run

```bash
# 1. Prepare the dataset (~1 hour)
python -m annotate_tech_debt --stage prepare

# 2. Annotate with Groq (~17 min at 30 rpm)
python -m annotate_tech_debt --stage annotate --rpm 30

# 3. View the report
open pipeline_output/annotation_report.md
```

After both stages complete, `pipeline_output/annotation_report.md` contains:

- Executive summary with key findings
- Statistical comparison of AI commits vs baseline commits (Wilcoxon signed-rank + sign test, effect sizes)
- Per-category breakdown by AI-use type and comment type
- Language breakdown (Python vs JavaScript)
- Data quality metrics
- Individual record details with real code diffs, complexity tables, evidence citations, and baseline comparisons
#   C S 4 0 9 9 - U n d e r g r a d R e s e a r c h  
 