# CS4099 — Gen AI Technical Debt Study

Research pipeline for studying whether generative AI tools introduce technical debt. Developer comments referencing AI tools (ChatGPT, Copilot, Claude, etc.) are sampled from real commits, enriched with code context, and manually annotated.

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Create `.env` in the project root:

```
GITHUB_TOKEN=your_github_token
GROQ_API_KEY=your_groq_key      # only needed for LLM inter-rater annotations
```

---

## Workflow

### 1. Prepare the dataset

Samples ~500 records from `combined_whole_dataset_per_comment.json`, fetches GitHub context (diffs, commit history, file content), computes cyclomatic complexity, and finds non-AI baseline commits for each record.

```bash
python -m annotate_tech_debt --stage prepare
```

Outputs to `pipeline_output/prepared_dataset.json`. Takes ~1 hour (GitHub API). Resume if interrupted:

```bash
python -m annotate_tech_debt --stage prepare --resume
```

### 2. Annotate manually (web UI)

```bash
python -m annotate_tech_debt.annotation_server
# → http://localhost:5000
```

The UI shows each record's code diff, complexity metrics, and commit context. Fill in the annotation form and click **Save & Next**. Annotations are saved back to `prepared_dataset.json` after each record.

Use **✦ Suggest with LLM** to auto-populate the form via Groq or Ollama — review each field before saving (for inter-rater reliability, not primary annotation).

### 3. Generate the report

```bash
python -m annotate_tech_debt --stage annotate --skip-failed
```

Generates `pipeline_output/annotation_report.md` with statistical comparisons (Wilcoxon signed-rank, effect sizes), per-category breakdowns, and a full section per record.

---

## Project Structure

```
annotate_tech_debt/
├── config.py              — constants, paths, API keys
├── models.py              — dataclasses (AnnotatedRecord, TechDebtAnnotation, …)
├── data_selector.py       — stratified sampling
├── github_context.py      — GitHub API: diffs, history, baselines
├── diff_parser.py         — unified diff → before/after code
├── complexity.py          — cyclomatic complexity via lizard
├── groq_annotator.py      — LLM prompt + annotation (Groq / Ollama)
├── pattern_analyzer.py    — statistical analysis
├── report_generator.py    — markdown report generation
├── annotation_server.py   — manual annotation web UI (Flask)
└── main.py                — CLI entry point
pipeline_output/           — all generated files (gitignored)
```

---

## Annotation Schema

Each record is annotated with:

| Field | Type | Description |
|-------|------|-------------|
| `debt_categories` | list | Up to 7 categories (complexity, testing, hallucination, etc.) |
| `overall_assessment` | positive / negative / mixed | — |
| `code_quality_score` | 1–5 | — |
| `ai_suggestion_quality` | 1–5 | Quality of the AI-generated code itself |
| `developer_awareness` | aware / partially_aware / unaware | Did the developer notice issues? |
| `confidence` | 0–1 | Annotator confidence |
| `no_debt_detected` | bool | Set when the AI change is clean |
