# CS4099 - Gen AI Technical Debt Study

## Setup

1. Create and activate a virtual environment.

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install dependencies.

```bash
pip install -r requirements.txt
```

3. Create a .env file in the project root.

```env
GITHUB_TOKEN=your_github_token

# Optional (for automatic annotation)
GROQ_API_KEY=your_groq_key
```

## Process: Filtering and Annotating the Data

### 1. Filter and prepare records

Run:

```bash
python -m tech_debt_pipeline --stage prepare
```

This stage:

- loads records from combined_whole_dataset_per_comment.json
- filters and samples records for the study
- fetches GitHub context (diffs, file snippet, commit history)
- finds baseline (non-AI) commits for comparison
- computes complexity metrics
- writes prepared records to pipeline_output/prepared_dataset.json

Detailed logic used in this stage:

1. Initial filtering (candidate quality control)

- Exclude if rq1a label is False Positive.
- Exclude if rq1b label is False Positive.
- Exclude if rq1b label is Generic Comment.
- Exclude if comment length is shorter than 10 characters.
- Exclude if first commit patch is missing or shorter than 20 characters.

2. Sampling strategy (how records are selected)

- Remaining candidates are grouped by rq1a predicted label.
- Categories Unspecified and Unknown are treated as exploratory pool.
- All other rq1a categories are primary pool.
- Target sizes are set in config:
	- primary target: 500
	- exploratory target: 40
	- minimum per primary category: 25
- Per-category primary allocation is proportional to category size, then balanced to match the overall target.
- Within each category, records are ranked using this priority key:
	- has subsequent commits (preferred)
	- moderate patch size, defined as 200 to 10000 chars (preferred)
	- higher rq1a confidence (preferred)
- Repo uniqueness is enforced during selection, so one repository does not dominate the sample.
- Records not selected are kept as reserves per category.

3. What context is fetched from GitHub

For each selected record:

- File content at the first commit SHA (if available).
- A file snippet around the annotated line (plus/minus 50 lines).
- Commit history for the same file path (up to 50 commits).
- If API history is unavailable, fallback to commit data already stored in the dataset.

4. How baseline commits are found

Baselines are historical commits from the same file, intended as non-AI comparison commits.

- Candidate baseline commit must be older than the AI commit date.
- Candidate is rejected if commit message matches AI keywords such as copilot, chatgpt, claude, openai, gpt-4, llm, and related variants.
- Up to 15 baseline candidates are fetched per record.
- Candidate is rejected if patch is missing/empty.
- Candidate is rejected if patch size differs too much from AI patch:
	- size_ratio = max(ai_patch_size, baseline_patch_size) / min(ai_patch_size, baseline_patch_size)
	- reject when size_ratio > 5
- Remaining candidates are scored by size similarity:
	- score = 1 / (1 + abs(log2(patch_size / ai_patch_size)))
	- plus 0.2 bonus if structural change is detected (function/class style declarations in diff lines)
- Top scored baselines are kept (up to 2).
- A prepared record must have at least 1 baseline. If not, that record is rejected and replaced using reserve records from the same category.

5. Complexity metrics computed

Complexity is computed with lizard using reconstructed before and after code from the patch.

Per side (before and after), the pipeline stores:

- function-level list with:
	- function name
	- cyclomatic complexity
	- NLOC (lines of code)
	- parameter count
	- start line
- aggregate metrics:
	- total_nloc
	- avg_complexity
	- max_complexity
	- num_functions

Delta metrics are then computed:

- delta_avg = after.avg_complexity - before.avg_complexity
- delta_max = after.max_complexity - before.max_complexity
- delta_nloc = after.total_nloc - before.total_nloc
- delta_functions = after.num_functions - before.num_functions
- complexity_increased = after.avg_complexity > before.avg_complexity

Line-change metrics are also extracted from the patch:

- lines_added
- lines_removed

6. Data written to prepared_dataset.json

Each prepared record includes:

- original dataset fields
- parsed diff (added/removed/context lines plus reconstructed before/after)
- GitHub context (file snippet/full content availability, commit history)
- complexity metrics and delta metrics
- baseline commits with metadata and complexity snapshot
- line counts, truncation flags, fetch errors, and sample group (primary or exploratory)

Re-running the same command skips records that already succeeded — resume is automatic.

### 2. Annotate the prepared records

You can annotate in two ways.

Automatic annotation (LLM):

```bash
python -m tech_debt_pipeline --stage annotate --backend groq
```

Other backend options: ollama.

Re-running the same command skips records that already succeeded. Groq stops once the daily token/request cap would be hit — re-run the next day to continue.

Flags:

- `--backend {groq,ollama}` — which LLM to use (default: groq)
- `--rpm N` — override Groq requests-per-minute pacing
- `--skip-failed` — mark failed records as skipped and generate the report anyway
- `--ids ID1 ID2 ...` — re-annotate specific record IDs only

Manual annotation (web UI):

```bash
python -m tech_debt_pipeline.annotation_server
```

Open http://localhost:5000 and annotate records there.

### 3. Outputs

- Main annotated dataset: pipeline_output/prepared_dataset.json
- Findings report: pipeline_output/annotation_report.md

