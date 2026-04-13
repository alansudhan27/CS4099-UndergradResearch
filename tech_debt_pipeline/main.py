import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .models import AnnotatedRecord, GitHubContext, ComplexityMetrics, TechDebtAnnotation
from .data_selector import load_dataset, select_data_points, print_selection_summary
from .github_context import (
    gather_github_context, fetch_baseline_commits, count_lines_from_patch,
    get_api_call_count,
)
from .llm_annotator import annotate_record, QuotaExhaustedError
from .diff_parser import parse_patch
from .complexity import compute_complexity_delta


PROGRESS_FILE = config.PIPELINE_DIR / ".prepare_progress.json"
ANNOTATE_PROGRESS_FILE = config.PIPELINE_DIR / ".annotate_progress.json"


def _ensure_pipeline_dir():
    config.PIPELINE_DIR.mkdir(parents=True, exist_ok=True)


def _load_progress(path: Path) -> set[str]:
    """Load completed IDs from a progress file."""
    if not path.exists():
        return set()
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return set(data.get("completed_ids", []))
    except (json.JSONDecodeError, KeyError):
        return set()


def _save_progress(path: Path, completed_ids: set[str]):
    """Atomically save completed IDs to a progress file.

    Writes to a temp file first then renames so a crash mid-write never
    corrupts the progress file.
    """
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump({"completed_ids": sorted(completed_ids), "updated": datetime.now(timezone.utc).isoformat()}, f)
    os.replace(tmp, path)  # atomic on both Unix and Windows


# ─── PREPARE ────────────────────────────────────────────────────────────────

def run_prepare():
    """Command 1: Select records, fetch GitHub context + baselines."""
    _ensure_pipeline_dir()

    print("Loading dataset...")
    data = load_dataset()
    print(f"Loaded {len(data)} records")

    # Select records
    selected, reserves_by_category, _ = select_data_points(data)
    print_selection_summary(selected)

    if not selected:
        print("No records selected. Exiting.")
        return

    # Check for resume
    completed_ids = _load_progress(PROGRESS_FILE)
    if completed_ids:
        print(f"\nResuming: {len(completed_ids)} records already fetched, skipping them.")

    # Fetch context for each record
    results = []
    rejected_count = 0
    processed_count = 0  # records actually fetched (not skipped from resume)
    total = len(selected)
    remaining_to_fetch = sum(1 for r in selected if r["id"] not in completed_ids)
    start_time = time.time()

    # Load previously fetched results if resuming
    previously_fetched = {}
    if completed_ids and config.PREPARED_JSON_PATH.exists():
        try:
            with open(config.PREPARED_JSON_PATH, "r") as f:
                for r in json.load(f):
                    previously_fetched[r["id"]] = r
        except (json.JSONDecodeError, KeyError):
            pass

    # Reserve iterators — advance as reserves are consumed
    reserve_iters: dict[str, int] = {cat: 0 for cat in reserves_by_category}

    for i, record in enumerate(selected, 1):
        record_id = record["id"]

        # Skip if already fetched (and data is available)
        if record_id in completed_ids:
            if record_id in previously_fetched:
                results.append(previously_fetched[record_id])
                continue
            # Progress file says done but data not in prepared JSON — re-fetch
            print(f"\n[{i}/{total}] {record['repo_full_name']} | re-fetching (progress file but no saved data)")
            completed_ids.discard(record_id)
            remaining_to_fetch += 1  # restore count since we're re-fetching this record

        elapsed = time.time() - start_time
        rate = processed_count / max(elapsed, 1) if processed_count > 0 else 0
        eta = remaining_to_fetch / rate if rate > 0 else 0

        print(f"\n[{i}/{total}] {record['repo_full_name']} | "
              f"accepted: {len(results)} | rejected: {rejected_count} | "
              f"API: {get_api_call_count()} | ETA: {eta/60:.0f}m")
        print(f"  File: {record['path']}")

        result_dict = _fetch_record(record)
        processed_count += 1
        remaining_to_fetch -= 1

        baselines = result_dict.get("baseline_commits", [])
        if len(baselines) >= config.MIN_REQUIRED_BASELINES:
            results.append(result_dict)
        else:
            rejected_count += 1
            print(f"  REJECTED: 0 baselines found — trying reserve for this category")
            rq1a_cat = (record.get("annotations", {}).get("rq1a") or {}).get("pred_label", "")
            reserve_key = "_exploratory" if record.get("_sample_group") == "exploratory" else rq1a_cat
            reserve_list = reserves_by_category.get(reserve_key, [])
            # Try reserves until one succeeds or we run out
            replaced = False
            while reserve_iters.get(reserve_key, 0) < len(reserve_list):
                idx = reserve_iters[reserve_key]
                reserve_iters[reserve_key] = idx + 1
                reserve_record = reserve_list[idx]
                reserve_id = reserve_record["id"]
                if reserve_id in completed_ids:
                    if reserve_id in previously_fetched:
                        results.append(previously_fetched[reserve_id])
                        replaced = True
                        break
                    continue
                print(f"  Reserve [{idx+1}]: {reserve_record['repo_full_name']}")
                reserve_result = _fetch_record(reserve_record)
                processed_count += 1
                completed_ids.add(reserve_id)
                _save_progress(PROGRESS_FILE, completed_ids)
                if len(reserve_result.get("baseline_commits", [])) >= config.MIN_REQUIRED_BASELINES:
                    results.append(reserve_result)
                    replaced = True
                    print(f"  Reserve accepted: {len(reserve_result['baseline_commits'])} baselines")
                    break
                else:
                    print(f"  Reserve also has 0 baselines, trying next...")

            if not replaced:
                print(f"  WARNING: No reserve with baselines found for category '{reserve_key}'")

        # Save progress
        completed_ids.add(record_id)
        _save_progress(PROGRESS_FILE, completed_ids)

    print(f"\n{'='*60}")
    print(f"Fetching complete: {len(results)} accepted, {rejected_count} rejected")
    print(f"Total API calls: {get_api_call_count()}")

    _save_prepared_json(results)

    # Clean up progress file
    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()

    print(f"\nOutputs saved to {config.PIPELINE_DIR}/")


def _fetch_record(record: dict) -> dict:
    """Fetch GitHub context, complexity, and baselines for a single record. Returns a record dict."""
    fetch_errors = []

    # Step 1: GitHub context
    context = gather_github_context(record)
    time.sleep(0.5)

    # Step 2: Parse diff and compute complexity
    first_commit = record.get("commits", {}).get("first_commit") or {}
    patch = first_commit.get("patch", "")
    was_truncated = len(patch) > config.MAX_PATCH_CHARS

    parsed = parse_patch(patch)

    # Use patch fragments for BOTH before and after so the comparison is symmetric.
    # The full file content is still kept in GitHubContext for the LLM prompt.
    before_source = parsed["before_code"]
    after_source = parsed["after_code"]

    delta = compute_complexity_delta(before_source, after_source, record["path"])
    complexity_metrics = ComplexityMetrics.from_delta(delta)

    print(f"  Complexity: avg {delta['before']['avg_complexity']} -> "
          f"{delta['after']['avg_complexity']} (delta: {delta['delta_avg']:+.2f})")
    if delta["error"]:
        print(f"  Complexity note: {delta['error']}")
        fetch_errors.append(f"complexity: {delta['error']}")

    # Step 3: Count lines from patch
    lines_added, lines_removed = count_lines_from_patch(patch)

    # Step 4: Fetch baselines
    ai_commit_date = first_commit.get("date", "")
    ai_patch_size = len(patch)
    print(f"  Fetching baseline commits...")
    baselines = fetch_baseline_commits(
        record["repo_full_name"], record["path"],
        ai_commit_date, ai_patch_size, context.commit_history,
    )
    print(f"  Found {len(baselines)} baseline commits")

    annotated = AnnotatedRecord(
        id=record["id"],
        original_comment=record.get("comment", ""),
        repo_full_name=record["repo_full_name"],
        path=record["path"],
        html_url=record.get("html_url", ""),
        line=record.get("line", 0),
        existing_labels={
            "rq1a": record.get("annotations", {}).get("rq1a", {}),
            "rq1b": record.get("annotations", {}).get("rq1b", {}),
        },
        original_commits=record.get("commits", {}),
        github_context=context,
        complexity_metrics=complexity_metrics,
        parsed_diff=parsed,
        baseline_commits=baselines,
        sample_group=record.get("_sample_group", "primary"),
        fetch_errors=fetch_errors,
        patch_was_truncated=was_truncated,
        lines_added=lines_added,
        lines_removed=lines_removed,
    )
    return annotated.to_dict()


def _save_prepared_json(results: list[dict]):
    with open(config.PREPARED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  JSON: {config.PREPARED_JSON_PATH} ({len(results)} records)")


# ─── ANNOTATE ───────────────────────────────────────────────────────────────

def _save_prepared(records: list[dict]) -> None:
    """Atomically write records back to prepared_dataset.json."""
    tmp = config.PREPARED_JSON_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    tmp.replace(config.PREPARED_JSON_PATH)


def run_annotate(rpm: int = config.GROQ_RATE_LIMIT_RPM,
                 skip_failed: bool = False, ids: list[str] | None = None,
                 backend: str = config.DEFAULT_BACKEND):
    """Annotate prepared records, saving each result back into prepared_dataset.json.

    Resume is automatic: records with annotation_status == 'success' are skipped.
    The web UI (annotation_server.py) reads the same file, so progress is visible live.
    """
    _ensure_pipeline_dir()

    if not config.PREPARED_JSON_PATH.exists():
        raise SystemExit(f"ERROR: {config.PREPARED_JSON_PATH} not found. Run --stage prepare first.")

    print("Loading prepared dataset...")
    with open(config.PREPARED_JSON_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} records")

    already_done = sum(1 for r in records if r.get("annotation_status") == "success")
    if already_done:
        print(f"Skipping {already_done} already-annotated records (remove annotation_status to re-annotate)")

    if backend == "ollama":
        print(f"Backend: Ollama ({config.OLLAMA_MODEL}) at {config.OLLAMA_BASE_URL}")
    else:
        print(f"Backend: Groq ({config.GROQ_MODEL})")

    # Filter to specific IDs if requested (force re-annotate by clearing their status first)
    if ids:
        id_set = set(ids)
        for r in records:
            if r["id"] in id_set:
                r["annotation_status"] = "pending"
                r["tech_debt_analysis"] = None
        print(f"Re-annotating {len(id_set)} specific records")

    # Determine what to annotate
    to_annotate = [r for r in records if r.get("annotation_status") != "success"]
    total_to_do = len(to_annotate)

    # Compute daily call cap for Groq to avoid hitting RPD / TPD limits
    if backend == "groq":
        max_calls_by_tpd = int(config.GROQ_TPD / config.GROQ_ESTIMATED_TOKENS_PER_CALL)
        max_calls_this_run = int(min(config.GROQ_RPD, max_calls_by_tpd) * 0.95)  # 5% safety buffer
        print(f"Groq daily cap: {max_calls_this_run} calls this run "
              f"(RPD={config.GROQ_RPD:,}, TPD={config.GROQ_TPD:,})")
    else:
        max_calls_this_run = None  # no daily cap for ollama

    success_count = already_done
    fail_count = 0
    processed_this_run = 0
    start_time = time.time()
    quota_exhausted = False

    for i, record in enumerate(to_annotate, 1):
        if max_calls_this_run is not None and processed_this_run >= max_calls_this_run:
            print(f"\nGroq daily limit reached ({processed_this_run} calls this run).")
            print("Re-run the same command tomorrow — already-annotated records will be skipped.")
            break
        elapsed = time.time() - start_time
        fallback_rate = rpm / 60 if backend != "ollama" else 0.1
        rate = processed_this_run / max(elapsed, 1) if processed_this_run > 0 else fallback_rate
        remaining = total_to_do - i + 1
        eta = remaining / rate if rate > 0 else 0

        print(f"\n[{i}/{total_to_do}] {record['repo_full_name']} | ETA: {eta/60:.0f}m")

        ctx_data = record.get("github_context") or {}
        context = GitHubContext(
            file_content_snippet=ctx_data.get("file_content_snippet", ""),
            file_available=ctx_data.get("file_available", False),
            commit_history=ctx_data.get("commit_history", []),
        )

        try:
            annotation = annotate_record(record, context, rpm=rpm, backend=backend)
        except QuotaExhaustedError as e:
            quota_exhausted = True
            print("  QUOTA EXHAUSTED: stopping early. Re-run to resume automatically.")
            print(f"  Details: {e}")
            break

        if annotation is not None:
            record["tech_debt_analysis"] = annotation.to_dict()
            record["annotation_status"] = "success"
            record["schema_violations"] = []

            _, violations = TechDebtAnnotation.validate_and_coerce(record["tech_debt_analysis"])
            if violations:
                record["schema_violations"] = violations
                print(f"  Schema violations: {violations}")

            success_count += 1
        else:
            record["annotation_status"] = "skipped"
            record["tech_debt_analysis"] = None
            record["schema_violations"] = []
            fail_count += 1
            print("  Skipped: retries exhausted")

        record["annotation_metadata"] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backend": backend,
            "model": config.OLLAMA_MODEL if backend == "ollama" else config.GROQ_MODEL,
        }

        processed_this_run += 1

        # Write the full dataset back after every record so the web UI sees live progress
        # and a crash loses at most one record's work.
        _save_prepared(records)

    if skip_failed:
        for record in records:
            if record.get("annotation_status") == "failed":
                record["annotation_status"] = "skipped"
        _save_prepared(records)

    print(f"\n{'='*60}")
    print(f"Annotation complete: {success_count} annotated, {fail_count} failed")
    if quota_exhausted:
        print("Re-run the same command to continue — already-annotated records will be skipped.")

    # Generate markdown report from all records
    from .report_generator import generate_markdown_report
    generate_markdown_report(records, config.ANNOTATION_REPORT_PATH)
    print(f"Report: {config.ANNOTATION_REPORT_PATH}")


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gen AI Technical Debt Annotation Pipeline")
    parser.add_argument("--stage", required=True, choices=["prepare", "annotate"],
                        help="Pipeline stage to run")
    parser.add_argument("--rpm", type=int, default=config.GROQ_RATE_LIMIT_RPM,
                        help=f"Groq requests per minute (default: {config.GROQ_RATE_LIMIT_RPM})")
    parser.add_argument("--skip-failed", action="store_true",
                        help="Mark failed annotations as skipped and generate report anyway")
    parser.add_argument("--ids", nargs="+",
                        help="Re-annotate specific record IDs only")
    parser.add_argument("--backend", choices=["groq", "ollama"],
                        default=config.DEFAULT_BACKEND,
                        help=f"Annotation backend (default: {config.DEFAULT_BACKEND})")
    args = parser.parse_args()

    if args.stage == "prepare":
        run_prepare()
    elif args.stage == "annotate":
        run_annotate(
            rpm=args.rpm,
            skip_failed=args.skip_failed,
            ids=args.ids,
            backend=args.backend,
        )


if __name__ == "__main__":
    main()
