import argparse
import csv
import json
import os
import statistics
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
from .groq_annotator import annotate_record, QuotaExhaustedError
from .diff_parser import parse_patch, detect_language
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
    selected, reserves_by_category, summary_stats = select_data_points(data)
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

    # Save outputs
    _save_sampled_json(results)
    _save_prepared_json(results)
    _save_prepared_csv(results)
    _save_preparation_summary(results, summary_stats)

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


def _save_sampled_json(results: list[dict]):
    """Save selected records in the original dataset schema (no enriched fields)."""
    sampled = []
    for r in results:
        original_commits = r.get("original_commits", {})
        record = {
            "id": r["id"],
            "comment": r.get("original_comment", ""),
            "repo_full_name": r["repo_full_name"],
            "path": r["path"],
            "html_url": r.get("html_url", ""),
            "line": r.get("line", 0),
            "annotations": r.get("existing_labels", {}),
            "commits": original_commits,
        }
        sampled.append(record)
    with open(config.SAMPLED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(sampled, f, indent=2, ensure_ascii=False)
    print(f"  Sampled JSON: {config.SAMPLED_JSON_PATH} ({len(sampled)} records, original schema)")


def _save_prepared_json(results: list[dict]):
    with open(config.PREPARED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  JSON: {config.PREPARED_JSON_PATH} ({len(results)} records)")


def _save_prepared_csv(results: list[dict]):
    """Save flat CSV for R/pandas analysis."""
    fieldnames = [
        "id", "repo", "path", "language", "line", "comment_first100",
        "commit_sha", "commit_date", "commit_message_first50", "html_url",
        "sample_group", "rq1a_label", "rq1b_label",
        "patch_size", "patch_was_truncated", "has_subsequent_commits",
        "lines_added", "lines_removed",
        "cc_before_avg", "cc_after_avg", "cc_delta_avg",
        "cc_before_max", "cc_after_max", "cc_delta_max",
        "nloc_before", "nloc_after", "nloc_delta",
        "num_functions_before", "num_functions_after",
        "cc_computation_succeeded",
        "baseline_count", "baseline_1_sha", "baseline_2_sha",
        "baseline_median_cc_avg", "baseline_median_lines_added",
        "baseline_cc_computation_succeeded",
    ]

    with open(config.PREPARED_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            commits = r.get("original_commits", {})
            first = commits.get("first_commit", {}) or {}
            labels = r.get("existing_labels", {})
            complexity = r.get("complexity_metrics", {}) or {}
            baselines = r.get("baseline_commits", [])
            history = (r.get("github_context") or {}).get("commit_history", [])

            # Count subsequent commits
            first_date = first.get("date", "")
            subsequent = [c for c in history if c.get("date", "") > first_date] if first_date else []

            # Baseline medians
            baseline_cc_avgs = [b.get("complexity_avg", 0) for b in baselines if not b.get("complexity_error")]
            baseline_lines = [b.get("lines_added", 0) for b in baselines]
            baseline_cc_succeeded = any(not b.get("complexity_error") for b in baselines)

            median_cc = statistics.median(baseline_cc_avgs) if baseline_cc_avgs else ""
            median_lines = statistics.median(baseline_lines) if baseline_lines else ""

            row = {
                "id": r["id"],
                "repo": r["repo_full_name"],
                "path": r["path"],
                "language": detect_language(r["path"]),
                "line": r.get("line", 0),
                "comment_first100": r.get("original_comment", "")[:100],
                "commit_sha": first.get("sha", ""),
                "commit_date": first.get("date", ""),
                "commit_message_first50": first.get("message", "")[:50],
                "html_url": r.get("html_url", ""),
                "sample_group": r.get("sample_group", ""),
                "rq1a_label": (labels.get("rq1a") or {}).get("pred_label", ""),
                "rq1b_label": (labels.get("rq1b") or {}).get("pred_label", ""),
                "patch_size": len(first.get("patch", "")),
                # Use lowercase strings for booleans so R/pandas parse them correctly
                "patch_was_truncated": str(r.get("patch_was_truncated", False)).lower(),
                "has_subsequent_commits": str(len(subsequent) > 0).lower(),
                "lines_added": r.get("lines_added", 0),
                "lines_removed": r.get("lines_removed", 0),
                "cc_before_avg": complexity.get("before_avg_complexity", ""),
                "cc_after_avg": complexity.get("after_avg_complexity", ""),
                "cc_delta_avg": complexity.get("delta_avg", ""),
                "cc_before_max": complexity.get("before_max_complexity", ""),
                "cc_after_max": complexity.get("after_max_complexity", ""),
                "cc_delta_max": complexity.get("delta_max", ""),
                "nloc_before": complexity.get("before_nloc", ""),
                "nloc_after": complexity.get("after_nloc", ""),
                "nloc_delta": complexity.get("delta_nloc", ""),
                "num_functions_before": len(complexity.get("before_functions", [])),
                "num_functions_after": len(complexity.get("after_functions", [])),
                "cc_computation_succeeded": str(not bool(complexity.get("error"))).lower(),
                "baseline_count": len(baselines),
                "baseline_1_sha": baselines[0].get("sha", "") if len(baselines) > 0 else "",
                "baseline_2_sha": baselines[1].get("sha", "") if len(baselines) > 1 else "",
                "baseline_median_cc_avg": median_cc,
                "baseline_median_lines_added": median_lines,
                # Consistent bool — False when no baselines rather than empty string
                "baseline_cc_computation_succeeded": str(baseline_cc_succeeded).lower(),
            }
            writer.writerow(row)

    print(f"  CSV: {config.PREPARED_CSV_PATH}")


def _save_preparation_summary(results: list[dict], stats: dict):
    """Save preparation summary markdown."""
    lines = []
    lines.append("# Data Preparation Summary")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append(f"**Seed:** {stats['seed']}")
    lines.append(f"**Sample hash:** `{stats['sample_hash']}`")
    lines.append("")

    lines.append("## Sample Size")
    lines.append("")
    lines.append(f"- **Total candidates:** {stats['total_candidates']}")
    lines.append(f"- **Total selected:** {stats['total_selected']}")
    lines.append(f"  - Primary: {stats['primary_count']}")
    lines.append(f"  - Exploratory: {stats['exploratory_count']}")
    lines.append("")

    # Category distribution
    lines.append("## Category Distribution")
    lines.append("")
    lines.append("| Category | Requested | Actual | Shortfall |")
    lines.append("|----------|-----------|--------|-----------|")
    allocation = stats.get("allocation", {})
    shortfalls = stats.get("shortfalls", {})
    cat_dist = stats.get("category_distribution", {})
    for cat in sorted(allocation, key=lambda c: -allocation[c]):
        requested = allocation[cat]
        actual = cat_dist.get(cat, {}).get("count", 0)
        shortfall = shortfalls.get(cat, ("", ""))
        sf_text = f"requested {shortfall[0]}, got {shortfall[1]}" if cat in shortfalls else "-"
        lines.append(f"| {cat} | {requested} | {actual} | {sf_text} |")
    lines.append("")

    # Language distribution
    lines.append("## Language Distribution")
    lines.append("")
    lang_dist = stats.get("language_distribution", {})
    for lang, count in sorted(lang_dist.items(), key=lambda x: -x[1]):
        lang_pct = count / stats["total_selected"] * 100 if stats["total_selected"] > 0 else 0
        lines.append(f"- **{lang}**: {count} ({lang_pct:.1f}%)")
    lines.append("")

    # Patch size
    lines.append("## Patch Size Distribution")
    lines.append("")
    lines.append(f"- P25: {stats['patch_size_p25']} chars")
    lines.append(f"- P50 (median): {stats['patch_size_p50']} chars")
    lines.append(f"- P75: {stats['patch_size_p75']} chars")
    lines.append(f"- % with subsequent commits: {stats['pct_with_subsequent']}%")
    lines.append("")

    # Fetch stats
    lines.append("## Fetch Statistics")
    lines.append("")
    total = len(results)
    with_baselines = sum(1 for r in results if r.get("baseline_commits"))
    with_file = sum(1 for r in results if (r.get("github_context") or {}).get("file_available"))
    cc_success = sum(1 for r in results if not (r.get("complexity_metrics") or {}).get("error"))
    errors = sum(1 for r in results if r.get("fetch_errors"))

    pct = lambda n: f"{n/total*100:.0f}%" if total > 0 else "N/A"
    lines.append(f"- **File content fetched:** {with_file}/{total} ({pct(with_file)})")
    lines.append(f"- **Records with baselines:** {with_baselines}/{total} ({pct(with_baselines)})")
    lines.append(f"- **CC computation succeeded:** {cc_success}/{total} ({pct(cc_success)})")
    lines.append(f"- **Records with errors:** {errors}")
    lines.append(f"- **Total API calls:** {get_api_call_count()}")
    lines.append("")

    # Baseline quality
    baseline_counts = [len(r.get("baseline_commits", [])) for r in results]
    lines.append("## Baseline Quality")
    lines.append("")
    lines.append(f"- Records with 2 baselines: {sum(1 for c in baseline_counts if c >= 2)}")
    lines.append(f"- Records with 1 baseline: {sum(1 for c in baseline_counts if c == 1)}")
    lines.append(f"- Records with 0 baselines: {sum(1 for c in baseline_counts if c == 0)}")
    lines.append("")

    with open(config.PREPARATION_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Summary: {config.PREPARATION_SUMMARY_PATH}")


# ─── ANNOTATE ───────────────────────────────────────────────────────────────

def run_annotate(resume: bool = False, rpm: int = config.GROQ_RATE_LIMIT_RPM,
                 skip_failed: bool = False, ids: list[str] | None = None,
                 backend: str = config.DEFAULT_BACKEND):
    """Command 2: Annotate prepared records and generate report."""
    _ensure_pipeline_dir()

    # Load prepared dataset
    if not config.PREPARED_JSON_PATH.exists():
        raise SystemExit(f"ERROR: {config.PREPARED_JSON_PATH} not found. Run --stage prepare first.")

    print("Loading prepared dataset...")
    with open(config.PREPARED_JSON_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} records")

    if backend == "ollama":
        print(f"Backend: Ollama ({config.OLLAMA_MODEL}) at {config.OLLAMA_BASE_URL}")
        print("Estimated time depends on local model speed")
    else:
        est_minutes = len(records) / rpm
        print(f"Backend: Groq ({config.GROQ_MODEL})")
        print(f"Estimated time: {est_minutes:.0f} min at {rpm} rpm (+ retries)")

    # Persist annotated records incrementally so resume never loses progress.
    annotated_json_path = config.ANNOTATION_REPORT_PATH.with_suffix(".json")

    # Handle resume
    completed_annotations = {}
    completed_ids = set()
    if resume:
        completed_ids = _load_progress(ANNOTATE_PROGRESS_FILE)
        # Load previously checkpointed annotation data
        if annotated_json_path.exists():
            try:
                with open(annotated_json_path, "r", encoding="utf-8") as f:
                    for r in json.load(f):
                        if r.get("annotation_status") == "success":
                            completed_annotations[r["id"]] = r
                        # Keep all prior statuses so resume preserves context.
                        elif r.get("annotation_status") in {"failed", "skipped"}:
                            completed_annotations[r["id"]] = r
            except (json.JSONDecodeError, KeyError):
                pass

        # Reconcile progress IDs with checkpointed records.
        # If a run was interrupted before JSON checkpoint was written, don't skip unseen IDs.
        checkpoint_ids = set(completed_annotations.keys())
        dangling_progress_ids = completed_ids - checkpoint_ids
        if dangling_progress_ids:
            print(
                f"Resume note: {len(dangling_progress_ids)} progress IDs had no checkpointed records; "
                "they will be re-annotated."
            )
            completed_ids -= dangling_progress_ids

        completed_ids |= checkpoint_ids
        if completed_ids:
            print(f"Resuming: {len(completed_ids)} records already annotated")

    # Filter to specific IDs if requested
    if ids:
        id_set = set(ids)
        records = [r for r in records if r["id"] in id_set]
        # Remove from completed so they get re-annotated
        for rid in id_set:
            completed_ids.discard(rid)
            completed_annotations.pop(rid, None)
        print(f"Re-annotating {len(records)} specific records")

    # Annotate
    annotated_results = dict(completed_annotations)
    total = len(records)
    previously_completed = len(completed_ids)
    success_count = previously_completed
    fail_count = 0
    processed_this_run = 0
    remaining_to_annotate = sum(1 for r in records if r["id"] not in completed_ids)
    start_time = time.time()
    quota_exhausted = False

    for i, record in enumerate(records, 1):
        record_id = record["id"]
        if record_id in completed_ids:
            continue

        elapsed = time.time() - start_time
        fallback_rate = rpm / 60 if backend != "ollama" else 0.1  # ~6 records/min fallback for Ollama
        rate = processed_this_run / max(elapsed, 1) if processed_this_run > 0 else fallback_rate
        eta = remaining_to_annotate / rate if rate > 0 else 0

        if backend == "ollama":
            print(f"\n[{i}/{total}] {record['repo_full_name']} | "
                  f"local | ETA: {eta/60:.0f}m")
        else:
            print(f"\n[{i}/{total}] {record['repo_full_name']} | "
                  f"{rpm}rpm | ETA: {eta/60:.0f}m")

        # Build context from stored data
        ctx_data = record.get("github_context") or {}
        context = GitHubContext(
            file_content_snippet=ctx_data.get("file_content_snippet", ""),
            file_available=ctx_data.get("file_available", False),
            full_file_content=ctx_data.get("full_file_content", ""),
            commit_history=ctx_data.get("commit_history", []),
        )

        # Annotate
        try:
            annotation = annotate_record(record, context, rpm=rpm, backend=backend)
        except QuotaExhaustedError as e:
            quota_exhausted = True
            print("  QUOTA EXHAUSTED: stopping early to preserve progress.")
            print(f"  Details: {e}")
            break

        if annotation is not None:
            # annotation is already coerced by validate_and_coerce in groq_annotator
            record["tech_debt_analysis"] = annotation.to_dict()
            record["annotation_status"] = "success"
            record["schema_violations"] = []

            # Re-validate to track any coercions that were applied
            _, violations = TechDebtAnnotation.validate_and_coerce(record["tech_debt_analysis"])
            if violations:
                record["schema_violations"] = violations
                print(f"  Schema violations: {violations}")

            success_count += 1
        else:
            record["annotation_status"] = "failed"
            record["tech_debt_analysis"] = None
            record["schema_violations"] = []
            fail_count += 1

        record["annotation_metadata"] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backend": backend,
            "model": config.OLLAMA_MODEL if backend == "ollama" else config.GROQ_MODEL,
        }

        annotated_results[record_id] = record
        processed_this_run += 1
        remaining_to_annotate -= 1

        # Save progress
        completed_ids.add(record_id)
        _save_progress(ANNOTATE_PROGRESS_FILE, completed_ids)

        # Save checkpoint after each record to survive rate limits/crashes.
        with open(annotated_json_path, "w", encoding="utf-8") as f:
            json.dump(list(annotated_results.values()), f, indent=2, ensure_ascii=False)

    # Handle skip-failed
    if skip_failed:
        for record in records:
            if record["id"] not in annotated_results:
                record["annotation_status"] = "skipped"
                record["tech_debt_analysis"] = None
                annotated_results[record["id"]] = record

    all_results = list(annotated_results.values())

    print(f"\n{'='*60}")
    print(f"Annotation complete: {success_count} success, {fail_count} failed")
    if quota_exhausted:
        print("Stopped early due to backend quota exhaustion. Resume with --resume after quota resets.")

    # Save annotated JSON (for resume capability)
    with open(annotated_json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Generate report
    from .report_generator import generate_markdown_report
    generate_markdown_report(all_results, config.ANNOTATION_REPORT_PATH)

    # Clean up progress
    if ANNOTATE_PROGRESS_FILE.exists():
        ANNOTATE_PROGRESS_FILE.unlink()

    print(f"\nReport: {config.ANNOTATION_REPORT_PATH}")


# ─── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gen AI Technical Debt Annotation Pipeline")
    parser.add_argument("--stage", required=True, choices=["prepare", "annotate"],
                        help="Pipeline stage to run")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from where the previous run left off")
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
            resume=args.resume,
            rpm=args.rpm,
            skip_failed=args.skip_failed,
            ids=args.ids,
            backend=args.backend,
        )


if __name__ == "__main__":
    main()
