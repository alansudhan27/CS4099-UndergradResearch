import json
import random
from pathlib import Path
from . import config


def load_dataset(path: Path = config.DATASET_PATH) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _has_commit_data(record: dict) -> bool:
    commits = record.get("commits", {})
    if not commits:
        return False
    first = commits.get("first_commit")
    if not first or not isinstance(first, dict):
        return False
    patch = first.get("patch", "")
    return len(patch) >= config.MIN_PATCH_CHARS


def _has_subsequent_commits(record: dict) -> bool:
    commits = record.get("commits", {})
    second = commits.get("second_commit")
    others = commits.get("other_commits", [])
    return bool(second) or bool(others)


def _patch_size(record: dict) -> int:
    patch = record.get("commits", {}).get("first_commit", {}).get("patch", "")
    return len(patch)


def _get_rq1a_label(record: dict) -> str:
    return ((record.get("annotations") or {}).get("rq1a") or {}).get("pred_label", "Unknown")


def _get_confidence(record: dict) -> float:
    return ((record.get("annotations") or {}).get("rq1a") or {}).get("confidence", 0.0)


def filter_candidates(data: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Apply the filtering pipeline to get high-quality candidates.

    Returns (candidates, exclusion_counts) where exclusion_counts maps
    reason -> number of records excluded for that reason (first match wins).
    """
    candidates = []
    exclusion_counts = {
        "false_positive": 0,
        "generic_comment": 0,
        "short_comment": 0,
        "no_patch_or_too_short": 0,
    }
    for record in data:
        annotations = record.get("annotations") or {}
        rq1a = (annotations.get("rq1a") or {}).get("pred_label", "")
        rq1b = (annotations.get("rq1b") or {}).get("pred_label", "")

        if rq1a == "False Positive" or rq1b == "False Positive":
            exclusion_counts["false_positive"] += 1
            continue
        if rq1b == "Generic Comment":
            exclusion_counts["generic_comment"] += 1
            continue
        comment = record.get("comment", "")
        if len(comment) < config.MIN_COMMENT_LEN:
            exclusion_counts["short_comment"] += 1
            continue
        if not _has_commit_data(record):
            exclusion_counts["no_patch_or_too_short"] += 1
            continue

        candidates.append(record)

    return candidates, exclusion_counts


EXCLUDED_CATEGORIES = {"Unspecified", "Unknown", "", None}


def select_data_points(
    data: list[dict],
    target: int = config.TARGET_SAMPLE_SIZE,
    exploratory_size: int = config.EXPLORATORY_SAMPLE_SIZE,
    seed: int = config.RANDOM_SEED,
) -> tuple[list[dict], dict[str, list[dict]], dict]:
    """Select diverse records using stratified sampling.

    Returns (selected_records, reserves_by_category, summary_stats).
    reserves_by_category maps each rq1a category to its ordered list of
    remaining candidates not in selected — used to replace records that
    end up with 0 baseline commits during the prepare stage.
    """
    random.seed(seed)
    candidates, exclusion_counts = filter_candidates(data)
    print(f"Filtered to {len(candidates)} candidates from {len(data)} records")
    for reason, n in exclusion_counts.items():
        print(f"  excluded ({reason}): {n}")

    # Group by rq1a category
    by_category: dict[str, list[dict]] = {}
    for record in candidates:
        cat = _get_rq1a_label(record)
        by_category.setdefault(cat, []).append(record)

    # Separate primary vs exploratory pools
    primary_pool: dict[str, list[dict]] = {}
    exploratory_pool: list[dict] = []
    for cat, recs in by_category.items():
        if cat in EXCLUDED_CATEGORIES:
            exploratory_pool.extend(recs)
        else:
            primary_pool[cat] = recs

    total_primary = sum(len(recs) for recs in primary_pool.values())
    print(f"\nPrimary pool: {total_primary} records across {len(primary_pool)} categories")
    print(f"Exploratory pool: {len(exploratory_pool)} records")

    # Compute proportional allocation with floor
    allocation: dict[str, int] = {}
    for cat, recs in primary_pool.items():
        raw = round(len(recs) / total_primary * target)
        allocation[cat] = max(config.MIN_PER_CATEGORY, raw)

    # Balance to hit target
    while sum(allocation.values()) > target:
        largest = max(allocation, key=lambda c: allocation[c])
        allocation[largest] -= 1
    while sum(allocation.values()) < target:
        largest = max(allocation, key=lambda c: len(primary_pool[c]))
        allocation[largest] += 1

    # Sort function for prioritization within each category
    def sort_key(r):
        has_subsequent = _has_subsequent_commits(r)
        patch_len = _patch_size(r)
        moderate_patch = 200 <= patch_len <= 10000
        confidence = _get_confidence(r)
        return (has_subsequent, moderate_patch, confidence)

    # Select from each category; track reserves (remaining sorted candidates)
    selected = []
    used_repos = set()
    shortfalls = {}
    reserves_by_category: dict[str, list[dict]] = {}

    print("\nCategory allocation:")
    for cat in sorted(allocation, key=lambda c: -allocation[c]):
        pool = primary_pool[cat]
        requested = allocation[cat]

        # Shuffle for tie-breaking, then sort by priority
        random.shuffle(pool)
        pool.sort(key=sort_key, reverse=True)

        picked = []
        reserve = []
        for record in pool:
            repo = record["repo_full_name"]
            if repo in used_repos:
                continue
            if len(picked) < requested:
                picked.append(record)
                used_repos.add(repo)
            else:
                reserve.append(record)

        reserves_by_category[cat] = reserve

        if len(picked) < requested:
            shortfalls[cat] = (requested, len(picked))
            print(f"  {cat}: requested {requested}, got {len(picked)} (repo uniqueness) | {len(reserve)} reserves")
        else:
            print(f"  {cat}: {len(picked)} | {len(reserve)} reserves")

        for r in picked:
            r["_sample_group"] = "primary"
        selected.extend(picked)

    # Exploratory selection
    random.shuffle(exploratory_pool)
    exploratory_pool.sort(key=sort_key, reverse=True)
    exploratory_selected = []
    exploratory_reserve = []
    for record in exploratory_pool:
        repo = record["repo_full_name"]
        if repo in used_repos:
            continue
        if len(exploratory_selected) < exploratory_size:
            record["_sample_group"] = "exploratory"
            exploratory_selected.append(record)
            used_repos.add(repo)
        else:
            exploratory_reserve.append(record)

    reserves_by_category["_exploratory"] = exploratory_reserve
    selected.extend(exploratory_selected)
    print(f"\nExploratory: {len(exploratory_selected)} | {len(exploratory_reserve)} reserves")
    print(f"Total selected: {len(selected)}")

    # ID uniqueness assertion
    ids = [r["id"] for r in selected]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate IDs in selection! {len(ids)} total, {len(set(ids))} unique")

    # Build summary stats
    summary = _build_summary_stats(selected, allocation, shortfalls, candidates, seed)
    summary["exclusion_counts"] = exclusion_counts
    summary["total_input_records"] = len(data)

    return selected, reserves_by_category, summary


def _build_summary_stats(selected, allocation, shortfalls, candidates, seed):
    """Build summary statistics for the preparation summary."""
    import hashlib

    # Category distribution
    cat_dist = {}
    for r in selected:
        cat = _get_rq1a_label(r)
        group = r.get("_sample_group", "primary")
        cat_dist.setdefault(cat, {"count": 0, "group": group})
        cat_dist[cat]["count"] += 1

    # Language distribution
    lang_dist = {}
    for r in selected:
        ext = r.get("path", "").rsplit(".", 1)[-1] if "." in r.get("path", "") else "unknown"
        lang_dist[ext] = lang_dist.get(ext, 0) + 1

    # Patch size stats
    patch_sizes = [_patch_size(r) for r in selected]
    patch_sizes.sort()
    n = len(patch_sizes)

    # % with subsequent commits
    with_subsequent = sum(1 for r in selected if _has_subsequent_commits(r))

    # Sample hash
    ids = sorted(r["id"] for r in selected)
    sample_hash = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]

    return {
        "total_selected": len(selected),
        "primary_count": sum(1 for r in selected if r.get("_sample_group") == "primary"),
        "exploratory_count": sum(1 for r in selected if r.get("_sample_group") == "exploratory"),
        "allocation": allocation,
        "shortfalls": shortfalls,
        "category_distribution": cat_dist,
        "language_distribution": lang_dist,
        "patch_size_p25": patch_sizes[n // 4] if n > 0 else 0,
        "patch_size_p50": patch_sizes[n // 2] if n > 0 else 0,
        "patch_size_p75": patch_sizes[3 * n // 4] if n > 0 else 0,
        "pct_with_subsequent": round(with_subsequent / len(selected) * 100, 1) if selected else 0,
        "total_candidates": len(candidates),
        "seed": seed,
        "sample_hash": sample_hash,
    }


def print_selection_summary(selected: list[dict]) -> None:
    """Print a summary of selected records."""
    print(f"\n{'='*80}")
    print(f"Selected {len(selected)} records for annotation:")
    print(f"{'='*80}")
    for i, record in enumerate(selected[:20], 1):
        rq1a = _get_rq1a_label(record)
        confidence = _get_confidence(record)
        has_sub = _has_subsequent_commits(record)
        patch_len = _patch_size(record)
        group = record.get("_sample_group", "?")
        print(f"\n{i}. [{rq1a}] (conf: {confidence:.3f}) [{group}]")
        print(f"   Repo: {record['repo_full_name']}")
        print(f"   File: {record['path']}")
        print(f"   Comment: {record['comment'][:100]}...")
        print(f"   Patch size: {patch_len} chars | Has subsequent commits: {has_sub}")
    if len(selected) > 20:
        print(f"\n... and {len(selected) - 20} more records")
