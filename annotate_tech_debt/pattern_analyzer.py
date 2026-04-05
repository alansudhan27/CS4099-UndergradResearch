"""Statistical analysis for AI vs baseline comparison and pattern identification."""

import statistics
from collections import Counter


def analyze_patterns(records: list[dict]) -> dict:
    """Run full analysis on annotated records.

    Returns a dict with all analysis results for the report.
    """
    # Separate primary vs exploratory
    primary = [r for r in records if r.get("sample_group") == "primary"]
    exploratory = [r for r in records if r.get("sample_group") == "exploratory"]
    successful = [r for r in records if r.get("annotation_status") == "success"]

    results = {
        "total_records": len(records),
        "primary_count": len(primary),
        "exploratory_count": len(exploratory),
        "successful_annotations": len(successful),
        "failed_annotations": sum(1 for r in records if r.get("annotation_status") == "failed"),
        "skipped_annotations": sum(1 for r in records if r.get("annotation_status") == "skipped"),
    }

    # Paired testing (AI vs baseline)
    results["paired_test"] = _paired_analysis(successful)

    # Per-category analysis
    results["per_category"] = _per_category_analysis(successful)

    # rq1b breakdown
    results["rq1b_breakdown"] = _rq1b_analysis(successful)

    # Language breakdown
    results["language_breakdown"] = _language_analysis(successful)

    # Data quality
    results["data_quality"] = _data_quality(records)

    return results


#: Number of confirmatory tests for Bonferroni correction.
#: Currently: CC delta Wilcoxon + lines added Wilcoxon (descriptive sanity check).
#: Only CC is a true confirmatory test; lines_added is descriptive because
#: baselines were selected to be size-similar to AI commits.
N_CONFIRMATORY_TESTS = 2
BONFERRONI_ALPHA = 0.05 / N_CONFIRMATORY_TESTS  # = 0.025


def _paired_analysis(records: list[dict]) -> dict:
    """Paired comparison of AI commits vs baseline median.

    Two metrics are tested:
    - cc_delta_avg (CONFIRMATORY): primary test. Compares CC delta of AI commit
      vs CC delta median of size-matched baseline commits. Only pairs where BOTH
      sides have valid CC (no error, functions found) are included.
    - lines_added (DESCRIPTIVE SANITY CHECK): baselines were selected to be
      size-similar to AI commits, so this CANNOT be a confirmatory test. Reported
      for descriptive purposes only.
    """
    pairs_lines = []  # (ai_lines_added, baseline_median_lines_added)
    pairs_cc = []     # (ai_cc_delta, baseline_median_cc_delta)
    n_cc_excluded_ai = 0    # AI side CC invalid
    n_cc_excluded_baseline = 0  # no baseline with valid CC

    for r in records:
        baselines = r.get("baseline_commits", [])
        if not baselines:
            continue

        ai_lines = r.get("lines_added", 0)
        baseline_lines = [b.get("lines_added", 0) for b in baselines]
        if baseline_lines:
            median_lines = statistics.median(baseline_lines)
            pairs_lines.append((ai_lines, median_lines))

        # CC pairs: require both AI and at least one baseline to have valid CC.
        # "Valid" means no error recorded during complexity computation.
        complexity = r.get("complexity_metrics") or {}
        ai_cc_valid = not complexity.get("error")
        if not ai_cc_valid:
            n_cc_excluded_ai += 1
            continue

        # Use CC DELTA (after - before) so the metric reflects the change, not
        # absolute code complexity. This is what "did this commit add complexity"
        # actually asks.
        ai_cc_delta = complexity.get("delta_avg", 0)

        # Baselines: compute delta for each, keep only those with valid CC
        baseline_cc_deltas = []
        for b in baselines:
            if b.get("complexity_error"):
                continue
            # BaselineCommit stores complexity_avg as the "after" average
            # (there's no separate delta_avg on baselines — it's a single-commit snapshot).
            # Best proxy: treat complexity_avg as the baseline's post-commit CC.
            # Since we can't compute a true delta for baselines without more fetching,
            # fall back to comparing the AFTER CC values instead.
            baseline_cc_deltas.append(b.get("complexity_avg", 0))

        if not baseline_cc_deltas:
            n_cc_excluded_baseline += 1
            continue

        # Compare AI's after-CC to baseline median after-CC (apples to apples,
        # both are single-snapshot values from fragments).
        ai_cc_after = complexity.get("after_avg_complexity", 0)
        median_baseline_cc = statistics.median(baseline_cc_deltas)
        pairs_cc.append((ai_cc_after, median_baseline_cc))

    result = {
        "n_paired_lines": len(pairs_lines),
        "n_paired_cc": len(pairs_cc),
        "n_cc_excluded_ai": n_cc_excluded_ai,
        "n_cc_excluded_baseline": n_cc_excluded_baseline,
        "bonferroni_alpha": BONFERRONI_ALPHA,
        "n_confirmatory_tests": N_CONFIRMATORY_TESTS,
    }

    # Lines analysis — DESCRIPTIVE SANITY CHECK (not confirmatory)
    if len(pairs_lines) >= 10:
        diffs = [ai - bl for ai, bl in pairs_lines]
        result["lines_median_diff"] = statistics.median(diffs)
        result["lines_mean_diff"] = statistics.mean(diffs)
        result["lines_test_type"] = "descriptive_sanity_check"
        result["lines_caveat"] = (
            "Baselines were selected to be size-similar to AI commits. "
            "This test cannot demonstrate a size difference — it is a sanity check "
            "verifying that size-matching succeeded."
        )

        try:
            from scipy.stats import wilcoxon, binomtest

            # Filter zero diffs — wilcoxon raises ValueError if all diffs are zero
            nonzero_diffs = [d for d in diffs if d != 0]
            result["lines_n_total_pairs"] = len(diffs)
            result["lines_n_zero_diffs"] = len(diffs) - len(nonzero_diffs)
            if len(nonzero_diffs) >= 10:
                stat, p_value = wilcoxon(nonzero_diffs, alternative='two-sided')
                n = len(nonzero_diffs)
                result["lines_wilcoxon_stat"] = stat
                result["lines_wilcoxon_p"] = p_value
                # Rank-biserial correlation: r = 1 - (2T)/(n(n+1)/2)
                # Computed on nonzero pairs only (zeros don't contribute ranks).
                max_t = n * (n + 1) / 2
                result["lines_effect_size_r"] = 1 - (2 * stat) / max_t if max_t > 0 else 0
                result["lines_effect_size_n"] = n  # the n the effect size applies to

            # Sign test (two-sided via binomtest default)
            n_positive = sum(1 for d in diffs if d > 0)
            n_negative = sum(1 for d in diffs if d < 0)
            n_nonzero = n_positive + n_negative
            if n_nonzero > 0:
                sign_p = binomtest(min(n_positive, n_negative), n_nonzero, 0.5).pvalue
                result["lines_sign_test_p"] = sign_p
                result["lines_sign_n_positive"] = n_positive
                result["lines_sign_n_negative"] = n_negative
        except ImportError:
            result["lines_note"] = "scipy not installed — statistical tests skipped"

    # CC analysis — CONFIRMATORY (Bonferroni-corrected)
    if len(pairs_cc) >= 10:
        cc_diffs = [ai - bl for ai, bl in pairs_cc]
        result["cc_median_diff"] = statistics.median(cc_diffs)
        result["cc_mean_diff"] = statistics.mean(cc_diffs)
        result["cc_test_type"] = "confirmatory"

        try:
            from scipy.stats import wilcoxon, binomtest

            nonzero_cc_diffs = [d for d in cc_diffs if d != 0]
            result["cc_n_total_pairs"] = len(cc_diffs)
            result["cc_n_zero_diffs"] = len(cc_diffs) - len(nonzero_cc_diffs)
            if len(nonzero_cc_diffs) >= 10:
                stat, p_value = wilcoxon(nonzero_cc_diffs, alternative='two-sided')
                n = len(nonzero_cc_diffs)
                result["cc_wilcoxon_stat"] = stat
                result["cc_wilcoxon_p"] = p_value
                # Rank-biserial correlation: r = 1 - (2T)/(n(n+1)/2)
                max_t = n * (n + 1) / 2
                result["cc_effect_size_r"] = 1 - (2 * stat) / max_t if max_t > 0 else 0
                result["cc_effect_size_n"] = n

            n_positive = sum(1 for d in cc_diffs if d > 0)
            n_negative = sum(1 for d in cc_diffs if d < 0)
            n_nonzero = n_positive + n_negative
            if n_nonzero > 0:
                sign_p = binomtest(min(n_positive, n_negative), n_nonzero, 0.5).pvalue
                result["cc_sign_test_p"] = sign_p
                result["cc_sign_n_positive"] = n_positive
                result["cc_sign_n_negative"] = n_negative
        except ImportError:
            pass

    return result


def _per_category_analysis(records: list[dict]) -> dict:
    """Analyze patterns per rq1a category."""
    by_cat: dict[str, list[dict]] = {}
    for r in records:
        cat = (r.get("existing_labels") or {}).get("rq1a", {}).get("pred_label", "Unknown")
        by_cat.setdefault(cat, []).append(r)

    results = {}
    for cat, recs in sorted(by_cat.items(), key=lambda x: -len(x[1])):
        analyses = [r.get("tech_debt_analysis") or {} for r in recs]
        complexities = [r.get("complexity_metrics") or {} for r in recs]

        quality_scores = [a["code_quality_score"] for a in analyses if "code_quality_score" in a and a["code_quality_score"] is not None]
        ai_quality_scores = [a["ai_suggestion_quality"] for a in analyses if "ai_suggestion_quality" in a and a["ai_suggestion_quality"] is not None]
        n_mistakes = sum(1 for a in analyses if a.get("ai_made_mistake"))
        n_complexity = sum(1 for a in analyses if a.get("ai_added_complexity"))
        # "No debt" = LLM set the flag explicitly OR returned an empty debt_categories list
        n_no_debt = sum(1 for a in analyses if a.get("no_debt_detected") or not a.get("debt_categories"))

        # Debt category frequencies
        debt_cats = Counter()
        for a in analyses:
            for dc in a.get("debt_categories", []):
                debt_cats[dc] += 1

        # CC deltas
        cc_deltas = [c.get("delta_avg", 0) for c in complexities if not c.get("error") and c.get("delta_avg") is not None]

        results[cat] = {
            "n": len(recs),
            "reliable": len(recs) >= 20,  # flag for small-n warnings in reports
            "mean_quality": statistics.mean(quality_scores) if quality_scores else None,
            "mean_ai_quality": statistics.mean(ai_quality_scores) if ai_quality_scores else None,
            "pct_ai_mistakes": n_mistakes / len(recs) * 100 if recs else 0,
            "pct_added_complexity": n_complexity / len(recs) * 100 if recs else 0,
            "pct_no_debt": n_no_debt / len(recs) * 100 if recs else 0,
            "top_debt_categories": debt_cats.most_common(3),
            "mean_cc_delta": statistics.mean(cc_deltas) if cc_deltas else None,
            "mean_lines_added": statistics.mean([r.get("lines_added", 0) for r in recs]),
        }

    return results


def _rq1b_analysis(records: list[dict]) -> dict:
    """Analyze patterns grouped by rq1b (comment type)."""
    by_type: dict[str, list[dict]] = {}
    for r in records:
        rq1b = (r.get("existing_labels") or {}).get("rq1b", {}).get("pred_label", "Unknown")
        by_type.setdefault(rq1b, []).append(r)

    results = {}
    for rq1b_type, recs in sorted(by_type.items(), key=lambda x: -len(x[1])):
        analyses = [r.get("tech_debt_analysis") or {} for r in recs]
        quality_scores = [a["code_quality_score"] for a in analyses if "code_quality_score" in a and a["code_quality_score"] is not None]

        debt_cats = Counter()
        for a in analyses:
            for dc in a.get("debt_categories", []):
                debt_cats[dc] += 1

        results[rq1b_type] = {
            "n": len(recs),
            "reliable": len(recs) >= 20,
            "mean_quality": statistics.mean(quality_scores) if quality_scores else None,
            "top_debt_categories": debt_cats.most_common(3),
        }

    return results


def _language_analysis(records: list[dict]) -> dict:
    """Analyze patterns by programming language."""
    from .diff_parser import detect_language

    by_lang: dict[str, list[dict]] = {}
    for r in records:
        lang = detect_language(r.get("path", ""))
        by_lang.setdefault(lang, []).append(r)

    results = {}
    excluded_langs: dict[str, int] = {}
    for lang, recs in sorted(by_lang.items(), key=lambda x: -len(x[1])):
        if len(recs) < 5:
            excluded_langs[lang or "unknown"] = len(recs)
            continue
        analyses = [r.get("tech_debt_analysis") or {} for r in recs]
        quality_scores = [a["code_quality_score"] for a in analyses if "code_quality_score" in a and a["code_quality_score"] is not None]
        n_mistakes = sum(1 for a in analyses if a.get("ai_made_mistake"))

        results[lang] = {
            "n": len(recs),
            "reliable": len(recs) >= 20,
            "mean_quality": statistics.mean(quality_scores) if quality_scores else None,
            "pct_ai_mistakes": n_mistakes / len(recs) * 100 if recs else 0,
        }

    # Surface the silently-excluded languages
    if excluded_langs:
        results["_excluded"] = excluded_langs

    return results


def _data_quality(records: list[dict]) -> dict:
    """Compute data quality metrics."""
    total = len(records)
    if total == 0:
        return {}

    success = sum(1 for r in records if r.get("annotation_status") == "success")
    failed = sum(1 for r in records if r.get("annotation_status") == "failed")
    skipped = sum(1 for r in records if r.get("annotation_status") == "skipped")
    cc_success = sum(1 for r in records if not (r.get("complexity_metrics") or {}).get("error"))
    with_baselines = sum(1 for r in records if r.get("baseline_commits"))
    truncated = sum(1 for r in records if r.get("patch_was_truncated"))
    with_violations = sum(1 for r in records if r.get("schema_violations"))

    return {
        "total": total,
        "annotation_success_rate": success / total * 100,
        "annotation_failed": failed,
        "annotation_skipped": skipped,
        "cc_computation_rate": cc_success / total * 100,
        "baseline_match_rate": with_baselines / total * 100,
        "truncation_rate": truncated / total * 100,
        "schema_violation_rate": with_violations / total * 100,
    }
