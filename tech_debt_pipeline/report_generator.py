"""Generate markdown annotation report with statistical analysis and individual records."""

from datetime import datetime, timezone
from pathlib import Path

from . import config
from .diff_parser import detect_language
from .pattern_analyzer import analyze_patterns


def generate_markdown_report(records: list[dict], output_path: Path = config.ANNOTATION_REPORT_PATH):
    """Generate the full annotation report as markdown."""
    analysis = analyze_patterns(records)
    successful = [r for r in records if r.get("annotation_status") == "success"]

    lines = []
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Derive model/backend from annotation metadata (use first annotated record as representative)
    annotated = [r for r in records if r.get("annotation_metadata")]
    meta = annotated[0].get("annotation_metadata", {}) if annotated else {}
    report_model = meta.get("model", config.GROQ_MODEL)
    report_backend = meta.get("backend", "groq")

    # ── Executive Summary ──
    lines.append("# Gen AI Technical Debt Annotation Report")
    lines.append("")
    lines.append(f"**Generated:** {timestamp}")
    lines.append(f"**Backend:** {report_backend} | **Model:** {report_model}")
    lines.append(f"**Records annotated:** {analysis['successful_annotations']} "
                 f"(of {analysis['total_records']} prepared)")
    lines.append(f"**Primary:** {analysis['primary_count']} | "
                 f"**Exploratory:** {analysis['exploratory_count']}")
    lines.append("")
    lines.append("> **Survivorship bias note:** This study only observes commits where developers "
                 "explicitly mentioned AI tools. Developers who used AI without commenting, or "
                 "whose AI-assisted code was rejected in review, are invisible to this analysis.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Statistical Comparison ──
    _write_statistical_comparison(lines, analysis)

    # ── Per-Category Breakdown ──
    _write_per_category(lines, analysis)

    # ── rq1b Breakdown ──
    _write_rq1b_breakdown(lines, analysis)

    # ── Language Breakdown ──
    _write_language_breakdown(lines, analysis)

    # ── Data Quality ──
    _write_data_quality(lines, analysis)

    # ── Individual Records ──
    lines.append("---")
    lines.append("")
    lines.append("# Individual Records")
    lines.append("")

    for i, r in enumerate(successful, 1):
        _write_individual_record(lines, r, i)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report saved to {output_path}")


def _fmt_delta(value, fmt=".2f") -> str:
    if isinstance(value, float):
        return f"{value:+{fmt}}"
    if isinstance(value, int):
        return f"{value:+d}"
    try:
        return f"{float(value):+{fmt}}"
    except (ValueError, TypeError):
        return str(value)


def _write_statistical_comparison(lines: list[str], analysis: dict):
    paired = analysis.get("paired_test", {})
    lines.append("## Statistical Comparison: AI vs Baseline")
    lines.append("")

    if paired.get("lines_note"):
        lines.append(f"> {paired['lines_note']}")
        lines.append("")
        return

    bonf_alpha = paired.get("bonferroni_alpha", 0.025)
    n_confirm = paired.get("n_confirmatory_tests", 2)
    lines.append(
        f"> **Multiple comparisons note:** {n_confirm} confirmatory tests were pre-designated. "
        f"Bonferroni-corrected significance threshold: α = {bonf_alpha:.4f}. "
        f"All per-category, per-language, and rq1b breakdowns below are *exploratory* analyses "
        f"and are **not** corrected for multiple comparisons."
    )
    lines.append("")

    # ── Cyclomatic Complexity (CONFIRMATORY) ──
    n_cc = paired.get("n_paired_cc", 0)
    n_cc_excl_ai = paired.get("n_cc_excluded_ai", 0)
    n_cc_excl_bl = paired.get("n_cc_excluded_baseline", 0)
    lines.append(f"### Cyclomatic Complexity — CONFIRMATORY TEST")
    lines.append("")
    lines.append(f"n = {n_cc} paired records "
                 f"(excluded: {n_cc_excl_ai} with invalid AI CC, "
                 f"{n_cc_excl_bl} with no valid baseline CC)")
    lines.append("")
    if n_cc >= 10:
        lines.append(f"- Median CC difference (AI - baseline): {paired.get('cc_median_diff', 'N/A')}")
        mean_diff = paired.get('cc_mean_diff')
        if isinstance(mean_diff, (int, float)):
            lines.append(f"- Mean difference: {mean_diff:.2f}")
        if "cc_wilcoxon_p" in paired:
            p = paired["cc_wilcoxon_p"]
            lines.append(f"- Wilcoxon signed-rank: p = {p:.4f} "
                         f"(Bonferroni-corrected threshold: α = {bonf_alpha:.4f})")
            r = paired.get('cc_effect_size_r', 0)
            n_eff = paired.get('cc_effect_size_n', 0)
            n_total = paired.get('cc_n_total_pairs', n_cc)
            n_zero = paired.get('cc_n_zero_diffs', 0)
            lines.append(f"- Effect size r = {r:.3f} "
                         f"(computed on {n_eff} nonzero of {n_total} pairs; "
                         f"{n_zero} pairs had zero difference)")
        if "cc_sign_test_p" in paired:
            lines.append(f"- Sign test: p = {paired['cc_sign_test_p']:.4f} "
                         f"(+{paired.get('cc_sign_n_positive', 0)}/"
                         f"-{paired.get('cc_sign_n_negative', 0)})")
    else:
        lines.append(f"- Insufficient pairs for statistical test (need >= 10, have {n_cc})")
    lines.append("")

    # ── Lines Added (DESCRIPTIVE SANITY CHECK) ──
    n_lines = paired.get("n_paired_lines", 0)
    lines.append(f"### Lines Added — DESCRIPTIVE SANITY CHECK")
    lines.append("")
    caveat = paired.get("lines_caveat", "")
    if caveat:
        lines.append(f"> *{caveat}*")
        lines.append("")
    lines.append(f"n = {n_lines} paired records")
    lines.append("")
    if n_lines >= 10:
        lines.append(f"- Median difference (AI - baseline): {paired.get('lines_median_diff', 'N/A')}")
        mean_diff = paired.get('lines_mean_diff')
        if isinstance(mean_diff, (int, float)):
            lines.append(f"- Mean difference: {mean_diff:.1f}")
        if "lines_wilcoxon_p" in paired:
            p = paired["lines_wilcoxon_p"]
            lines.append(f"- Wilcoxon signed-rank: p = {p:.4f}")
            r = paired.get('lines_effect_size_r', 0)
            n_eff = paired.get('lines_effect_size_n', 0)
            n_total = paired.get('lines_n_total_pairs', n_lines)
            n_zero = paired.get('lines_n_zero_diffs', 0)
            lines.append(f"- Effect size r = {r:.3f} "
                         f"(computed on {n_eff} nonzero of {n_total} pairs; "
                         f"{n_zero} pairs had zero difference)")
        if "lines_sign_test_p" in paired:
            lines.append(f"- Sign test: p = {paired['lines_sign_test_p']:.4f} "
                         f"(+{paired.get('lines_sign_n_positive', 0)}/"
                         f"-{paired.get('lines_sign_n_negative', 0)})")
    else:
        lines.append(f"- Insufficient pairs for statistical test (need >= 10, have {n_lines})")
    lines.append("")
    lines.append("---")
    lines.append("")


def _write_per_category(lines: list[str], analysis: dict):
    per_cat = analysis.get("per_category", {})
    if not per_cat:
        return

    lines.append("## Per-Category Breakdown (rq1a)")
    lines.append("")
    lines.append("| Category | n | Mean Quality | AI Quality | % Mistakes | % +Complexity | % No Debt | Top Debt Types |")
    lines.append("|----------|---|-------------|------------|------------|---------------|-----------|----------------|")

    for cat, stats in sorted(per_cat.items(), key=lambda x: -x[1]["n"]):
        n = stats["n"]
        name = f"{cat}*" if n < 20 else cat
        if n < 10:
            lines.append(f"| {name} | {n} | — | — | — | — | — | (n too small for reliable statistics) |")
            continue
        q = f"{stats['mean_quality']:.1f}" if stats["mean_quality"] is not None else "-"
        aq = f"{stats['mean_ai_quality']:.1f}" if stats["mean_ai_quality"] is not None else "-"
        top_cats = ", ".join(f"{c[0][:20]}" for c in stats["top_debt_categories"][:2])
        lines.append(f"| {name} | {n} | {q} | {aq} | "
                     f"{stats['pct_ai_mistakes']:.0f}% | {stats['pct_added_complexity']:.0f}% | "
                     f"{stats['pct_no_debt']:.0f}% | {top_cats} |")
    lines.append("")
    lines.append("*\\* n < 20 — interpret with caution; n < 10 — statistics suppressed*")
    lines.append("")
    lines.append("---")
    lines.append("")


def _write_rq1b_breakdown(lines: list[str], analysis: dict):
    breakdown = analysis.get("rq1b_breakdown", {})
    if not breakdown:
        return

    lines.append("## Comment Type Breakdown (rq1b)")
    lines.append("")
    lines.append("| Comment Type | n | Mean Quality | Top Debt Types |")
    lines.append("|-------------|---|-------------|----------------|")

    for rq1b_type, stats in sorted(breakdown.items(), key=lambda x: -x[1]["n"]):
        n = stats["n"]
        name = f"{rq1b_type}*" if n < 20 else rq1b_type
        if n < 10:
            lines.append(f"| {name} | {n} | — | (n too small for reliable statistics) |")
            continue
        q = f"{stats['mean_quality']:.1f}" if stats["mean_quality"] is not None else "-"
        top_cats = ", ".join(f"{c[0][:25]}" for c in stats["top_debt_categories"][:2])
        lines.append(f"| {name} | {n} | {q} | {top_cats} |")
    lines.append("")
    lines.append("*\\* n < 20 — interpret with caution; n < 10 — statistics suppressed*")
    lines.append("")
    lines.append("---")
    lines.append("")


def _write_language_breakdown(lines: list[str], analysis: dict):
    langs = analysis.get("language_breakdown", {})
    if not langs:
        return

    lines.append("## Language Breakdown")
    lines.append("")
    excluded = langs.get("_excluded") or {}
    lines.append("| Language | n | Mean Quality | % AI Mistakes |")
    lines.append("|----------|---|-------------|---------------|")

    lang_rows = [(k, v) for k, v in langs.items() if k != "_excluded"]
    for lang, stats in sorted(lang_rows, key=lambda x: -x[1]["n"]):
        n = stats["n"]
        name = f"{lang}*" if n < 20 else lang
        if n < 10:
            lines.append(f"| {name} | {n} | — | (n too small for reliable statistics) |")
            continue
        q = f"{stats['mean_quality']:.1f}" if stats["mean_quality"] is not None else "-"
        lines.append(f"| {name} | {n} | {q} | {stats['pct_ai_mistakes']:.0f}% |")
    lines.append("")
    lines.append("*\\* n < 20 — interpret with caution; n < 10 — statistics suppressed*")
    lines.append("")
    if excluded:
        excl_str = ", ".join(f"{lang or 'unknown'} ({cnt})"
                             for lang, cnt in sorted(excluded.items(), key=lambda x: -x[1]))
        lines.append(f"> *Languages with fewer than 5 records excluded: {excl_str}*")
        lines.append("")
    lines.append("---")
    lines.append("")


def _write_data_quality(lines: list[str], analysis: dict):
    dq = analysis.get("data_quality", {})
    if not dq:
        return

    lines.append("## Data Quality")
    lines.append("")
    lines.append(f"- **Annotation success rate:** {dq.get('annotation_success_rate', 0):.1f}%")
    lines.append(f"- **Failed:** {dq.get('annotation_failed', 0)} | "
                 f"**Skipped:** {dq.get('annotation_skipped', 0)}")
    lines.append(f"- **CC computation rate:** {dq.get('cc_computation_rate', 0):.1f}%")
    lines.append(f"- **Baseline match rate:** {dq.get('baseline_match_rate', 0):.1f}%")
    lines.append(f"- **Patch truncation rate:** {dq.get('truncation_rate', 0):.1f}%")
    lines.append(f"- **Schema violation rate:** {dq.get('schema_violation_rate', 0):.1f}%")
    lines.append("")
    lines.append(
        "> **Sampling note:** Records labeled \"False Positive\" (rq1a/rq1b) or "
        "\"Generic Comment\" (rq1b) were excluded during candidate filtering. "
        "The \"Generic Comment\" exclusion may bias the sample toward comments "
        "where developers expressed an opinion about AI-generated code. "
        "See `preparation_summary.md` for exact exclusion counts."
    )
    lines.append("")
    lines.append("---")
    lines.append("")


def _write_individual_record(lines: list[str], r: dict, index: int):
    """Write a single annotated record with full code context."""
    analysis = r.get("tech_debt_analysis") or {}
    cats = analysis.get("debt_categories", [])
    reasoning = analysis.get("debt_category_reasoning", {})
    commits = r.get("original_commits", {})
    first_commit = commits.get("first_commit") or {}
    second_commit = commits.get("second_commit")
    other_commits = commits.get("other_commits", [])
    parsed = r.get("parsed_diff") or {}
    complexity = r.get("complexity_metrics") or {}
    labels = r.get("existing_labels") or {}
    rq1a = labels.get("rq1a") or {}
    lang = detect_language(r.get("path", ""))

    # ── Header ──
    lines.append(f"## {index}. {r['repo_full_name']}")
    lines.append("")

    # ── Context ──
    lines.append("### Context")
    lines.append("")
    lines.append(f"**File:** `{r['path']}` (line {r.get('line', '?')})")
    lines.append(f"**Developer Comment:**")
    lines.append(f"> {r.get('original_comment', '')}")
    lines.append("")
    lines.append(f"**Commit:** \"{first_commit.get('message', 'N/A')}\"")
    lines.append(f"**Author:** {first_commit.get('author', 'N/A')} | "
                 f"**Date:** {first_commit.get('date', 'N/A')}")
    rq1a_conf = rq1a.get('confidence', 0)
    try:
        rq1a_conf_str = f"{float(rq1a_conf):.3f}"
    except (ValueError, TypeError):
        rq1a_conf_str = str(rq1a_conf)
    lines.append(f"**AI Use Type:** {rq1a.get('pred_label', 'N/A')} "
                 f"(confidence: {rq1a_conf_str})")
    lines.append(f"**Sample Group:** {r.get('sample_group', 'N/A')}")
    lines.append("")

    # ── Code: Full Diff ──
    raw_patch = first_commit.get("patch", "")
    if raw_patch:
        lines.append("### Full Diff")
        lines.append("")
        lines.append("```diff")
        lines.append(raw_patch)
        lines.append("```")
        lines.append("")

    # ── Code: Before/After ──
    removed = parsed.get("removed_lines", [])
    added = parsed.get("added_lines", [])

    if removed:
        lines.append("### Before (Removed Lines)")
        lines.append("")
        lines.append(f"```{lang}")
        lines.append("\n".join(removed))
        lines.append("```")
        lines.append("")
    else:
        lines.append("### Before (Removed Lines)")
        lines.append("")
        lines.append("*No lines removed — purely additive change.*")
        lines.append("")

    if added:
        lines.append("### After (Added Lines)")
        lines.append("")
        lines.append(f"```{lang}")
        lines.append("\n".join(added))
        lines.append("```")
        lines.append("")
    else:
        lines.append("### After (Added Lines)")
        lines.append("")
        lines.append("*No lines added — purely subtractive change.*")
        lines.append("")

    # ── Complexity Metrics ──
    lines.append("### Complexity Metrics")
    lines.append("")

    if complexity.get("error"):
        lines.append(f"*Note: {complexity['error']}*")
        lines.append("")

    before_avg = complexity.get("before_avg_complexity", 0)
    after_avg = complexity.get("after_avg_complexity", 0)
    before_max = complexity.get("before_max_complexity", 0)
    after_max = complexity.get("after_max_complexity", 0)
    before_nloc = complexity.get("before_nloc", 0)
    after_nloc = complexity.get("after_nloc", 0)
    delta_avg = complexity.get("delta_avg", 0)
    delta_max = complexity.get("delta_max", 0)
    delta_nloc = complexity.get("delta_nloc", 0)
    delta_funcs = complexity.get("delta_functions", 0)

    lines.append("| Metric | Before | After | Delta |")
    lines.append("|--------|--------|-------|-------|")
    lines.append(f"| CC (avg) | {before_avg} | {after_avg} | **{_fmt_delta(delta_avg)}** |")
    lines.append(f"| CC (max) | {before_max} | {after_max} | **{_fmt_delta(delta_max)}** |")
    lines.append(f"| Lines of Code | {before_nloc} | {after_nloc} | **{_fmt_delta(delta_nloc)}** |")

    before_funcs = complexity.get("before_functions", [])
    after_funcs = complexity.get("after_functions", [])
    lines.append(f"| Functions | {len(before_funcs)} | {len(after_funcs)} | **{_fmt_delta(delta_funcs)}** |")
    lines.append(f"| Lines Added/Removed | | | **+{r.get('lines_added', 0)}/-{r.get('lines_removed', 0)}** |")
    lines.append("")

    # ── AI Change Analysis ──
    lines.append("### AI Change Analysis")
    lines.append("")
    ai_mistake = analysis.get("ai_made_mistake", False)
    ai_complexity = analysis.get("ai_added_complexity", False)
    ai_quality = analysis.get("ai_suggestion_quality", 0)

    lines.append(f"- **AI Made Mistake:** {'Yes' if ai_mistake else 'No'}")
    if analysis.get("ai_mistake_description"):
        lines.append(f"  - {analysis['ai_mistake_description']}")
    lines.append(f"- **AI Added Complexity:** {'Yes' if ai_complexity else 'No'}")
    if analysis.get("ai_complexity_description"):
        lines.append(f"  - {analysis['ai_complexity_description']}")
    lines.append(f"- **AI Suggestion Quality:** {ai_quality}/5")
    if analysis.get("ai_suggestion_quality_reasoning"):
        lines.append(f"  - {analysis['ai_suggestion_quality_reasoning']}")
    lines.append("")

    # ── Developer Analysis ──
    lines.append("### Developer Analysis")
    lines.append("")
    assessment = analysis.get("overall_assessment", "N/A")
    quality = analysis.get("code_quality_score", "N/A")
    awareness = analysis.get("developer_awareness", "N/A")
    confidence = analysis.get("confidence", 0)
    try:
        confidence_str = f"{float(confidence):.0%}"
    except (ValueError, TypeError):
        confidence_str = str(confidence)

    lines.append(f"- **Overall Assessment:** {assessment}")
    lines.append(f"- **Code Quality:** {quality}/5")
    lines.append(f"- **Developer Awareness:** {awareness}")
    lines.append(f"- **Confidence:** {confidence_str}")
    lines.append("")

    if analysis.get("overall_reasoning"):
        lines.append(f"**Reasoning:** {analysis['overall_reasoning']}")
        lines.append("")

    if analysis.get("developer_awareness_reasoning"):
        lines.append(f"**Awareness Detail:** {analysis['developer_awareness_reasoning']}")
        lines.append("")

    if analysis.get("complexity_explanation"):
        lines.append(f"**Complexity:** {analysis['complexity_explanation']}")
        lines.append("")

    if analysis.get("fix_description"):
        lines.append(f"**Fix Description:** {analysis['fix_description']}")
        lines.append("")

    # ── Evidence Citations ──
    evidence = analysis.get("evidence_citations", [])
    if evidence:
        lines.append("### Evidence Citations")
        lines.append("")
        for j, e in enumerate(evidence, 1):
            claim = e.get("claim", "")
            code_ref = e.get("code_reference", "")
            explanation = e.get("explanation", "")
            lines.append(f"{j}. *\"{claim}\"*")
            if code_ref:
                lines.append(f"   ```{lang}")
                lines.append(f"   {code_ref}")
                lines.append(f"   ```")
            if explanation:
                lines.append(f"   {explanation}")
            lines.append("")

    # ── Tech Debt Categories ──
    if cats:
        lines.append("### Tech Debt Categories")
        lines.append("")
        for cat in cats:
            explanation = reasoning.get(cat, "")
            lines.append(f"- **{cat}:** {explanation}")
        lines.append("")

    # ── Baseline Comparison ──
    baselines = r.get("baseline_commits", [])
    if baselines:
        lines.append("### Baseline Comparison")
        lines.append("")
        for bi, b in enumerate(baselines, 1):
            lines.append(f"**Baseline {bi}:** `{b.get('sha', '?')[:7]}` — "
                         f"\"{b.get('message', '')[:80]}\"")
            lines.append(f"- Author: {b.get('author', 'N/A')} | Date: {b.get('date', 'N/A')}")
            lines.append(f"- Lines: +{b.get('lines_added', 0)}/-{b.get('lines_removed', 0)} | "
                         f"CC avg: {b.get('complexity_avg', 0)} | "
                         f"Structural: {'Yes' if b.get('is_structural') else 'No'}")
            if b.get("patch"):
                lines.append("")
                lines.append("```diff")
                lines.append(b["patch"][:2000])
                lines.append("```")
            lines.append("")

    # ── Subsequent Commits ──
    has_subsequent = False
    if second_commit and isinstance(second_commit, dict):
        if not has_subsequent:
            lines.append("### Subsequent Commits")
            lines.append("")
        has_subsequent = True
        lines.append(f"**Commit `{second_commit.get('sha', '?')[:7]}`** — "
                     f"\"{second_commit.get('message', '')}\" by "
                     f"{second_commit.get('author', '?')} on "
                     f"{second_commit.get('date', '?')}")
        patch2 = second_commit.get("patch", "")
        if patch2:
            lines.append("")
            lines.append("```diff")
            lines.append(patch2[:config.MAX_PATCH_CHARS])
            lines.append("```")
        lines.append("")

    if other_commits and isinstance(other_commits, list):
        for oc in other_commits[:3]:
            if isinstance(oc, dict):
                if not has_subsequent:
                    lines.append("### Subsequent Commits")
                    lines.append("")
                    has_subsequent = True
                lines.append(f"**Commit `{oc.get('sha', '?')[:7]}`** — "
                             f"\"{oc.get('message', '')}\" by "
                             f"{oc.get('author', '?')} on "
                             f"{oc.get('date', '?')}")
                oc_patch = oc.get("patch", "")
                if oc_patch:
                    lines.append("")
                    lines.append("```diff")
                    lines.append(oc_patch[:config.MAX_PATCH_CHARS])
                    lines.append("```")
                lines.append("")

    # Commit history from GitHub (metadata only — no patches)
    ctx = r.get("github_context") or {}
    history = ctx.get("commit_history", [])
    if history:
        first_date = first_commit.get("date", "")
        subsequent_from_api = [c for c in history if c.get("date", "") > first_date] if first_date else []
        if subsequent_from_api and not has_subsequent:
            lines.append("### Subsequent Commits")
            lines.append("")
            has_subsequent = True
        if subsequent_from_api:
            lines.append(f"**{len(subsequent_from_api)} commit(s) found via GitHub API:**")
            lines.append("")
            for c in subsequent_from_api[:5]:
                lines.append(f"- `{c.get('sha', '?')[:7]}` {c.get('date', '?')}: "
                             f"{c.get('message', '?')[:120]}")
            lines.append("")

    if not has_subsequent:
        lines.append("### Subsequent Commits")
        lines.append("")
        lines.append("*No subsequent commits found.*")
        lines.append("")

    lines.append("---")
    lines.append("")
