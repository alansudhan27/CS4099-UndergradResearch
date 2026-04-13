"""Compute cyclomatic complexity metrics using lizard (multi-language)."""

import lizard


def compute_complexity(source_code: str, filename: str) -> dict:
    """Compute cyclomatic complexity metrics for a code string.

    Args:
        source_code: The source code to analyze.
        filename: Used as a language hint (e.g. "file.py", "app.js").

    Returns:
        {
            'functions': [{'name', 'complexity', 'nloc', 'params', 'start_line'}],
            'total_nloc': int,
            'avg_complexity': float,
            'max_complexity': int,
            'num_functions': int,
        }
    """
    if not source_code or not source_code.strip():
        return {
            "functions": [],
            "total_nloc": 0,
            "avg_complexity": 0.0,
            "max_complexity": 0,
            "num_functions": 0,
        }

    try:
        result = lizard.analyze_file.analyze_source_code(filename, source_code)
    except Exception:
        return {
            "functions": [],
            "total_nloc": 0,
            "avg_complexity": 0.0,
            "max_complexity": 0,
            "num_functions": 0,
            "error": f"lizard could not analyze {filename}",
        }

    functions = []
    for func in result.function_list:
        functions.append({
            "name": func.name,
            "complexity": func.cyclomatic_complexity,
            "nloc": func.nloc,
            "params": func.parameter_count,
            "start_line": func.start_line,
        })

    complexities = [f["complexity"] for f in functions]

    return {
        "functions": functions,
        "total_nloc": result.nloc,
        "avg_complexity": round(sum(complexities) / len(complexities), 2) if complexities else 0.0,
        "max_complexity": max(complexities) if complexities else 0,
        "num_functions": len(functions),
    }


def compute_complexity_delta(before_code: str, after_code: str, filename: str) -> dict:
    """Compare cyclomatic complexity before and after a change.

    Returns:
        {
            'before': {metrics},
            'after': {metrics},
            'delta_avg': float,
            'delta_max': int,
            'delta_nloc': int,
            'delta_functions': int,
            'complexity_increased': bool,
            'error': str,  # non-empty if analysis was limited
        }
    """
    before = compute_complexity(before_code, filename)
    after = compute_complexity(after_code, filename)

    # If EITHER side has no complete functions, the CC comparison is unreliable:
    # a 0 from "no functions found" is not the same as "genuinely simple code".
    # Excluding these records prevents biased paired tests downstream.
    if "error" in before or "error" in after:
        error = before.get("error", "") or after.get("error", "")
    elif before["num_functions"] == 0 and after["num_functions"] == 0:
        error = "insufficient context — no complete functions found in before or after code"
    elif before["num_functions"] == 0:
        error = "insufficient context — no complete functions found in before code"
    elif after["num_functions"] == 0:
        error = "insufficient context — no complete functions found in after code"
    else:
        error = ""

    return {
        "before": before,
        "after": after,
        "delta_avg": round(after["avg_complexity"] - before["avg_complexity"], 2),
        "delta_max": after["max_complexity"] - before["max_complexity"],
        "delta_nloc": after["total_nloc"] - before["total_nloc"],
        "delta_functions": after["num_functions"] - before["num_functions"],
        "complexity_increased": after["avg_complexity"] > before["avg_complexity"],
        "error": error,
    }
