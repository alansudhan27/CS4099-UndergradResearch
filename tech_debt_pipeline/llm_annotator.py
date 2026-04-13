import json
import re
import time
from groq import Groq
from openai import OpenAI
from . import config
from .models import TechDebtAnnotation, GitHubContext


class QuotaExhaustedError(RuntimeError):
    """Raised when backend quota is exhausted and retries should stop."""

SYSTEM_PROMPT = """You are a software engineering researcher assessing code changes that reference generative AI tools (ChatGPT, GPT-4, Claude, Copilot, etc.).

Assess whether the AI-assisted change introduces, maintains, or reduces technical debt. Be neutral — no debt is a valid finding.

Respond with valid JSON matching this exact schema:
{
  "no_debt_detected": true/false,
  "debt_categories": ["applicable categories or empty list"],
  "debt_category_reasoning": {"category": "why it applies"},
  "developer_awareness": "aware | partially_aware | unaware",
  "developer_awareness_reasoning": "explanation",
  "overall_assessment": "positive | negative | mixed",
  "overall_reasoning": "explanation",
  "adds_complexity": true/false,
  "complexity_explanation": "explanation",
  "evidence_of_subsequent_fixes": true/false,
  "fix_description": "description or null",
  "code_quality_score": 1-5,
  "code_quality_reasoning": "explanation",
  "confidence": 0.0-1.0,
  "evidence_citations": [{"claim": "assertion", "code_reference": "exact code snippet", "explanation": "how it proves the claim"}],
  "ai_made_mistake": true/false,
  "ai_mistake_description": "description or empty string",
  "ai_added_complexity": true/false,
  "ai_complexity_description": "explanation",
  "ai_suggestion_quality": 1-5,
  "ai_suggestion_quality_reasoning": "explanation"
}

RATINGS (1–5 scale, applies to both code_quality_score and ai_suggestion_quality):
  1 = Broken — bugs, doesn't compile, security holes, wrong results
  2 = Works but significant issues — logic errors, performance problems
  3 = Functional but not idiomatic — minor style/structure issues
  4 = Good — correct, readable, minor improvements possible
  5 = Excellent — idiomatic, efficient, well-structured

confidence: 0.0–0.3 very uncertain | 0.4–0.6 moderate | 0.7–0.8 confident | 0.9–1.0 very confident

RULES:
1. If no_debt_detected is true, debt_categories must be empty.
2. Every claim needs at least one evidence_citation quoting actual code from the patch.
3. Analyse AI output (ai_* fields) and developer behaviour (developer_awareness, overall_assessment) separately.
4. Respond ONLY with valid JSON — no prose before or after."""


def _extract_first_json_object(text: str) -> str | None:
    """Return the first top-level JSON object found in text, if any."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None


def _parse_llm_json(content: str) -> dict:
    """Parse model output as JSON with light recovery for common formatting issues."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    cleaned = content.strip()

    # Some models wrap JSON in Markdown fences.
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\\s*", "", cleaned)
        cleaned = re.sub(r"\\s*```$", "", cleaned)

    # Common malformed escape from model outputs: apostrophe written as \' inside JSON.
    cleaned = cleaned.replace("\\'", "'")

    # If extra prose exists, isolate the first full JSON object.
    candidate = _extract_first_json_object(cleaned)
    if candidate is not None:
        cleaned = candidate

    return json.loads(cleaned)


def _truncate_patch_at_hunk(patch: str, max_chars: int) -> tuple[str, bool]:
    """Truncate patch at a hunk boundary if possible."""
    if len(patch) <= max_chars:
        return patch, False
    hunk_starts = [m.start() for m in re.finditer(r'^@@', patch, re.MULTILINE)]
    if len(hunk_starts) > 1:
        valid = [s for s in hunk_starts if s < max_chars]
        if valid:
            return patch[:valid[-1]].rstrip() + "\n... (truncated)", True
    return patch[:max_chars] + "\n... (truncated)", True


_PROMPT_PATCH_CHARS = 1500  # tighter limit than MAX_PATCH_CHARS (used for storage)


def _build_user_prompt(record: dict, context: GitHubContext) -> str:
    """Build a compact user prompt containing only information the LLM needs."""
    comment = record.get("comment") or record.get("original_comment", "")
    repo = record["repo_full_name"]
    path = record["path"]
    annotations = record.get("annotations") or record.get("existing_labels", {})
    rq1a = annotations.get("rq1a") or {}
    rq1b = annotations.get("rq1b") or {}
    commits = record.get("commits") or record.get("original_commits", {})
    first_commit = commits.get("first_commit") or {}

    sections = []

    # Header — comment + file + existing classification on one block
    sections.append(
        f"## Context\n"
        f"Repo: {repo} | File: {path}\n"
        f"Developer comment: \"{comment}\"\n"
        f"AI use type (rq1a): {rq1a.get('pred_label', 'N/A')} | "
        f"Comment type (rq1b): {rq1b.get('pred_label', 'N/A')}"
    )

    # Complexity metrics — compact one-liner, high signal for adds_complexity fields
    complexity = record.get("complexity_metrics") or {}
    if not complexity.get("error") and complexity.get("delta_avg") is not None:
        sections.append(
            f"## Complexity Metrics\n"
            f"CC avg delta: {complexity['delta_avg']:+.2f} | "
            f"CC max delta: {complexity.get('delta_max', 0):+d} | "
            f"Lines: +{record.get('lines_added', 0)}/-{record.get('lines_removed', 0)}"
        )

    # First commit patch (capped at _PROMPT_PATCH_CHARS)
    if first_commit:
        patch, _ = _truncate_patch_at_hunk(first_commit.get("patch", ""), _PROMPT_PATCH_CHARS)
        sections.append(
            f"## Commit: \"{first_commit.get('message', '')}\" "
            f"({first_commit.get('date', '')[:10]})\n"
            f"```diff\n{patch}\n```"
        )

    # Second commit patch — only if it has a patch
    second = commits.get("second_commit")
    if second and isinstance(second, dict) and second.get("patch"):
        patch2, _ = _truncate_patch_at_hunk(second.get("patch", ""), _PROMPT_PATCH_CHARS)
        sections.append(
            f"## Subsequent Commit: \"{second.get('message', '')}\" "
            f"({second.get('date', '')[:10]})\n"
            f"```diff\n{patch2}\n```"
        )

    # Commit history — 3 entries max, no SHA
    if context.commit_history:
        first_date = first_commit.get("date", "")
        subsequent = [c for c in context.commit_history if c.get("date", "") > first_date] if first_date else []
        if subsequent:
            history_lines = [f"## Commit History ({len(subsequent)} after initial)"]
            for c in subsequent[:3]:
                history_lines.append(f"- {c.get('date','')[:10]}: {c.get('message','')[:80]}")
            sections.append("\n".join(history_lines))

    # Debt categories list + task directive
    categories_list = " | ".join(f"{i+1}.{c}" for i, c in enumerate(config.TECH_DEBT_CATEGORIES))
    sections.append(
        f"## Task\n"
        f"Assess this change. Categories: {categories_list}\n"
        f"Respond with JSON only."
    )

    return "\n\n".join(sections)


_groq_client = None
_ollama_client = None


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=config.GROQ_API_KEY)
    return _groq_client


def _get_ollama_client() -> OpenAI:
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OpenAI(
            base_url=config.OLLAMA_BASE_URL,
            api_key="ollama",
            timeout=config.OLLAMA_REQUEST_TIMEOUT_SECONDS,
        )
    return _ollama_client


def annotate_record(
    record: dict,
    context: GitHubContext,
    rpm: int = config.GROQ_RATE_LIMIT_RPM,
    backend: str = config.DEFAULT_BACKEND,
) -> TechDebtAnnotation | None:
    """Send record + context to LLM and return a TechDebtAnnotation.

    backend: "groq" | "ollama"
    """
    user_prompt = _build_user_prompt(record, context)
    print(f"  Sending to {backend}...")
    print(f"  Prompt: ~{len(user_prompt)} chars (~{len(user_prompt)//4} tokens)")

    if backend == "ollama":
        client = _get_ollama_client()
        model = config.OLLAMA_MODEL
        max_tokens = config.OLLAMA_MAX_COMPLETION_TOKENS
        print(f"  Model: {model} | timeout: {config.OLLAMA_REQUEST_TIMEOUT_SECONDS:.0f}s")
    else:
        client = _get_groq_client()
        model = config.GROQ_MODEL
        max_tokens = config.GROQ_MAX_COMPLETION_TOKENS
        print(f"  Model: {model}")

    use_json_mode = True

    for attempt in range(3):
        try:
            start = time.time()
            call_kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=max_tokens,
            )
            if use_json_mode:
                call_kwargs["response_format"] = {"type": "json_object"}

            response = client.chat.completions.create(**call_kwargs)
            elapsed = time.time() - start

            # Rate-limit pacing for Groq
            if backend == "groq":
                sleep_time = max(60 / rpm - elapsed, 0)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            content = response.choices[0].message.content
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            data = _parse_llm_json(content)
            annotation, _ = TechDebtAnnotation.validate_and_coerce(data)

            usage = response.usage
            if usage:
                print(f"  Response: {usage.prompt_tokens} prompt + {usage.completion_tokens} completion tokens")

            return annotation

        except json.JSONDecodeError as e:
            print(f"  JSON parse error (attempt {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(2)
        except Exception as e:
            err_str = str(e).lower()
            if backend == "ollama" and ("timeout" in err_str or "timed out" in err_str):
                print(f"  Ollama timed out; skipping record")
                return None
            if backend == "ollama" and use_json_mode and ("response_format" in err_str or "not supported" in err_str or "unknown" in err_str):
                print(f"  Ollama: JSON mode unsupported, retrying without it...")
                use_json_mode = False
                continue
            if backend == "groq" and use_json_mode and (
                "json_validate_failed" in err_str
                or "failed to generate json" in err_str
            ):
                print("  Groq JSON validation failed; retrying without JSON mode...")
                use_json_mode = False
                continue
            if backend == "groq" and (
                "tokens per day" in err_str
                or ("service tier" in err_str and "tokens" in err_str)
                or ("rate_limit_exceeded" in err_str and "requested" in err_str)
            ):
                raise QuotaExhaustedError(str(e))
            if backend == "groq" and ("rate limit" in err_str or "429" in err_str or "too many requests" in err_str):
                if attempt < 2:
                    wait = 60 * (attempt + 1)
                    print(f"  Groq rate limited; waiting {wait}s...")
                    time.sleep(wait)
                    continue
            print(f"  {backend} error (attempt {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))

    print(f"  WARNING: All {backend} attempts failed")
    return None
