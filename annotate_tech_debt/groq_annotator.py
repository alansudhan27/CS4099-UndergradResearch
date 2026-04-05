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

You will be given:
- A developer's code comment that references an AI tool
- The surrounding code context and/or commit patches
- File commit history showing how the code evolved

Your task is to assess whether this AI-assisted code change introduces, maintains, or reduces technical debt. Be neutral — it is equally valid to find no technical debt. If the code is clean and correct, say so. Do not force categories that don't apply.

You MUST respond with valid JSON matching this exact schema:
{
  "no_debt_detected": true/false,
  "debt_categories": ["list applicable categories, or empty list if none apply"],
  "debt_category_reasoning": {"category": "explanation for why this category applies or doesn't"},
  "developer_awareness": "aware | partially_aware | unaware",
  "developer_awareness_reasoning": "explanation of whether the developer recognized any AI-introduced issue (or note if no issue was introduced)",
  "overall_assessment": "positive | negative | mixed",
  "overall_reasoning": "explanation of overall impact",
  "adds_complexity": true/false,
  "complexity_explanation": "explanation",
  "evidence_of_subsequent_fixes": true/false,
  "fix_description": "description of fixes or null",
  "code_quality_score": 1-5,
  "code_quality_reasoning": "explanation of code quality rating",
  "confidence": 0.0-1.0,
  "evidence_citations": [
    {
      "claim": "what you are asserting (e.g. 'adds complexity via nested conditionals' OR 'code is correct and idiomatic')",
      "code_reference": "the exact code snippet from the patch or file that proves this claim",
      "explanation": "how this code proves the claim"
    }
  ],
  "ai_made_mistake": true/false,
  "ai_mistake_description": "description of any mistake or incorrect pattern the AI tool introduced, or empty string if none",
  "ai_added_complexity": true/false,
  "ai_complexity_description": "how the AI suggestion specifically added, reduced, or did not affect complexity",
  "ai_suggestion_quality": 1-5,
  "ai_suggestion_quality_reasoning": "assessment of the AI tool's output quality as code, independent of how the developer used it"
}

RATING RUBRICS:

Rate AI suggestion quality (ai_suggestion_quality, 1-5):
  1 = Code has bugs, doesn't compile/run, or produces wrong results
  2 = Code works but has significant issues (security holes, performance problems, logic errors)
  3 = Code works correctly but is not idiomatic or has minor style/structure issues
  4 = Code is correct and reasonably well-written with minor improvements possible
  5 = Code is correct, idiomatic, efficient, and well-structured

Rate overall code quality (code_quality_score, 1-5):
  1 = Very poor — introduces bugs, security issues, or fundamentally broken logic
  2 = Below average — works but with significant maintainability or correctness concerns
  3 = Average — functional code with typical minor issues
  4 = Good — clean, readable, well-structured with minor improvements possible
  5 = Excellent — exemplary code quality, good patterns, well-tested

Rate your confidence (confidence, 0.0-1.0) in the OVERALL assessment:
  0.0-0.3 = Very uncertain — limited evidence, ambiguous code, or insufficient context
  0.4-0.6 = Moderate — some evidence but significant uncertainty
  0.7-0.8 = Confident — clear evidence, minor ambiguity
  0.9-1.0 = Very confident — unambiguous evidence, clear-cut case

IMPORTANT INSTRUCTIONS:
1. Set "no_debt_detected": true when the AI-assisted change is clean and introduces no technical debt. In that case, "debt_categories" must be an empty list.
2. For EVERY claim you make (positive OR negative), you MUST provide at least one evidence citation quoting specific code from the patch or file. Cite code for "code is clean" findings just as you would for "code has debt" findings.
3. Separate your analysis into TWO perspectives:
   - **AI tool analysis**: Evaluate the AI-generated code ITSELF. Was the suggestion correct? Did it introduce bugs or unnecessary complexity? Rate the quality of what the AI produced (ai_suggestion_quality, ai_made_mistake, ai_added_complexity).
   - **Developer analysis**: Evaluate the DEVELOPER's behavior. Were they aware of issues? Did they blindly accept or critically review? (developer_awareness, overall_assessment)."""


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


def _build_user_prompt(record: dict, context: GitHubContext) -> str:
    """Build the user prompt with all available context."""
    # Support both original dataset keys and prepared record (AnnotatedRecord) keys
    comment = record.get("comment") or record.get("original_comment", "")
    repo = record["repo_full_name"]
    path = record["path"]
    line = record.get("line", 0)
    annotations = record.get("annotations") or record.get("existing_labels", {})
    rq1a = annotations.get("rq1a") or {}
    rq1b = annotations.get("rq1b") or {}
    commits = record.get("commits") or record.get("original_commits", {})
    first_commit = commits.get("first_commit") or {}

    sections = []

    # Header
    sections.append(f"## Developer Comment\n\"{comment}\"")
    sections.append(f"## File: {path} (in {repo}, line {line})")

    # File content snippet
    if context.file_content_snippet:
        snippet = context.file_content_snippet[:config.MAX_FILE_CONTENT_CHARS]
        sections.append(f"## File Content (around line {line})\n```\n{snippet}\n```")

    # First commit patch (with hunk-boundary truncation)
    if first_commit:
        patch = first_commit.get("patch", "")
        patch, was_truncated = _truncate_patch_at_hunk(patch, config.MAX_PATCH_CHARS)
        msg = first_commit.get("message", "")
        date = first_commit.get("date", "")
        author = first_commit.get("author", "")
        sections.append(
            f"## Original Commit (introducing the AI-assisted code)\n"
            f"- Message: {msg}\n- Author: {author}\n- Date: {date}\n"
            f"### Patch:\n```diff\n{patch}\n```"
        )

    # Second commit
    second = commits.get("second_commit")
    if second and isinstance(second, dict):
        patch2 = second.get("patch", "")
        patch2, _ = _truncate_patch_at_hunk(patch2, config.MAX_PATCH_CHARS)
        sections.append(
            f"## Subsequent Commit\n"
            f"- Message: {second.get('message', '')}\n"
            f"- Author: {second.get('author', '')}\n"
            f"- Date: {second.get('date', '')}\n"
            f"### Patch:\n```diff\n{patch2}\n```"
        )
    else:
        sections.append("## Subsequent Commit\nNo subsequent commit in dataset.")

    # Commit history from GitHub
    if context.commit_history:
        first_date = first_commit.get("date", "")
        subsequent = [c for c in context.commit_history if c.get("date", "") > first_date] if first_date else []
        if subsequent:
            history_lines = [f"## File Commit History ({len(subsequent)} commits after initial)"]
            for c in subsequent[:5]:
                history_lines.append(f"- [{c.get('sha', '?')[:7]}] {c.get('date', '?')}: {c.get('message', '?')[:100]}")
            sections.append("\n".join(history_lines))

    # Existing classification
    sections.append(
        f"## Existing Classification\n"
        f"- AI Use Type (rq1a): {rq1a.get('pred_label', 'N/A')} "
        f"(confidence: {(rq1a.get('confidence') or 0):.3f})\n"
        f"- Comment Type (rq1b): {rq1b.get('pred_label', 'N/A')} "
        f"(confidence: {(rq1b.get('confidence') or 0):.3f})"
    )

    # Instructions
    categories_list = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(config.TECH_DEBT_CATEGORIES))
    sections.append(
        f"## Instructions\n"
        f"Assess this AI-assisted code change. It may introduce, maintain, or reduce technical debt — all three outcomes are valid findings. "
        f"For each of the categories below, determine if it applies and explain why (or why not). Only include a category in `debt_categories` if it genuinely applies. If none apply, set `no_debt_detected` to true and leave `debt_categories` empty.\n"
        f"{categories_list}\n\n"
        f"## Part A: AI Tool Analysis\n"
        f"Evaluate the AI-generated code ITSELF, independent of the developer:\n"
        f"- Did the AI suggestion contain a mistake or incorrect pattern? (ai_made_mistake)\n"
        f"- Did the AI suggestion add unnecessary complexity? (ai_added_complexity)\n"
        f"- Rate the quality of the AI's output as code, 1-5 (ai_suggestion_quality)\n\n"
        f"## Part B: Developer Analysis\n"
        f"Evaluate the developer's behavior:\n"
        f"- Developer awareness: Was the developer aware of potential issues with the AI suggestion? "
        f"(aware = they noted the problem, partially_aware = they used it but with caveats, "
        f"unaware = they accepted it without question)\n"
        f"- Was the change good overall? (positive/negative/mixed)\n"
        f"- Did the change truly add complexity to the codebase?\n"
        f"- Is there evidence of subsequent fixes needed?\n"
        f"- Rate overall code quality 1-5\n\n"
        f"Respond ONLY with valid JSON matching the schema described in the system prompt."
    )

    return "\n\n".join(sections)


_groq_client = None
_ollama_client = None


def _get_groq_client() -> Groq:
    """Return a reusable Groq client instance."""
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=config.GROQ_API_KEY)
    return _groq_client


def _get_ollama_client() -> OpenAI:
    """Return a reusable Ollama client (OpenAI-compatible)."""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OpenAI(
            base_url=config.OLLAMA_BASE_URL,
            api_key="ollama",  # Ollama doesn't require a real key
            timeout=config.OLLAMA_REQUEST_TIMEOUT_SECONDS,
        )
    return _ollama_client


def annotate_record(
    record: dict,
    context: GitHubContext,
    rpm: int = config.GROQ_RATE_LIMIT_RPM,
    backend: str = config.DEFAULT_BACKEND,
) -> TechDebtAnnotation | None:
    """Send record + context to LLM and parse the annotation response.

    Args:
        backend: "groq" for Groq cloud API, "ollama" for local Ollama model.

    Returns TechDebtAnnotation on success, None on total failure.
    """
    if backend == "ollama":
        client = _get_ollama_client()
        model = config.OLLAMA_MODEL
        max_tokens = config.OLLAMA_MAX_COMPLETION_TOKENS
    else:
        client = _get_groq_client()
        model = config.GROQ_MODEL
        max_tokens = config.GROQ_MAX_COMPLETION_TOKENS

    user_prompt = _build_user_prompt(record, context)

    print(f"  Sending to {backend} ({model})...")
    print(f"  Prompt length: ~{len(user_prompt)} chars")
    if backend == "ollama":
        print(f"  Ollama timeout: {config.OLLAMA_REQUEST_TIMEOUT_SECONDS:.0f}s | max tokens: {max_tokens}")

    # Ollama models vary in JSON mode support — try with response_format first, fall back without
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

            # Rate limiting (Groq only — local models don't need it)
            if backend != "ollama":
                sleep_time = max(60 / rpm - elapsed, 0)
                if sleep_time > 0:
                    time.sleep(sleep_time)

            content = response.choices[0].message.content
            # Strip <think>...</think> blocks emitted by DeepSeek R1 and similar
            # reasoning models before the actual JSON response.
            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
            data = json.loads(content)
            annotation, _ = TechDebtAnnotation.validate_and_coerce(data)

            usage = response.usage
            if usage:
                print(f"  Response: {usage.prompt_tokens} prompt + {usage.completion_tokens} completion tokens")
            else:
                print(f"  Response received (token usage not reported)")

            return annotation

        except json.JSONDecodeError as e:
            print(f"  JSON parse error (attempt {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(2)
        except Exception as e:
            err_str = str(e).lower()
            if backend == "ollama" and ("timeout" in err_str or "timed out" in err_str):
                print(
                    f"  Ollama request timed out after {config.OLLAMA_REQUEST_TIMEOUT_SECONDS:.0f}s; skipping this record"
                )
                return None
            # Some Ollama models don't support response_format — retry without it
            if backend == "ollama" and use_json_mode and ("response_format" in err_str or "not supported" in err_str or "unknown" in err_str):
                print(f"  Ollama: JSON mode not supported, retrying without response_format...")
                use_json_mode = False
                continue
            if backend == "groq" and (
                "tokens per day" in err_str
                or "service tier" in err_str and "tokens" in err_str
                or "code': 'rate_limit_exceeded'" in err_str and "requested" in err_str
            ):
                raise QuotaExhaustedError(str(e))
            if backend == "groq" and ("rate limit" in err_str or "429" in err_str or "too many requests" in err_str):
                if attempt < 2:
                    wait_seconds = 60 * (attempt + 1)
                    print(f"  Groq rate limited (attempt {attempt+1}); waiting {wait_seconds}s before retry...")
                    time.sleep(wait_seconds)
                    continue
            print(f"  {backend} API error (attempt {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(2 ** (attempt + 1))

    print(f"  WARNING: All {backend} attempts failed")
    return None
