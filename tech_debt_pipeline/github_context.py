import base64
import math
import re
import time
from urllib.parse import quote as url_quote
import requests
from . import config
from .models import GitHubContext, BaselineCommit
from .complexity import compute_complexity_delta

# Global API call counter
_api_call_count = 0
_api_remaining = 5000


def get_api_call_count() -> int:
    return _api_call_count


def get_api_remaining() -> int:
    return _api_remaining


def _headers() -> dict:
    h = {"Accept": "application/vnd.github.v3+json"}
    if config.GITHUB_TOKEN:
        h["Authorization"] = f"token {config.GITHUB_TOKEN}"
    return h


def _rate_limit_wait(response: requests.Response) -> None:
    global _api_remaining
    remaining = int(response.headers.get("X-RateLimit-Remaining", 100))
    _api_remaining = remaining
    if remaining < config.API_QUOTA_WARNING_THRESHOLD:
        print(f"  WARNING: GitHub API quota low ({remaining} remaining)")
    if remaining < 10:
        reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
        wait = max(reset_time - int(time.time()), 1)
        print(f"  Rate limit low ({remaining} remaining), waiting {wait}s...")
        time.sleep(min(wait, 60))


def _api_get(url: str, retries: int = 3) -> requests.Response | None:
    global _api_call_count
    for attempt in range(retries):
        try:
            _api_call_count += 1
            if _api_call_count % 100 == 0:
                print(f"  [API] {_api_call_count} calls made, ~{_api_remaining} remaining")
            resp = requests.get(url, headers=_headers(), timeout=15)
            if resp.status_code == 200:
                _rate_limit_wait(resp)
                return resp
            if resp.status_code in (403, 429):
                _rate_limit_wait(resp)  # update remaining count even on rate-limit responses
                wait = 2 ** (attempt + 1)
                print(f"  Rate limited (HTTP {resp.status_code}), retrying in {wait}s...")
                time.sleep(wait)
                continue
            if resp.status_code == 404:
                return None
            print(f"  Unexpected HTTP {resp.status_code} from {url}")
            return None
        except requests.RequestException as e:
            print(f"  Request error: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    return None


def fetch_file_at_commit(repo: str, path: str, sha: str) -> str | None:
    """Fetch full file content at a specific commit."""
    url = f"{config.GITHUB_API_BASE}/repos/{repo}/contents/{url_quote(path, safe='')}?ref={sha}"
    resp = _api_get(url)
    if resp is None:
        return None
    data = resp.json()
    content = data.get("content", "")
    if not content:
        return None
    try:
        return base64.b64decode(content).decode("utf-8", errors="replace")
    except Exception:
        return None


def fetch_commit_history(repo: str, path: str, per_page: int = config.BASELINE_COMMIT_HISTORY_DEPTH) -> list[dict]:
    """Fetch recent commits touching a specific file."""
    url = f"{config.GITHUB_API_BASE}/repos/{repo}/commits?path={url_quote(path, safe='')}&per_page={per_page}"
    resp = _api_get(url)
    if resp is None:
        return []
    data = resp.json()
    if not isinstance(data, list):
        return []
    commits = []
    for c in data:
        commits.append({
            "sha": c.get("sha", ""),
            "message": c.get("commit", {}).get("message", ""),
            "author": c.get("commit", {}).get("author", {}).get("name", ""),
            "date": c.get("commit", {}).get("author", {}).get("date", ""),
        })
    return commits


def fetch_commit_patch(repo: str, sha: str, path: str) -> dict | None:
    """Fetch patch for a specific file in a specific commit."""
    url = f"{config.GITHUB_API_BASE}/repos/{repo}/commits/{sha}"
    resp = _api_get(url)
    if resp is None:
        return None
    data = resp.json()
    files = data.get("files", [])
    for f in files:
        if f.get("filename") == path:
            return {
                "patch": f.get("patch", ""),
                "additions": f.get("additions", 0),
                "deletions": f.get("deletions", 0),
            }
    # File not found — may be truncated (>300 files)
    return None


def _extract_snippet(content: str, line: int, context: int = config.FILE_CONTEXT_LINES) -> str:
    """Extract ±context lines around the target line."""
    lines = content.split("\n")
    start = max(0, line - context - 1)
    end = min(len(lines), line + context)
    snippet_lines = []
    for i, l in enumerate(lines[start:end], start=start + 1):
        marker = ">>>" if i == line else "   "
        snippet_lines.append(f"{marker} {i:4d} | {l}")
    return "\n".join(snippet_lines)


def count_lines_from_patch(patch: str) -> tuple[int, int]:
    """Count added/removed lines from a unified diff patch."""
    patch_lines = patch.split('\n')
    added = sum(1 for l in patch_lines if l.startswith('+') and not l.startswith('+++'))
    removed = sum(1 for l in patch_lines if l.startswith('-') and not l.startswith('---'))
    return added, removed


STRUCTURAL_PATTERN = re.compile(
    r'^[+-]\s*(def |function |class |async |export |const |let |var |public |private |static )')


def is_structural_change(patch: str) -> bool:
    """Check if a patch modifies structural elements (function/class definitions)."""
    return any(STRUCTURAL_PATTERN.match(l) for l in patch.split('\n'))


_AI_KEYWORD_PATTERN = re.compile(
    r'\b(' + '|'.join(re.escape(kw) for kw in config.AI_KEYWORDS) + r')\b',
    re.IGNORECASE,
)


def _is_ai_keyword_commit(message: str) -> bool:
    """Check if a commit message references AI tools (word boundary match)."""
    return bool(_AI_KEYWORD_PATTERN.search(message))


def fetch_baseline_commits(
    repo: str,
    path: str,
    ai_commit_date: str,
    ai_patch_size: int,
    commit_history: list[dict],
    count: int = config.BASELINE_COMMITS_TO_KEEP,
) -> list[dict]:
    """Fetch and score baseline (non-AI) commits for comparison.

    Returns list of BaselineCommit dicts, scored by similarity to the AI commit.
    """
    from .diff_parser import parse_patch

    # Filter: before AI commit, not AI-related
    candidates = []
    for c in commit_history:
        if not c.get("date") or not ai_commit_date:
            continue
        if c["date"] >= ai_commit_date:
            continue
        if _is_ai_keyword_commit(c.get("message", "")):
            continue
        candidates.append(c)

    if not candidates:
        return []

    # Fetch patches for up to BASELINE_CANDIDATES_TO_FETCH candidates
    scored = []
    for c in candidates[:config.BASELINE_CANDIDATES_TO_FETCH]:
        patch_data = fetch_commit_patch(repo, c["sha"], path)
        if patch_data is None or not patch_data.get("patch"):
            continue

        patch = patch_data["patch"]
        patch_size = len(patch)
        added, removed = count_lines_from_patch(patch)
        structural = is_structural_change(patch)

        # Hard reject baselines too different in size (or if AI patch is empty)
        if ai_patch_size == 0 or patch_size == 0:
            continue
        size_ratio = max(patch_size, ai_patch_size) / min(patch_size, ai_patch_size)
        if size_ratio > config.MAX_BASELINE_SIZE_RATIO:
            continue

        # Score by size similarity
        size_score = 1.0 / (1 + abs(math.log2(max(patch_size, 1) / max(ai_patch_size, 1))))

        # Structural tiebreaker bonus
        if structural:
            size_score += 0.2

        # Compute complexity
        parsed = parse_patch(patch)
        before_code = parsed.get("before_code", "")
        after_code = parsed.get("after_code", "")

        delta = compute_complexity_delta(before_code, after_code, path)

        baseline = BaselineCommit(
            sha=c["sha"],
            message=c.get("message", ""),
            author=c.get("author", ""),
            date=c.get("date", ""),
            patch=patch[:config.MAX_PATCH_CHARS],
            patch_size=patch_size,
            is_structural=structural,
            complexity_avg=delta.get("after", {}).get("avg_complexity", 0.0),
            complexity_max=delta.get("after", {}).get("max_complexity", 0),
            nloc=delta.get("after", {}).get("total_nloc", 0),
            lines_added=added,
            lines_removed=removed,
            complexity_error=delta.get("error", ""),
        )

        scored.append((size_score, baseline))

    # Sort by score descending, take top `count`
    scored.sort(key=lambda x: x[0], reverse=True)
    return [b.to_dict() for _, b in scored[:count]]


def gather_github_context(record: dict) -> GitHubContext:
    """Gather all GitHub context for a record."""
    repo = record["repo_full_name"]
    path = record["path"]
    line = record.get("line", 0)
    commits = record.get("commits", {})
    first_commit = commits.get("first_commit") or {}
    sha = first_commit.get("sha", "")

    ctx = GitHubContext()

    # 1. Fetch file content at commit (kept only for the ±50-line snippet)
    if sha:
        print(f"  Fetching file content for {repo}/{path}@{sha[:7]}...")
        content = fetch_file_at_commit(repo, path, sha)
        if content:
            ctx.file_available = True
            ctx.file_content_snippet = _extract_snippet(content, line)
        else:
            print(f"  File not available via API, using patch from dataset")

    # 2. Fetch commit history
    print(f"  Fetching commit history for {path}...")
    history = fetch_commit_history(repo, path)
    if history:
        ctx.commit_history = history
    else:
        # Fall back to dataset commit info
        fallback = []
        if first_commit:
            fallback.append({
                "sha": first_commit.get("sha", ""),
                "message": first_commit.get("message", ""),
                "author": first_commit.get("author", ""),
                "date": first_commit.get("date", ""),
            })
        second = commits.get("second_commit")
        if second and isinstance(second, dict):
            fallback.append({
                "sha": second.get("sha", ""),
                "message": second.get("message", ""),
                "author": second.get("author", ""),
                "date": second.get("date", ""),
            })
        for o in commits.get("other_commits", []):
            if isinstance(o, dict):
                fallback.append({
                    "sha": o.get("sha", ""),
                    "message": o.get("message", ""),
                    "author": o.get("author", ""),
                    "date": o.get("date", ""),
                })
        ctx.commit_history = fallback

    return ctx
