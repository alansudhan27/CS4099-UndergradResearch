"""Parse unified diff patches into before/after code and detect languages."""

import re


EXTENSION_TO_LANGUAGE = {
    "py": "python",
    "js": "javascript",
    "ts": "typescript",
    "jsx": "javascript",
    "tsx": "typescript",
    "java": "java",
    "c": "c",
    "h": "c",
    "cpp": "cpp",
    "cc": "cpp",
    "cxx": "cpp",
    "hpp": "cpp",
    "cs": "csharp",
    "go": "go",
    "rs": "rust",
    "rb": "ruby",
    "php": "php",
    "swift": "swift",
    "kt": "kotlin",
    "kts": "kotlin",
    "scala": "scala",
    "r": "r",
    "R": "r",
    "sh": "bash",
    "bash": "bash",
    "zsh": "bash",
    "html": "html",
    "css": "css",
    "scss": "scss",
    "sql": "sql",
    "yaml": "yaml",
    "yml": "yaml",
    "json": "json",
    "xml": "xml",
    "md": "markdown",
    "dart": "dart",
    "lua": "lua",
    "pl": "perl",
    "m": "objectivec",
    "vue": "vue",
    "svelte": "svelte",
}


def detect_language(path: str) -> str:
    """Map file extension to markdown language identifier for syntax highlighting."""
    ext = path.rsplit(".", 1)[-1] if "." in path else ""
    return EXTENSION_TO_LANGUAGE.get(ext, "")


def parse_patch(patch: str) -> dict:
    """Parse a unified diff patch into structured components.

    Returns:
        {
            'removed_lines': list[str],  # Lines removed (without '-' prefix)
            'added_lines': list[str],    # Lines added (without '+' prefix)
            'context_lines': list[str],  # Unchanged context lines
            'before_code': str,          # Reconstructed "before" (context + removed, in order)
            'after_code': str,           # Reconstructed "after" (context + added, in order)
            'raw_patch': str,            # Original patch string
            'hunks': list[dict],         # Per-hunk breakdown
        }
    """
    if not patch or not patch.strip():
        return {
            "removed_lines": [],
            "added_lines": [],
            "context_lines": [],
            "before_code": "",
            "after_code": "",
            "raw_patch": "",
            "hunks": [],
        }

    removed_lines = []
    added_lines = []
    context_lines = []
    before_lines = []
    after_lines = []
    hunks = []

    current_hunk = None

    for line in patch.split("\n"):
        # Hunk header
        if line.startswith("@@"):
            if current_hunk:
                hunks.append(current_hunk)
            header_match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)", line)
            current_hunk = {
                "header": line,
                "old_start": int(header_match.group(1)) if header_match else 0,
                "new_start": int(header_match.group(3)) if header_match else 0,
                "removed": [],
                "added": [],
                "context": [],
            }
            continue

        # Skip file-level headers
        if line.startswith("---") or line.startswith("+++"):
            continue

        if line.startswith("-"):
            content = line[1:]
            removed_lines.append(content)
            before_lines.append(content)
            if current_hunk:
                current_hunk["removed"].append(content)
        elif line.startswith("+"):
            content = line[1:]
            added_lines.append(content)
            after_lines.append(content)
            if current_hunk:
                current_hunk["added"].append(content)
        elif line.startswith(" ") or (line == "" and current_hunk):
            content = line[1:] if line.startswith(" ") else line
            context_lines.append(content)
            before_lines.append(content)
            after_lines.append(content)
            if current_hunk:
                current_hunk["context"].append(content)

    if current_hunk:
        hunks.append(current_hunk)

    return {
        "removed_lines": removed_lines,
        "added_lines": added_lines,
        "context_lines": context_lines,
        "before_code": "\n".join(before_lines),
        "after_code": "\n".join(after_lines),
        "raw_patch": patch,
        "hunks": hunks,
    }
