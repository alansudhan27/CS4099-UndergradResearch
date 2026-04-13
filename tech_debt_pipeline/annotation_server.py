"""Local Flask server for manual tech-debt annotation.

Run with:
    python -m tech_debt_pipeline.annotation_server

Then open http://localhost:5000
"""

import json
import os
import shutil
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from . import config

app = Flask(__name__)

DATA_PATH: Path = config.PREPARED_JSON_PATH

# In-memory store: loaded once at startup
_records: list[dict] = []
_index: dict[str, int] = {}  # record id → position in _records


def load_data() -> None:
    global _records, _index
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Prepared dataset not found: {DATA_PATH}")
    with open(DATA_PATH, encoding="utf-8") as f:
        _records = json.load(f)
    _index = {r["id"]: i for i, r in enumerate(_records)}
    print(f"[annotator] Loaded {len(_records)} records from {DATA_PATH}")


def save_data() -> None:
    tmp = DATA_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_records, f, indent=2, ensure_ascii=False)
    shutil.move(str(tmp), str(DATA_PATH))


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("annotator.html")


@app.route("/api/records")
def api_records():
    """Sidebar list — minimal fields only. Includes numeric idx for URL safety."""
    out = []
    for i, r in enumerate(_records):
        out.append({
            "idx": i,
            "id": r["id"],
            "status": r.get("annotation_status", "pending"),
            "repo": r.get("repo_full_name", ""),
            "path": r.get("path", ""),
            "line": r.get("line", 0),
            "sample_group": r.get("sample_group", ""),
        })
    return jsonify(out)


@app.route("/api/record/<int:idx>")
def api_record(idx: int):
    """Full record for display. Uses integer index to avoid URL-encoding issues."""
    if idx < 0 or idx >= len(_records):
        return jsonify({"error": "not found"}), 404
    return jsonify(_records[idx])


@app.route("/api/record/<int:idx>/annotate", methods=["POST"])
def api_annotate(idx: int):
    """Save manual annotation back to the dataset."""
    if idx < 0 or idx >= len(_records):
        return jsonify({"error": "not found"}), 404

    body = request.get_json(force=True)
    if not body:
        return jsonify({"error": "empty body"}), 400

    _records[idx]["tech_debt_analysis"] = body
    _records[idx]["annotation_status"] = "success"

    save_data()
    return jsonify({"ok": True})


@app.route("/api/record/<int:idx>/skip", methods=["POST"])
def api_skip(idx: int):
    """Mark a record as skipped."""
    if idx < 0 or idx >= len(_records):
        return jsonify({"error": "not found"}), 404
    _records[idx]["annotation_status"] = "skipped"
    save_data()
    return jsonify({"ok": True})


@app.route("/api/record/<int:idx>/suggest", methods=["POST"])
def api_suggest(idx: int):
    """Call LLM (Groq or Ollama) and return annotation suggestion without saving."""
    if idx < 0 or idx >= len(_records):
        return jsonify({"error": "not found"}), 404

    body = request.get_json(force=True) or {}
    backend = body.get("backend", "groq")

    from .llm_annotator import annotate_record
    from .models import GitHubContext

    record = _records[idx]
    ctx = GitHubContext()
    gh = record.get("github_context") or {}
    ctx.file_content_snippet = gh.get("file_content_snippet", "")
    ctx.commit_history = gh.get("commit_history", [])
    ctx.file_available = gh.get("file_available", False)

    annotation = annotate_record(record, ctx, backend=backend)
    if annotation is None:
        return jsonify({"error": "LLM annotation failed — check API key or model availability"}), 500

    return jsonify(annotation.to_dict())


@app.route("/api/progress")
def api_progress():
    total = len(_records)
    annotated = sum(1 for r in _records if r.get("annotation_status") == "success")
    skipped = sum(1 for r in _records if r.get("annotation_status") == "skipped")
    pending = total - annotated - skipped
    return jsonify({
        "total": total,
        "annotated": annotated,
        "skipped": skipped,
        "pending": pending,
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_data()
    app.run(debug=False, port=5000)


# Support `python -m tech_debt_pipeline.annotation_server`
def main():
    load_data()
    print("[annotator] Open http://localhost:5000")
    app.run(debug=False, port=5000)
