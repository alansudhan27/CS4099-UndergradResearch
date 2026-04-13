import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "combined_whole_dataset_per_comment.json"

# Pipeline output paths
PIPELINE_DIR = BASE_DIR / "pipeline_output"
SAMPLED_JSON_PATH = PIPELINE_DIR / "sampled_dataset.json"      # original schema, new dataset
PREPARED_JSON_PATH = PIPELINE_DIR / "prepared_dataset.json"    # enriched, for annotate stage
PREPARED_CSV_PATH = PIPELINE_DIR / "prepared_dataset.csv"
PREPARATION_SUMMARY_PATH = PIPELINE_DIR / "preparation_summary.md"
ANNOTATION_REPORT_PATH = PIPELINE_DIR / "annotation_report.md"

GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
GROQ_MAX_COMPLETION_TOKENS = int(os.getenv("GROQ_MAX_COMPLETION_TOKENS", "800"))
GITHUB_API_BASE = "https://api.github.com"

# Ollama (local model) config
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_REQUEST_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "120"))
OLLAMA_MAX_COMPLETION_TOKENS = int(os.getenv("OLLAMA_MAX_COMPLETION_TOKENS", "800"))

# Backend selection: "groq" or "ollama"
DEFAULT_BACKEND = "groq"

# Sampling
TARGET_SAMPLE_SIZE = 500
EXPLORATORY_SAMPLE_SIZE = 40
MIN_PER_CATEGORY = 25
RANDOM_SEED = 42

# Baselines
BASELINE_CANDIDATES_TO_FETCH = 15   # was 5 — try more candidates before giving up
BASELINE_COMMITS_TO_KEEP = 2
BASELINE_COMMIT_HISTORY_DEPTH = 50  # was 10 — fetch more history per file
MAX_BASELINE_SIZE_RATIO = 5         # hard reject baselines >5x different in patch size
MIN_REQUIRED_BASELINES = 1          # records with 0 baselines are rejected and replaced

# Pre-selection filters
MIN_COMMENT_LEN = 10                # was 20
MIN_PATCH_CHARS = 20                # was 50

# Groq rate limits (qwen/qwen3-32b tier)
GROQ_RATE_LIMIT_RPM = 3               # TPM=6K ÷ ~1900 tokens/call → floor to 3
GROQ_RPD = 1_000                      # requests per day
GROQ_TPD = 500_000                    # tokens per day
GROQ_ESTIMATED_TOKENS_PER_CALL = 1_900  # observed ~1891; rounded up for safety
API_QUOTA_WARNING_THRESHOLD = 100

TECH_DEBT_CATEGORIES = [
    "Complexity of change",
    "Size of change",
    "Programmer didn't know the codebase",
    "Programmer didn't know the skill/language",
    "AI hallucination / incorrect suggestion",
    "Lack of testing",
    "Over-reliance on AI without review",
]

DEVELOPER_AWARENESS_LEVELS = ["aware", "partially_aware", "unaware"]

# Content truncation limits
FILE_CONTEXT_LINES = 50  # ±50 lines around relevant line
MAX_PATCH_CHARS = 3000

# AI keyword detection for baseline filtering
AI_KEYWORDS = [
    "copilot", "chatgpt", "gpt-4", "gpt-3", "claude",
    "ai-generated", "github copilot", "openai", "ai generated",
    "generative ai", "gen ai", "llm", "language model",
]
