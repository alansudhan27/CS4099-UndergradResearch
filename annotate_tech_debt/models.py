from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class TechDebtAnnotation:
    no_debt_detected: bool = False
    debt_categories: list[str] = field(default_factory=list)
    debt_category_reasoning: dict[str, str] = field(default_factory=dict)
    developer_awareness: str = ""  # aware, partially_aware, unaware
    developer_awareness_reasoning: str = ""
    overall_assessment: str = ""  # positive, negative, mixed
    overall_reasoning: str = ""
    adds_complexity: bool = False
    complexity_explanation: str = ""
    evidence_of_subsequent_fixes: bool = False
    fix_description: Optional[str] = None
    code_quality_score: int = 0  # 1-5
    code_quality_reasoning: str = ""
    confidence: float = 0.0
    evidence_citations: list[dict] = field(default_factory=list)
    # Each: {"claim": "...", "code_reference": "...", "explanation": "..."}

    # AI Change Analysis (separate from developer behavior)
    ai_made_mistake: bool = False
    ai_mistake_description: str = ""
    ai_added_complexity: bool = False
    ai_complexity_description: str = ""
    ai_suggestion_quality: int = 0  # 1-5
    ai_suggestion_quality_reasoning: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TechDebtAnnotation":
        """Simple constructor from dict — no coercion, filters unknown keys."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    @classmethod
    def validate_and_coerce(cls, data: dict) -> tuple["TechDebtAnnotation", list[str]]:
        """Coerce types + collect violations. Used by annotation pipeline."""
        violations = []
        cleaned = {}
        bool_fields = {"no_debt_detected", "adds_complexity", "evidence_of_subsequent_fixes",
                       "ai_made_mistake", "ai_added_complexity"}
        int_fields = {"code_quality_score", "ai_suggestion_quality"}
        float_fields = {"confidence"}

        for k, v in data.items():
            if k in bool_fields and not isinstance(v, bool):
                cleaned[k] = str(v).lower() in ("true", "yes", "1")
                violations.append(f"coerced {k}: {type(v).__name__} '{v}' -> bool")
            elif k in int_fields and not isinstance(v, int):
                try:
                    cleaned[k] = int(float(str(v)))
                    violations.append(f"coerced {k}: {type(v).__name__} '{v}' -> int")
                except (ValueError, TypeError):
                    cleaned[k] = 0  # dataclass default; value was unusable
                    violations.append(f"failed to coerce {k} from '{v}' to int, defaulting to 0")
            elif k in float_fields and not isinstance(v, (int, float)):
                try:
                    cleaned[k] = float(str(v))
                    violations.append(f"coerced {k}: {type(v).__name__} '{v}' -> float")
                except (ValueError, TypeError):
                    cleaned[k] = 0.0  # dataclass default; value was unusable
                    violations.append(f"failed to coerce {k} from '{v}' to float, defaulting to 0.0")
            else:
                cleaned[k] = v

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in cleaned.items() if k in valid_fields}
        return cls(**filtered), violations


@dataclass
class ComplexityMetrics:
    before_avg_complexity: float = 0.0
    after_avg_complexity: float = 0.0
    before_max_complexity: int = 0
    after_max_complexity: int = 0
    before_nloc: int = 0
    after_nloc: int = 0
    before_functions: list[dict] = field(default_factory=list)
    after_functions: list[dict] = field(default_factory=list)
    delta_avg: float = 0.0
    delta_max: int = 0
    delta_nloc: int = 0
    delta_functions: int = 0
    complexity_increased: bool = False
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_delta(cls, delta: dict) -> "ComplexityMetrics":
        before = delta.get("before", {})
        after = delta.get("after", {})
        return cls(
            before_avg_complexity=before.get("avg_complexity", 0.0),
            after_avg_complexity=after.get("avg_complexity", 0.0),
            before_max_complexity=before.get("max_complexity", 0),
            after_max_complexity=after.get("max_complexity", 0),
            before_nloc=before.get("total_nloc", 0),
            after_nloc=after.get("total_nloc", 0),
            before_functions=before.get("functions", []),
            after_functions=after.get("functions", []),
            delta_avg=delta.get("delta_avg", 0.0),
            delta_max=delta.get("delta_max", 0),
            delta_nloc=delta.get("delta_nloc", 0),
            delta_functions=delta.get("delta_functions", 0),
            complexity_increased=delta.get("complexity_increased", False),
            error=delta.get("error", ""),
        )


@dataclass
class BaselineCommit:
    sha: str = ""
    message: str = ""
    author: str = ""
    date: str = ""
    patch: str = ""
    patch_size: int = 0
    is_structural: bool = False
    complexity_avg: float = 0.0
    complexity_max: int = 0
    nloc: int = 0
    lines_added: int = 0
    lines_removed: int = 0
    complexity_error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GitHubContext:
    file_content_snippet: str = ""
    file_available: bool = False
    full_file_content: str = ""
    commit_history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AnnotatedRecord:
    id: str
    original_comment: str
    repo_full_name: str
    path: str
    html_url: str = ""
    line: int = 0
    existing_labels: dict = field(default_factory=dict)
    original_commits: dict = field(default_factory=dict)
    github_context: Optional[GitHubContext] = None
    tech_debt_analysis: Optional[TechDebtAnnotation] = None
    complexity_metrics: Optional[ComplexityMetrics] = None
    parsed_diff: dict = field(default_factory=dict)
    annotation_metadata: dict = field(default_factory=dict)
    baseline_commits: list[dict] = field(default_factory=list)
    sample_group: str = ""  # "primary" or "exploratory"
    fetch_errors: list[str] = field(default_factory=list)
    patch_was_truncated: bool = False
    lines_added: int = 0
    lines_removed: int = 0
    annotation_status: str = "pending"  # "pending", "success", "failed", "skipped"
    schema_violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d
