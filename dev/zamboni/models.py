"""
models.py — Structured Data Contracts, Route Decisions, & Trajectory Telemetry for Agent Zamboni.

Incorporates advanced agentic patterns:
- Pattern 3 (Adaptive Multi-Pipeline Router): `RouteDecision` with `pipeline_ids` and `extracted_entities`.
- Pattern 2 & 5 (State Projection & Strategic Reflection): `TrajectoryTelemetry` tracking
  context projection size, deterministic critic validation, and pruned reflection attempts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


class ZamboniDomainRejectionError(ValueError):
    """Raised when a user query violates the Hockey-Only Domain Guardrail."""

    def __init__(self, message: str, error_code: str = "NON_HOCKEY_QUERY"):
        super().__init__(message)
        self.error_code = error_code
        self.message = message


class ZamboniRateLimitError(RuntimeError):
    """Raised when Hugging Face global cross-user rate limit (1 req/sec) is exceeded."""

    def __init__(self, retry_after_sec: float):
        msg = (
            f"⏱️ Global Rate Limit Active (Hugging Face Protection): Agent Zamboni is limited to "
            f"1 response per second across all internet users. Please wait {retry_after_sec:.2f}s and try again."
        )
        super().__init__(msg)
        self.retry_after_sec = retry_after_sec
        self.error_code = "GLOBAL_RATE_LIMIT_EXCEEDED"
        self.message = msg


@dataclass
class GuardrailDecision:
    """Result of the 3-Layer Hockey-Only Domain Guardrail evaluation."""

    allowed: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    matched_signals: List[str] = field(default_factory=list)
    similarity_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RouteDecision:
    """
    Pattern 3 (Adaptive Router Agent) Structured Decision Contract.
    Supports mixed-intent queries by returning a list of `pipeline_ids` to execute in parallel.
    """

    intent: str                                # "METHODOLOGY" | "DATA_QUERY" | "HYBRID"
    pipeline_ids: List[str]                    # e.g. ["methodology_rag", "skill_skater_analytics"]
    target_domains: List[str]                  # e.g. ["xg_gsax", "player_ppi"]
    extracted_entities: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrajectoryTelemetry:
    """
    Trajectory Analysis & Context Engineering Telemetry for Observability & Label Studio HITL.
    Tracks tool selection, projected context window size, and reflection loop behavior.
    """

    pipeline_ids: List[str] = field(default_factory=list)
    projected_context_chars: int = 0
    projected_tokens_est: int = 0
    reflection_attempts: int = 0
    critic_passed: bool = True
    critic_feedback: Optional[str] = None
    schema_compliant: bool = True
    prompt_prefix_cached: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class KnowledgeChunk:
    """Canonical Rinklytics methodology and calculation reference chunk."""

    chunk_id: str
    title: str
    domain: str
    page_view: str
    source_files: List[str]
    hf_dataset_tables: List[str]
    formulas_latex: List[str]
    summary: str
    methodology_markdown: str
    keywords: List[str]
    score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ComputeSkillResult:
    """Structured output from a deterministic DuckDB / Pandas skill execution."""

    skill_name: str
    title: str
    dataset_source: str
    summary_markdown: str
    records: List[Dict[str, Any]] = field(default_factory=list)
    columns: List[str] = field(default_factory=list)
    query_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ZamboniResponse:
    """Complete verified response package returned by Agent Zamboni."""

    query: str
    allowed: bool
    answer_markdown: str
    intent: str = "METHODOLOGY"
    persona: str = "zamboni"
    target_domains: List[str] = field(default_factory=list)
    entities: Dict[str, Any] = field(default_factory=dict)
    retrieved_chunks: List[KnowledgeChunk] = field(default_factory=list)
    computed_results: List[ComputeSkillResult] = field(default_factory=list)
    generator_model: str = "zamboni-guardrails-v1"
    error_code: Optional[str] = None
    latency_ms: float = 0.0
    is_local: bool = True
    lm_studio_online: bool = False
    trajectory: TrajectoryTelemetry = field(default_factory=TrajectoryTelemetry)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "allowed": self.allowed,
            "answer_markdown": self.answer_markdown,
            "intent": self.intent,
            "persona": self.persona,
            "target_domains": self.target_domains,
            "entities": self.entities,
            "retrieved_chunks": [c.to_dict() for c in self.retrieved_chunks],
            "computed_results": [r.to_dict() for r in self.computed_results],
            "generator_model": self.generator_model,
            "error_code": self.error_code,
            "latency_ms": self.latency_ms,
            "is_local": self.is_local,
            "lm_studio_online": self.lm_studio_online,
            "trajectory": self.trajectory.to_dict(),
        }
