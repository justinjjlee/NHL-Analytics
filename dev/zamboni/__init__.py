"""
dev/zamboni — Agent Zamboni: Hockey Analytics Methodology & Hugging Face Dataset Q&A Engine.
"""

from .agent import ZamboniAgent
from .config import AGENT_PERSONAS, NHL_TEAMS
from .guardrails import evaluate_hockey_guardrail, clean_zamboni_prose
from .models import (
    ComputeSkillResult,
    GuardrailDecision,
    KnowledgeChunk,
    ZamboniDomainRejectionError,
    ZamboniRateLimitError,
    ZamboniResponse,
)
from .rate_limiter import GlobalRateLimiter, get_global_rate_limiter
from .recap_engine import (
    clean_llm_narrative,
    expand_first_mention_abbreviations,
    extract_headline_and_narrative,
    format_milestone_badge,
    reconcile_game_record,
    reconcile_slate_and_games_records,
    verify_game_narrative_facts,
    verify_slate_narrative_facts,
)

__all__ = [
    "ZamboniAgent",
    "AGENT_PERSONAS",
    "NHL_TEAMS",
    "evaluate_hockey_guardrail",
    "clean_zamboni_prose",
    "ComputeSkillResult",
    "GuardrailDecision",
    "KnowledgeChunk",
    "ZamboniDomainRejectionError",
    "ZamboniRateLimitError",
    "ZamboniResponse",
    "GlobalRateLimiter",
    "get_global_rate_limiter",
    "clean_llm_narrative",
    "expand_first_mention_abbreviations",
    "extract_headline_and_narrative",
    "format_milestone_badge",
    "reconcile_game_record",
    "reconcile_slate_and_games_records",
    "verify_game_narrative_facts",
    "verify_slate_narrative_facts",
]
