"""
agent.py — Main Orchestrator & Composite State-Graph Engine for Agent Zamboni.

Executes the 6-Node Composite Architecture (Patterns 2, 3, 4 & 5):
  Node 1: Global Rate Limiter (1 req/sec on HF Spaces; unlimited locally) & Hockey-Only Guardrail
  Node 2: Pattern 3 Adaptive Multi-Pipeline Router (`route_query_decision`)
  Node 3: Pattern 4 Parallel Specialist Workers: Hybrid BM25 RAG + DuckDB/Pandas Compute Skills
  Node 4: Label Studio Golden Q&A Cache Lookup (`insights/zamboni/golden_qa.parquet`)
  Node 5: Pattern 2 State Projection & Prompt Prefix Caching (`ZamboniSynthesizer`)
  Node 6: Pattern 5 Strategic Deterministic Reflection (with Context Pruning via Diffing)
"""

from __future__ import annotations

import time
from typing import Optional

from .compute_engine import ZamboniComputeEngine
from .config import AGENT_PERSONAS
from .dataset_store import HFZamboniStore
from .env_utils import is_local_streamlit
from .guardrails import evaluate_hockey_guardrail
from .knowledge_rag import ZamboniKnowledgeRAG
from .models import TrajectoryTelemetry, ZamboniResponse
from .rate_limiter import get_global_rate_limiter
from .router import route_query_decision
from .synthesizer import ZamboniSynthesizer


class ZamboniAgent:
    """
    Production Hockey Analytics & Methodology Reference Agent for Rinklytics.
    Combines Pattern 3 (Adaptive Multi-Pipeline Router), Pattern 2 (State Projection),
    and Pattern 5 (Strategic Deterministic Reflection with Context Pruning).
    """

    def __init__(
        self,
        persona: str = "zamboni",
        is_local: Optional[bool] = None,
        log_interactions: bool = False,
    ):
        self.persona = persona if persona in AGENT_PERSONAS else "zamboni"
        self.is_local = is_local_streamlit() if is_local is None else bool(is_local)
        # Ephemeral privacy mandate: Never persist user chat history across sessions by default
        self.log_interactions = log_interactions

        self.rate_limiter = get_global_rate_limiter()
        self.rag = ZamboniKnowledgeRAG()
        self.compute = ZamboniComputeEngine()
        self.synthesizer = ZamboniSynthesizer(persona=self.persona)
        self.store = HFZamboniStore()

    def ask(
        self,
        query: str,
        lang: str = "EN",
        persona: Optional[str] = None,
        force_mode: str = "auto",
        top_k_rag: int = 3,
    ) -> ZamboniResponse:
        """
        End-to-end execution of the 6-Node Agent Zamboni pipeline with Trajectory Telemetry.
        """
        t0 = time.perf_counter()
        active_persona = persona if persona in AGENT_PERSONAS else self.persona
        self.synthesizer.refresh_health()

        # ── Node 1A: Global Cross-User Rate Limiter (1 req/sec on HF Spaces) ──
        allowed_rate, retry_after = self.rate_limiter.try_acquire(is_local=self.is_local)
        if not allowed_rate:
            msg = (
                f"⏱️ **Global Rate Limit Active (`GLOBAL_RATE_LIMIT_EXCEEDED`)**: "
                f"In Hugging Face production, Agent Zamboni is guarded at **1 response per second across all internet users** "
                f"to prevent abuse and protect compute resources. Please wait **{retry_after:.2f}s** and submit again."
                if lang == "EN"
                else f"⏱️ **Limite de débit globale active (`GLOBAL_RATE_LIMIT_EXCEEDED`)** : "
                f"Sur Hugging Face, l'Agent Zamboni est limité à **1 réponse par seconde pour tous les utilisateurs** "
                f"afin de protéger les ressources. Veuillez patienter **{retry_after:.2f}s** avant de réessayer."
            )
            return ZamboniResponse(
                query=query,
                allowed=False,
                answer_markdown=msg,
                persona=active_persona,
                error_code="GLOBAL_RATE_LIMIT_EXCEEDED",
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                is_local=self.is_local,
                lm_studio_online=self.synthesizer.lm_studio_online,
            )

        # ── Node 1B: Strict Hockey-Only Domain Guardrail ──────────────────────
        guard = evaluate_hockey_guardrail(query, lang=lang)
        if not guard.allowed:
            resp = ZamboniResponse(
                query=query,
                allowed=False,
                answer_markdown=guard.error_message or "Non-hockey query rejected.",
                persona=active_persona,
                error_code=guard.error_code,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                is_local=self.is_local,
                lm_studio_online=self.synthesizer.lm_studio_online,
            )
            if self.log_interactions:
                self.store.record_interaction(resp)
            return resp

        # ── Node 2: Pattern 3 Adaptive Multi-Pipeline Router ──────────────────
        route_dec = route_query_decision(query)
        intent = route_dec.intent
        target_domains = route_dec.target_domains
        entities = route_dec.extracted_entities
        pipeline_ids = route_dec.pipeline_ids

        # ── Node 3: Retrieve Canonical Methodology Chunks (RAG Worker) ────────
        chunks = self.rag.retrieve(query=query, target_domains=target_domains, top_k=top_k_rag)

        # ── Node 4: Execute Deterministic DuckDB/Pandas Compute Workers ───────
        computed_results = self.compute.execute_skills(
            intent=intent,
            domains=target_domains,
            entities=entities,
        )

        # ── Node 5: Check Label Studio Golden Q&A Cache ───────────────────────
        golden_hit = self.store.lookup_golden_qa(query)
        if golden_hit and force_mode != "force_llm":
            answer_md = golden_hit["golden_answer"]
            model_stamp = "golden-cache/label-studio-verified"
            trajectory = TrajectoryTelemetry(
                pipeline_ids=pipeline_ids,
                projected_context_chars=0,
                projected_tokens_est=0,
                reflection_attempts=0,
                critic_passed=True,
                schema_compliant=True,
                prompt_prefix_cached=True,
            )
        else:
            # ── Node 6: State Projection + Synthesis + Strategic Reflection ───
            answer_md, model_stamp, trajectory = self.synthesizer.synthesize(
                query=query,
                intent=intent,
                chunks=chunks,
                compute_results=computed_results,
                persona=active_persona,
                force_mode=force_mode,
                pipeline_ids=pipeline_ids,
                lang=lang,
            )

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        response = ZamboniResponse(
            query=query,
            allowed=True,
            answer_markdown=answer_md,
            intent=intent,
            persona=active_persona,
            target_domains=target_domains,
            entities=entities,
            retrieved_chunks=chunks,
            computed_results=computed_results,
            generator_model=model_stamp,
            error_code=None,
            latency_ms=elapsed_ms,
            is_local=self.is_local,
            lm_studio_online=self.synthesizer.lm_studio_online,
            trajectory=trajectory,
        )

        if self.log_interactions:
            self.store.record_interaction(response)

        return response
