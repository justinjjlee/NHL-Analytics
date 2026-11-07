"""
synthesizer.py — Multi-Tier Narrative & Reference Synthesizer with Context Engineering & Strategic Reflection.

Implements:
1. Context Engineering via State Projection & Observation Truncation (`_project_state_for_llm`, `truncate_observation`)
   to prevent context bloat and "Lost in the Middle" degradation on 4B small models.
2. Prompt Prefix Caching Layout: Static system instructions sit at `messages[0]` so LM Studio / ZeroGPU
   KV-caches reuse precomputed attention states across turns.
3. Pattern 5 (Strategic Deterministic Reflection with Context Pruning via Diffing):
   `_verify_zamboni_draft` validates numerical anchors and infrastructure citations. If a flaw is detected,
   executes a single pruned revision call (`T=0.10`) passing ONLY `[Original Task + Projected Ledger + Critic Diff]`
   before falling back to `zamboni-guardrails-v1`.
"""

from __future__ import annotations

import logging
import os
import re
from typing import List, Optional, Tuple
import requests

from .config import (
    AGENT_PERSONAS,
    DEFAULT_ALGORITHMIC_STAMP,
    DEFAULT_LM_STUDIO_URL,
    DEFAULT_TEMPERATURE,
    DEFAULT_ZEROGPU_MODEL,
    MAX_GENERATION_TOKENS,
    RETRY_TEMPERATURE,
)
from .env_utils import (
    discover_hf_token,
    discover_model_name,
    is_local_streamlit,
    resolve_lm_studio_endpoint,
)
from .guardrails import clean_zamboni_prose
from .knowledge_base import KNOWLEDGE_CHUNKS_FR
from .models import ComputeSkillResult, KnowledgeChunk, TrajectoryTelemetry

logger = logging.getLogger("ZamboniAgent.Synthesizer")

MAX_OBSERVATION_CHARS = 1500
MAX_PROJECTED_CHUNKS = 2


def is_french_language(query: str, lang: str = "EN") -> bool:
    """Detects whether user query or session context should be answered in French."""
    if str(lang).upper() == "FR":
        return True
    q_low = str(query or "").lower()
    french_cues = (
        "comment", "pourquoi", "quel", "quelle", "quels", "quelles", "qui est", "qui sont",
        "explique", "expliquer", "est-ce", "sont les", "dans le", "dans la", "avec le",
        "pour les", "calculé", "calculer", "donne-moi", "montre-moi", "meilleur", "meilleurs",
        "patineur", "patineurs", "gardien", "gardiens", "buts attendus", "but attendu",
        "mise en échec", "mises en échec", "repêchage", "classement", "saison régulière",
        "séries éliminatoires"
    )
    return any(re.search(rf"\b{re.escape(fc)}\b", q_low) for fc in french_cues)


def truncate_observation(text: str, max_chars: int = MAX_OBSERVATION_CHARS) -> str:
    """
    Context Engineering (Observation Truncation):
    Caps individual tool/RAG observations so a large table never bloats the 4B model window.
    """
    if not text:
        return ""
    s = str(text).strip()
    if len(s) <= max_chars:
        return s
    return s[:max_chars].rsplit("\n", 1)[0] + "\n...[Truncated observation for context efficiency]"


class ZamboniSynthesizer:
    """
    Synthesizes citation-backed answers from projected Rinklytics KnowledgeChunks
    and deterministic Hugging Face ComputeSkillResults, enforced by a deterministic Critic.
    """

    def __init__(
        self,
        lm_studio_url: str = DEFAULT_LM_STUDIO_URL,
        persona: str = "zamboni",
    ):
        self.lm_studio_url = lm_studio_url
        self.persona = persona if persona in AGENT_PERSONAS else "zamboni"
        self.is_local = is_local_streamlit()
        self.resolved_lm_url, self.lm_studio_online, self.active_lm_model = resolve_lm_studio_endpoint(
            self.lm_studio_url
        )

    def refresh_health(self) -> Tuple[bool, Optional[str]]:
        """Re-check LM Studio server status."""
        self.is_local = is_local_streamlit()
        self.resolved_lm_url, self.lm_studio_online, self.active_lm_model = resolve_lm_studio_endpoint(
            self.lm_studio_url
        )
        return self.lm_studio_online, self.active_lm_model

    def _build_static_system_prompt(self, persona: str) -> str:
        """
        Context Engineering (Prompt Prefix Caching):
        100% static system prompt per persona placed at `messages[0]` so KV-caches
        (LM Studio llama.cpp / MLX / vLLM) cache attention states across calls.
        """
        p_info = AGENT_PERSONAS.get(persona, AGENT_PERSONAS["zamboni"])
        return (
            f"{p_info['system_voice']}\n\n"
            "STRICT OPERATING RULES:\n"
            "1. ONLY answer questions about NHL hockey, Rinklytics analytics methodology, and the provided Hugging Face dataset evidence.\n"
            "2. NEVER invent or hallucinate player stats, team records, or formulas. Use ONLY the exact numbers and equations in the Projected Evidence Ledger.\n"
            "3. Always cite the canonical source code file (e.g., `src/data/apinhle/function/procs_xg.py`, `app/exe/player_stats.py`) or Hugging Face dataset table (`jjerlee/nhl-analytics`).\n"
            "4. Include LaTeX mathematical formulas (`$$ ... $$`) when explaining how metrics are calculated.\n"
            "5. Do NOT output `<think>` tags or internal drafting notes."
        )

    def _project_state_for_llm(
        self,
        chunks: List[KnowledgeChunk],
        compute_results: List[ComputeSkillResult],
    ) -> str:
        """
        Context Engineering (Pattern 2 State Projection):
        Projects only the top-2 methodology chunks and truncated top-N summary tables
        into the LLM working memory, keeping raw DataFrames out of the prompt window.
        """
        sections: List[str] = []
        if chunks:
            sections.append("=== PROJECTED METHODOLOGY LEDGER ===")
            for ch in chunks[:MAX_PROJECTED_CHUNKS]:
                formulas = "\n".join(f"  $$ {f} $$" for f in ch.formulas_latex) if ch.formulas_latex else "  (See methodology text)"
                method_trunc = truncate_observation(ch.methodology_markdown, max_chars=1200)
                sections.append(
                    f"[Reference ID: {ch.chunk_id}] Title: {ch.title}\n"
                    f"Application View: {ch.page_view}\n"
                    f"Source Code Files: {', '.join(ch.source_files)}\n"
                    f"Hugging Face Dataset Tables: {', '.join(ch.hf_dataset_tables)}\n"
                    f"Formulas:\n{formulas}\n"
                    f"Summary: {ch.summary}\n"
                    f"Methodology:\n{method_trunc}\n"
                )

        if compute_results:
            sections.append("=== PROJECTED HUGGING FACE DATA LEDGER (GROUND TRUTH) ===")
            for res in compute_results:
                obs_trunc = truncate_observation(res.summary_markdown, max_chars=MAX_OBSERVATION_CHARS)
                sections.append(
                    f"[Skill: {res.skill_name}] {res.title}\n"
                    f"Dataset Source: {res.dataset_source}\n"
                    f"{obs_trunc}\n"
                )

        return "\n\n".join(sections)

    def _verify_zamboni_draft(
        self,
        draft: str,
        chunks: List[KnowledgeChunk],
        compute_results: List[ComputeSkillResult],
    ) -> Tuple[bool, Optional[str]]:
        """
        Pattern 5 (Strategic Deterministic Critic):
        Fast deterministic unit check verifying that the LLM draft:
        1. Is non-empty and free of leaked `<think>` tags.
        2. Cites at least one canonical Rinklytics file, view, or dataset when chunks are present.
        3. Includes the top-ranked entity/player/team name when `compute_results` are present.
        """
        if not draft or len(draft.strip()) < 45:
            return False, "Draft is too short or empty."

        d_low = draft.lower()
        if "<think>" in d_low or "</think>" in d_low:
            return False, "Draft contains unstripped <think> scratchpad tags."

        # Check citation anchor if methodology chunks were provided
        if chunks:
            top_ch = chunks[0]
            cite_candidates = (
                [os.path.basename(f).lower() for f in top_ch.source_files]
                + [os.path.basename(t).lower() for t in top_ch.hf_dataset_tables]
                + ["nhl-analytics", "procs_xg", "procs_boxscore", "player_stats", "rebuild_tracker", "rinklytics"]
            )
            if not any(c in d_low for c in cite_candidates):
                return (
                    False,
                    f"Missing canonical infrastructure citation. Must cite `{top_ch.source_files[0]}` or `{top_ch.hf_dataset_tables[0] if top_ch.hf_dataset_tables else 'jjerlee/nhl-analytics'}`.",
                )

        # Check ground-truth entity anchor if computed data was provided
        if compute_results:
            first_res = compute_results[0]
            if first_res.records:
                top_row = first_res.records[0]
                anchor_candidates = []
                for key in ("fullName", "team", "franchise_id", "tricode_home", "lastName"):
                    if key in top_row and top_row[key]:
                        val = str(top_row[key]).strip()
                        anchor_candidates.append(val.lower())
                        if " " in val:
                            anchor_candidates.append(val.split()[-1].lower())
                if anchor_candidates and not any(a in d_low for a in anchor_candidates):
                    return (
                        False,
                        f"Draft omitted the #1 ground-truth record (`{anchor_candidates[0]}`) from `{first_res.skill_name}`.",
                    )

        return True, None

    def build_algorithmic_response(
        self,
        query: str,
        intent: str,
        chunks: List[KnowledgeChunk],
        compute_results: List[ComputeSkillResult],
        persona: Optional[str] = None,
        lang: str = "EN",
    ) -> Tuple[str, str]:
        """
        Deterministic Guardrail Synthesizer (`zamboni-guardrails-v1`).
        Produces publication-ready markdown combining exact methodology explanations,
        LaTeX equations, code/dataset references, and live computed tables in English or French.
        """
        is_fr = is_french_language(query, lang=lang)
        p_info = AGENT_PERSONAS.get(persona or self.persona, AGENT_PERSONAS["zamboni"])
        parts: List[str] = []

        if compute_results and intent in ("DATA_QUERY", "HYBRID"):
            for res in compute_results:
                parts.append(f"### 📊 {res.title}")
                parts.append(res.summary_markdown)
                src_label = "Table source" if is_fr else "Source Table"
                parts.append(f"*{src_label}*: `{res.dataset_source}`\n")

        if chunks:
            primary_chunks = chunks[:2] if intent != "DATA_QUERY" else chunks[:1]
            for ch in primary_chunks:
                fr_info = KNOWLEDGE_CHUNKS_FR.get(ch.chunk_id, {}) if is_fr else {}
                ch_title = fr_info.get("title", ch.title)
                ch_summary = fr_info.get("summary", ch.summary)
                parts.append(f"### 📐 {ch_title}")
                parts.append(ch_summary)
                if ch.formulas_latex:
                    form_label = "**Formulation mathématique canonique :**" if is_fr else "**Canonical Mathematical Formulation:**"
                    parts.append(form_label)
                    for f in ch.formulas_latex:
                        parts.append(f"$$\n{f}\n$$")
                parts.append(ch.methodology_markdown)
                src_list = ", ".join(f"`{s}`" for s in ch.source_files)
                hf_list = ", ".join(f"`{t}`" for t in ch.hf_dataset_tables)
                if is_fr:
                    parts.append(
                        f"**Références de l'infrastructure Rinklytics :**\n"
                        f"- **Vue du tableau de bord** : `{ch.page_view}`\n"
                        f"- **Code de calcul** : {src_list}\n"
                        f"- **Jeu de données Hugging Face (`jjerlee/nhl-analytics`)** : {hf_list}"
                    )
                else:
                    parts.append(
                        f"**Infrastructure Citations:**\n"
                        f"- **Dashboard View**: `{ch.page_view}`\n"
                        f"- **Calculation Code**: {src_list}\n"
                        f"- **Hugging Face Dataset (`jjerlee/nhl-analytics`)**: {hf_list}"
                    )

        if compute_results and intent == "METHODOLOGY":
            for res in compute_results[:1]:
                snap_title = f"---\n### 📊 Aperçu empirique en direct : {res.title}" if is_fr else f"---\n### 📊 Live Empirical Snapshot: {res.title}"
                parts.append(snap_title)
                parts.append(res.summary_markdown)

        if not parts:
            if is_fr:
                parts.append(
                    f"{p_info['icon']} **{p_info['display_name']}** : Veuillez préciser une équipe, un joueur, un gardien "
                    f"ou une métrique de Rinklytics (comme `xG`, `ixG`, `GSAx`, `PPI`, `RPE`, `RPI`, `Rang par pairs`, "
                    f"`Régimes de reconstruction`, `Impact des mises en échec` ou `Tirs bloqués`) pour analyse."
                )
            else:
                parts.append(
                    f"{p_info['icon']} **{p_info['display_name']}**: Please specify an NHL team, player, goalie, "
                    f"or Rinklytics metric (such as `xG`, `ixG`, `GSAx`, `PPI`, `RPE`, `RPI`, `Pairwise Rank`, "
                    f"`Rebuild Regimes`, `Hit on Impact`, or `Shot Blocking`) to inspect."
                )

        return "\n\n".join(parts), DEFAULT_ALGORITHMIC_STAMP

    def _call_lm_studio(
        self,
        messages: List[dict],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = MAX_GENERATION_TOKENS,
    ) -> Optional[Tuple[str, str]]:
        """Calls local LM Studio server (`http://127.0.0.1:1234/v1/chat/completions`)."""
        self.refresh_health()
        if not self.lm_studio_online:
            return None

        url = f"{self.resolved_lm_url}/chat/completions"
        model_id = self.active_lm_model or "local-model"
        payload = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            resp = requests.post(url, json=payload, timeout=90)
            if resp.status_code == 200:
                data = resp.json()
                raw_content = data["choices"][0]["message"]["content"]
                cleaned = clean_zamboni_prose(raw_content)
                if cleaned:
                    return cleaned, f"lm-studio/{model_id}"
        except Exception as e:
            logger.warning("LM Studio request failed: %s", e)

        return None

    def _call_hf_cloud(
        self,
        messages: List[dict],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = MAX_GENERATION_TOKENS,
    ) -> Optional[Tuple[str, str]]:
        """Calls Hugging Face Inference / ZeroGPU model in production when available."""
        model_id = discover_model_name(DEFAULT_ZEROGPU_MODEL)

        if os.environ.get("SPACES_ZERO_GPU") or os.environ.get("SPACE_ID"):
            try:
                from dev.agent.scoreboard.narrator import _zerogpu_infer
                raw = _zerogpu_infer(
                    messages=messages,
                    model_id=model_id,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                )
                cleaned = clean_zamboni_prose(raw)
                if cleaned:
                    return cleaned, f"hf-space/{model_id}"
            except Exception as e:
                logger.debug("ZeroGPU inference fallback: %s", e)

        token = discover_hf_token()
        if token:
            try:
                from huggingface_hub import InferenceClient
                client = InferenceClient(token=token, timeout=25)
                out = client.chat_completion(
                    model="meta-llama/Llama-3.2-3B-Instruct",
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                raw = out.choices[0].message.content
                cleaned = clean_zamboni_prose(raw)
                if cleaned:
                    return cleaned, "hf-inference/meta-llama/Llama-3.2-3B-Instruct"
            except Exception as e:
                logger.debug("HF InferenceClient fallback: %s", e)

        return None

    def _dispatch_llm(
        self,
        messages: List[dict],
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> Optional[Tuple[str, str]]:
        if self.is_local:
            return self._call_lm_studio(messages, temperature=temperature)
        return self._call_hf_cloud(messages, temperature=temperature)

    def synthesize(
        self,
        query: str,
        intent: str,
        chunks: List[KnowledgeChunk],
        compute_results: List[ComputeSkillResult],
        persona: Optional[str] = None,
        force_mode: str = "auto",
        pipeline_ids: Optional[List[str]] = None,
        lang: str = "EN",
    ) -> Tuple[str, str, TrajectoryTelemetry]:
        """
        Synthesizes the final answer using State Projection, Prompt Prefix Caching,
        and Pattern 5 Strategic Deterministic Reflection (with Context Pruning via Diffing).
        Adapts seamlessly to English or French.
        Returns `(answer_markdown, generator_model_stamp, trajectory_telemetry)`.
        """
        is_fr = is_french_language(query, lang=lang)
        active_persona = persona if persona in AGENT_PERSONAS else self.persona
        algo_answer, algo_stamp = self.build_algorithmic_response(
            query=query,
            intent=intent,
            chunks=chunks,
            compute_results=compute_results,
            persona=active_persona,
            lang=lang,
        )

        projected_context = self._project_state_for_llm(chunks, compute_results)
        proj_chars = len(projected_context)
        proj_tokens_est = max(1, proj_chars // 4)

        telemetry = TrajectoryTelemetry(
            pipeline_ids=pipeline_ids or ["methodology_rag"],
            projected_context_chars=proj_chars,
            projected_tokens_est=proj_tokens_est,
            reflection_attempts=0,
            critic_passed=True,
            critic_feedback=None,
            schema_compliant=True,
            prompt_prefix_cached=True,
        )

        if force_mode == "algorithmic":
            return algo_answer, algo_stamp, telemetry

        system_prompt = self._build_static_system_prompt(active_persona)
        lang_note = (
            "\n\nDIRECTIVE DE LANGUE (RÉPONDRE EN FRANÇAIS) :\n"
            "La question de l'utilisateur est en français ou la session est en français. "
            "Tu DOIS impérativement rédiger TOUTE ta réponse, explications, titres et synthèses analytiques en français fluide et soigné. "
            "Conserve intacts les chemins de fichiers (ex. `src/data/apinhle/function/procs_xg.py`), les formules mathématiques LaTeX, "
            "et les tables Hugging Face (`jjerlee/nhl-analytics`)."
            if is_fr
            else ""
        )
        user_prompt = (
            f"USER QUESTION:\n{query}\n\n"
            f"DETECTED INTENT: {intent}\n\n"
            f"PROJECTED EVIDENCE LEDGER:\n{projected_context}\n\n"
            "Provide a clear, well-structured Markdown answer grounded strictly in the Projected Evidence Ledger above."
            f"{lang_note}"
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        llm_res = self._dispatch_llm(messages, temperature=DEFAULT_TEMPERATURE)
        if llm_res is None:
            return algo_answer, algo_stamp, telemetry

        draft_text, model_stamp = llm_res
        passed, critique = self._verify_zamboni_draft(draft_text, chunks, compute_results)
        if passed:
            return draft_text, model_stamp, telemetry

        # ── Pattern 5: Strategic Reflection with Context Pruning via Diffing ──
        # Replace conversation history with ONLY [Original Task + Projected Ledger + Broken Draft + Specific Critic Diff]
        telemetry.reflection_attempts = 1
        telemetry.critic_feedback = critique
        revision_user_prompt = (
            f"ORIGINAL USER QUESTION:\n{query}\n\n"
            f"PROJECTED EVIDENCE LEDGER:\n{projected_context}\n\n"
            f"CURRENT BROKEN DRAFT:\n{draft_text}\n\n"
            f"AUTOMATED CRITIC FEEDBACK (FIX REQUIRED):\n{critique}\n\n"
            "Output ONLY the fully corrected Markdown response that resolves the Critic Feedback using exact facts and citations from the Projected Evidence Ledger."
            f"{lang_note}"
        )
        pruned_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": revision_user_prompt},
        ]
        retry_res = self._dispatch_llm(pruned_messages, temperature=RETRY_TEMPERATURE)
        if retry_res is not None:
            retry_text, retry_stamp = retry_res
            passed_retry, critique_retry = self._verify_zamboni_draft(retry_text, chunks, compute_results)
            if passed_retry:
                telemetry.critic_passed = True
                return retry_text, retry_stamp, telemetry
            telemetry.critic_feedback = critique_retry

        # If reflection retry still fails deterministic verification, guarantee safety via algorithmic guardrail
        telemetry.critic_passed = False
        return algo_answer, algo_stamp, telemetry
