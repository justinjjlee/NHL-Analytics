"""
zamboni.py — Agent Zamboni Interactive Chat & Methodology Reference View.

Placed right below "Home" in the Rinklytics navigation bar (`app/app.py`).
Provides:
1. Boxed, Fixed-Height Scrollable Chat Window (`st.container(height=520, border=True)`) so the page never stretches down.
2. Strictly Ephemeral Per-Tab Session State: Zero chat history is saved to disk or shared across browser sessions/users.
   Refreshing the page or opening a new session starts completely fresh.
3. Strict Hockey-Only Domain Guardrail (rejects non-hockey queries with an explicit error).
4. Global Cross-User Rate Limiter (Unlimited locally; strictly 1 response / 1.0s across all internet users on Hugging Face).
5. Multi-Tier Compute Telemetry (Local LM Studio `http://127.0.0.1:1234/v1` with Alarm Protocol -> Cloud ZeroGPU -> Deterministic `zamboni-guardrails-v1`).
"""

from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Ensure repository root is on sys.path so `dev.zamboni` imports cleanly
_CURRENT_DIR = Path(__file__).resolve().parent
_APP_DIR = _CURRENT_DIR.parent
_REPO_ROOT = _APP_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from dev.zamboni.agent import ZamboniAgent
from dev.zamboni.config import AGENT_PERSONAS
from dev.zamboni.label_studio_bridge import LABEL_STUDIO_XML_CONFIG

MAX_EPHEMERAL_TURNS = 20


@st.cache_resource
def get_zamboni_agent() -> ZamboniAgent:
    """
    Stateless ZamboniAgent compute & RAG singleton.
    `log_interactions=False` ensures zero chat history is ever written to disk or shared across sessions.
    """
    return ZamboniAgent(persona="zamboni", log_interactions=False)


def format_engine_badge(model_stamp: str) -> str:
    if not model_stamp:
        return "Algorithmic Guardrail Engine (zamboni-guardrails-v1)"
    m = str(model_stamp).strip()
    m_low = m.lower()
    if m_low.startswith("lm-studio/"):
        return f"LM Studio Local ({m.replace('lm-studio/', '')})"
    if m_low.startswith("hf-space/"):
        return f"Hugging Face ZeroGPU ({m.replace('hf-space/', '')})"
    if m_low.startswith("hf-inference/"):
        return f"Hugging Face Inference ({m.replace('hf-inference/', '')})"
    if "golden-cache" in m_low:
        return "Label Studio Verified Golden Cache (Tier 0)"
    return "Deterministic Reference & Compute Engine (zamboni-guardrails-v1)"


def render_evidence_drawer(meta: dict, is_fr: bool = False) -> None:
    """Renders the expandable methodology citations, formulas, and computed data drawer inside a chat bubble."""
    if not meta or not meta.get("allowed", True):
        return
    drawer_label = (
        "🔬 View Methodology Citations, Formulas, Live Data & Trajectory Telemetry"
        if not is_fr
        else "🔬 Voir les citations méthodologiques, formules, données et télémétrie"
    )
    with st.expander(drawer_label, expanded=False):
        traj = meta.get("trajectory", {})
        st.caption(
            f"**Engine**: `{format_engine_badge(meta.get('generator_model', ''))}` | "
            f"**Intent**: `{meta.get('intent')}` | "
            f"**Pipelines**: `{', '.join(traj.get('pipeline_ids', meta.get('target_domains', [])))}` | "
            f"**Projected Context**: `~{traj.get('projected_tokens_est', 0)} tokens` | "
            f"**Critic Verified**: `{traj.get('critic_passed', True)} (Retries: {traj.get('reflection_attempts', 0)})` | "
            f"**Latency**: `{meta.get('latency_ms')} ms`"
        )
        for comp in meta.get("computed_results", []):
            st.markdown(f"**📊 Computed Table: {comp['title']}** (`{comp['dataset_source']}`)")
            if comp.get("records"):
                st.dataframe(pd.DataFrame(comp["records"]), use_container_width=True, hide_index=True)

        for ch in meta.get("retrieved_chunks", []):
            st.markdown(
                f"**📚 Knowledge Reference: {ch['title']}** (`{ch['chunk_id']}`, score=`{ch['score']}`)  \n"
                f"- **Page**: `{ch['page_view']}`  \n"
                f"- **Code**: `{', '.join(ch['source_files'])}`  \n"
                f"- **HF Dataset**: `{', '.join(ch['hf_dataset_tables'])}`"
            )


# ── Language Resolution ───────────────────────────────────────────────────────
lang = st.session_state.get("lang", "EN")
is_fr = lang == "FR"

agent = get_zamboni_agent()
lm_online, active_lm_model = agent.synthesizer.refresh_health()
rate_telemetry = agent.rate_limiter.get_telemetry(is_local=agent.is_local)

# ── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("---")
    st.subheader("🚜 Agent Configuration" if not is_fr else "🚜 Configuration de l'Agent")

    persona_options = list(AGENT_PERSONAS.keys())
    selected_persona = st.selectbox(
        "Agent Persona" if not is_fr else "Persona de l'Agent",
        options=persona_options,
        format_func=lambda k: f"{AGENT_PERSONAS[k]['icon']} {AGENT_PERSONAS[k]['display_name']}",
        key="zamboni_persona_select",
    )
    st.caption(AGENT_PERSONAS[selected_persona]["tagline"])

    synth_mode = st.radio(
        "Synthesis Mode" if not is_fr else "Mode de synthèse",
        options=["auto", "algorithmic"],
        format_func=lambda m: (
            "Auto (LLM + Guardrails)" if m == "auto" else "Deterministic Reference Only (Fast)"
        ),
        index=0,
        key="zamboni_synth_mode",
    )

    if st.button("🗑️ Clear Chat History" if not is_fr else "🗑️ Effacer la conversation", use_container_width=True):
        st.session_state.zamboni_messages = []
        st.rerun()

    st.caption(
        "🔒 **Ephemeral Session**: Chat history is never saved to disk or shared across users/sessions."
        if not is_fr
        else "🔒 **Session éphémère** : L'historique n'est jamais sauvegardé ni partagé entre sessions."
    )

p_meta = AGENT_PERSONAS[selected_persona]

# ── Header & Telemetry Banner ─────────────────────────────────────────────────
st.title(f"{p_meta['icon']} {p_meta['display_name']} — Rinklytics AI")
st.markdown(
    (
        "**Hockey-Specific Analytics Methodology Reference & Live Hugging Face Dataset (`jjerlee/nhl-analytics`) Q&A System.**  \n"
        "*Guarded by a strict Hockey-Only Domain Filter, Ephemeral Per-Tab Privacy, and Compute-First Verification.*"
    )
    if not is_fr
    else (
        "**Système de référence méthodologique et d'analyse en direct sur le jeu de données Hugging Face (`jjerlee/nhl-analytics`).**  \n"
        "*Protégé par un garde-fou strict réservé au hockey, une session éphémère privée et une vérification déterministe.*"
    )
)

# Alarm Protocol if Local LM Studio Server is Unreachable
if agent.is_local and not lm_online:
    st.warning(
        "⚠️ **LM Studio Local Server Alarm (`http://127.0.0.1:1234/v1` Unreachable)**: "
        "The local LM Studio model server could not be reached. AGENT ZAMBONI has automatically "
        "activated the deterministic **Algorithmic Guardrail & Reference Engine (`zamboni-guardrails-v1`)** "
        "so all methodology formulas and Hugging Face dataset queries continue without interruption.",
        icon="🚨",
    )

# Telemetry Status Pills
col_t1, col_t2 = st.columns(2)
with col_t1:
    env_label = (
        f"💻 Local (`{active_lm_model}`)" if (agent.is_local and lm_online)
        else ("💻 Local (`zamboni-guardrails-v1`)" if agent.is_local else "☁️ Hugging Face Space")
    )
    st.markdown(f"**Runtime**: {env_label}")
with col_t2:
    st.markdown(f"**Rate Limit**: `{rate_telemetry['mode']}`")

# ── Interactive Chat View (Single-Page Layout) ────────────────────────────────
# Initialize strictly ephemeral per-websocket session state (isolated to current browser tab only)
if "zamboni_messages" not in st.session_state:
    welcome_text = (
        f"Hello! I am **{p_meta['display_name']}** ({p_meta['icon']} *{p_meta['tagline']}*). "
        "I can explain **how any metric or model is calculated** across all 11 Rinklytics dashboard pages "
        "(with exact LaTeX formulas and source code citations) and query our **Hugging Face Dataset (`jjerlee/nhl-analytics`)** "
        "for live player, goalie, team, rebuild, draft, and sequence analytics.\n\n"
        "*Note: I am strictly restricted to NHL hockey and Rinklytics analytics—non-hockey questions are rejected. "
        "This chat is 100% ephemeral and resets whenever your session renews.*"
        if not is_fr
        else f"Bonjour ! Je suis **{p_meta['display_name']}** ({p_meta['icon']}). "
        "Je peux expliquer **comment chaque métrique ou modèle est calculé** dans Rinklytics "
        "et interroger notre **jeu de données Hugging Face (`jjerlee/nhl-analytics`)** en direct.\n\n"
        "*Note : Je réponds exclusivement aux questions sur le hockey de la LNH et l'analytique Rinklytics. "
        "Cette session est 100 % éphémère et aucune conversation n'est enregistrée.*"
    )
    st.session_state.zamboni_messages = [
        {
            "role": "assistant",
            "content": welcome_text,
            "allowed": True,
            "meta": None,
        }
    ]

# Quick-Starter Question Pills (Stacked Vertically by Question Type for Full-Width Readability)
prompt_to_run = None
with st.expander(
    "⚡ Quick-Starter Questions (Click to Ask)" if not is_fr else "⚡ Questions rapides (Cliquez pour demander)",
    expanded=False,
):
    st.markdown("**📐 1. Analytics Methodology & Mathematical Formulas**")
    if st.button("How are Expected Goals (xG, ixG) and Goalie GSAx calculated using the Dual-Stage Logistic model?", use_container_width=True):
        prompt_to_run = "How are Expected Goals (xG, ixG) and Goalie GSAx calculated using the Dual-Stage Logistic model?"
    if st.button("How is the Player Performance Index (PPI) calculated for skaters and goalies?", use_container_width=True):
        prompt_to_run = "How is the Player Performance Index (PPI) calculated for skaters and goalies?"
    if st.button("How are Pythagorean Expectation (RPE), RPI, and Common-Opponent Pairwise Win Rank calculated in Team Season Stats?", use_container_width=True):
        prompt_to_run = "How are Pythagorean Expectation (RPE), RPI, and Common-Opponent Pairwise Win Rank calculated in Team Season Stats?"

    st.markdown("**📊 2. Live Hugging Face Dataset (`jjerlee/nhl-analytics`) Queries**")
    if st.button("Who are the top 5 skaters by PPI and ixG in the dataset, and how is Skater PPI calculated?", use_container_width=True):
        prompt_to_run = "Who are the top 5 skaters by PPI and ixG in the dataset, and how is Skater PPI calculated?"
    if st.button("Which goalies lead the NHL in Goals Saved Above Expected (GSAx) and Goalie PPI?", use_container_width=True):
        prompt_to_run = "Which goalies lead the NHL in GSAx and Goalie PPI?"
    if st.button("Compare Montréal (MTL) and Chicago (CHI) in the Franchise Rebuild Tracker (R0–R5 regimes, ALI, YIS, Cap Surplus)", use_container_width=True):
        prompt_to_run = "Compare MTL and CHI in the Franchise Rebuild Tracker and explain the R0-R5 regimes, ALI, YIS, and Cap Surplus."

    st.markdown("**💥 3. Tactical Sequences & Guardrail Verification**")
    if st.button("How does Hit on Impact measure event chains (+1 to +4), Puck Win Rate, and 30-second goal windows?", use_container_width=True):
        prompt_to_run = "How does Hit on Impact measure event chains (+1 to +4), Puck Win Rate, and 30-second goal windows?"
    if st.button("How does the Blocking Shots Sankey diagram model downstream possession recovery vs. opponent continuation?", use_container_width=True):
        prompt_to_run = "How does the Blocking Shots Sankey diagram model downstream possession recovery vs. opponent continuation?"
    if st.button("🚫 Test Hockey-Only Guardrail (Ask an off-topic non-hockey question)", use_container_width=True):
        prompt_to_run = "How do I bake a sourdough bread recipe?"

# ── Boxed, Fixed-Height Scrollable Chat Container (520px) ─────────────────
# Keeps the chat area strictly contained inside a scrollable box so the page never stretches down
chat_box = st.container(height=520, border=True)

# Chat Input Bar (Pinned directly below the boxed scrollable chat container)
chat_input_val = st.chat_input(
    "Ask AGENT ZAMBONI about NHL analytics methodology, formulas, or Hugging Face dataset stats..."
    if not is_fr
    else "Posez une question à AGENT ZAMBONI sur la méthodologie hockey ou le jeu de données Hugging Face..."
)

active_query = prompt_to_run or chat_input_val

# Render existing ephemeral messages & process new query inside the scrollable box
with chat_box:
    for msg in st.session_state.zamboni_messages:
        role = msg.get("role", "assistant")
        avatar = "👤" if role == "user" else p_meta["icon"]
        with st.chat_message(role, avatar=avatar):
            if not msg.get("allowed", True):
                st.error(msg["content"])
            else:
                st.markdown(msg["content"])
            render_evidence_drawer(msg.get("meta"), is_fr=is_fr)

    if active_query:
        st.session_state.zamboni_messages.append({
            "role": "user",
            "content": active_query,
            "allowed": True,
            "meta": None,
        })
        with st.chat_message("user", avatar="👤"):
            st.markdown(active_query)

        with st.chat_message("assistant", avatar=p_meta["icon"]):
            with st.spinner(
                "Resurfacing the ice & querying Hugging Face telemetry..."
                if not is_fr
                else "Surfaçage de la glace et requête Hugging Face..."
            ):
                resp = agent.ask(
                    query=active_query,
                    lang=lang,
                    persona=selected_persona,
                    force_mode=synth_mode,
                )

            if not resp.allowed:
                st.error(resp.answer_markdown)
            else:
                st.markdown(resp.answer_markdown)

            meta_dict = resp.to_dict()
            render_evidence_drawer(meta_dict, is_fr=is_fr)

            st.session_state.zamboni_messages.append({
                "role": "assistant",
                "content": resp.answer_markdown,
                "allowed": resp.allowed,
                "meta": meta_dict,
            })
            # Keep ephemeral in-memory list bounded
            if len(st.session_state.zamboni_messages) > MAX_EPHEMERAL_TURNS:
                st.session_state.zamboni_messages = (
                    [st.session_state.zamboni_messages[0]]
                    + st.session_state.zamboni_messages[-(MAX_EPHEMERAL_TURNS - 1):]
                )
