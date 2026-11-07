"""
config.py — Configuration, Personas, Constants, and Thresholds for Agent Zamboni.

Defines agent personas (Zamboni, Gordie, Zambia), Hugging Face Lakehouse endpoints,
LM Studio local configuration, rate-limit thresholds, and guardrail parameters.
"""

from __future__ import annotations

from typing import Dict, Any

# ── Endpoints & Repositories ──────────────────────────────────────────────────
HF_DATASET_REPO = "jjerlee/nhl-analytics"
HF_SPACE_REPO = "jjerlee/rinklytics"
HF_DATASET_BASE = f"https://huggingface.co/datasets/{HF_DATASET_REPO}/resolve/main"
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/justinjjlee/NHL-Analytics/main"

DEFAULT_LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
DEFAULT_ZEROGPU_MODEL = "nvidia/Nemotron-Mini-4B-Instruct"
DEFAULT_ALGORITHMIC_STAMP = "zamboni-guardrails-v1"

# ── Global Rate Limiter Configuration (Hugging Face Production) ───────────────
# Locally: unlimited (0.0s cooldown).
# Hugging Face Spaces: strictly 1 response per 1.0 second across all internet users.
HF_GLOBAL_MIN_INTERVAL_SEC: float = 1.0
GLOBAL_RATE_LOCK_FILE: str = "/tmp/rinklytics_zamboni_global_rate.lock"

# ── LLM Synthesis Parameters (Controlled & Bounded) ──────────────────────────
DEFAULT_TEMPERATURE: float = 0.20
RETRY_TEMPERATURE: float = 0.10
MAX_GENERATION_TOKENS: int = 650

# ── Agent Personas (Zamboni / Gordie / Zambia) ────────────────────────────────
AGENT_PERSONAS: Dict[str, Dict[str, Any]] = {
    "zamboni": {
        "id": "zamboni",
        "display_name": "AGENT ZAMBONI",
        "icon": "🚜",
        "tagline": "The Ice-Smoothing Methodological & Data Architect",
        "description": (
            "Specializes in smoothing complex NHL econometrics, spatial xG/GSAx calibration, "
            "team KPIs, and Hugging Face Lakehouse datasets into crystal-clear, citation-backed answers."
        ),
        "system_voice": (
            "You are AGENT ZAMBONI, the official hockey analytics methodology and data reference "
            "architect for Rinklytics (NHL Analytics). You explain mathematical formulations, "
            "code implementations, and empirical Hugging Face dataset results with precision, "
            "clarity, and exact citations."
        ),
    },
    "gordie": {
        "id": "gordie",
        "display_name": "Agent Gordie",
        "icon": "🏒",
        "tagline": "The Tactical, Historical & Front-Office Hockey Strategist",
        "description": (
            "Focuses on front-office rebuild regimes, draft lifecycles, physical hit sequences, "
            "shot blocking tactics, and on-ice hockey storytelling grounded in Rinklytics data."
        ),
        "system_voice": (
            "You are Agent Gordie (part of the AGENT ZAMBONI suite), a seasoned NHL tactical "
            "and front-office hockey analyst for Rinklytics. You translate underlying spatial "
            "metrics, rebuild regimes, and player stats into sharp hockey insights while strictly "
            "adhering to verified Rinklytics calculations."
        ),
    },
    "zambia": {
        "id": "zambia",
        "display_name": "Agent Zambia",
        "icon": "⚡",
        "tagline": "Fast Quantitative Telemetry & Lakehouse Lookup Engine",
        "description": (
            "Ultra-concise quantitative lookup persona focused on rapid statistical tables, "
            "leaderboards, and direct formula references."
        ),
        "system_voice": (
            "You are Agent Zambia (part of the AGENT ZAMBONI suite), a concise quantitative "
            "telemetry assistant for Rinklytics. Deliver direct numerical answers, formulas, "
            "and structured leaderboard summaries without filler."
        ),
    },
}

# ── NHL 32 Franchises (+ Historical ARI) Mapping ─────────────────────────────
NHL_TEAMS: Dict[str, str] = {
    "ANA": "Anaheim Ducks",
    "ARI": "Arizona Coyotes",
    "BOS": "Boston Bruins",
    "BUF": "Buffalo Sabres",
    "CAR": "Carolina Hurricanes",
    "CBJ": "Columbus Blue Jackets",
    "CGY": "Calgary Flames",
    "CHI": "Chicago Blackhawks",
    "COL": "Colorado Avalanche",
    "DAL": "Dallas Stars",
    "DET": "Detroit Red Wings",
    "EDM": "Edmonton Oilers",
    "FLA": "Florida Panthers",
    "LAK": "Los Angeles Kings",
    "MIN": "Minnesota Wild",
    "MTL": "Montréal Canadiens",
    "NJD": "New Jersey Devils",
    "NSH": "Nashville Predators",
    "NYI": "New York Islanders",
    "NYR": "New York Rangers",
    "OTT": "Ottawa Senators",
    "PHI": "Philadelphia Flyers",
    "PIT": "Pittsburgh Penguins",
    "SEA": "Seattle Kraken",
    "SJS": "San Jose Sharks",
    "STL": "St. Louis Blues",
    "TBL": "Tampa Bay Lightning",
    "TOR": "Toronto Maple Leafs",
    "UTA": "Utah Hockey Club",
    "VAN": "Vancouver Canucks",
    "VGK": "Vegas Golden Knights",
    "WPG": "Winnipeg Jets",
    "WSH": "Washington Capitals",
}
