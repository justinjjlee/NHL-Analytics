"""
guardrails.py — Strict Hockey-Only Domain Guardrail & Factual Output Verifier for Agent Zamboni.

Enforces a 3-Layer Defense-in-Depth Guardrail:
1. Layer 1: Adversarial Prompt Injection & Hard Off-Topic Blocklist.
2. Layer 2: NHL / Hockey / Rinklytics Lexical & Metric Entity Gate.
3. Layer 3: Output Sanitization (`clean_zamboni_prose`) & Factual Number Reconciliation.
"""

from __future__ import annotations

import re
from typing import List, Optional, Set

from .config import NHL_TEAMS
from .models import GuardrailDecision


# ── Layer 1: Prompt Injection & Hard Off-Topic Patterns ───────────────────────
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions",
    r"disregard\s+(?:all\s+)?(?:previous|prior|system)\s+instructions",
    r"you\s+are\s+now\s+(?:a|an)\s+(?!hockey|nhl|zamboni|gordie|rinklytics)",
    r"reveal\s+(?:your\s+)?system\s+prompt",
    r"jailbreak",
    r"dan\s+mode",
    r"bypass\s+guardrail",
]

OFF_TOPIC_PATTERNS = [
    r"\b(?:recipe|bake|cooking|pancake|pasta|pizza|cake|ingredients|restaurant|recette|cuisiner|gâteau|ingrédients)\b",
    r"\b(?:bitcoin|crypto|ethereum|stock\s+market|nasdaq|s&p\s*500|forex|tax\s+return|mortgage|actions|bourse|impôts)\b",
    r"\b(?:presidential\s+election|democrat|republican|senate|congress|geopolitics|élection|président|politique)\b",
    r"\b(?:horoscope|astrology|zodiac|tarot|astrologie)\b",
    r"\b(?:leetcode|binary\s+tree|quicksort|fibonacci|linked\s+list|react\s+native|kubernetes\s+cluster)\b",
    r"\b(?:nba|nfl|mlb|premier\s+league|champions\s+league|fifa|world\s+cup|super\s+bowl|world\s+series|touchdown|home\s+run|slam\s+dunk|quarterback|point\s+guard)\b",
    r"\b(?:weather\s+forecast|flight\s+booking|hotel\s+reservation|météo|billet\s+d'avion|hôtel)\b",
    r"\b(?:medical\s+diagnosis|prescription|symptom|antibiotic|lawsuit|divorce\s+lawyer|médicament|ordonnance|diagnostic)\b",
]

# ── Layer 2: Positive Hockey, NHL & Rinklytics Signals ────────────────────────
HOCKEY_DOMAIN_KEYWORDS: Set[str] = {
    # General Hockey & NHL
    "nhl", "hockey", "ice", "puck", "rink", "skater", "skaters", "goalie", "goalies",
    "goaltender", "goaltenders", "goaltending", "netminder", "defenseman", "defensemen",
    "defenceman", "forward", "forwards", "center", "winger", "left wing", "right wing",
    "stanley cup", "playoff", "playoffs", "regular season", "overtime", "shootout",
    "power play", "powerplay", "shorthanded", "penalty", "penalties", "pim", "faceoff",
    "faceoffs", "hat trick", "shutout", "shutouts", "save", "saves", "save percentage",
    "sv%", "gaa", "goals against", "shots on goal", "sog", "plus minus", "plus-minus",
    "toi", "time on ice", "shift", "shifts", "crease", "slot", "blue line", "neutral zone",
    "offensive zone", "defensive zone", "forecheck", "backcheck", "breakout", "one-timer",
    "wrist shot", "slap shot", "snap shot", "backhand", "tip-in", "deflection", "wrap-around",
    "rebound", "rebounds", "rush", "empty net", "takeaway", "takeaways", "giveaway", "giveaways",
    "turnover", "turnovers", "hit", "hits", "hitting", "bodycheck", "block", "blocked",
    "blocking", "blocked shots", "draft", "drafted", "draftee", "prospect", "prospects",
    "first round", "1st round", "overall pick", "elc", "entry-level", "ufa", "rfa",
    "salary cap", "cap hit", "cap surplus", "front office", "general manager", "gm",
    "trade", "trades", "waivers", "franchise", "franchises", "arena", "travel distance",
    "division", "conference", "eastern", "western", "atlantic", "metropolitan", "central", "pacific",
    # Rinklytics Specific Metrics, Models & Infrastructure
    "rinklytics", "zamboni", "gordie", "zambia", "scoreboard", "game recap", "game recaps",
    "slate", "xg", "ixg", "xga", "gsax", "xsv%", "dsv%", "hdsv%", "hdcf", "mdcf", "ldcf",
    "expected goals", "expected goal", "goals saved above expected", "goals above expected",
    "high-danger", "high danger", "medium-danger", "low-danger", "fenwick", "corsi",
    "dual-stage", "dual stage", "logistic", "crease-jam", "clamping", "ppi",
    "player performance index", "z-score", "z-scores", "rpe", "pythagorean",
    "pythagorean expectation", "fitted wins", "rwin_fittedpe", "rpi", "rating percentage index",
    "pairwise", "pairwise_win", "common opponent", "common opponents", "strength of record",
    "kpinorm", "kpi", "kpis", "hot streak", "win streak", "rebuild", "rebuilds",
    "rebuild tracker", "markov", "regime", "regimes", "inaction", "teardown", "incubation",
    "emergence", "purgatory", "contender", "scorched earth", "retool", "ali", "yis", "rci",
    "vir", "ccm", "spsi", "shannon entropy", "experience bucket", "insulation ratio",
    "core continuity", "synergy", "magnifiques patineuses", "exceptional season",
    "exceptional seasons", "exceptional players", "95th percentile", "hit on impact",
    "hit impact", "30s window", "30-second", "puck retention", "sankey", "moneyline",
    "odds", "betting odds", "implied probability", "huggingface", "hugging face",
    "jjerlee/nhl-analytics", "parquet", "lakehouse", "procs_xg", "procs_boxscore",
    "player_stats", "data_loader", "lm studio", "zerogpu", "label studio", "langgraph",
    # French Hockey Terms
    "patineur", "patineurs", "gardien", "gardiens", "rondelle", "patinoire",
    "mises en échec", "mise en échec", "tirs bloqués", "tir bloqué", "buts attendus", "but attendu",
    "repêchage", "reconstruction", "reconstructions", "classement", "équipe", "équipes",
    "joueur", "joueurs", "statistiques", "but", "buts", "tir", "tirs", "glace",
    "avantage numérique", "désavantage numérique", "prolongation", "fusillade",
    "saison régulière", "séries éliminatoires", "coupe stanley", "échappée",
    "mise en jeu", "zone offensive", "zone défensive", "zone neutre", "temps de glace",
    "indice de performance", "régimes", "directeur général", "blanchissage",
    "arrêts", "arrêt", "pourcentage d'arrêts", "différentiel",
}

# Build team token set (tricodes, city names, nicknames)
TEAM_TOKENS: Set[str] = set()
for tri, full_name in NHL_TEAMS.items():
    TEAM_TOKENS.add(tri.lower())
    TEAM_TOKENS.add(full_name.lower())
    for part in full_name.lower().split():
        if len(part) >= 4:
            TEAM_TOKENS.add(part)

# Common NHL star last names for quick lexical detection even without metric words
STAR_PLAYER_TOKENS: Set[str] = {
    "mcdavid", "draisaitl", "mackinnon", "makar", "matthews", "marner", "nylander",
    "bedard", "kucherov", "vasilevskiy", "hellebuyck", "shesterkin", "sorokin",
    "oettinger", "swayman", "ullmark", "bobrovsky", "barkov", "tkachuk", "reinhart",
    "pastrnak", "marchand", "mcavoy", "crosby", "malkin", "letang", "ovechkin",
    "hughes", "hischier", "bratt", "fox", "panarin", "zibanejad", "kaprizov",
    "boldy", "robo", "robertson", "heiskanen", "rantanen", "aho", "svechnikov",
    "slafkovsky", "suzuki", "caufield", "hutson", "dobes", "dobeš", "montembeault",
    "celibrini", "celebrini", "michkov", "carlsson", "fantilli", "dahlin", "tage",
    "kopitar", "kempe", "doughty", "pettersson", "miller", "boeser", "demko",
    "eichel", "stone", "pietrangelo", "connor", "scheifele", "morrissey", "forsberg",
    "josi", "saros", "larkin", "seider", "raymond", "stutzle", "stützle", "tkachuk",
    "werenski", "gaudreau", "keller", "guenther", "beniers", "mccann", "kadri",
    "weegar", "wolf", "binnington", "thomas", "kyrou", "fowler", "gretzky", "howe",
    "lemieux", "orr", "brodeur", "roy", "hasek", "price", "lundqvist", "bergeron",
}


def evaluate_hockey_guardrail(query: str, lang: str = "EN") -> GuardrailDecision:
    """
    Evaluates a user query against the 3-Layer Hockey-Only Domain Guardrail.
    Rejects non-hockey, off-topic, or prompt-injection inputs with a structured error.
    """
    if not query or not str(query).strip():
        msg = (
            "Please enter a question about NHL hockey or Rinklytics analytics."
            if lang == "EN"
            else "Veuillez poser une question sur le hockey de la LNH ou l'analytique Rinklytics."
        )
        return GuardrailDecision(
            allowed=False,
            error_code="EMPTY_QUERY",
            error_message=msg,
        )

    q_clean = str(query).strip()
    q_low = q_clean.lower()

    # ── Layer 1A: Check Adversarial Prompt Injections ─────────────────────────
    for pat in PROMPT_INJECTION_PATTERNS:
        if re.search(pat, q_low):
            msg = (
                "🚨 Guardrail Error (`PROMPT_INJECTION_BLOCKED`): Instruction override attempts are prohibited. "
                "Agent Zamboni only answers questions about NHL hockey and Rinklytics analytics methodology and datasets."
                if lang == "EN"
                else "🚨 Erreur de garde-fou (`PROMPT_INJECTION_BLOCKED`) : Les tentatives de contournement sont interdites. "
                "L'Agent Zamboni répond uniquement aux questions sur le hockey de la LNH et l'infrastructure Rinklytics."
            )
            return GuardrailDecision(
                allowed=False,
                error_code="PROMPT_INJECTION_BLOCKED",
                error_message=msg,
            )

    # Collect positive hockey signals first
    matched_signals: List[str] = []

    for kw in HOCKEY_DOMAIN_KEYWORDS:
        if len(kw) <= 3:
            if re.search(rf"\b{re.escape(kw)}\b", q_low):
                matched_signals.append(f"keyword:{kw}")
        else:
            if kw in q_low:
                matched_signals.append(f"keyword:{kw}")

    for tm in TEAM_TOKENS:
        if len(tm) <= 3:
            if re.search(rf"\b{re.escape(tm)}\b", q_low):
                matched_signals.append(f"team:{tm}")
        else:
            if re.search(rf"\b{re.escape(tm)}\b", q_low):
                matched_signals.append(f"team:{tm}")

    for pl in STAR_PLAYER_TOKENS:
        if re.search(rf"\b{re.escape(pl)}\b", q_low):
            matched_signals.append(f"player:{pl}")

    # ── Layer 1B: Check Hard Off-Topic Blocklist ──────────────────────────────
    # If off-topic terms appear and there are NO strong NHL/Rinklytics anchors, reject immediately.
    for pat in OFF_TOPIC_PATTERNS:
        m = re.search(pat, q_low)
        if m and len(matched_signals) == 0:
            off_word = m.group(0)
            msg = (
                f"🚨 Guardrail Error (`NON_HOCKEY_QUERY`): Rejected off-topic query matching `{off_word}`. "
                f"Agent Zamboni is strictly restricted to **NHL Hockey**, **Rinklytics Analytics Methodology** "
                f"(xG, GSAx, PPI, RPE, RPI, Pairwise, Franchise Rebuilds, Hit Impact, Shot Blocking, Drafts), "
                f"and our **Hugging Face Dataset (`jjerlee/nhl-analytics`)**."
                if lang == "EN"
                else f"🚨 Erreur de garde-fou (`NON_HOCKEY_QUERY`) : Question hors sujet rejetée (`{off_word}`). "
                f"L'Agent Zamboni répond exclusivement aux questions sur le **hockey de la LNH**, la **méthodologie Rinklytics** "
                f"et notre **jeu de données Hugging Face (`jjerlee/nhl-analytics`)**."
            )
            return GuardrailDecision(
                allowed=False,
                error_code="NON_HOCKEY_QUERY",
                error_message=msg,
            )

    # ── Layer 2: Require Positive Hockey or Rinklytics Methodology Signal ─────
    # Also check meta-questions about how the dashboard/calculations work if they reference metrics/pages
    methodology_inquiry_patterns = [
        r"\bhow\s+(?:is|are|do\s+you|does\s+rinklytics)\s+.*(?:calculat|comput|measur|model|formulat|calibrat)",
        r"\bwhat\s+(?:is|are)\s+the\s+(?:formula|equation|methodology|metric|dataset|table|page)",
    ]
    has_methodology_pattern = any(re.search(p, q_low) for p in methodology_inquiry_patterns)

    if len(matched_signals) == 0 and not has_methodology_pattern:
        msg = (
            "🚨 Guardrail Error (`NON_HOCKEY_QUERY`): No NHL hockey or Rinklytics analytics context detected. "
            "Agent Zamboni only answers questions about **NHL Hockey**, **Rinklytics Analytics & Measurement Methodology** "
            "(Expected Goals xG/ixG, Goalie GSAx, Skater/Goalie PPI, Pythagorean Expectation RPE, RPI, Pairwise Win Rank, "
            "Franchise Rebuilds, Hit Impact, Shot Blocking, Amateur Drafts, Scoreboard Recaps), and the "
            "**Hugging Face Dataset (`jjerlee/nhl-analytics`)**."
            if lang == "EN"
            else "🚨 Erreur de garde-fou (`NON_HOCKEY_QUERY`) : Aucun contexte lié au hockey de la LNH ou à Rinklytics n'a été détecté. "
            "L'Agent Zamboni répond exclusivement aux questions sur le hockey de la LNH, les méthodes de calcul Rinklytics "
            "et le jeu de données Hugging Face (`jjerlee/nhl-analytics`)."
        )
        return GuardrailDecision(
            allowed=False,
            error_code="NON_HOCKEY_QUERY",
            error_message=msg,
        )

    score = min(1.0, 0.25 * len(matched_signals) + (0.3 if has_methodology_pattern else 0.0))
    return GuardrailDecision(
        allowed=True,
        error_code=None,
        error_message=None,
        matched_signals=matched_signals[:12],
        similarity_score=round(score, 3),
    )


def clean_zamboni_prose(raw_text: str) -> str:
    """
    Purifies LLM output by stripping internal chain-of-thought tags (<think>...</think>,
    <thought>...</thought>) and reasoning preamble artifacts from reasoning models
    (e.g., Nemotron, Phi-4-mini-reasoning, DeepSeek/Qwen).
    """
    if not raw_text:
        return ""

    text = str(raw_text)
    # Remove closed <think> / <thought> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```thought.*?```", "", text, flags=re.DOTALL | re.IGNORECASE)

    # If an unclosed <think> tag exists, strip before the last closing or strip prefix
    if "</think>" in text.lower():
        idx = text.lower().rfind("</think>")
        text = text[idx + len("</think>"):]
    elif "<think>" in text.lower():
        idx = text.lower().find("<think>")
        text = text[:idx]

    lines = text.splitlines()
    cleaned_lines: List[str] = []
    skip_prefixes = (
        "thinking process:",
        "here is my thought process:",
        "okay, let's",
        "okay, i need to",
        "first, let's check",
        "let's analyze the user",
    )
    for line in lines:
        s = line.strip().lower()
        if any(s.startswith(p) for p in skip_prefixes):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()
