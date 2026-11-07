"""
router.py — Pattern 3: Adaptive Multi-Pipeline Router & Entity Extractor for Agent Zamboni.

Outputs a strict `RouteDecision` containing:
- `intent`: `METHODOLOGY` | `DATA_QUERY` | `HYBRID`
- `pipeline_ids`: List of specialist pipelines to execute in parallel (handles mixed queries)
- `target_domains`: Canonical Rinklytics knowledge domains
- `extracted_entities`: Teams, Season Year, Date, Top-N, Metric sort key
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from .config import NHL_TEAMS
from .models import RouteDecision


# Map common team names/nicknames/cities to 3-letter NHL tricode
TEAM_ALIAS_TO_TRICODE: Dict[str, str] = {
    "anaheim": "ANA", "ducks": "ANA",
    "arizona": "ARI", "coyotes": "ARI",
    "boston": "BOS", "bruins": "BOS",
    "buffalo": "BUF", "sabres": "BUF",
    "carolina": "CAR", "hurricanes": "CAR", "canes": "CAR",
    "columbus": "CBJ", "blue jackets": "CBJ", "jackets": "CBJ",
    "calgary": "CGY", "flames": "CGY",
    "chicago": "CHI", "blackhawks": "CHI", "hawks": "CHI",
    "colorado": "COL", "avalanche": "COL", "avs": "COL",
    "dallas": "DAL", "stars": "DAL",
    "detroit": "DET", "red wings": "DET", "wings": "DET",
    "edmonton": "EDM", "oilers": "EDM",
    "florida": "FLA", "panthers": "FLA", "cats": "FLA",
    "los angeles": "LAK", "kings": "LAK",
    "minnesota": "MIN", "wild": "MIN",
    "montreal": "MTL", "montréal": "MTL", "canadiens": "MTL", "habs": "MTL",
    "new jersey": "NJD", "devils": "NJD",
    "nashville": "NSH", "predators": "NSH", "preds": "NSH",
    "islanders": "NYI", "isles": "NYI",
    "rangers": "NYR",
    "ottawa": "OTT", "senators": "OTT", "sens": "OTT",
    "philadelphia": "PHI", "flyers": "PHI",
    "pittsburgh": "PIT", "penguins": "PIT", "pens": "PIT",
    "seattle": "SEA", "kraken": "SEA",
    "san jose": "SJS", "sharks": "SJS",
    "st. louis": "STL", "st louis": "STL", "blues": "STL",
    "tampa bay": "TBL", "tampa": "TBL", "lightning": "TBL", "bolts": "TBL",
    "toronto": "TOR", "maple leafs": "TOR", "leafs": "TOR",
    "utah": "UTA",
    "vancouver": "VAN", "canucks": "VAN", "nucks": "VAN",
    "vegas": "VGK", "golden knights": "VGK", "knights": "VGK",
    "winnipeg": "WPG", "jets": "WPG",
    "washington": "WSH", "capitals": "WSH", "caps": "WSH",
}

DOMAIN_TO_PIPELINE_ID: Dict[str, str] = {
    "player_ppi": "skill_skater_analytics",
    "goalie_ppi": "skill_goalie_analytics",
    "team_season": "skill_team_season_kpis",
    "rebuild": "skill_rebuild_econometrics",
    "hits": "skill_hit_impact",
    "blocking_shots": "skill_blocking_shots",
    "draft": "skill_draft_lifecycle",
    "map_travel": "skill_map_travel",
    "scoreboard_recaps": "skill_scoreboard_and_recaps",
}


def route_query_decision(query: str) -> RouteDecision:
    """
    Pattern 3 (Adaptive Multi-Pipeline Router):
    Analyzes a hockey query and returns a structured `RouteDecision` with `pipeline_ids`,
    `target_domains`, and `extracted_entities`.
    """
    q_low = (query or "").lower()

    # ── 1. Extract Team Tricodes ──────────────────────────────────────────────
    teams: List[str] = []
    for tri in NHL_TEAMS.keys():
        if re.search(rf"\b{tri}\b", query) or re.search(rf"\b{tri.lower()}\b", q_low):
            if tri in ("VAN", "SEA", "CAR", "MIN") and not re.search(rf"\b{tri}\b", query):
                continue
            if tri not in teams:
                teams.append(tri)

    for alias, tri in sorted(TEAM_ALIAS_TO_TRICODE.items(), key=lambda x: -len(x[0])):
        if re.search(rf"\b{re.escape(alias)}\b", q_low):
            if tri not in teams:
                teams.append(tri)

    # ── 2. Extract Season Year ────────────────────────────────────────────────
    season_year = None
    m_span = re.search(r"\b(20[0-2]\d)\s*[-/]\s*(?:20)?(\d{2})\b", q_low)
    if m_span:
        season_year = int(m_span.group(1))
    else:
        m_yr = re.search(r"\b(201[3-9]|202[0-6])\b", q_low)
        if m_yr:
            season_year = int(m_yr.group(1))

    # ── 3. Extract Date (YYYY-MM-DD) ──────────────────────────────────────────
    date_str = None
    m_date = re.search(r"\b(202\d-\d{2}-\d{2})\b", q_low)
    if m_date:
        date_str = m_date.group(1)

    # ── 4. Extract Top-N ──────────────────────────────────────────────────────
    top_n = 10
    m_top = re.search(r"\btop\s+(\d{1,2})\b", q_low)
    if m_top:
        top_n = max(1, min(int(m_top.group(1)), 32))

    # ── 5. Detect Target Domains ──────────────────────────────────────────────
    domains: List[str] = []

    if any(k in q_low for k in ("rebuild", "reconstruction", "régime", "regime", "inaction", "teardown", "incubation", "emergence", "purgatory", "purgatoire", "ali", "yis", "rci", "vir", "ccm", "spsi", "shannon", "surplus", "scorched earth", "retool", "gm lever", "b1", "b2", "b3", "b4")):
        domains.append("rebuild")

    if any(k in q_low for k in ("hit", "hits", "hitting", "bodycheck", "mise en échec", "mises en échec", "puck win rate", "puck retention", "30s", "30-second", "forecheck", "échec avant")):
        domains.append("hits")

    if any(k in q_low for k in ("block", "blocked", "blocking", "tir bloqué", "tirs bloqués", "bloquer", "sankey")):
        domains.append("blocking_shots")

    if any(k in q_low for k in ("draft", "drafted", "draftee", "repêchage", "repêché", "espoir", "espoirs", "magnifiques", "exceptional season", "exceptional player", "95th percentile", "not so magnificent", "first round", "1st round")):
        domains.append("draft")

    if any(k in q_low for k in ("distance", "travel", "map", "arena", "miles", "kilometers", "km", "folium", "voyage", "déplacement", "carte")):
        domains.append("map_travel")

    if any(k in q_low for k in ("scoreboard", "recap", "recaps", "odds", "moneyline", "game script", "comeback", "wire_to_wire", "blowout", "slate", "tableau des scores", "résumé", "résumés")):
        domains.append("scoreboard_recaps")

    if any(k in q_low for k in ("rpe", "pythagorean", "rwin_fittedpe", "rpi", "rating percentage index", "pairwise", "pairwise_win", "common opponent", "strength of record", "kpinorm", "kpi_pe", "kpi_pairwise", "hot streak", "win streak", "team season", "standings", "classement")):
        domains.append("team_season")

    if any(k in q_low for k in ("goalie", "goalies", "goaltender", "goaltending", "gardien", "gardiens", "gsax", "xga", "xsave_pct", "dsave_pct", "xsv%", "dsv%", "hd_save_pct", "save percentage", "gaa", "shutout", "blanchissage", "vasilevskiy", "hellebuyck", "shesterkin", "sorokin", "dobes", "dobeš", "montembeault", "swayman", "oettinger", "bobrovsky", "saros")):
        domains.append("goalie_ppi")

    if any(k in q_low for k in ("skater", "skaters", "patineur", "patineurs", "forward", "defenseman", "ppi", "player performance index", "indice de performance", "ixg", "goals above expected", "goals_above_expected", "points_per_game", "hd_shots", "mcdavid", "draisaitl", "mackinnon", "makar", "matthews", "bedard", "kucherov", "kaprizov", "suzuki", "caufield", "hutson", "celebrini", "crosby", "ovechkin", "pastrnak", "hughes")):
        domains.append("player_ppi")

    if any(k in q_low for k in ("xg", "ixg", "xga", "gsax", "expected goal", "buts attendus", "but attendu", "dual-stage", "dual stage", "logistic", "crease-jam", "clamping", "slot", "procs_xg", "fenwick", "high-danger", "hdcf")):
        if "xg_gsax" not in domains:
            domains.append("xg_gsax")

    if any(k in q_low for k in ("zamboni", "gordie", "zambia", "architecture", "platform", "rinklytics", "huggingface", "hugging face", "lakehouse", "lm studio", "zerogpu", "rate limit", "guardrail", "label studio", "langgraph", "langchain")):
        domains.append("platform_architecture")

    if teams and not domains:
        domains.append("team_season")

    if not domains:
        domains = ["platform_architecture", "xg_gsax"]

    # ── 6. Detect Metric Sort Target ──────────────────────────────────────────
    metric = "PPI"
    if "gsax" in q_low or "goals saved above expected" in q_low:
        metric = "gsax"
    elif "dsv%" in q_low or "dsave_pct" in q_low:
        metric = "dsave_pct"
    elif "xsv%" in q_low or "xsave_pct" in q_low:
        metric = "xsave_pct"
    elif "save percentage" in q_low or "sv%" in q_low:
        metric = "savePercentage"
    elif "goals above expected" in q_low or "g - ixg" in q_low:
        metric = "goals_above_expected"
    elif "ixg" in q_low or "expected goals" in q_low or "xg" in q_low:
        metric = "ixg" if "player_ppi" in domains else "xg_pct"
    elif "high-danger" in q_low or "hd_shots" in q_low or "hdcf" in q_low:
        metric = "hd_shots" if "player_ppi" in domains else "hdcf_pct"
    elif "goals" in q_low and "expected" not in q_low and "against" not in q_low:
        metric = "goals"
    elif "assists" in q_low:
        metric = "assists"
    elif "points" in q_low and "team_season" not in domains:
        metric = "points"
    elif "rpe" in q_low or "pythagorean" in q_low:
        metric = "rpe"
    elif "rpi" in q_low:
        metric = "rpi"
    elif "pairwise" in q_low:
        metric = "pairwise_win"

    # ── 7. Classify Intent: METHODOLOGY vs DATA_QUERY vs HYBRID ───────────────
    methodology_cues = any(
        re.search(p, q_low)
        for p in [
            r"\bhow\s+(?:is|are|do|does)\b",
            r"\bexplain\b",
            r"\bwhat\s+(?:is|are)\s+(?:the\s+)?(?:formula|equation|method|calculation|difference|definition|meaning|approach|architecture|guardrail)",
            r"\bwhy\s+(?:do|does|is)\b",
            r"\bformula\b",
            r"\bequation\b",
            r"\bcalculat(?:e|ed|ion|ing)\b",
            r"\bcalibrat(?:e|ed|ion)\b",
            r"\breference\b",
            # French cues
            r"\bcomment\b",
            r"\bexpliqu(?:er|ez)\b",
            r"\bformule\b",
            r"\béquation\b",
            r"\bcalcul(?:er|é|ée|er)?\b",
            r"\bpourquoi\b",
            r"\bdéfinition\b",
            r"\bméthode\b",
            r"\bapproche\b",
        ]
    )

    data_cues = any(
        re.search(p, q_low)
        for p in [
            r"\bwho\s+(?:is|are|leads|led|has|had|won)\b",
            r"\bwhich\s+(?:player|skater|goalie|team|franchise)\b",
            r"\btop\s+\d+\b",
            r"\bleader(?:s|board)?\b",
            r"\bcompare\b",
            r"\bstats\s+for\b",
            r"\bhow\s+many\b",
            r"\bwhat\s+(?:was|were|is|are)\s+.*(?:'s|stats|record|score|distance|regime|ppi|gsax|ixg|rpe)\b",
            # French data cues
            r"\bqui\b",
            r"\bquel(?:s|le|les)?\b",
            r"\bmeilleur(?:s|es)?\b",
            r"\bcompar(?:er|ez)\b",
            r"\bcombien\b",
            r"\bclassement\b",
            r"\bstatistiques?\b",
            r"\bstatut\b",
            r"\bstat\b",
        ]
    ) or bool(teams) or any(
        pl in q_low for pl in (
            "mcdavid", "draisaitl", "mackinnon", "makar", "matthews", "bedard", "kucherov",
            "vasilevskiy", "hellebuyck", "shesterkin", "sorokin", "dobes", "dobeš", "montembeault",
            "suzuki", "caufield", "hutson", "kaprizov", "crosby", "ovechkin", "pastrnak", "hughes",
            "celebrini", "oettinger", "swayman", "bobrovsky",
        )
    )

    if methodology_cues and data_cues:
        intent = "HYBRID"
    elif data_cues and not methodology_cues:
        intent = "DATA_QUERY"
    else:
        intent = "METHODOLOGY"

    # Build explicit list of pipeline_ids for Pattern 3 multi-pipeline routing
    pipeline_ids: List[str] = ["methodology_rag"]
    for dom in domains:
        pid = DOMAIN_TO_PIPELINE_ID.get(dom)
        if pid and pid not in pipeline_ids:
            pipeline_ids.append(pid)
    if "xg_gsax" in domains and intent in ("DATA_QUERY", "HYBRID"):
        if "skill_skater_analytics" not in pipeline_ids:
            pipeline_ids.append("skill_skater_analytics")

    entities = {
        "teams": teams,
        "season_year": season_year,
        "date_str": date_str,
        "top_n": top_n,
        "metric": metric,
        "raw_query": query,
    }
    return RouteDecision(
        intent=intent,
        pipeline_ids=pipeline_ids,
        target_domains=domains,
        extracted_entities=entities,
    )


def route_query(query: str) -> Tuple[str, List[str], Dict[str, Any]]:
    """Backward-compatible tuple unpacker around `route_query_decision`."""
    dec = route_query_decision(query)
    return dec.intent, dec.target_domains, dec.extracted_entities
