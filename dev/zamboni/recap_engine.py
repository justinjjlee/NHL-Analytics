"""
recap_engine.py — Unified Agent Zamboni Editorial & Game/Slate Recap Engine.

Unifies the narrative generation, Context Engineering (State Projection, Prompt Prefix Caching,
Structured Scratchpad Sanitization), Strategic Deterministic Reflection (with Context Pruning
via Diffing), and Live Slate/Game Record Reconciliation across `dev/zamboni/` and
`dev/agent/scoreboard/`.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("Zamboni.RecapEngine")

# =============================================================================
# 1. NHL Franchise Lookup & Shared Constants
# =============================================================================

NHL_TEAM_NAMES: Dict[str, str] = {
    "ANA": "Anaheim Ducks",
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
    "MTL": "Montreal Canadiens",
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
    "UTA": "Utah Mammoth",
    "VAN": "Vancouver Canucks",
    "VGK": "Vegas Golden Knights",
    "WPG": "Winnipeg Jets",
    "WSH": "Washington Capitals",
}

ALGORITHMIC_MODEL_STAMP = "algorithmic-guardrails-v1"

# Static Prefix-Cached System Prompts (Context Engineering Technique 5)
ZAMBONI_GAME_RECAP_SYSTEM_PROMPT = (
    "You are Agent Zamboni, senior NHL beat columnist and tactical hockey analyst for Rinklytics.\n"
    "Write an authoritative sports headline and a 5-paragraph NHL game recap in crisp, broadcast-grade hockey prose.\n\n"
    "STRICT EDITORIAL & FACTUAL RULES:\n"
    "1. Line 1 MUST be the headline: HEADLINE: <Sharp journalistic headline under 15 words with final score>\n"
    "2. Follow with exactly 5 distinct, non-overlapping paragraphs separated by blank lines:\n"
    "   - Paragraph 1 (The Lede): Final score, winning/losing teams, venue, decision type (Regulation/OT/SO), game script, and #1 star catalyst.\n"
    "   - Paragraph 2 (Scoring Progression): Chronological period-by-period scoring flow (1st intermission score, 2nd intermission score, key goals, and game-winning strike). DO NOT repeat the opening final-score summary from Paragraph 1.\n"
    "   - Paragraph 3 (5v5 Line Shift Chemistry & L5 Trends): On-ice three-forward (3F) trios and two-defenseman (2D) pairings—Time on Ice (TOI), Corsi For percentage (CF%), Shots on Goal (SOG), Goals For vs. Goals Against (GF-GA), and comparison to their last five games (L5) baseline.\n"
    "   - Paragraph 4 (Goaltending GSAx, Shot Quality xG & Defense): Winning and losing goaltender saves, Save Percentage (SV%), Goals Saved Above Expected (GSAx), team Expected Goals (xG) & high-danger chances, blocked shots, hits, and power-play conversions.\n"
    "   - Paragraph 5 (Season Standings, Pythagorean Expectation & L5 Form): Updated season records, points percentage, Pythagorean expectation, and rolling team form over the last five games (L5).\n"
    "3. EXACT PLAYER NAMES: Use player names EXACTLY as written in the roster ledger (e.g. if listed as 'V. Husso' or 'D. Wolf', write 'V. Husso' or 'Husso'—NEVER invent or guess first names).\n"
    "4. STRICT SHUTOUT RULE: A shutout ONLY occurs when the losing team scored 0 total goals in the game. If a relief goalie stopped 10/10 shots in a 5-3 game, call it scoreless relief—NEVER call it a shutout.\n"
    "5. STRICT GOALS VS. ASSISTS RULE: Players with 0 goals and multiple assists dished out assists/points; NEVER state they scored or netted a goal.\n"
    "6. FIRST-MENTION ABBREVIATION RULE: Spell out abbreviations on first mention only (e.g., 'last five games (L5)', 'Time on Ice (TOI)', 'Shots on Goal (SOG)', 'Corsi For percentage (CF%, share of all shot attempts)', 'Save Percentage (SV%)').\n"
    "7. NO PROMPT ECHOING OR MONOLOGUE: Never output <think> blocks, planning notes, markdown headers, bullet points, or internal archetype labels."
)

ZAMBONI_SLATE_REVIEW_SYSTEM_PROMPT = (
    "You are Agent Zamboni, Executive NHL Editor writing 'The NHL Night in Review' for Rinklytics.\n"
    "Write a sharp Day Title (headline) and a 3-paragraph executive summary covering the completed NHL slate.\n\n"
    "STRICT EDITORIAL & FACTUAL RULES:\n"
    "1. Line 1 MUST be the Day Title formatted as: HEADLINE: <Authoritative slate headline citing the top star/storyline and the exact game/goal scope>\n"
    "2. Follow with exactly 3 paragraphs separated by blank lines:\n"
    "   - Paragraph 1 (Slate Macro Ledger): Cite the EXACT number of games played, EXACT total goals scored, average goals per game, home wins vs. road wins split, and number of overtime/shootout finishes.\n"
    "   - Paragraph 2 (Three Stars of the Night): Highlight the #1, #2, and #3 Stars of the Night with their exact teams and stat lines.\n"
    "   - Paragraph 3 (Signature Matchups & Tactical Storylines): Summarize the standout team victories, comebacks, overtime walk-offs, or goaltending clinics from the matchup ledger.\n"
    "3. ZERO HALLUCINATIONS: Never claim a night had 'no goals' or was 'scoreless'. Never assign a player to the wrong team. Never invent first names for initialed players. No <think> tags or meta-commentary."
)


# =============================================================================
# 2. Text Purification & First-Mention Abbreviation Expansion
# =============================================================================

def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()


def _paragraph_jaccard(p1: str, p2: str) -> float:
    """Computes content-word Jaccard similarity between two paragraphs to detect duplicate ledes."""
    stop = {"the", "and", "for", "with", "over", "from", "into", "after", "that", "this", "was", "were", "their", "his"}
    w1 = {w for w in re.findall(r"[a-z0-9]+", p1.lower()) if len(w) >= 3 and w not in stop}
    w2 = {w for w in re.findall(r"[a-z0-9]+", p2.lower()) if len(w) >= 3 and w not in stop}
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / float(len(w1 | w2))


def _are_duplicate_lede_paragraphs(p1: str, p2: str) -> bool:
    """Detects when Paragraph 1 and Paragraph 2 both state the opening final-score lede."""
    if _paragraph_jaccard(p1, p2) >= 0.42:
        return True
    scores_1 = set(re.findall(r"\b\d+[–-]\d+\b", p1))
    scores_2 = set(re.findall(r"\b\d+[–-]\d+\b", p2))
    lede_verbs = r"\b(?:secured|defeated|emerged victorious|triumph|victory over|beat|edged|outlasted|routed)\b"
    if scores_1 and (scores_1 & scores_2) and re.search(lede_verbs, p1.lower()) and re.search(lede_verbs, p2.lower()):
        # If P2 starts by restating the winner and final score in its first sentence
        first_sent_p2 = p2.split(".")[0].lower()
        if any(sc in first_sent_p2 for sc in scores_2) and re.search(lede_verbs, first_sent_p2):
            return True
    return False


def clean_llm_narrative(text: str) -> str:
    """Strips reasoning tokens, thinking blocks, echoed prompt ledgers, archetype labels, and duplicate lede paragraphs."""
    if not text:
        return ""

    cleaned = str(text)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<thought>.*?</thought>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in cleaned.lower():
        idx = cleaned.lower().rfind("</think>")
        cleaned = cleaned[idx + len("</think>"):]
    cleaned = re.sub(r"```(?:thought|thinking).*?```", "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    # Strip inline Paragraph N (...): prefixes and raw uppercase enum codes like (TERRITORIAL_EXECUTION)
    cleaned = re.sub(
        r"(?m)^(?:#{1,4}\s*)?(?:\*\*)?Paragraph\s*\d+\s*(?:\([^\n]*\))?\s*(?:\*\*)?\s*:\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"[ \t]*\([A-Z]{3,}_[A-Z0-9_]+\)", "", cleaned)

    lines = cleaned.strip().splitlines()
    filtered_lines: List[str] = []
    skipping_preamble = True

    reasoning_prefixes = (
        "okay, i need to",
        "okay, let's",
        "alright, i need",
        "alright, let's",
        "let me start by",
        "let's start by",
        "thinking process:",
        "here is my thought",
        "i need to write",
        "first, the first paragraph",
        "second paragraph deals",
        "third paragraph is",
        "fourth paragraph is",
        "fifth paragraph is",
        "now checking each paragraph",
        "structuring each paragraph",
        "wait, but",
        "wait, let's",
        "first paragraph:",
        "paragraph 1:",
        "the lede:",
        "sure, here is",
        "here is the headline",
    )

    echoed_ledger_prefixes = (
        "final score & venue:",
        "first star / catalyst:",
        "verified scoring & roster ledger",
        "chronological goals scored:",
        "goaltenders (never credit",
        "goaltenders:",
        "primary editorial archetype:",
        "editorial angle rationale:",
        "paragraph-by-paragraph writing plan",
        "write the article now:",
        "season trajectory (",
        "line shift chemistry (",
        "goaltending composure,",
    )

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if stripped in ("---", "***", "___"):
            continue
        if any(lower.startswith(p) for p in echoed_ledger_prefixes):
            continue
        if re.match(r"^(?:#{1,4}\s*)?(?:\*\*)?paragraph\s*\d+.*:\s*$", lower):
            continue
        if re.match(r"^[a-z .]+\(\d+\)\s+defeated\s+[a-z .]+\(\d+\)\s+in\s+final", lower):
            continue
        if re.match(r"^top\s+[a-z]{2,4}\s+skaters\s*:", lower):
            continue
        if re.match(r"^\d+\.\s+(?:1st|2nd|3rd|ot|overtime|so|shootout)\b", lower):
            continue
        if re.match(r"^-\s+[^\n]+\bgoaltender\b.*saves", lower):
            continue
        if stripped.endswith("**") and len(stripped.split()) <= 14 and not stripped.endswith(".**"):
            continue

        if skipping_preamble:
            if not stripped:
                continue
            if any(lower.startswith(p) for p in reasoning_prefixes):
                continue
            if re.match(r"^(?:#{1,4}\s*)?(?:\*\*)?paragraph\s*\d.*", lower):
                continue
            skipping_preamble = False
        filtered_lines.append(line)

    # Deduplicate consecutive paragraphs if Paragraph 1 and Paragraph 2 repeat the lede
    deduped_paras: List[str] = []
    for p in filtered_lines:
        s = p.strip()
        if not s:
            if deduped_paras and deduped_paras[-1] != "":
                deduped_paras.append("")
            continue
        if s.lower().startswith("in summary,") and len([x for x in deduped_paras if x]) >= 5:
            continue
        non_empty = [x for x in deduped_paras if x and not x.upper().startswith("HEADLINE:")]
        if non_empty and _are_duplicate_lede_paragraphs(non_empty[-1], s):
            if len(s) > len(non_empty[-1]):
                for idx in range(len(deduped_paras) - 1, -1, -1):
                    if deduped_paras[idx] == non_empty[-1]:
                        deduped_paras[idx] = s
                        break
            continue
        deduped_paras.append(s)

    return "\n".join(deduped_paras).strip()


def format_milestone_badge(category: str, title: str, description: str = "") -> str:
    """Formats a milestone into a clean, human-readable string without brackets or emoji clutter."""
    clean_title = re.sub(r"^\[.*?\]\s*", "", str(title)).strip()
    if not description:
        return f"**{clean_title}**"
    clean_desc = str(description).strip()
    return f"**{clean_title}** — {clean_desc}"


def expand_first_mention_abbreviations(text: str) -> str:
    """
    Idempotently expands hockey/analytical abbreviations (L5, TOI, SOG, CF-CA, CF%, FF%,
    GF-GA, GF/G, GA/G, SF-SA, SV%, OZS%, PIM, FO%, 3F, 2D) on their first mention only.
    """
    if not text or not text.strip():
        return text

    out = text
    normalization_rules = [
        (r"(?:[Ll]ast|[Pp]ast)\s+(?:five|5)\s+[Gg]ames\s*\(L5\)", "L5"),
        (r"[Ll]ast-five-games\s*\(L5\)", "L5"),
        (r"[Tt]ime\s+on\s+[Ii]ce\s*\(TOI\)", "TOI"),
        (r"[Ss]hots\s+on\s+[Gg]oal\s*\(SOG\)", "SOG"),
        (r"Corsi\s+For\s+percentage\s*\(CF%[^)]*\)", "CF%"),
        (r"Corsi\s+For\s+vs\.?\s*(?:Corsi\s+)?Against\s*\(CF-CA[^)]*\)", "CF-CA"),
        (r"Corsi\s+For\s*\(CF[^)]*\)", "CF"),
        (r"Fenwick\s+For\s+percentage\s*\(FF%[^)]*\)", "FF%"),
        (r"[Gg]oals\s+[Ff]or\s+vs\.?\s*[Gg]oals\s+[Aa]gainst\s*\(GF-GA\)", "GF-GA"),
        (r"[Gg]oals\s+[Ff]or\s+per\s+game\s*\(GF/G\)", "GF/G"),
        (r"[Gg]oals\s+[Aa]gainst\s+per\s+game\s*\(GA/G\)", "GA/G"),
        (r"[Ss]hots\s+[Ff]or\s+vs\.?\s*[Ss]hots\s+[Aa]gainst\s*\(SF-SA\)", "SF-SA"),
        (r"[Ss]ave\s+[Pp]ercentage\s*\(SV%\)", "SV%"),
        (r"[Oo]ffensive\s+[Zz]one\s+[Ss]tart\s+percentage\s*\(OZS%\)", "OZS%"),
        (r"[Pp]enalty\s+[Mm]inutes\s*\(PIM\)", "PIM"),
        (r"[Ff]aceoff\s+[Ww]in\s+percentage\s*\(FO%\)", "FO%"),
        (r"three-forward\s*\(3F\)", "3F"),
        (r"two-defenseman\s*\(2D\)", "2D"),
    ]
    for pattern, base_token in normalization_rules:
        out = re.sub(pattern, base_token, out)

    first_mention_expansions = [
        (r"\bL5\b", "last five games (L5)"),
        (r"\bTOI\b", "Time on Ice (TOI)"),
        (r"\bSOG\b", "Shots on Goal (SOG)"),
        (r"\bCF-CA\b", "Corsi For vs. Corsi Against (CF-CA, total shot attempts for and against)"),
        (r"\bCF%(?!\w)", "Corsi For percentage (CF%, share of all shot attempts)"),
        (r"\bFF%(?!\w)", "Fenwick For percentage (FF%, share of unblocked shot attempts)"),
        (r"\bGF-GA\b", "Goals For vs. Goals Against (GF-GA)"),
        (r"\bGF/G\b", "Goals For per game (GF/G)"),
        (r"\bGA/G\b", "Goals Against per game (GA/G)"),
        (r"\bSF-SA\b", "Shots For vs. Shots Against (SF-SA)"),
        (r"\bSV%(?!\w)", "Save Percentage (SV%)"),
        (r"\bOZS%(?!\w)", "Offensive Zone Start percentage (OZS%)"),
        (r"\bPIM\b", "Penalty Minutes (PIM)"),
        (r"\bFO%(?!\w)", "Faceoff Win percentage (FO%)"),
        (r"\b3F\b", "three-forward (3F)"),
        (r"\b2D\b", "two-defenseman (2D)"),
    ]
    for pattern, expanded_str in first_mention_expansions:
        out = re.sub(pattern, expanded_str, out, count=1)

    if "Corsi For" not in out and re.search(r"\bCF\b(?![%-])", out):
        out = re.sub(r"\bCF\b(?![%-])", "Corsi For (CF, total shot attempts)", out, count=1)

    return out


def extract_headline_and_narrative(
    raw_text: str, default_headline: str, period_type: str = "REG"
) -> Tuple[str, str]:
    """Separates the LLM-generated headline from the narrative paragraphs and expands first-mention abbreviations."""
    cleaned = clean_llm_narrative(raw_text).strip()
    lines = cleaned.splitlines()
    headline = default_headline
    narrative_lines: List[str] = []
    found_headline = False

    for line in lines:
        s = line.strip()
        if not s:
            continue
        s_unwrapped = s.strip('"*# ').strip()
        if not s_unwrapped:
            continue

        if not found_headline:
            if re.match(r"^(?:headline|title)\s*:?\s*$", s_unwrapped, re.IGNORECASE):
                continue
            if re.match(r"^(?:headline|title)\s*:\s*", s_unwrapped, re.IGNORECASE):
                candidate = re.sub(r"^(?:headline|title)\s*:\s*", "", s_unwrapped, flags=re.IGNORECASE).strip().strip('"*# ')
                if candidate and candidate.lower() not in ("headline", "headline:", "title", "title:"):
                    headline = candidate
                    found_headline = True
                continue
            elif (
                len(s_unwrapped.split()) <= 18
                and not s_unwrapped.endswith((".", ";"))
                and not s_unwrapped.lower().startswith(("the ", "action ", "in ", "across "))
            ):
                if s_unwrapped.lower() not in ("headline", "headline:", "title", "title:"):
                    headline = s_unwrapped
                    found_headline = True
                continue

        if re.match(r"^(?:recap|article|headline|title)\s*:?\s*$", s_unwrapped, re.IGNORECASE):
            continue
        if s.startswith("#") and len(s_unwrapped.split()) <= 16:
            continue
        if s.startswith("**") and s.endswith("**") and len(s_unwrapped.split()) <= 16 and not s_unwrapped.endswith("."):
            continue
        narrative_lines.append(line)

    h_lower = headline.lower()
    if h_lower in ("headline", "headline:", "title", "title:") or len(headline.strip()) < 6:
        headline = default_headline
    elif period_type == "REG" and ("overtime" in h_lower or "shootout" in h_lower or " ot " in h_lower):
        headline = default_headline
    elif period_type == "OT" and "shootout" in h_lower:
        headline = default_headline

    raw_paras = [p.strip() for p in narrative_lines if p.strip()]
    final_paras: List[str] = []
    for p in raw_paras:
        if final_paras and _are_duplicate_lede_paragraphs(final_paras[-1], p):
            if len(p) > len(final_paras[-1]):
                final_paras[-1] = p
            continue
        final_paras.append(p)

    narrative = "\n\n".join(final_paras).strip()
    narrative = expand_first_mention_abbreviations(narrative)
    return headline, narrative


# =============================================================================
# 3. Context Engineering: State Projection Prompts (Games & Slates)
# =============================================================================

def build_projected_game_prompt(game: Any) -> str:
    """
    Context Engineering Technique 1 (State Projection):
    Projects a compact, high-signal, non-redundant fact sheet organized strictly by the
    5 paragraph roles so 3B-4B models never repeat Paragraph 1 or hallucinate names/teams.
    """
    w = game.winner
    l = game.loser
    flow = game.period_flow
    w_is_home = (w.team_abbrev == game.home_team.team_abbrev)
    w_tr = game.home_shift_trend_profile if w_is_home else game.away_shift_trend_profile
    l_tr = game.away_shift_trend_profile if w_is_home else game.home_shift_trend_profile
    w_g = game.winning_goalie_profile
    l_g = game.losing_goalie_profile
    w_def = game.home_defense_profile if w_is_home else game.away_defense_profile
    l_def = game.away_defense_profile if w_is_home else game.home_defense_profile
    w_ctx = game.home_season_context if w_is_home else game.away_season_context
    l_ctx = game.away_season_context if w_is_home else game.home_season_context

    dec_str = (
        "Regulation"
        if game.period_type == "REG"
        else ("Overtime" if game.period_type == "OT" else "Shootout")
    )
    star1 = f"{game.three_stars[0].name} ({game.three_stars[0].team_abbrev}, {game.three_stars[0].stat_summary})" if game.three_stars else "Balanced team attack"

    scoring_items = []
    for idx, sp in enumerate(game.scoring_plays or [], 1):
        assists_str = f" [Assists: {', '.join(sp.assists)}]" if sp.assists else " [Unassisted]"
        scoring_items.append(
            f"  {idx}. {sp.period_descriptor} {sp.time_in_period} ({sp.strength}): {sp.goal_scorer} ({sp.scoring_team}){assists_str} -> {game.away_team.team_abbrev} {sp.away_score}, {game.home_team.team_abbrev} {sp.home_score}"
        )
    scoring_str = "\n".join(scoring_items) if scoring_items else "  Scoreless regulation/OT."

    shift_lines = []
    if w_tr and w_tr.has_shift_data:
        shift_lines.append(
            f"  - {w.team_abbrev} 5v5 Line Continuity: {w_tr.line_continuity_pct:.1f}% L5 continuity ({w_tr.new_lines_deployed_count} new/shuffled trios)."
        )
        if w_tr.best_line:
            shift_lines.append(f"  - {w.team_abbrev} Standout Forward Trio: {w_tr.best_line.line_story_summary}")
        if w_tr.top_pair:
            shift_lines.append(f"  - {w.team_abbrev} Top Defense Pair: {w_tr.top_pair.line_story_summary}")
    if l_tr and l_tr.has_shift_data and l_tr.best_line:
        shift_lines.append(f"  - {l.team_abbrev} Top Forward Trio: {l_tr.best_line.line_story_summary}")
    if not shift_lines:
        top_w = ", ".join(f"{s.name} ({s.goals}G, {s.assists}A, {s.sog} SOG, {s.toi} TOI)" for s in (game.home_skaters if w_is_home else game.away_skaters)[:3])
        top_l = ", ".join(f"{s.name} ({s.goals}G, {s.assists}A, {s.sog} SOG, {s.toi} TOI)" for s in (game.away_skaters if w_is_home else game.home_skaters)[:2])
        shift_lines.append(f"  - {w.team_abbrev} Skater Leaders: {top_w}")
        shift_lines.append(f"  - {l.team_abbrev} Skater Leaders: {top_l}")

    g_lines = []
    if w_g:
        shutout_note = "TRUE SHUTOUT" if (l.score == 0 and w_g.is_shutout) else ("Relief outing, NOT a shutout" if w_g.goals_against == 0 and l.score > 0 else "Standard decision")
        g_lines.append(
            f"  - Winning Goalie: {w_g.name} ({w.team_abbrev}, {shutout_note}) — {w_g.saves}/{w_g.shots_against} saves ({w_g.save_pct*100:.1f}% SV%, {w_g.gsax:+.2f} GSAx)."
        )
    if l_g:
        g_lines.append(
            f"  - Losing Goalie: {l_g.name} ({l.team_abbrev}) — {l_g.saves}/{l_g.shots_against} saves ({l_g.save_pct*100:.1f}% SV%, {l_g.gsax:+.2f} GSAx)."
        )
    g_lines.append(
        f"  - Shot Quality & Expected Goals (xG): {w.team_abbrev} {w.xg:.2f} xG ({w.xg_pct:.1f}% share, {w.hdcf} high-danger chances, {w.sog} SOG) vs {l.team_abbrev} {l.xg:.2f} xG ({l.xg_pct:.1f}% share, {l.hdcf} high-danger chances, {l.sog} SOG)."
    )
    if w_def and l_def:
        g_lines.append(
            f"  - Defense & Special Teams: {w.team_abbrev} ({w_def.blocked_shots} blocks, {w_def.hits} hits, {w_def.turnover_diff:+d} turnover diff, PP {w.power_play_conv}) vs {l.team_abbrev} ({l_def.blocked_shots} blocks, {l_def.hits} hits, {l_def.turnover_diff:+d} turnover diff, PP {l.power_play_conv})."
        )

    s_lines = []
    if w_ctx and l_ctx:
        s_lines.append(
            f"  - {w.team_name} Season Standing: {w_ctx.wins}-{w_ctx.losses}-{w_ctx.ot_losses} ({w_ctx.points} pts, {w_ctx.points_pct*100:.1f}% pts, Pythagorean {w_ctx.pythagorean_pct*100:.1f}%)."
        )
        s_lines.append(
            f"  - {l.team_name} Season Standing: {l_ctx.wins}-{l_ctx.losses}-{l_ctx.ot_losses} ({l_ctx.points} pts, {l_ctx.points_pct*100:.1f}% pts, Pythagorean {l_ctx.pythagorean_pct*100:.1f}%)."
        )
    if w_tr and w_tr.l5_games > 0:
        s_lines.append(
            f"  - {w.team_abbrev} L5 Rolling Form: {w_tr.l5_record} ({w_tr.l5_gf_per_game:.2f} GF/G, {w_tr.l5_cf_pct:.1f}% L5 CF%, tonight {w_tr.game_cf_pct:.1f}% CF%)."
        )
    if l_tr and l_tr.l5_games > 0:
        s_lines.append(
            f"  - {l.team_abbrev} L5 Rolling Form: {l_tr.l5_record} ({l_tr.l5_gf_per_game:.2f} GF/G, {l_tr.l5_cf_pct:.1f}% L5 CF%, tonight {l_tr.game_cf_pct:.1f}% CF%)."
        )

    flow_desc = flow.script_summary if flow else f"{w.team_abbrev} defeated {l.team_abbrev} {w.score}-{l.score}."
    p1_score = flow.score_after_p1 if flow else ""
    p2_score = flow.score_after_p2 if flow else ""

    return "\n".join([
        f"GAME DATE & VENUE: {game.game_date} at {game.venue}",
        f"FINAL RESULT: {w.team_name} ({w.team_abbrev}) {w.score}, {l.team_name} ({l.team_abbrev}) {l.score} — Final ({dec_str})",
        f"PARAGRAPH 1 FACTS (Lede & Script): {flow_desc} First Star: {star1}.",
        f"PARAGRAPH 2 FACTS (Chronological Scoring — Intermissions P1: {p1_score}, P2: {p2_score}):\n{scoring_str}",
        "PARAGRAPH 3 FACTS (5v5 Line Shifts & L5 Chemistry):\n" + "\n".join(shift_lines),
        "PARAGRAPH 4 FACTS (Goaltending GSAx, xG & Defense):\n" + "\n".join(g_lines),
        "PARAGRAPH 5 FACTS (Season Standings, Pythagorean & L5 Form):\n" + "\n".join(s_lines),
        "",
        "Write Line 1 as HEADLINE: <headline>, followed by the 5 distinct paragraphs. Keep player names exactly as shown above.",
    ])


def build_projected_slate_prompt(slate: Any) -> str:
    """Projects a compact, audited state block for the evening slate review."""
    stars_lines = []
    for s in getattr(slate, "slate_three_stars", []):
        team_full = NHL_TEAM_NAMES.get(s.team_abbrev, s.team_abbrev)
        stars_lines.append(f"  - #{s.star} Star: {s.name} ({team_full} / {s.team_abbrev}, {s.position}) — {s.stat_summary}")

    matchup_lines = []
    for g in getattr(slate, "games", []):
        dec = "Final" if g.period_type == "REG" else ("Final/OT" if g.period_type == "OT" else "Final/SO")
        w = g.winner
        l = g.loser
        matchup_lines.append(
            f"  - {g.away_team.team_abbrev} {g.away_team.score} @ {g.home_team.team_abbrev} {g.home_team.score} ({dec}): "
            f"{w.team_name} defeated {l.team_name} {w.score}-{l.score} ({g.headline})"
        )

    return "\n".join([
        f"SLATE DATE: {slate.date}",
        "PARAGRAPH 1 MACRO LEDGER (MUST CITE EXACT NUMBERS):",
        f"  - Completed Games: {slate.total_games}",
        f"  - Combined Goals Scored: {slate.total_goals} goals ({slate.avg_goals_per_game:.1f} goals per game)",
        f"  - Venue Split: {slate.home_wins} Home Wins vs. {slate.away_wins} Road Wins",
        f"  - Extra-Time Contests (OT/SO): {slate.ot_so_games}",
        "PARAGRAPH 2 THREE STARS OF THE NIGHT:",
        "\n".join(stars_lines) if stars_lines else "  - Balanced scoring across all clubs.",
        "PARAGRAPH 3 MATCHUP RESULTS:",
        "\n".join(matchup_lines),
        "",
        "Write Line 1 as HEADLINE: <Day Title>, followed by 3 cohesive paragraphs matching these exact numbers.",
    ])


# =============================================================================
# 4. Strategic Deterministic Reflection Critics & Context-Pruned Diff Prompts
# =============================================================================

def verify_game_narrative_facts(
    narrative: str,
    game: Any,
    headline: str = "",
) -> Tuple[bool, List[str]]:
    """
    Strategic Reflection Critic for single-game recaps.
    Returns (is_valid: bool, violations: List[str]).
    """
    violations: List[str] = []
    if not narrative or len(narrative.strip()) < 950:
        violations.append("Narrative is too short (<950 chars).")
        return False, violations

    paras = [p.strip() for p in narrative.strip().splitlines() if len(p.strip()) >= 80]
    if len(paras) < 4:
        violations.append(f"Narrative has fewer than 4 substantive paragraphs ({len(paras)} found).")
        return False, violations

    if len(paras) >= 2 and _are_duplicate_lede_paragraphs(paras[0], paras[1]):
        violations.append("Paragraph 1 and Paragraph 2 repeat the same opening final-score lede.")
        return False, violations

    combined = f"{headline}\n{narrative}" if headline else narrative
    lower = combined.lower()

    leak_phrases = (
        "i need to write",
        "alright, i need",
        "okay, i need",
        "let me start by",
        "first paragraph should",
        "second paragraph deals",
        "third paragraph is on",
        "fourth paragraph is",
        "fifth paragraph is",
        "the script narrative says",
        "now checking each paragraph",
        "wait, but",
        "wait, let's",
        "structuring each paragraph",
        "deciding hero is",
        "so i should mention",
        "<think>",
        "</think>",
        "<thought>",
        "thinking process",
        "in the reg period",
        "of reg.",
        "verified scoring & roster ledger",
        "chronological goals scored:",
        "away top skaters:",
        "home top skaters:",
        "archetype",
        "accounted for 100% of the team",
        "100% of their shots on goal",
        "period.0%",
    )
    for phrase in leak_phrases:
        if phrase in lower:
            violations.append(f"Contains leaked prompt/reasoning token or corrupted phrase: '{phrase}'.")
            return False, violations

    if "halftime" in lower:
        violations.append("Contains non-hockey term 'halftime'.")
        return False, violations

    if re.search(r"\b(?:1st|2nd|3rd|4th|first|second|third|fourth)\s+quarter\b", lower):
        if not re.search(r"\b(?:1st|2nd|3rd|4th|first|second|third|fourth)\s+quarter\s+of\s+the\s+(?:season|schedule|year|campaign)\b", lower):
            violations.append("Contains non-hockey period term 'quarter'.")
            return False, violations

    if game.period_type in ("REG", "OT") and ("penalty shootout" in lower or "in a shootout" in lower or "shootout win" in lower or "shootout victory" in lower):
        violations.append(f"Game ended in {game.period_type} but text claims a shootout.")
        return False, violations

    if game.loser.score > 0 and re.search(r"\b(?:shutout|clean sheet)\b", lower):
        if not re.search(r"\b(?:breaking|broke|spoiled|snapped|prevented|avoided|no)\s+(?:a\s+|the\s+)?shutout\b", lower):
            violations.append(f"Game final was {game.winner.score}-{game.loser.score} ({game.loser.score} goals allowed), so claiming a shutout is false.")
            return False, violations

    winner_name = game.winner.team_name.lower()
    winner_abbrev = game.winner.team_abbrev.lower()
    name_parts = [p.lower() for p in game.winner.team_name.split() if len(p) >= 3]
    if winner_name not in lower and winner_abbrev not in lower and not any(p in lower for p in name_parts):
        violations.append(f"Winning team ({game.winner.team_abbrev}) not mentioned.")
        return False, violations

    all_skaters = list(game.away_skaters or []) + list(game.home_skaters or [])
    all_goalies = list(game.away_goalies or []) + list(game.home_goalies or [])
    scorer_names = {sp.goal_scorer.split()[-1].lower() for sp in (game.scoring_plays or []) if sp.goal_scorer}
    skater_last_names = {s.name.split()[-1].lower() for s in all_skaters if s.name}
    goalie_last_names = {g.name.split()[-1].lower() for g in all_goalies if g.name}

    # Prevent fake first-name expansion for initialed roster players (e.g. 'V. Husso' -> 'Vasilisa Husso')
    for p in all_skaters + all_goalies:
        m_init = re.match(r"^([A-Z])\.\s+([A-Za-zÀ-ÿ'-]+)$", str(p.name).strip())
        if m_init:
            last_n = m_init.group(2)
            fake_first_pat = rf"\b([A-Z][a-zÀ-ÿ]{{2,}})\s+{re.escape(last_n)}\b"
            for fm in re.finditer(fake_first_pat, combined):
                candidate_first = fm.group(1)
                if candidate_first.lower() not in (
                    "the", "and", "for", "with", "over", "past", "from", "into", "after", "before",
                    "while", "when", "where", "though", "although", "because", "since", "until",
                    "goalie", "goaltender", "netminder", "forward", "center", "winger", "defenseman",
                    "captain", "rookie", "veteran", "starter", "reliever", "opposite", "meanwhile",
                    "conversely", "notably", "specifically", "ultimately",
                ) and candidate_first.lower() not in {t.lower() for t in game.away_team.team_name.split() + game.home_team.team_name.split()}:
                    violations.append(
                        f"Invented full first name '{candidate_first} {last_n}' for initialed player '{p.name}'."
                    )
                    return False, violations

    non_scorer_skaters = {
        s.name.split()[-1].lower()
        for s in all_skaters
        if s.name and s.goals == 0 and s.name.split()[-1].lower() not in scorer_names
    }
    for ns_last in non_scorer_skaters:
        if ns_last and len(ns_last) >= 3:
            non_scorer_patterns = (
                rf"\b{re.escape(ns_last)}['’]s\s+(?:[a-z0-9-]+\s+){{0,2}}(?:goal|winner|strike|tally|marker)\b",
                rf"\b{re.escape(ns_last)}\s+(?:scored|netted|buried|potted)\b",
                rf"\b{re.escape(ns_last)}\s+(?:had|recorded|tallied)\s+(?:a|\d+|two|three)\s+goals?\b",
            )
            for pat in non_scorer_patterns:
                if re.search(pat, lower):
                    violations.append(f"Player '{ns_last}' had 0 goals but was credited with scoring a goal.")
                    return False, violations

    for g_last in goalie_last_names:
        if g_last and g_last not in scorer_names and g_last not in skater_last_names:
            goal_patterns = (
                rf"\b{re.escape(g_last)}['’]s\s+(?:[a-z0-9-]+\s+){{0,3}}(?:goal|winner|strike|tally|marker)\b",
                rf"\b{re.escape(g_last)}\s+(?:scored|netted|tallied|buried|potted)\b",
            )
            for pat in goal_patterns:
                if re.search(pat, lower):
                    violations.append(f"Goaltender '{g_last}' falsely credited with scoring a goal.")
                    return False, violations

    for s_last in skater_last_names:
        if s_last and s_last not in goalie_last_names:
            save_pat = rf"\b{re.escape(s_last)}\s+(?:made|makes|stopped|turned aside)\s+\d*\s*(?:[a-z-]+\s+){{0,2}}saves?\b"
            if re.search(save_pat, lower):
                violations.append(f"Skater '{s_last}' falsely credited with making goaltender saves.")
                return False, violations

    roster_tokens: Set[str] = set()
    for p in all_skaters + all_goalies:
        for part in re.findall(r"[A-Za-zÀ-ÿ-]+", p.name):
            if len(part) >= 3:
                roster_tokens.add(part.lower())
                roster_tokens.add(_strip_accents(part))
    for sp in (game.scoring_plays or []):
        for part in re.findall(r"[A-Za-zÀ-ÿ-]+", sp.goal_scorer):
            if len(part) >= 3:
                roster_tokens.add(part.lower())
                roster_tokens.add(_strip_accents(part))
        for a_name in (sp.assists or []):
            for part in re.findall(r"[A-Za-zÀ-ÿ-]+", a_name):
                if len(part) >= 3:
                    roster_tokens.add(part.lower())
                    roster_tokens.add(_strip_accents(part))

    team_tokens = {game.away_team.team_abbrev.lower(), game.home_team.team_abbrev.lower()}
    for t_name in (game.away_team.team_name, game.home_team.team_name, game.venue):
        for part in re.findall(r"[A-Za-zÀ-ÿ-]+", t_name):
            team_tokens.add(part.lower())
            team_tokens.add(_strip_accents(part))

    common_words = {
        "the", "and", "for", "with", "over", "past", "from", "into", "after", "before",
        "first", "second", "third", "period", "overtime", "shootout", "regulation", "final",
        "late", "early", "rally", "comeback", "thriller", "battle", "clash", "showdown",
        "victory", "win", "wins", "defeat", "defeats", "edge", "edges", "outlast", "outlasts",
        "stun", "stuns", "top", "tops", "beat", "beats", "sink", "sinks", "rout", "routs",
        "surge", "surges", "power", "powers", "lift", "lifts", "lead", "leads", "guide", "guides",
        "propel", "propels", "carry", "carries", "fuel", "fuels", "spark", "sparks", "clinch", "clinches",
        "net", "nets", "score", "scores", "tally", "tallies", "bury", "buries", "deliver", "delivers",
        "secure", "secures", "claim", "claims", "capture", "captures", "seize", "seizes", "hold", "holds",
        "stave", "staves", "off", "push", "effort", "total", "team", "home", "road", "away", "ice",
        "season", "opener", "night", "game", "goal", "goals", "point", "points", "save", "saves",
        "shutout", "hat", "trick", "star", "stars", "line", "pair", "power-play", "shorthanded",
        "sudden-death", "wire-to-wire", "back-to-back", "see-saw", "deadlock", "break", "breaks",
        "offensive", "defensive", "explosion", "clinic", "masterclass", "wall", "brick", "siege",
        "dispatch", "dispatches", "overwhelm", "overwhelms", "prevail", "prevails", "triumph", "triumphs",
        "mount", "mounts", "erase", "erases", "deficit", "streak", "form", "steady", "dominant",
        "nhl", "hockey", "eastern", "western", "division", "conference", "arena", "center", "centre",
        "club", "squad", "bench", "coach", "trio", "unit", "group", "frame", "contest", "matchup",
        "evening", "visitor", "host", "winner", "loser", "opponent", "penalty", "forward", "defense",
        "goalie", "goaltender", "netminder", "captain", "veteran", "rookie", "corsi", "fenwick", "pythagorean",
        "tonight", "today", "yesterday", "tomorrow", "stretch", "trip", "stand", "homestand", "recap",
        "action", "showcase", "meeting", "franchise", "organization", "roster", "lineup", "schedule",
        "campaign", "race", "standings", "rival", "rivalry", "canada", "america", "blueliner", "skater",
        "defenseman", "pairing", "corps", "depth", "tandem", "milestone", "record", "history", "momentum",
        "intermission", "outburst", "barrage", "onslaught", "flurry", "blitz", "heroics", "masterpiece",
        "display", "performance", "gem", "finish", "playmaking", "scoring", "offense", "goaltending",
        "special", "teams", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
        "october", "november", "december", "january", "february", "march", "april", "may", "june",
    }
    allowed_tokens = roster_tokens | team_tokens | common_words

    for m in re.finditer(r"\b([A-Z][a-zÀ-ÿ-]+)['’]s\b", headline):
        w_low = m.group(1).lower()
        w_norm = _strip_accents(m.group(1))
        if w_low not in allowed_tokens and w_norm not in allowed_tokens:
            violations.append(f"Non-roster possessive name '{m.group(1)}' in headline.")
            return False, violations

    cred_pat = (
        r"\b([A-Z][a-zÀ-ÿ-]+)['’]s\s+(?:[a-z0-9-]+\s+){0,2}"
        r"(?:goal|winner|tally|marker|strike|assist|point|save|shutout|hat[- ]trick)\b"
    )
    for m in re.finditer(cred_pat, narrative, re.IGNORECASE):
        w_low = m.group(1).lower()
        w_norm = _strip_accents(m.group(1))
        if w_low not in allowed_tokens and w_norm not in allowed_tokens:
            violations.append(f"Non-roster player '{m.group(1)}' credited in narrative.")
            return False, violations

    return True, []


def verify_slate_narrative_facts(
    narrative: str,
    slate: Any,
    headline: str = "",
) -> Tuple[bool, List[str]]:
    """
    Strategic Reflection Critic for League Slate Reviews (Day Title + Executive Summary).
    Ensures 100% alignment between the Day Title / Slate Narrative and measured slate details.
    """
    violations: List[str] = []
    if not narrative or len(narrative.strip()) < 220:
        violations.append("Slate narrative is too short (<220 chars).")
        return False, violations

    combined = f"{headline}\n{narrative}" if headline else narrative
    lower = combined.lower()

    if any(
        p in lower
        for p in (
            "i need to write",
            "alright, i need",
            "okay, i need",
            "let me start by",
            "thinking process",
            "<think>",
            "</think>",
            "halftime",
        )
    ):
        violations.append("Slate narrative contains internal reasoning leakage or non-hockey terminology.")
        return False, violations

    corrupt_phrases = (
        "per contests",
        "3-games",
        "accounted for 100%",
        "archetype",
        "wire-clock",
        "bricks wall",
    )
    for cp in corrupt_phrases:
        if cp in lower:
            violations.append(f"Slate narrative contains typographical defect: '{cp}'.")
            return False, violations

    total_goals = int(getattr(slate, "total_goals", 0))
    total_games = int(getattr(slate, "total_games", 0))
    if total_goals > 0:
        false_zero_goal_phrases = (
            "no goals",
            "non-scoring",
            "devoid of goals",
            "scoreless night",
            "without a goal",
            "zero goals",
        )
        for fp in false_zero_goal_phrases:
            if fp in lower:
                violations.append(f"Claimed '{fp}' on a slate with {total_goals} measured goals.")
                return False, violations

    # Require that the slate narrative explicitly grounds the measured total_games and total_goals
    if str(total_goals) not in narrative or str(total_games) not in narrative:
        violations.append(
            f"Slate narrative omits the measured macro scope ({total_games} games, {total_goals} total goals)."
        )
        return False, violations

    if headline:
        h_low = headline.strip().lower()
        generic_patterns = (
            "nhl action highlights goals, standouts",
            "nhl highlights: stars shine",
            "leads dramatic nights",
            "non-scoring night",
            "wire-clock",
        )
        if any(gp in h_low for gp in generic_patterns):
            violations.append(f"Headline '{headline}' is generic/corrupted and does not match measured slate details.")
            return False, violations

    for m_games in re.finditer(r"\b(\d+)(?:-|\s+)game\s+slate\b", lower):
        claimed_g = int(m_games.group(1))
        if claimed_g != total_games:
            violations.append(f"Claimed {claimed_g}-game slate but measured total_games is {total_games}.")
            return False, violations

    for m_goals in re.finditer(r"\bcombined\s+(?:for\s+)?(?:\*\*)?(\d+)\s+(?:total\s+)?goals\b", lower):
        claimed_goals = int(m_goals.group(1))
        if abs(claimed_goals - total_goals) > 1:
            violations.append(f"Claimed {claimed_goals} combined goals but measured total_goals is {total_goals}.")
            return False, violations

    return True, []


# =============================================================================
# 5. Broadcast-Grade Deterministic Zamboni Headlines & Recaps
# =============================================================================

def build_zamboni_game_headline(game: Any) -> str:
    """Creates an authoritative, broadcast-grade game headline verified against period flow and stars."""
    w = game.winner
    l = game.loser
    w_score = w.score
    l_score = l.score
    hero = game.three_stars[0].name if getattr(game, "three_stars", None) else w.team_name
    hero_is_winner = (
        bool(getattr(game, "three_stars", None))
        and game.three_stars[0].team_abbrev == w.team_abbrev
    )
    flow = getattr(game, "period_flow", None)
    script = flow.game_script_type if flow else "REGULATION"

    w_g = getattr(game, "winning_goalie_profile", None)
    if l_score == 0 and w_g:
        return f"{w_g.name} Posts {w_g.saves}-Save Shutout as {w.team_name} Blank {l.team_name} {w_score}-0"

    if script == "WALK_OFF_OT_SO":
        if game.period_type == "SO":
            return f"{w.team_name} Outlast {l.team_name} {w_score}-{l_score} in Shootout Thriller"
        scorer = game.scoring_plays[-1].goal_scorer if getattr(game, "scoring_plays", None) else hero
        return f"{scorer} Buries Overtime Winner as {w.team_name} Edge {l.team_name} {w_score}-{l_score}"
    elif script == "COMEBACK_WIN":
        if hero_is_winner:
            return f"{hero} Sparks 3rd-Period Comeback as {w.team_name} Rally Past {l.team_name} {w_score}-{l_score}"
        return f"{w.team_name} Erase 3rd-Period Deficit to Stun {l.team_name} {w_score}-{l_score}"
    elif script == "BLOWOUT":
        if hero_is_winner:
            return f"{hero} Fuels Offensive Surge as {w.team_name} Rout {l.team_name} {w_score}-{l_score}"
        return f"{w.team_name} Overwhelm {l.team_name} {w_score}-{l_score} in Offensive Showcase"
    elif script == "CONTROLLED_LATE_PUSH":
        return f"{w.team_name} Withstand Late 3rd-Period Surge to Edge {l.team_name} {w_score}-{l_score}"
    elif script == "TRUE_THRILLER":
        if hero_is_winner:
            return f"{hero} Breaks Late Deadlock as {w.team_name} Outduel {l.team_name} {w_score}-{l_score}"
        return f"{w.team_name} Prevail {w_score}-{l_score} Over {l.team_name} in Back-and-Forth Battle"
    elif script == "WIRE_TO_WIRE":
        if hero_is_winner:
            return f"{hero} Sets the Tone as {w.team_name} Command {l.team_name} {w_score}-{l_score} Wire-to-Wire"
        return f"{w.team_name} Dictate Tempo Wire-to-Wire in {w_score}-{l_score} Win Over {l.team_name}"
    else:
        if hero_is_winner:
            return f"{hero} Leads {w.team_name} Past {l.team_name} {w_score}-{l_score}"
        return f"{w.team_name} Secure {w_score}-{l_score} Victory Over {l.team_name}"


def build_zamboni_slate_headline(
    date_str: str,
    total_games: int,
    total_goals: int,
    home_wins: int,
    away_wins: int,
    ot_so_games: int,
    three_stars: List[Dict[str, Any]],
    game_headlines: Optional[List[str]] = None,
) -> str:
    """
    Constructs a Day Title (`slate_headline`) that is 100% reconciled with the date's
    measured details (games count, total goals, overtime drama, and #1 Star of the Night).
    """
    if three_stars:
        star1 = three_stars[0]
        s_name = star1.get("name", "Top Performers")
        s_team = star1.get("team", "")
        s_stat = re.sub(r"\s*\(([^)]+)\)", r", \1", str(star1.get("stat_summary", ""))).strip()
        if ot_so_games >= 2:
            return (
                f"{s_name} ({s_team} • {s_stat}) Headlines {total_games}-Game, "
                f"{total_goals}-Goal NHL Slate With {ot_so_games} Overtime Thrillers"
            )
        elif ot_so_games == 1:
            return (
                f"{s_name} ({s_team} • {s_stat}) Leads {total_games}-Game, "
                f"{total_goals}-Goal NHL Slate Featuring Extra-Time Drama"
            )
        else:
            return (
                f"{s_name} ({s_team} • {s_stat}) Paces {total_games}-Game, "
                f"{total_goals}-Goal NHL Slate ({home_wins}-{away_wins} Home/Road Split)"
            )
    return (
        f"NHL Night in Review ({date_str}): {total_goals} Goals Across {total_games}-Game Slate "
        f"({home_wins} Home Wins, {away_wins} Road Wins)"
    )


def build_zamboni_slate_narrative_from_records(
    date_str: str,
    total_games: int,
    total_goals: int,
    avg_goals_per_game: float,
    home_wins: int,
    away_wins: int,
    ot_so_games: int,
    three_stars: List[Dict[str, Any]],
    game_records: List[Dict[str, Any]],
    milestones: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Deterministic, 100% reconciled 3-paragraph League Slate Media Review built directly
    from the date's measured KPIs, Three Stars podium, and individual game outcomes.
    """
    if home_wins > away_wins:
        venue_clause = f"Home clubs defended their ice with a **{home_wins}–{away_wins}** advantage over visiting squads"
    elif away_wins > home_wins:
        venue_clause = f"Visiting road clubs seized territorial control, claiming a **{away_wins}–{home_wins}** edge over home teams"
    else:
        venue_clause = f"Home and road clubs split the evening evenly at **{home_wins}–{away_wins}**"

    if ot_so_games == 0:
        ot_clause = "with all matchups settled inside sixty minutes of regulation."
    elif ot_so_games == 1:
        ot_clause = "with **1 matchup** requiring extra time beyond regulation to determine a winner."
    else:
        ot_clause = f"with **{ot_so_games} matchups** extending into overtime or a shootout."

    game_art = "an" if str(total_games).startswith("8") or str(total_games).startswith("11") or total_games in (8, 11, 18) else "a"
    p1 = (
        f"Across {game_art} **{total_games}-game** National Hockey League slate on **{date_str}**, offenses combined for "
        f"**{total_goals} total goals**—an average of **{avg_goals_per_game:.1f} goals per contest**. "
        f"{venue_clause}, {ot_clause}"
    )

    p2_parts: List[str] = []
    if three_stars:
        s1 = three_stars[0]
        s1_team_full = NHL_TEAM_NAMES.get(str(s1.get("team", "")), str(s1.get("team", "")))
        p2_parts.append(
            f"Individual honours across the league were led by #1 Star **{s1.get('name')}** of the **{s1_team_full}** "
            f"({s1.get('team')}), who set the standard with **{s1.get('stat_summary')}**."
        )
        if len(three_stars) > 1:
            s2 = three_stars[1]
            s2_team_full = NHL_TEAM_NAMES.get(str(s2.get("team", "")), str(s2.get("team", "")))
            p2_parts.append(
                f"Second-star recognition went to **{s2.get('name')}** of the **{s2_team_full}** ({s2.get('team')}) "
                f"following a standout outing ({s2.get('stat_summary')})"
            )
        if len(three_stars) > 2:
            s3 = three_stars[2]
            s3_team_full = NHL_TEAM_NAMES.get(str(s3.get("team", "")), str(s3.get("team", "")))
            if len(p2_parts) >= 2:
                p2_parts[-1] += (
                    f", while **{s3.get('name')}** of the **{s3_team_full}** ({s3.get('team')}) rounded out the "
                    f"Three Stars podium at #3 ({s3.get('stat_summary')})."
                )
        elif len(p2_parts) >= 2:
            p2_parts[-1] += "."

    p3_parts: List[str] = []
    if game_records:
        capsule_items: List[str] = []
        for g in game_records[:6]:
            a_tri = str(g.get("away_team", ""))
            h_tri = str(g.get("home_team", ""))
            a_sc = int(g.get("away_score", 0))
            h_sc = int(g.get("home_score", 0))
            pe = str(g.get("period_ending", "REG"))
            dec_lbl = "in overtime" if pe == "OT" else ("in a shootout" if pe == "SO" else "in regulation")
            w_tri = a_tri if a_sc > h_sc else h_tri
            l_tri = h_tri if a_sc > h_sc else a_tri
            w_full = NHL_TEAM_NAMES.get(w_tri, w_tri)
            l_full = NHL_TEAM_NAMES.get(l_tri, l_tri)
            w_sc = max(a_sc, h_sc)
            l_sc = min(a_sc, h_sc)
            script = str(g.get("game_script_type", "REGULATION"))
            if script == "COMEBACK_WIN":
                capsule_items.append(f"the **{w_full}** rallied in the third period to defeat the {l_full} {w_sc}–{l_sc}")
            elif script == "WIRE_TO_WIRE":
                capsule_items.append(f"the **{w_full}** controlled tempo wire-to-wire past the {l_full} {w_sc}–{l_sc}")
            elif script == "BLOWOUT":
                capsule_items.append(f"the **{w_full}** overwhelmed the {l_full} {w_sc}–{l_sc}")
            else:
                capsule_items.append(f"the **{w_full}** topped the {l_full} {w_sc}–{l_sc} {dec_lbl}")
        if capsule_items:
            if len(capsule_items) == 1:
                p3_parts.append(f"In the night's featured matchup, {capsule_items[0]}.")
            else:
                joined = "; ".join(capsule_items[:-1]) + f"; and {capsule_items[-1]}"
                p3_parts.append(f"Across the scoreboard, {joined}.")

    if milestones:
        top_m = milestones[0]
        m_desc = str(top_m.get("description", "")).strip()
        if m_desc and "shutout" not in m_desc.lower():
            p3_parts.append(f"Tactical takeaway: {m_desc}")

    paragraphs = [p1]
    if p2_parts:
        paragraphs.append(" ".join(p2_parts))
    if p3_parts:
        paragraphs.append(" ".join(p3_parts))

    return expand_first_mention_abbreviations("\n\n".join(paragraphs))


def build_zamboni_game_narrative_from_record(game_rec: Dict[str, Any]) -> str:
    """
    Builds a crisp, broadcast-grade 5-paragraph NHL game recap directly from a game record's
    audited JSON telemetry (`period_flow_json`, `three_stars_json`, `goalie_performance_json`,
    `defensive_effort_json`, `season_standings_json`, `xg_summary_json`, `milestones_json`,
    and `ShiftAndTrendDataProvider`).
    """
    gid = int(game_rec.get("game_id", 0))
    a_tri = str(game_rec.get("away_team", ""))
    h_tri = str(game_rec.get("home_team", ""))
    a_score = int(game_rec.get("away_score", 0))
    h_score = int(game_rec.get("home_score", 0))
    pe = str(game_rec.get("period_ending", "REG"))
    w_is_home = h_score > a_score
    w_tri = h_tri if w_is_home else a_tri
    l_tri = a_tri if w_is_home else h_tri
    w_full = NHL_TEAM_NAMES.get(w_tri, w_tri)
    l_full = NHL_TEAM_NAMES.get(l_tri, l_tri)
    w_score = max(a_score, h_score)
    l_score = min(a_score, h_score)

    def _safe_json(val: Any, default: Any) -> Any:
        if not val:
            return default
        if isinstance(val, (dict, list)):
            return val
        try:
            return json.loads(val)
        except Exception:
            return default

    flow = _safe_json(game_rec.get("period_flow_json"), {})
    stars = _safe_json(game_rec.get("three_stars_json"), [])
    goalies = _safe_json(game_rec.get("goalie_performance_json"), {})
    defense = _safe_json(game_rec.get("defensive_effort_json"), {})
    standings = _safe_json(game_rec.get("season_standings_json"), {})
    a_stats = _safe_json(game_rec.get("away_stats_json"), {})
    h_stats = _safe_json(game_rec.get("home_stats_json"), {})
    xg_sum = _safe_json(game_rec.get("xg_summary_json"), {})
    milestones = _safe_json(game_rec.get("milestones_json"), [])

    w_stats = h_stats if w_is_home else a_stats
    l_stats = a_stats if w_is_home else h_stats

    # Shift profiles if available
    w_tr, l_tr = None, None
    try:
        from dev.agent.scoreboard.skills import ShiftAndTrendDataProvider
        away_sp = ShiftAndTrendDataProvider.get_team_shift_trend_profile(gid, a_tri, h_tri, is_home=False)
        home_sp = ShiftAndTrendDataProvider.get_team_shift_trend_profile(gid, h_tri, a_tri, is_home=True)
        w_tr = home_sp if w_is_home else away_sp
        l_tr = away_sp if w_is_home else home_sp
    except Exception:
        pass

    paragraphs: List[str] = []

    # Paragraph 1: Lede, Game Script & Three Stars
    dec_clause = " in overtime" if pe == "OT" else (" in a shootout" if pe == "SO" else "")
    venue_phrase = "on home ice" if w_is_home else "on the road"
    if w_tr:
        if not w_is_home and getattr(w_tr, "consec_road_games", 0) >= 2:
            venue_phrase = f"on the road in game #{w_tr.consec_road_games} of their current trip"
        if getattr(w_tr, "is_back_to_back", False):
            venue_phrase += " on the second night of a back-to-back"

    script_type = str(flow.get("game_script_type") or game_rec.get("game_script_type") or "REGULATION")
    if script_type == "WIRE_TO_WIRE":
        script_sentence = f"The {w_full} dictated terms from the opening period and never trailed, executing a wire-to-wire victory."
    elif script_type == "COMEBACK_WIN":
        script_sentence = f"Trailing entering the third period, the {w_full} staged a composed final-frame rally to flip the scoreboard."
    elif script_type == "CONTROLLED_LATE_PUSH":
        script_sentence = f"The {w_full} built a multi-goal cushion through two periods and withstood a late third-period push from the {l_full}."
    elif script_type == "WALK_OFF_OT_SO":
        script_sentence = f"After sixty minutes of tight-checking regulation play left the clubs deadlocked, the {w_full} captured the extra point in sudden-death fashion."
    elif script_type == "BLOWOUT":
        script_sentence = f"The {w_full} controlled territorial pace throughout, pulling away for a decisive {w_score - l_score}-goal victory."
    else:
        script_sentence = f"The {w_full} combined structured five-on-five execution with timely finishing to secure the two points."

    star_sentence = ""
    if stars:
        s1 = stars[0]
        star_sentence = f" **{s1.get('name')}** ({s1.get('team')}) paced all performers as the #1 Star of the Game ({s1.get('stat_summary')})"
        if len(stars) > 1:
            s2 = stars[1]
            star_sentence += f", while **{s2.get('name')}** ({s2.get('team')}, {s2.get('stat_summary')}) earned second-star honours."
        else:
            star_sentence += "."

    p1 = (
        f"The **{w_full}** ({w_tri}) defeated the **{l_full}** ({l_tri}) **{w_score}–{l_score}**{dec_clause} "
        f"{venue_phrase}. {script_sentence}{star_sentence}"
    )
    paragraphs.append(p1.strip())

    # Paragraph 2: Period-by-Period Scoring Progression & Expected Goals (xG) Danger Zones
    p2_parts: List[str] = []
    p1_sc = flow.get("score_after_p1")
    p2_sc = flow.get("score_after_p2")
    ties = int(flow.get("ties_count", 0))
    leads = int(flow.get("lead_changes", 0))
    if p1_sc and p2_sc:
        p2_parts.append(
            f"Chronological period progression saw the clubs reach the first intermission at **{p1_sc}** before moving to "
            f"**{p2_sc}** through forty minutes ({ties} ties, {leads} lead changes)."
        )
    w_xg = float(w_stats.get("xg", 0.0) or 0.0)
    l_xg = float(l_stats.get("xg", 0.0) or 0.0)
    w_xg_pct = float(w_stats.get("xg_pct", 50.0) or 50.0)
    l_xg_pct = float(l_stats.get("xg_pct", 50.0) or 50.0)
    dz = xg_sum.get("danger_zones", {}) if isinstance(xg_sum, dict) else {}
    w_hd = int(dz.get(w_tri, {}).get("HIGH", w_stats.get("hdcf", 0)) or 0)
    l_hd = int(dz.get(l_tri, {}).get("HIGH", l_stats.get("hdcf", 0)) or 0)
    w_md = int(dz.get(w_tri, {}).get("MEDIUM", 0) or 0)
    l_md = int(dz.get(l_tri, {}).get("MEDIUM", 0) or 0)
    if w_xg > 0 or l_xg > 0:
        p2_parts.append(
            f"In underlying shot-quality telemetry, {w_tri} generated **{w_xg:.2f} Expected Goals (xG)** ({w_xg_pct:.1f}% share) "
            f"with **{w_hd} high-danger slot chances** and {w_md} medium-danger looks, while {l_tri} produced "
            f"**{l_xg:.2f} xG** ({l_xg_pct:.1f}% share) across **{l_hd} high-danger** and {l_md} medium-danger attempts."
        )
    if milestones:
        notable_descs = [
            str(m.get("description", "")).strip()
            for m in milestones[:2]
            if m.get("description") and "shutout" not in str(m.get("description", "")).lower()
        ]
        if notable_descs:
            p2_parts.append(" ".join(notable_descs))
    if p2_parts:
        paragraphs.append(" ".join(p2_parts).strip())

    # Paragraph 3: 5v5 Line Shift Chemistry & L5 Continuity
    p3_parts: List[str] = []
    if w_tr and getattr(w_tr, "has_shift_data", False):
        if w_tr.new_lines_deployed_count > 0:
            p3_parts.append(
                f"At five-on-five, {w_tri} deployed {w_tr.new_lines_deployed_count} newly assembled or reshuffled forward combinations "
                f"({w_tr.line_continuity_pct:.1f}% line continuity over the last five games (L5))."
            )
        else:
            p3_parts.append(
                f"Five-on-five deployment saw {w_tri} lean on established unit continuity ({w_tr.line_continuity_pct:.1f}% of forward Time on Ice (TOI) played by regular L5 trios)."
            )
        if w_tr.best_line:
            p3_parts.append(w_tr.best_line.line_story_summary)
        if w_tr.top_pair:
            p3_parts.append(f"On the blue line, {w_tr.top_pair.line_story_summary}")
    if l_tr and getattr(l_tr, "has_shift_data", False) and l_tr.best_line:
        p3_parts.append(f"For {l_tri}, {l_tr.best_line.line_story_summary}")
    if not p3_parts:
        w_sog = int(w_stats.get("sog", 0))
        l_sog = int(l_stats.get("sog", 0))
        w_fo = float(w_stats.get("faceoff_pct", 50.0))
        l_fo = float(l_stats.get("faceoff_pct", 50.0))
        p3_parts.append(
            f"Across even-strength and special-teams rotations, {w_tri} put **{w_sog} Shots on Goal (SOG)** on net and captured "
            f"**{w_fo:.1f}%** of faceoffs, while {l_tri} countered with **{l_sog} SOG** and a **{l_fo:.1f}%** faceoff share."
        )
    paragraphs.append(" ".join(p3_parts).strip())

    # Paragraph 4: Goaltending GSAx, Defensive Sacrifice & Special Teams
    p4_parts: List[str] = []
    w_g = goalies.get("winner") if isinstance(goalies, dict) else None
    l_g = goalies.get("loser") if isinstance(goalies, dict) else None
    if w_g:
        wg_saves = int(w_g.get("saves", 0))
        wg_sa = int(w_g.get("shots_against", 0))
        wg_sv = float(w_g.get("save_pct", 0.0)) * 100.0
        wg_gsax = float(w_g.get("gsax", 0.0))
        wg_ga = int(w_g.get("goals_against", 0))
        if l_score == 0 and wg_ga == 0:
            p4_parts.append(
                f"In the crease, **{w_g.get('name')}** ({w_tri}) delivered a {wg_saves}-save shutout ({wg_sv:.1f}% Save Percentage (SV%), **{wg_gsax:+.2f} Goals Saved Above Expected (GSAx)**)."
            )
        elif wg_ga == 0 and l_score > 0:
            p4_parts.append(
                f"In the crease, **{w_g.get('name')}** ({w_tri}) anchored the win with a faultless relief outing, stopping all {wg_saves} shots faced ({wg_sv:.1f}% Save Percentage (SV%), **{wg_gsax:+.2f} Goals Saved Above Expected (GSAx)**)."
            )
        else:
            p4_parts.append(
                f"In the crease, **{w_g.get('name')}** ({w_tri}) earned the victory with **{wg_saves} saves on {wg_sa} shots** ({wg_sv:.1f}% Save Percentage (SV%), **{wg_gsax:+.2f} Goals Saved Above Expected (GSAx)**)."
            )
    if l_g:
        lg_saves = int(l_g.get("saves", 0))
        lg_sa = int(l_g.get("shots_against", 0))
        lg_sv = float(l_g.get("save_pct", 0.0)) * 100.0
        lg_gsax = float(l_g.get("gsax", 0.0))
        p4_parts.append(
            f"Opposite him, **{l_g.get('name')}** ({l_tri}) finished with **{lg_saves} saves on {lg_sa} shots** ({lg_sv:.1f}% SV%, **{lg_gsax:+.2f} GSAx**)."
        )

    w_def = defense.get("home" if w_is_home else "away") if isinstance(defense, dict) else None
    l_def = defense.get("away" if w_is_home else "home") if isinstance(defense, dict) else None
    if w_def and l_def:
        p4_parts.append(
            f"Defensively, {w_tri} logged **{w_def.get('blocked_shots', 0)} blocked shots** and **{w_def.get('hits', 0)} hits** "
            f"({int(w_def.get('turnover_diff', 0)):+d} turnover differential), compared to {l_tri}'s **{l_def.get('blocked_shots', 0)} blocks** "
            f"and **{l_def.get('hits', 0)} hits** ({int(l_def.get('turnover_diff', 0)):+d} turnover differential)."
        )
    if p4_parts:
        paragraphs.append(" ".join(p4_parts).strip())

    # Paragraph 5: Season Standings, Pythagorean Expectation & L5 Rolling Form
    w_st = standings.get("home" if w_is_home else "away") if isinstance(standings, dict) else None
    l_st = standings.get("away" if w_is_home else "home") if isinstance(standings, dict) else None
    p5_parts: List[str] = []
    if w_st and l_st:
        stg = str(w_st.get("stage_narrative", "")).strip()
        if stg:
            p5_parts.append(stg)
        p5_parts.append(
            f"With the result, the **{w_full}** move to **{w_st.get('record')}** ({w_st.get('pts')} points, "
            f"{float(w_st.get('pts_pct', 0.0))*100:.1f}% points share, **{float(w_st.get('pyth_pct', 0.5))*100:.1f}% Pythagorean expectation**), "
            f"while the **{l_full}** stand at **{l_st.get('record')}** ({l_st.get('pts')} points, "
            f"{float(l_st.get('pts_pct', 0.0))*100:.1f}% points share, **{float(l_st.get('pyth_pct', 0.5))*100:.1f}% Pythagorean expectation**)."
        )
    if w_tr and getattr(w_tr, "l5_games", 0) > 0:
        p5_parts.append(
            f"Over the rolling L5 window, {w_tri} entered at {w_tr.l5_record} ({w_tr.l5_gf_per_game:.2f} GF/G, {w_tr.l5_cf_pct:.1f}% L5 CF%) "
            f"and posted a {w_tr.game_cf_pct:.1f}% CF% ({w_tr.game_vs_l5_cf_delta:+.1f}% vs. L5)."
        )
    if p5_parts:
        paragraphs.append(" ".join(p5_parts).strip())

    return expand_first_mention_abbreviations("\n\n".join(paragraphs))


# =============================================================================
# 6. Live Slate & Game Record Reconciler (Repairs Cached Mismatches On-The-Fly)
# =============================================================================

def _has_legacy_game_narrative_defects(narr: str, l_score: int) -> bool:
    """Detects legacy crude writing, duplicated lede paragraphs, or corrupted artifacts in cached game recaps."""
    if not narr or len(narr.strip()) < 850:
        return True
    low = narr.lower()
    corrupt_phrases = (
        "accounted for 100% of the team",
        "100% of their shots on goal",
        "period.0%",
        "season trajectory (",
        "line shift chemistry (",
        "goaltending composure,",
        "archetype",
        "ahead 2-3",
        "ahead 1-2",
        "ahead 3-4",
    )
    if any(cp in low for cp in corrupt_phrases):
        return True
    if l_score > 0 and "shutout" in low and not re.search(r"\b(?:no|broke|spoiled|avoided)\s+shutout\b", low):
        return True
    paras = [p.strip() for p in narr.strip().splitlines() if len(p.strip()) >= 60]
    if len(paras) < 4:
        return True
    if len(paras) >= 2 and _are_duplicate_lede_paragraphs(paras[0], paras[1]):
        return True
    return False


def reconcile_game_record(game_rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Audits and reconciles a cached game record dictionary from `game_recaps.parquet`.
    Fixes:
    1. Relief-goalie false shutouts when opponent score > 0 (e.g. V. Husso 10/10 saves in ANA 5 @ CGY 3).
    2. Fake invented first names for initialed players (e.g. 'Vasilisa Husso' -> 'V. Husso', 'Domenik Wolf' -> 'D. Wolf').
    3. Legacy crude writing, duplicated Paragraph 1 / Paragraph 2 ledes, or corrupted artifacts.
    4. Malformed headlines ('Wire-Clock', 'Bricks Wall').
    """
    out = dict(game_rec)
    a_team = str(out.get("away_team", ""))
    h_team = str(out.get("home_team", ""))
    a_score = int(out.get("away_score", 0))
    h_score = int(out.get("home_score", 0))
    w_score = max(a_score, h_score)
    l_score = min(a_score, h_score)
    w_team = a_team if a_score > h_score else h_team
    l_team = h_team if a_score > h_score else a_team
    w_full = NHL_TEAM_NAMES.get(w_team, w_team)
    l_full = NHL_TEAM_NAMES.get(l_team, l_team)

    # 1. Reconcile goalie_performance_json for false relief shutouts
    initialed_names: List[Tuple[str, str]] = []
    goalie_str = out.get("goalie_performance_json")
    if goalie_str:
        try:
            g_data = json.loads(goalie_str) if isinstance(goalie_str, str) else dict(goalie_str)
            modified_g = False
            for role in ("winner", "loser"):
                if role in g_data and isinstance(g_data[role], dict):
                    p = g_data[role]
                    g_name = str(p.get("name", ""))
                    m_init = re.match(r"^([A-Z])\.\s+([A-Za-zÀ-ÿ'-]+)$", g_name.strip())
                    if m_init:
                        initialed_names.append((g_name.strip(), m_init.group(2)))
                    opp_goals = l_score if role == "winner" else w_score
                    if opp_goals > 0 and "shutout" in str(p.get("assessment", "")).lower():
                        saves = int(p.get("saves", 0))
                        sa = int(p.get("shots_against", 0))
                        gsax = float(p.get("gsax", 0.0))
                        p["rating"] = "STEADY" if saves < 25 else "BRICK_WALL"
                        p["assessment"] = (
                            f"Faultless relief outing: stopped all {saves} shots faced ({gsax:+.2f} GSAx)."
                            if int(p.get("goals_against", 0)) == 0
                            else f"Composed outing ({saves}/{sa} saves, {gsax:+.2f} GSAx)."
                        )
                        modified_g = True
            if modified_g:
                out["goalie_performance_json"] = json.dumps(g_data, ensure_ascii=False)
        except Exception:
            pass

    for json_col in ("three_stars_json", "milestones_json"):
        raw_j = out.get(json_col)
        if raw_j:
            try:
                items = json.loads(raw_j) if isinstance(raw_j, str) else raw_j
                if isinstance(items, list):
                    for item in items:
                        for field_k in ("name", "player_or_team", "title"):
                            val = str(item.get(field_k, ""))
                            for m in re.finditer(r"\b([A-Z]\.\s+[A-Za-zÀ-ÿ'-]+)\b", val):
                                full_i = m.group(1)
                                last_i = full_i.split()[-1]
                                initialed_names.append((full_i, last_i))
            except Exception:
                pass

    if l_score > 0 and out.get("milestones_json"):
        try:
            ms = json.loads(out["milestones_json"]) if isinstance(out["milestones_json"], str) else out["milestones_json"]
            if isinstance(ms, list):
                filtered_ms = [
                    m for m in ms
                    if "shutout" not in str(m.get("category", "")).lower()
                    and "shutout" not in str(m.get("title", "")).lower()
                ]
                if len(filtered_ms) != len(ms):
                    out["milestones_json"] = json.dumps(filtered_ms, ensure_ascii=False)
        except Exception:
            pass

    # 2. Audit narrative_recap; if legacy crude/defective, rebuild from verified telemetry
    raw_narr = str(out.get("narrative_recap", ""))
    if _has_legacy_game_narrative_defects(raw_narr, l_score):
        out["narrative_recap"] = build_zamboni_game_narrative_from_record(out)
    else:
        narr = clean_llm_narrative(raw_narr)
        for full_init, last_n in initialed_names:
            pat = rf"\b([A-Z][a-zÀ-ÿ]{{2,}})\s+{re.escape(last_n)}\b"
            def _fix_name(m: re.Match, target_init: str = full_init) -> str:
                w1 = m.group(1)
                if w1.lower() in ("the", "and", "for", "with", "over", "while", "when", "goalie", "goaltender", "netminder", "forward", "defenseman", "starter"):
                    return m.group(0)
                return target_init
            narr = re.sub(pat, _fix_name, narr)
        out["narrative_recap"] = expand_first_mention_abbreviations(narr)

    # 3. Clean and repair headline
    h = str(out.get("headline", "")).strip()
    if (
        not h
        or "Wire-Clock" in h
        or "Bricks Wall" in h
        or (l_score > 0 and "shutout" in h.lower())
        or str(w_score) not in h
    ):
        script = str(out.get("game_script_type", "REGULATION"))
        star1_name = ""
        try:
            stars = json.loads(out.get("three_stars_json", "[]"))
            if stars and stars[0].get("team") == w_team:
                star1_name = stars[0].get("name", "")
        except Exception:
            pass
        if script == "COMEBACK_WIN":
            h = f"{star1_name + ' Sparks ' if star1_name else ''}{w_full} 3rd-Period Rally Past {l_full} {w_score}-{l_score}"
        elif script == "WIRE_TO_WIRE":
            h = f"{star1_name + ' Paces ' if star1_name else ''}{w_full} in {w_score}-{l_score} Wire-to-Wire Win Over {l_full}"
        elif out.get("period_ending") == "OT":
            h = f"{star1_name + ' Lifts ' if star1_name else ''}{w_full} Past {l_full} {w_score}-{l_score} in Overtime"
        elif out.get("period_ending") == "SO":
            h = f"{w_full} Outlast {l_full} {w_score}-{l_score} in Shootout"
        elif script == "BLOWOUT":
            h = f"{star1_name + ' Fuels ' if star1_name else ''}{w_full} {w_score}-{l_score} Rout of {l_full}"
        else:
            h = f"{star1_name + ' Leads ' if star1_name else ''}{w_full} Past {l_full} {w_score}-{l_score}"
        out["headline"] = h

    return out


def reconcile_slate_and_games_records(
    target_date: str,
    slate_record: Optional[Dict[str, Any]],
    game_records: List[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Guarantees 100% mathematical and editorial reconciliation between the Day Title
    (`slate_headline`), Slate Executive Summary (`slate_narrative`), KPI cards, and
    the individual `game_records` displayed on the Game Recaps (by Zamboni) page.
    """
    reconciled_games = [reconcile_game_record(g) for g in (game_records or [])]
    if not slate_record and not reconciled_games:
        return slate_record, reconciled_games

    s_rec = dict(slate_record) if slate_record else {"game_date": target_date}

    if reconciled_games:
        total_games = len(reconciled_games)
        total_goals = sum(int(g.get("away_score", 0)) + int(g.get("home_score", 0)) for g in reconciled_games)
        avg_goals = total_goals / float(total_games) if total_games > 0 else 0.0
        home_wins = sum(1 for g in reconciled_games if int(g.get("home_score", 0)) > int(g.get("away_score", 0)))
        away_wins = sum(1 for g in reconciled_games if int(g.get("away_score", 0)) > int(g.get("home_score", 0)))
        ot_games = sum(1 for g in reconciled_games if str(g.get("period_ending", "REG")) in ("OT", "SO"))

        s_rec["total_games"] = total_games
        s_rec["total_goals"] = total_goals
        s_rec["avg_goals_per_game"] = avg_goals
        s_rec["home_wins"] = home_wins
        s_rec["away_wins"] = away_wins
        s_rec["overtime_games"] = ot_games
    else:
        total_games = int(s_rec.get("total_games", 0))
        total_goals = int(s_rec.get("total_goals", 0))
        avg_goals = float(s_rec.get("avg_goals_per_game", 0.0))
        home_wins = int(s_rec.get("home_wins", 0))
        away_wins = int(s_rec.get("away_wins", 0))
        ot_games = int(s_rec.get("overtime_games", 0))

    three_stars: List[Dict[str, Any]] = []
    try:
        raw_stars = s_rec.get("three_stars_json")
        if raw_stars:
            three_stars = json.loads(raw_stars) if isinstance(raw_stars, str) else list(raw_stars)
    except Exception:
        pass

    milestones: List[Dict[str, Any]] = []
    try:
        raw_ms = s_rec.get("milestones_json")
        if raw_ms:
            milestones = json.loads(raw_ms) if isinstance(raw_ms, str) else list(raw_ms)
            shutout_teams = {
                str(g.get("away_team")) if int(g.get("home_score", 0)) == 0 else str(g.get("home_team"))
                for g in reconciled_games
                if int(g.get("away_score", 0)) == 0 or int(g.get("home_score", 0)) == 0
            }
            milestones = [
                m for m in milestones
                if "shutout" not in str(m.get("category", "")).lower()
                or any(t in str(m.get("description", "")) or t in str(m.get("player_or_team", "")) for t in shutout_teams)
            ]
            s_rec["milestones_json"] = json.dumps(milestones, ensure_ascii=False)
    except Exception:
        pass

    cur_headline = str(s_rec.get("slate_headline", "")).strip()
    cur_narrative = str(s_rec.get("slate_narrative", "")).strip()

    class _SlateProxy:
        pass

    proxy = _SlateProxy()
    proxy.date = target_date
    proxy.total_games = total_games
    proxy.total_goals = total_goals
    proxy.avg_goals_per_game = avg_goals
    proxy.home_wins = home_wins
    proxy.away_wins = away_wins
    proxy.ot_so_games = ot_games

    is_valid_slate, _ = verify_slate_narrative_facts(cur_narrative, proxy, headline=cur_headline)

    any_true_shutout = any(
        int(g.get("away_score", 0)) == 0 or int(g.get("home_score", 0)) == 0
        for g in reconciled_games
    )
    if not any_true_shutout and "shutout" in cur_narrative.lower():
        is_valid_slate = False

    verified_day_title = build_zamboni_slate_headline(
        date_str=target_date,
        total_games=total_games,
        total_goals=total_goals,
        home_wins=home_wins,
        away_wins=away_wins,
        ot_so_games=ot_games,
        three_stars=three_stars,
    )

    # Always use the Zamboni verified Day Title and reconciled 3-paragraph Slate Review when cached text is invalid or ungrounded
    s_rec["slate_headline"] = verified_day_title
    if not is_valid_slate:
        s_rec["slate_narrative"] = build_zamboni_slate_narrative_from_records(
            date_str=target_date,
            total_games=total_games,
            total_goals=total_goals,
            avg_goals_per_game=avg_goals,
            home_wins=home_wins,
            away_wins=away_wins,
            ot_so_games=ot_games,
            three_stars=three_stars,
            game_records=reconciled_games,
            milestones=milestones,
        )
    else:
        s_rec["slate_narrative"] = expand_first_mention_abbreviations(clean_llm_narrative(cur_narrative))

    return s_rec, reconciled_games
