"""narrator.py — Sports journalism and narrative recap generator for the Scoreboard Agent.

Synthesizes structured game and slate analyses into polished editorial recaps,
incorporating:
1. Verified Game Scripts (audited period-by-period flow; no false 'thrillers').
2. Defensive Sacrifice & Goalie Performance evaluations.
3. Games Played sample-size context (early vs mid vs late-season race).
4. Pythagorean expectation comparisons & team standings.

Engine Architecture:
- LLM Mode: Uses `meta-llama/Llama-3.2-3B-Instruct` via Hugging Face InferenceClient
  with structured, plain-language prompts and strict anti-hallucination guardrails.
- Algorithmic Mode (Fallback): Deterministic, zero-cost rules-based NLG that guarantees
  100% mathematical reconciliation with official NHL feeds.
- Auto Mode: Attempts LLM generation if HF_TOKEN is present; seamlessly falls back
  to the algorithmic engine if offline, rate-limited, or unauthenticated.
"""
from __future__ import annotations

import logging
import os
import re
import sys
from typing import List, Optional, Tuple

import json
import requests

try:
    from huggingface_hub import InferenceClient, get_token
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

try:
    from .models import AnalyzedGame, AnalyzedSlate
    from .env_utils import is_local_streamlit, discover_hf_token, discover_model_name
    from .skills import EditorialSkillsOrchestrator, load_editorial_playbook
except (ImportError, ValueError):
    from models import AnalyzedGame, AnalyzedSlate
    from env_utils import is_local_streamlit, discover_hf_token, discover_model_name
    from skills import EditorialSkillsOrchestrator, load_editorial_playbook

logger = logging.getLogger("NarratorEngine")


DEFAULT_LLM_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
DEFAULT_HF_SPACE_URL = "https://jjerlee-rinklytics.hf.space"
DEFAULT_ZEROGPU_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8"
ALGORITHMIC_MODEL_STAMP = "algorithmic-guardrails-v1"


def resolve_zerogpu_model(model_override: Optional[str] = None) -> str:
    """Resolves active Space ZeroGPU model from user public Space variables, secrets, .env, or defaults."""
    if model_override and str(model_override).strip():
        return str(model_override).strip()
    return discover_model_name(DEFAULT_ZEROGPU_MODEL)

# Safe import for Hugging Face Spaces ZeroGPU
try:
    import spaces
    SPACES_AVAILABLE = True
except (ImportError, Exception):
    SPACES_AVAILABLE = False
    class spaces:  # type: ignore
        @staticmethod
        def GPU(func=None, duration=60):
            if func is None:
                def decorator(f):
                    return f
                return decorator
            return func


_ZEROGPU_CACHE: dict = {}


def _get_zerogpu_pipeline(model_id: Optional[str] = None):
    """Lazy loader for in-process ZeroGPU model on Hugging Face Spaces with dynamic model caching."""
    global _ZEROGPU_CACHE
    target_model = resolve_zerogpu_model(model_id)

    if target_model not in _ZEROGPU_CACHE:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Initializing in-process ZeroGPU model: %s...", target_model)
        tokenizer = AutoTokenizer.from_pretrained(target_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            target_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        _ZEROGPU_CACHE[target_model] = (model, tokenizer)
    return _ZEROGPU_CACHE[target_model], target_model


@spaces.GPU(duration=120)
def _zerogpu_infer(
    messages: list,
    model_id: Optional[str] = None,
    max_new_tokens: int = 1800,
    temperature: float = 0.35,
) -> str:
    """Runs generation inside Hugging Face Spaces ZeroGPU allocation."""
    import torch
    (model, tokenizer), target_model = _get_zerogpu_pipeline(model_id)

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except (TypeError, Exception):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=pad_id,
        )
    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(gen_tokens, skip_special_tokens=True)
    return response


def clean_llm_narrative(text: str) -> str:
    """Strips reasoning tokens, thinking blocks, and internal monologue from LLM output."""
    if not text:
        return ""

    # 1. Strip <think>...</think> and <thought>...</thought> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<thought>.*?</thought>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"```(?:thought|thinking).*?```", "", cleaned, flags=re.DOTALL | re.IGNORECASE)

    # 2. Split lines and filter out thinking/reasoning prefixes
    lines = cleaned.strip().splitlines()
    filtered_lines = []
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

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()
        if skipping_preamble:
            if not stripped:
                continue
            if any(lower.startswith(p) for p in reasoning_prefixes):
                continue
            # Also skip markdown header labels like "### Paragraph 1: ...", "**Paragraph 1:**"
            if re.match(r"^(?:#{1,4}\s*)?(?:\*\*)?paragraph\s*\d.*", lower):
                continue
            skipping_preamble = False
        filtered_lines.append(line)

    result = "\n".join(filtered_lines).strip()
    return result


def format_milestone_badge(category: str, title: str, description: str = "") -> str:
    """Formats a milestone into a clean, human-readable string without brackets or emoji clutter."""
    # Strip any brackets or legacy prefixes in title like [SHOT_BLOCKER]
    clean_title = re.sub(r"^\[.*?\]\s*", "", str(title)).strip()

    if not description:
        return f"**{clean_title}**"

    clean_desc = str(description).strip()
    return f"**{clean_title}** — {clean_desc}"


def expand_first_mention_abbreviations(text: str) -> str:
    """
    Ensures that the first time any hockey or analytical abbreviation (such as L5, CF%,
    CF-CA, FF%, TOI, SOG, GF-GA, GF/G, GA/G, SV%, OZS%, PIM, FO%, 3F, 2D) appears in
    a narrative, its full meaning is written out before the abbreviation in parentheses,
    while all subsequent mentions in the same text remain concise abbreviations.
    Idempotent: safe to call on text that already has partial or repeated expansions.
    """
    if not text or not text.strip():
        return text

    out = text

    # Step 1: Normalize any pre-existing inline expansions back to their base token
    # so we never double-expand and so only the true FIRST mention in the article is expanded.
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

    # Step 2: Expand strictly the first occurrence (count=1) of each abbreviation
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

    # If standalone 'CF' appears and Corsi has not yet been defined via CF% or CF-CA:
    if "Corsi For" not in out and re.search(r"\bCF\b(?![%-])", out):
        out = re.sub(r"\bCF\b(?![%-])", "Corsi For (CF, total shot attempts)", out, count=1)

    return out


def extract_headline_and_narrative(
    raw_text: str, default_headline: str, period_type: str = "REG"
) -> Tuple[str, str]:
    """Separates the LLM-generated headline from the 5-paragraph narrative text and expands first-mention abbreviations."""
    cleaned = clean_llm_narrative(raw_text).strip()
    lines = cleaned.splitlines()
    headline = default_headline
    narrative_lines = []
    found_headline = False

    for line in lines:
        s = line.strip()
        if not s:
            continue
        if not found_headline:
            if re.match(r"^(?:headline|title)\s*:\s*", s, re.IGNORECASE):
                candidate = re.sub(r"^(?:headline|title)\s*:\s*", "", s, flags=re.IGNORECASE).strip().strip('"*#')
                if candidate:
                    headline = candidate
                    found_headline = True
                    continue
            elif len(s.split()) <= 15 and not s.endswith((".", ";")) and not s.lower().startswith(("the ", "action ", "in ")):
                candidate = s.strip('"*#')
                if candidate:
                    headline = candidate
                    found_headline = True
                    continue

        # Strip any "RECAP:" or "ARTICLE:" markers
        if re.match(r"^(?:recap|article)\s*:\s*$", s, re.IGNORECASE):
            continue
        narrative_lines.append(line)

    # Validate headline against game period_type (prevent overtime/shootout hallucinations)
    h_lower = headline.lower()
    if period_type == "REG" and ("overtime" in h_lower or "shootout" in h_lower or " ot " in h_lower):
        headline = default_headline
    elif period_type == "OT" and "shootout" in h_lower:
        headline = default_headline

    raw_paras = [p.strip() for p in "\n".join(narrative_lines).split("\n\n") if p.strip()]
    if len(raw_paras) == 1 and "\n" in raw_paras[0]:
        sub_lines = [l.strip() for l in raw_paras[0].split("\n") if l.strip()]
        if len(sub_lines) >= 3 and all(len(l) >= 60 for l in sub_lines):
            raw_paras = sub_lines

    narrative = "\n\n".join(raw_paras).strip()
    narrative = expand_first_mention_abbreviations(narrative)
    return headline, narrative


class NarratorEngine:
    """Generates human-readable, sports-journalistic game recaps and slate summaries.
    
    Supports:
    1. Local LM Studio Server (OpenAI-compatible API at http://127.0.0.1:1234, zero fees, GPU-accelerated)
    2. Hugging Face Space Runtime (When deployed remotely; pings own HF Space host)
    3. Remote Router LLM (Meta-Llama-3.2-3B-Instruct via huggingface_hub)
    4. Deterministic Algorithmic Guardrail Engine (100% mathematically reconciled NLG)
    """

    def __init__(
        self,
        model_mode: str = "auto",  # 'auto', 'lm-studio', 'llm'/'router-llm', or 'algorithmic'
        lm_studio_url: str = DEFAULT_LM_STUDIO_URL,
        lm_model_id: Optional[str] = None,
        llm_model_id: str = DEFAULT_LLM_MODEL,
        hf_space_url: Optional[str] = None,
        token: Optional[str] = None,
    ):
        self.model_mode = model_mode
        self.lm_studio_url = os.environ.get("LM_STUDIO_URL", lm_studio_url).rstrip("/")
        self.lm_model_id = lm_model_id or os.environ.get("LM_STUDIO_MODEL")
        self.llm_model_id = llm_model_id
        
        # Configure Hugging Face Space Host URL for non-local environments
        env_space_host = os.environ.get("SPACE_HOST")
        derived_space_url = f"https://{env_space_host}" if env_space_host else DEFAULT_HF_SPACE_URL
        self.hf_space_url = os.environ.get("HF_SPACE_HOST") or os.environ.get("HF_SPACE_URL") or hf_space_url or derived_space_url
        self.hf_llm_endpoint = os.environ.get("HF_LLM_URL")
        self.zerogpu_model_id = resolve_zerogpu_model()

        self.token = discover_hf_token(token)
        self._inference_client = None
        self._active_lm_studio_model = None

    def _get_inference_client(self) -> Optional[InferenceClient]:
        """Lazy initializer for Hugging Face InferenceClient."""
        if self._inference_client is not None:
            return self._inference_client

        if not HF_HUB_AVAILABLE:
            logger.debug("huggingface_hub is not installed. Router LLM mode unavailable.")
            return None

        token = discover_hf_token(self.token)
        if not token:
            logger.debug("No HF_TOKEN found. Router LLM unavailable.")
            return None

        try:
            self._inference_client = InferenceClient(
                model=self.llm_model_id,
                token=token,
                timeout=45.0,
            )
            return self._inference_client
        except Exception as e:
            logger.warning("Failed to initialize Hugging Face InferenceClient: %s", e)
            return None

    def check_lm_studio_health(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """Verify connectivity to the local LM Studio instance at http://127.0.0.1:1234.
        
        Only pings LM Studio when running in a local development environment (`local_streamlit=True`).
        Automatically checks `host.docker.internal` fallback if running inside a local Docker container.
        
        Returns:
            Tuple[is_reachable: bool, detected_model: Optional[str], error_message: Optional[str]]
        """
        # In deployed / remote environments (e.g. Hugging Face Spaces), skip LM Studio ping entirely
        if not is_local_streamlit():
            return False, None, "Remote deployment (LM Studio ping skipped in Hugging Face environment)."

        candidate_urls = [self.lm_studio_url]
        if os.path.exists("/.dockerenv") and ("127.0.0.1" in self.lm_studio_url or "localhost" in self.lm_studio_url):
            docker_url = self.lm_studio_url.replace("127.0.0.1", "host.docker.internal").replace("localhost", "host.docker.internal")
            if docker_url not in candidate_urls:
                candidate_urls.append(docker_url)

        last_err = ""
        for base_url in candidate_urls:
            try:
                url = f"{base_url}/models"
                resp = requests.get(url, timeout=3.0)
                if resp.status_code == 200:
                    self.lm_studio_url = base_url
                    data = resp.json()
                    raw_models = [m.get("id") for m in data.get("data", []) if m.get("id")]
                    # Filter out pure text embedding models
                    chat_models = [m for m in raw_models if "embed" not in m.lower()]

                    # Prioritize instruct / non-reasoning chat models for sports beat reporting
                    instruct_models = [m for m in chat_models if "reasoning" not in m.lower()]
                    reasoning_models = [m for m in chat_models if "reasoning" in m.lower()]
                    ranked_models = instruct_models + reasoning_models + raw_models

                    selected = self.lm_model_id or (ranked_models[0] if ranked_models else None)
                    self._active_lm_studio_model = selected
                    return True, selected, None
                else:
                    last_err = f"LM Studio server at {base_url} returned HTTP status {resp.status_code}."
            except Exception as e:
                last_err = f"Unable to reach LM Studio at {base_url} ({type(e).__name__}: {e})"

        if is_local_streamlit():
            self._trigger_alarm(last_err)
        return False, None, last_err

    def check_hf_space_health(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """Verify connectivity to own Hugging Face Space host environment.
        
        Returns:
            Tuple[is_reachable: bool, space_name: Optional[str], status_note: Optional[str]]
        """
        space_url = self.hf_space_url.rstrip("/")
        try:
            resp = requests.get(space_url, timeout=5.0)
            if resp.status_code in (200, 301, 302):
                return True, "jjerlee/rinklytics", "Space host reachable"
            return False, "jjerlee/rinklytics", f"Space returned HTTP status {resp.status_code}"
        except Exception as e:
            return False, "jjerlee/rinklytics", f"Unable to reach Space at {space_url}: {e}"

    def _trigger_alarm(self, reason: str):
        """Emits an urgent alarm when the configured LM Studio model cannot be reached."""
        alarm_banner = (
            "\n" + "🚨" * 38 + "\n"
            f"🚨 [ALARM: LM STUDIO UNREACHABLE]\n"
            f"🚨 Details: {reason}\n"
            f"🚨 Action: Please verify LM Studio is running and the local server is started at {self.lm_studio_url}\n"
            f"🚨 Safeguard: Falling back to Algorithmic Guardrail Engine to guarantee verified box-score recaps.\n"
            + "🚨" * 38 + "\n"
        )
        logger.error(alarm_banner)
        sys.stderr.write(alarm_banner)
        sys.stderr.flush()

    def _verify_game_narrative_facts(self, narrative: str, game: AnalyzedGame, headline: str = "") -> bool:
        """Audits the generated narrative text and headline to ensure zero hallucinations or reasoning leaks.
        
        Checks:
        1. Length: Must be substantial (>120 chars).
        2. Reasoning filter: Rejects internal model monologue, planning notes, or thinking leaks.
        3. Winner presence: Must mention the winning team name or abbreviation.
        4. Hockey terms & period formatting: Must not contain 'halftime', 'quarter', or 'in the reg period'.
        5. Decision consistency: Must not claim shootout/overtime when inconsistent with period_type.
        6. Roster & Role Integrity: Rejects goalie-scoring-goal or skater-making-saves hallucinations
           and rejects hallucinated non-roster player names in headlines or possessives.
        """
        if not narrative or len(narrative.strip()) < 120:
            return False

        combined = f"{headline}\n{narrative}" if headline else narrative
        lower = combined.lower()

        # REASONING LEAKAGE FILTER:
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
            "first goal, maybe breaking a shutout",
            "<think>",
            "</think>",
            "<thought>",
            "thinking process",
            "in the reg period",
            "of reg.",
        )
        if any(phrase in lower for phrase in leak_phrases):
            logger.warning("Guardrail trigger: Narrative contains internal model reasoning leakage or malformed period label.")
            return False

        if "halftime" in lower:
            logger.warning("Guardrail trigger: Narrative contains non-hockey term 'halftime'.")
            return False

        if re.search(r"\b(?:1st|2nd|3rd|4th|first|second|third|fourth)\s+quarter\b", lower):
            if not re.search(r"\b(?:1st|2nd|3rd|4th|first|second|third|fourth)\s+quarter\s+of\s+the\s+(?:season|schedule|year|campaign)\b", lower):
                logger.warning("Guardrail trigger: Narrative contains non-hockey period term 'quarter'.")
                return False

        if game.period_type in ("REG", "OT") and ("penalty shootout" in lower or "in a shootout" in lower):
            logger.warning("Guardrail trigger: Game ended in %s but narrative claimed shootout.", game.period_type)
            return False

        winner_name = game.winner.team_name.lower()
        winner_abbrev = game.winner.team_abbrev.lower()
        name_parts = [p.lower() for p in game.winner.team_name.split() if len(p) >= 3]
        if winner_name not in lower and winner_abbrev not in lower and not any(p in lower for p in name_parts):
            logger.warning("Guardrail trigger: Winner (%s) not found in recap text.", game.winner.team_abbrev)
            return False

        # ROSTER & ROLE INTEGRITY AUDIT:
        import unicodedata

        def _strip_accents(text: str) -> str:
            nfkd = unicodedata.normalize("NFKD", text)
            return "".join(c for c in nfkd if not unicodedata.combining(c)).lower()

        all_skaters = list(game.away_skaters or []) + list(game.home_skaters or [])
        all_goalies = list(game.away_goalies or []) + list(game.home_goalies or [])
        scorer_names = {sp.goal_scorer.split()[-1].lower() for sp in (game.scoring_plays or []) if sp.goal_scorer}
        skater_last_names = {s.name.split()[-1].lower() for s in all_skaters if s.name}
        goalie_last_names = {g.name.split()[-1].lower() for g in all_goalies if g.name}

        # 6a. Prevent non-goal-scorer skaters from being falsely credited with scoring a goal (e.g. 0G, 3A player hallucinated as scoring)
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
                        logger.warning("Guardrail trigger: Player '%s' (0 goals in boxscore) falsely credited with scoring a goal.", ns_last)
                        return False

        # 6b. Prevent goalie-scored-goal hallucinations (unless goalie actually scored)
        for g_last in goalie_last_names:
            if g_last and g_last not in scorer_names and g_last not in skater_last_names:
                goal_patterns = (
                    rf"\b{re.escape(g_last)}['’]s\s+(?:[a-z0-9-]+\s+){{0,3}}(?:goal|winner|strike|tally|marker)\b",
                    rf"\b{re.escape(g_last)}\s+(?:scored|netted|tallied|buried|potted)\b",
                )
                for pat in goal_patterns:
                    if re.search(pat, lower):
                        logger.warning("Guardrail trigger: Goalie '%s' hallucinated as scoring a goal.", g_last)
                        return False

        # 6c. Prevent skater-making-saves hallucinations
        for s_last in skater_last_names:
            if s_last and s_last not in goalie_last_names:
                save_pat = rf"\b{re.escape(s_last)}\s+(?:made|makes|stopped|turned aside)\s+\d*\s*(?:[a-z-]+\s+){{0,2}}saves?\b"
                if re.search(save_pat, lower):
                    logger.warning("Guardrail trigger: Skater '%s' hallucinated as making goalie saves.", s_last)
                    return False

        # 6c. Prevent non-roster player hallucinations in headlines and possessives
        roster_tokens = set()
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

        team_tokens = {
            game.away_team.team_abbrev.lower(),
            game.home_team.team_abbrev.lower(),
        }
        for t_name in (game.away_team.team_name, game.home_team.team_name, game.venue):
            for part in re.findall(r"[A-Za-zÀ-ÿ-]+", t_name):
                team_tokens.add(part.lower())
                team_tokens.add(_strip_accents(part))

        common_headline_words = {
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
            "mount", "mounts", "erase", "erases", "deficit", "lead", "streak", "form", "steady", "dominant",
            "nhl", "hockey", "eastern", "western", "division", "conference", "arena", "center", "centre",
            "club", "squad", "bench", "coach", "trio", "unit", "group", "frame", "contest", "matchup",
            "evening", "visitor", "host", "winner", "loser", "opponent", "penalty", "forward", "defense",
            "goalie", "goaltender", "netminder", "captain", "veteran", "rookie", "corsi", "fenwick", "pythagorean",
            "tonight", "today", "yesterday", "tomorrow", "stretch", "trip", "stand", "homestand", "recap",
            "action", "showcase", "meeting", "franchise", "organization", "roster", "lineup", "schedule",
            "campaign", "race", "standings",
            "it", "its", "there", "here", "that", "what", "who", "how", "let", "one", "someone", "anyone", "nobody", "everyone", "everybody",
            "rival", "rivalry", "canada", "america", "blueliner", "skater", "defenseman", "pairing", "corps", "depth", "tandem", "milestone", "record", "history", "momentum", "intermission",
            "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
            "october", "november", "december", "january", "february", "march", "april", "may", "june",
        }
        allowed_headline_tokens = roster_tokens | team_tokens | common_headline_words

        # Check possessive proper nouns in headline (e.g. "Tuch’s", "Knies's", "Coleman's")
        for m in re.finditer(r"\b([A-Z][a-zÀ-ÿ-]+)['’]s\b", headline):
            word_low = m.group(1).lower()
            word_norm = _strip_accents(m.group(1))
            if word_low not in allowed_headline_tokens and word_norm not in allowed_headline_tokens:
                logger.warning("Guardrail trigger: Non-roster possessive name '%s' detected in headline.", m.group(1))
                return False

        # In narrative, check possessive proper nouns credited with goals/assists/saves/milestones
        cred_pat = (
            r"\b([A-Z][a-zÀ-ÿ-]+)['’]s\s+(?:[a-z0-9-]+\s+){0,2}"
            r"(?:goal|winner|tally|marker|strike|assist|point|save|shutout|hat[- ]trick)\b"
        )
        for m in re.finditer(cred_pat, narrative, re.IGNORECASE):
            word_low = m.group(1).lower()
            word_norm = _strip_accents(m.group(1))
            if word_low not in allowed_headline_tokens and word_norm not in allowed_headline_tokens:
                logger.warning("Guardrail trigger: Non-roster player '%s' credited with play in narrative.", m.group(1))
                return False

        if headline:
            # Check subject capitalized words before action verbs in headline (e.g. "Knies Steers", "Coleman Powers")
            verb_pat = (
                r"\b([A-Z][a-zÀ-ÿ-]+)\s+(?i:nets|scores|tallies|buries|pots|powers|propels|lifts|guides|leads|"
                r"sparks|delivers|breaks|stops|steers|seals|flips|clinches|caps|opens|strikes|fires|dishes|"
                r"anchors|dominates|outduels|shuts)\b"
            )
            for m in re.finditer(verb_pat, headline):
                word_low = m.group(1).lower()
                word_norm = _strip_accents(m.group(1))
                if word_low not in allowed_headline_tokens and word_norm not in allowed_headline_tokens:
                    logger.warning("Guardrail trigger: Non-roster player '%s' detected in headline '%s'.", m.group(1), headline)
                    return False

        return True

    def _verify_slate_narrative_facts(self, narrative: str, slate: AnalyzedSlate) -> bool:
        """Audits the generated slate overview to ensure zero hallucinations or reasoning leaks."""
        if not narrative or len(narrative.strip()) < 120:
            return False
        lower = narrative.lower()
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
            )
        ):
            return False
        if "halftime" in lower:
            return False
        return True

    # =========================================================================
    # 1. Headline Generator (Verified by Period Scoring Flow)
    # =========================================================================

    def build_game_headline(self, game: AnalyzedGame) -> str:
        """Create an expressive, context-aware sports headline verified by period scoring flow."""
        winner = game.winner
        loser = game.loser
        w_score = winner.score
        l_score = loser.score
        hero = game.three_stars[0].name if game.three_stars else "Total Team Effort"
        flow = game.period_flow

        script = flow.game_script_type if flow else "REGULATION"

        if script == "WALK_OFF_OT_SO":
            if game.period_type == "SO":
                return f"{winner.team_name} Outlast {loser.team_name} {w_score}-{l_score} in Dramatic Shootout"
            return f"{hero} Nets Overtime Winner as {winner.team_name} Edge {loser.team_name} {w_score}-{l_score}"
        elif script == "BLOWOUT":
            return f"Offensive Explosion: {winner.team_name} Overwhelm {loser.team_name} {w_score}-{l_score}"
        elif script == "COMEBACK_WIN":
            return f"{winner.team_name} Mount 3rd-Period Rally to Stun {loser.team_name} {w_score}-{l_score}"
        elif script == "CONTROLLED_LATE_PUSH":
            return f"{winner.team_name} Stave Off Late 3rd-Period Push by {loser.team_name} in {w_score}-{l_score} Victory"
        elif script == "TRUE_THRILLER":
            return f"{hero} Breaks Deadlock in See-Saw Battle as {winner.team_name} Edge {loser.team_name} {w_score}-{l_score}"
        elif script == "WIRE_TO_WIRE":
            return f"{winner.team_name} Dictate Tempo Wire-to-Wire, Dispatch {loser.team_name} {w_score}-{l_score}"
        else:
            return f"{winner.team_name} Secure {w_score}-{l_score} Victory Over {loser.team_name}"

    # =========================================================================
    # 2. Plain-Language Prompt Construction for LLM Generation
    # =========================================================================

    def build_game_llm_system_prompt(self) -> str:
        """Plain-language system instructions dynamically incorporating editorial_playbook.md."""
        playbook_md = load_editorial_playbook()
        return (
            "You are a senior NHL beat columnist and tactical hockey analyst writing for Rinklytics and the Associated Press.\n"
            "Your mission is to write an engaging, publication-ready sports headline and a 5-paragraph game recap in fluid, authoritative hockey prose.\n\n"
            "CRITICAL OUTPUT FORMAT MANDATE:\n"
            "1. Line 1 MUST be the article headline formatted exactly as: HEADLINE: <Engaging journalistic headline under 15 words>\n"
            "2. Follow immediately with 5 continuous, publication-ready paragraphs separated by blank lines.\n"
            "3. FIRST-MENTION ABBREVIATION RULE: The first time you use any hockey or analytical abbreviation in the article, write out what it stands for and means before placing the abbreviation in parentheses—for example: 'last five games (L5)', 'Time on Ice (TOI)', 'Shots on Goal (SOG)', 'Corsi For percentage (CF%, share of all shot attempts)', 'Goals For vs. Goals Against (GF-GA)', 'Goals For per game (GF/G)', and 'Save Percentage (SV%)'. On subsequent mentions in the same article, use the short abbreviation.\n"
            "4. NEVER echo prompt labels, skill headers, or field names (never start sentences with 'Winner:', 'Skill 1:', 'Standout Forward Unit:', or 'Season Stage:'). Weave all insights naturally into journalistic prose.\n"
            "5. NEVER output any internal thinking process, chain-of-thought, reasoning monologue, planning notes, or meta-commentary (e.g. absolutely no 'Okay, I need to write...').\n"
            "6. Do NOT include markdown headers, section titles, or bullet points in your output.\n"
            "7. STRICT GOALS VS. POINTS MANDATE: Points are goals PLUS assists. If a player recorded 0 goals and 3 assists, they had 3 points but ZERO goals. NEVER state that a player 'scored', 'netted', or 'factored into goals' unless they actually have goals in the Chronological Goals Scored ledger. Refer to players with 0 goals as playmakers who dished out assists or recorded points.\n\n"
            "EDITORIAL PLAYBOOK & MULTI-SKILLSET GUIDELINES:\n"
            f"{playbook_md}\n"
        )

    def build_game_llm_user_prompt(self, game: AnalyzedGame) -> str:
        """Multi-skillset user prompt combining the synthesized EditorialDossier and audited game telemetry."""
        winner = game.winner
        loser = game.loser
        flow = game.period_flow
        hero = game.three_stars[0].name if game.three_stars else "Total Team Effort"

        if game.editorial_dossier is None:
            game.editorial_dossier = EditorialSkillsOrchestrator.evaluate_game(game)
        dossier = game.editorial_dossier

        dec_str = (
            "Final (Regulation)"
            if game.period_type == "REG"
            else ("Final (Overtime)" if game.period_type == "OT" else "Final (Shootout)")
        )

        # Build strict verified scoring & roster ledger so small models never invent player names or swap roles
        scoring_ledger = []
        for idx, sp in enumerate(game.scoring_plays or [], 1):
            assists_str = f" (Assists: {', '.join(sp.assists)})" if sp.assists else " (Unassisted)"
            scoring_ledger.append(
                f"  {idx}. {sp.period_descriptor} {sp.time_in_period} [{sp.strength}] — Goal Scorer: {sp.goal_scorer} ({sp.scoring_team}){assists_str} -> Score: {game.away_team.team_abbrev} {sp.away_score}, {game.home_team.team_abbrev} {sp.home_score}"
            )
        scoring_block = "\n".join(scoring_ledger) if scoring_ledger else "  No regulation/OT goals recorded."

        goalie_ledger = []
        for g in (game.away_goalies or []) + (game.home_goalies or []):
            if g.shots_against > 0 or g.starter:
                goalie_ledger.append(
                    f"  - {g.name} ({g.team_abbrev}, GOALTENDER — saves only, did NOT score goals): {g.saves}/{g.shots_against} saves ({g.save_pct*100:.1f}% SV%, TOI {g.toi})"
                )
        goalie_block = "\n".join(goalie_ledger)

        away_top_skaters = ", ".join(
            f"{s.name} ({s.goals}G, {s.assists}A, {s.sog} SOG)" for s in (game.away_skaters or [])[:6]
        )
        home_top_skaters = ", ".join(
            f"{s.name} ({s.goals}G, {s.assists}A, {s.sog} SOG)" for s in (game.home_skaters or [])[:6]
        )

        steps = [
            "Write an authoritative headline and 5-paragraph NHL game column using this audited Multi-Skillset Editorial Dossier:",
            f"FINAL SCORE & VENUE: {winner.team_name} ({winner.score}) defeated {loser.team_name} ({loser.score}) in {dec_str} at {game.venue}.",
            f"FIRST STAR / CATALYST: {hero}",
            "",
            "VERIFIED SCORING & ROSTER LEDGER (STRICT — DO NOT INVENT OR SUBSTITUTE ANY PLAYER NAMES):",
            "Chronological Goals Scored:",
            scoring_block,
            "Goaltenders (NEVER credit a goaltender with scoring a goal; NEVER credit a skater with making saves):",
            goalie_block,
            f"Top {game.away_team.team_abbrev} Skaters: {away_top_skaters}",
            f"Top {game.home_team.team_abbrev} Skaters: {home_top_skaters}",
            "",
            dossier.synthesized_brief_text,
            "",
            "PARAGRAPH-BY-PARAGRAPH WRITING PLAN (DO NOT OUTPUT LABELS OR HEADERS):",
            "- Paragraph 1 (Lede, Archetype & Home/Away Adversity): Lead with the Primary Editorial Archetype, final score, decision type, venue, whether the winner overcame road-trip/back-to-back adversity or leveraged home last-change, and the primary star.",
            "- Paragraph 2 (Period Progression & Sequential Momentum): Trace the chronological scoring flow across periods (citing the ice-breaker, intermission scores, and decisive strike) and how momentum shifted.",
            "- Paragraph 3 (Line Shift Chemistry, Bench Adjustments & L5 Line Trends): Analyze the on-ice three-forward (3F) trios and two-defenseman (2D) pairings from shift data. Highlight which line drove play or struggled—spelling out Time on Ice (TOI), Corsi For percentage (CF%, share of all shot attempts), Shots on Goal (SOG), and Goals For vs. Goals Against (GF-GA) on first mention—and state whether any notable unit was established over the last five games (L5) or a new line change / mid-game shuffle, explaining whether that adjustment improved or regressed performance relative to their L5 baseline.",
            "- Paragraph 4 (Goaltending Composure, Defensive Sacrifice & Special Teams): Contrast the goaltenders' workloads and Save Percentage (SV%) alongside shot-blocking sacrifice, physicality, turnover margins, and power-play execution.",
            "- Paragraph 5 (Season Stage Trajectory & L5 Rolling Team Form): Contrast both teams' full-season standing (Record, Points%, Season Corsi%, Pythagorean expectation) against their rolling form over the last five games (L5) within the context of the current stage of the season.",
            "",
            "WRITE THE ARTICLE NOW:",
            "Start immediately with HEADLINE: <headline> on Line 1, followed by 5 continuous journalistic paragraphs in fluid hockey prose. Use ONLY player names from the Verified Roster Ledger above. Write out every abbreviation on its first mention. No preamble, no markdown headers, and no reasoning monologue."
        ]
        return "\n".join(steps)

    def _generate_game_narrative_with_llm(self, game: AnalyzedGame, client: InferenceClient) -> str:
        """Invokes Meta-Llama-3.2-3B-Instruct to draft the journalistic article."""
        system_prompt = self.build_game_llm_system_prompt()
        user_prompt = self.build_game_llm_user_prompt(game)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = client.chat.completions.create(
            messages=messages,
            max_tokens=1500,
            temperature=0.65,
        )
        content = response.choices[0].message.content or ""
        cleaned = clean_llm_narrative(content)
        return cleaned.strip()

    def _generate_game_narrative_with_custom_endpoint(self, game: AnalyzedGame, endpoint_url: str) -> str:
        """Invokes a custom remote OpenAI-compatible model server (e.g. Groq, remote LM Studio, or vLLM)."""
        system_prompt = self.build_game_llm_system_prompt()
        user_prompt = self.build_game_llm_user_prompt(game)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        url = f"{endpoint_url.rstrip('/')}/chat/completions"
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY") or self.token or "none"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": self.llm_model_id,
            "messages": messages,
            "max_tokens": 1500,
            "temperature": 0.35,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        content = choice["message"].get("content") or ""
        cleaned = clean_llm_narrative(content)
        return cleaned.strip()

    def _generate_game_narrative_with_lm_studio(
        self,
        game: AnalyzedGame,
        model_name: str,
        temperature: float = 0.35,
        max_tokens: int = 1800,
    ) -> str:
        """Invokes local LM Studio instance (e.g. nemotron / phi-4 / gemma) via OpenAI-compatible API without leaking reasoning."""
        system_prompt = self.build_game_llm_system_prompt()
        user_prompt = self.build_game_llm_user_prompt(game)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        url = f"{self.lm_studio_url}/chat/completions"
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        resp = requests.post(url, json=payload, timeout=180.0)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        content = choice["message"].get("content") or ""

        # NEVER fall back to reasoning_content: internal monologue is not article text
        cleaned = clean_llm_narrative(content)
        return cleaned.strip()

    def _can_use_zerogpu(self) -> bool:
        """Returns True if running in a Hugging Face Space environment with ZeroGPU support."""
        if SPACES_AVAILABLE:
            return True
        return bool(os.environ.get("SPACES_ZERO_GPU") or os.environ.get("SPACE_ID"))

    def _generate_game_narrative_with_zerogpu(
        self,
        game: AnalyzedGame,
        model_id: Optional[str] = None,
        temperature: float = 0.35,
        max_tokens: int = 1800,
    ) -> str:
        """Invokes in-process ZeroGPU allocation to generate game narrative."""
        target_model = resolve_zerogpu_model(model_id or self.zerogpu_model_id)
        system_prompt = self.build_game_llm_system_prompt()
        user_prompt = self.build_game_llm_user_prompt(game)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw = _zerogpu_infer(messages, model_id=target_model, max_new_tokens=max_tokens, temperature=temperature)
        cleaned = clean_llm_narrative(raw)
        return cleaned.strip()

    def _generate_slate_narrative_with_zerogpu(
        self,
        slate: AnalyzedSlate,
        model_id: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1200,
    ) -> str:
        """Invokes in-process ZeroGPU allocation for slate review."""
        target_model = resolve_zerogpu_model(model_id or self.zerogpu_model_id)
        system_prompt = self.build_slate_llm_system_prompt()
        user_prompt = self.build_slate_llm_user_prompt(slate)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw = _zerogpu_infer(messages, model_id=target_model, max_new_tokens=max_tokens, temperature=temperature)
        cleaned = clean_llm_narrative(raw)
        return cleaned.strip()

    # =========================================================================
    # 3. Algorithmic NLG Engine (Deterministic, Zero-Cost Fallback)
    # =========================================================================

    def build_game_narrative_algorithmic(self, game: AnalyzedGame) -> str:
        """Deterministic 5-paragraph generator incorporating Line Shifts, L5 Trends, and Schedule Context."""
        paragraphs = []
        winner = game.winner
        loser = game.loser
        w_score = winner.score
        l_score = loser.score
        flow = game.period_flow
        w_is_home = (winner.team_abbrev == game.home_team.team_abbrev)
        w_tr = game.home_shift_trend_profile if w_is_home else game.away_shift_trend_profile
        l_tr = game.away_shift_trend_profile if w_is_home else game.home_shift_trend_profile

        # Paragraph 1: The Lede, Game Archetype & Home/Away Adversity
        ot_clause = f" in {game.period_type}" if game.period_type in ("OT", "SO") else ""
        star1 = game.three_stars[0].name if game.three_stars else None

        venue_context = "on home ice" if w_is_home else "on the road"
        if w_tr:
            if not w_is_home and w_tr.consec_road_games >= 2:
                venue_context = f"on the road in game #{w_tr.consec_road_games} of their current trip"
            if w_tr.is_back_to_back:
                venue_context += " despite playing on the back end of a back-to-back"
            elif w_tr.rest_advantage < 0:
                venue_context += f" while overcoming a rest disadvantage ({w_tr.rest_days}d vs. {w_tr.opponent_rest_days}d)"

        script_desc = ""
        if flow:
            if flow.game_script_type == "WIRE_TO_WIRE":
                script_desc = f"{winner.team_name} dictated terms from the opening period and never relinquished control, executing a wire-to-wire clinic. "
            elif flow.game_script_type == "CONTROLLED_LATE_PUSH":
                script_desc = f"{winner.team_name} built a multi-goal cushion through forty minutes before weathering a desperate third-period push from {loser.team_name}. "
            elif flow.game_script_type == "COMEBACK_WIN":
                script_desc = f"{winner.team_name} displayed remarkable poise, erasing a third-period deficit to flip the script in the final frame. "
            elif flow.game_script_type == "TRUE_THRILLER":
                script_desc = f"In a back-and-forth battle featuring {flow.ties_count} ties and {flow.lead_changes} lead swings, {winner.team_name} delivered the decisive late counter-punch. "
            elif flow.game_script_type == "WALK_OFF_OT_SO":
                script_desc = f"After sixty minutes of tight-checking hockey left the clubs deadlocked, {winner.team_name} captured the extra point in sudden-death fashion. "

        lede = (
            f"The **{winner.team_name}** secured a {w_score}–{l_score} victory{ot_clause} {venue_context} "
            f"over the **{loser.team_name}** at {game.venue}. {script_desc}"
        )
        if star1:
            lede += f"**{star1}** headlined the victory with a pace-setting performance."
        paragraphs.append(lede.strip())

        # Paragraph 2: Scoring Progression & Period-by-Period Momentum
        if game.scoring_plays:
            first_goal = game.scoring_plays[0]
            last_goal = game.scoring_plays[-1]
            p_desc_low = first_goal.period_descriptor.lower()
            if p_desc_low in ("overtime", "shootout"):
                open_clause = f"Scoring opened in {p_desc_low}"
            elif "period" in p_desc_low:
                open_clause = f"Scoring opened in the {p_desc_low}"
            else:
                open_clause = f"Scoring opened in the {p_desc_low} period"

            prog_text = (
                f"{open_clause} when **{first_goal.goal_scorer}** "
                f"({first_goal.scoring_team}) broke the ice at {first_goal.time_in_period} ({first_goal.strength}). "
            )
            if flow:
                prog_text += (
                    f"The clubs hit the first intermission at **{flow.score_after_p1}** before moving to "
                    f"**{flow.score_after_p2}** through forty minutes. "
                )
            if len(game.scoring_plays) > 2:
                prog_text += f"Across the contest, the offenses combined for {len(game.scoring_plays)} goals. "

            if game.period_type in ("OT", "SO"):
                prog_text += (
                    f"With regulation unable to settle the issue, **{last_goal.goal_scorer}** buried the decisive marker "
                    f"at {last_goal.time_in_period} of {last_goal.period_descriptor}."
                )
            else:
                prog_text += f"**{last_goal.goal_scorer}** capped the scoring at {last_goal.time_in_period} of the third period."
            paragraphs.append(prog_text.strip())

        # Paragraph 3: Line Shift Chemistry, Bench Management & Past 5 Games (L5) Line Trends
        shift_parts = []
        if w_tr and w_tr.has_shift_data:
            if w_tr.new_lines_deployed_count > 0:
                shift_parts.append(
                    f"On the bench, {winner.team_abbrev} deployed {w_tr.new_lines_deployed_count} newly assembled or reshuffled forward combinations "
                    f"({w_tr.line_continuity_pct:.1f}% line continuity over the last five games (L5))."
                )
            else:
                shift_parts.append(
                    f"Five-on-five deployment saw {winner.team_abbrev} lean on line continuity ({w_tr.line_continuity_pct:.1f}% of forward Time on Ice (TOI) played by established trios from the last five games (L5))."
                )
            if w_tr.best_line:
                shift_parts.append(w_tr.best_line.line_story_summary)
            if w_tr.top_pair:
                shift_parts.append(f"On the blue line, {w_tr.top_pair.line_story_summary}")
        if l_tr and l_tr.has_shift_data and l_tr.worst_line:
            shift_parts.append(f"For {loser.team_abbrev}, {l_tr.worst_line.line_story_summary}")

        if not shift_parts and game.editorial_dossier and "line_shifts" in game.editorial_dossier.skill_briefs:
            shift_parts.extend(game.editorial_dossier.skill_briefs["line_shifts"].narrative_points)

        if shift_parts:
            paragraphs.append(" ".join(shift_parts).strip())

        # Paragraph 4: Goaltending Composure, Defensive Sacrifice & Special Teams
        w_goalie = game.winning_goalie_profile
        l_goalie = game.losing_goalie_profile
        w_def = game.home_defense_profile if w_is_home else game.away_defense_profile
        l_def = game.away_defense_profile if w_is_home else game.home_defense_profile

        def_parts = []
        if w_goalie:
            def_parts.append(f"In the crease, **{w_goalie.name}** {w_goalie.assessment.lower()}")
        if l_goalie and l_goalie.name != getattr(w_goalie, "name", ""):
            def_parts.append(f"Opposite him, **{l_goalie.name}** {l_goalie.assessment.lower()}")

        if w_def and l_def:
            def_parts.append(
                f"Defensively, {winner.team_abbrev} committed to **{w_def.blocked_shots}** blocked shots "
                f"and **{w_def.hits}** hits ({w_def.turnover_diff:+d} turnover differential), while {loser.team_abbrev} registered "
                f"**{l_def.blocked_shots}** blocks and **{l_def.hits}** hits ({l_def.turnover_diff:+d} turnover differential)."
            )

        pp_text = []
        if winner.power_play_conv and winner.power_play_conv != "0/0":
            pp_text.append(f"{winner.team_abbrev} went {winner.power_play_conv} ({winner.power_play_pct:.1f}%) on the power play")
        if loser.power_play_conv and loser.power_play_conv != "0/0":
            pp_text.append(f"{loser.team_abbrev} finished {loser.power_play_conv} with the man advantage")
        if pp_text:
            def_parts.append("On special teams, " + " while ".join(pp_text) + ".")

        paragraphs.append(" ".join(def_parts).strip())

        # Paragraph 5: Season Trajectory & Past 5 Games (L5) Rolling Form
        w_ctx = game.home_season_context if w_is_home else game.away_season_context
        l_ctx = game.away_season_context if w_is_home else game.home_season_context

        if w_ctx and l_ctx:
            stage_note = w_ctx.stage_narrative
            rec_line = (
                f"The victory lifts the **{winner.team_name}** to {w_ctx.wins}–{w_ctx.losses}–{w_ctx.ot_losses} "
                f"({w_ctx.points} pts, {w_ctx.points_pct*100:.1f}% pts, Pythagorean expectation {w_ctx.pythagorean_pct*100:.1f}%), "
                f"while the **{loser.team_name}** move to {l_ctx.wins}–{l_ctx.losses}–{l_ctx.ot_losses} "
                f"({l_ctx.points} pts, {l_ctx.points_pct*100:.1f}% pts, Pythagorean expectation {l_ctx.pythagorean_pct*100:.1f}%). "
            )
            l5_notes = []
            if w_tr and w_tr.l5_games > 0:
                l5_notes.append(
                    f"Entering the night at {w_tr.l5_record} over their L5 ({w_tr.l5_cf_pct:.1f}% L5 CF%, {w_tr.l5_gf_per_game:.2f} GF/G), "
                    f"{winner.team_abbrev} posted a {w_tr.game_cf_pct:.1f}% CF% ({w_tr.game_vs_l5_cf_delta:+.1f}% vs. L5)"
                )
            if l_tr and l_tr.l5_games > 0:
                l5_notes.append(
                    f"{loser.team_abbrev} ({l_tr.l5_record} in L5, {l_tr.l5_cf_pct:.1f}% L5 CF%) finished at {l_tr.game_cf_pct:.1f}% CF% ({l_tr.game_vs_l5_cf_delta:+.1f}% vs. L5)"
                )
            l5_sentence = ("; ".join(l5_notes) + ".") if l5_notes else ""
            paragraphs.append(f"{stage_note} {rec_line}{l5_sentence}".strip())

        return expand_first_mention_abbreviations("\n\n".join(paragraphs))

    # =========================================================================
    # 4. Unified Narrative Dispatcher
    # =========================================================================

    def build_game_narrative(
        self,
        game: AnalyzedGame,
        force_mode: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Dispatches narrative generation to LM Studio, Hugging Face Space, or Algorithmic engine.
        
        Returns:
            Tuple[narrative_text, generator_model_stamp]
        """
        mode = force_mode or self.model_mode
        is_local = is_local_streamlit()
        default_h = game.headline or self.build_game_headline(game)

        # 1. Local LM Studio Server (Only active when running locally)
        if is_local and mode in ("lm-studio", "local-llm", "local", "auto"):
            is_healthy, detected_model, alarm_err = self.check_lm_studio_health()
            if is_healthy and detected_model:
                try:
                    logger.info("Generating game recap via local LM Studio model (%s)...", detected_model)
                    raw_output = self._generate_game_narrative_with_lm_studio(game, detected_model)
                    llm_headline, narrative = extract_headline_and_narrative(raw_output, default_h, game.period_type)
                    if self._verify_game_narrative_facts(narrative, game, headline=llm_headline):
                        game.headline = llm_headline
                        model_stamp = f"lm-studio/{detected_model}"
                        return narrative, model_stamp

                    logger.info("LM Studio first attempt failed verification checks. Retrying once with focused temperature (0.2)...")
                    raw_retry = self._generate_game_narrative_with_lm_studio(game, detected_model, temperature=0.2, max_tokens=1800)
                    retry_h, retry_narrative = extract_headline_and_narrative(raw_retry, default_h, game.period_type)
                    if self._verify_game_narrative_facts(retry_narrative, game, headline=retry_h):
                        game.headline = retry_h
                        model_stamp = f"lm-studio/{detected_model}"
                        return retry_narrative, model_stamp

                    logger.warning("LM Studio output failed factual or reasoning verification checks after retry. Using algorithmic fallback.")
                except Exception as e:
                    self._trigger_alarm(f"LM Studio generation error: {e}")
            elif mode in ("lm-studio", "local-llm", "local"):
                logger.warning("LM Studio unavailable. Using algorithmic fallback.")

        # 2. Remote Hugging Face Space Runtime / Router LLM (When deployed remotely)
        if not is_local:
            # When running on Hugging Face Spaces, ping the Space host
            is_space_up, space_name, space_note = self.check_hf_space_health()
            logger.info("HF Space Status [%s]: %s (local_streamlit=False)", space_name, space_note)

            # 2a. Priority: In-process ZeroGPU inside Hugging Face Space
            if self._can_use_zerogpu():
                target_model = resolve_zerogpu_model(self.zerogpu_model_id)
                try:
                    logger.info("Generating game recap via Space ZeroGPU (%s)...", target_model)
                    raw_output = self._generate_game_narrative_with_zerogpu(game, target_model)
                    llm_headline, narrative = extract_headline_and_narrative(raw_output, default_h, game.period_type)
                    if self._verify_game_narrative_facts(narrative, game, headline=llm_headline):
                        game.headline = llm_headline
                        model_stamp = f"hf-space/{target_model}"
                        return narrative.strip(), model_stamp
                    logger.warning("ZeroGPU output failed verification. Retrying once with focused temperature (0.2)...")
                    raw_retry = self._generate_game_narrative_with_zerogpu(game, target_model, temperature=0.2)
                    retry_h, retry_n = extract_headline_and_narrative(raw_retry, default_h, game.period_type)
                    if self._verify_game_narrative_facts(retry_n, game, headline=retry_h):
                        game.headline = retry_h
                        return retry_n.strip(), f"hf-space/{target_model}"
                except Exception as e:
                    logger.warning("Space ZeroGPU narrative generation failed: %s", e)

            # Check if custom OpenAI-compatible endpoint is configured (e.g. HF_LLM_URL or OPENAI_BASE_URL)
            custom_endpoint = self.hf_llm_endpoint or os.environ.get("HF_LLM_URL") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL")
            if custom_endpoint:
                try:
                    logger.info("Attempting narrative generation via remote endpoint (%s)...", custom_endpoint)
                    raw_output = self._generate_game_narrative_with_custom_endpoint(game, custom_endpoint)
                    llm_headline, narrative = extract_headline_and_narrative(raw_output, default_h, game.period_type)
                    if self._verify_game_narrative_facts(narrative, game, headline=llm_headline):
                        game.headline = llm_headline
                        return narrative.strip(), f"remote-endpoint/{self.llm_model_id}"
                except Exception as e:
                    logger.warning("Remote endpoint generation failed: %s", e)

            if mode in ("auto", "llm", "router-llm"):
                client = self._get_inference_client()
                if client:
                    try:
                        logger.info("Attempting LLM narrative generation via %s...", self.llm_model_id)
                        raw_output = self._generate_game_narrative_with_llm(game, client)
                        llm_headline, narrative = extract_headline_and_narrative(raw_output, default_h, game.period_type)
                        if self._verify_game_narrative_facts(narrative, game, headline=llm_headline):
                            game.headline = llm_headline
                            return narrative.strip(), self.llm_model_id
                        logger.warning("Router LLM failed factual verification checks. Using algorithmic fallback.")
                    except Exception as e:
                        err_str = str(e)
                        if "402" in err_str or "Payment Required" in err_str or "credits" in err_str.lower():
                            logger.warning("Hugging Face Router LLM requires pre-paid credits (HTTP 402). Falling back to Algorithmic Guardrail Engine.")
                        else:
                            logger.warning("LLM inference call failed: %s. Using algorithmic fallback.", e)

        # 3. Explicit Remote Router LLM in local mode if requested
        elif mode in ("llm", "router-llm"):
            client = self._get_inference_client()
            if client:
                try:
                    logger.info("Attempting LLM narrative generation via %s...", self.llm_model_id)
                    raw_output = self._generate_game_narrative_with_llm(game, client)
                    llm_headline, narrative = extract_headline_and_narrative(raw_output, default_h, game.period_type)
                    if self._verify_game_narrative_facts(narrative, game, headline=llm_headline):
                        game.headline = llm_headline
                        return narrative.strip(), self.llm_model_id
                    logger.warning("Router LLM failed factual verification checks. Using algorithmic fallback.")
                except Exception as e:
                    logger.warning("LLM inference call failed: %s. Using algorithmic fallback.", e)

        # 4. Algorithmic Guardrail Engine (Deterministic, instant, 100% mathematically reconciled)
        if not game.headline:
            game.headline = self.build_game_headline(game)
        narrative = self.build_game_narrative_algorithmic(game)
        if not is_local:
            target_model = resolve_zerogpu_model(self.zerogpu_model_id)
            return narrative.strip(), f"hf-space/{target_model}"
        return narrative.strip(), ALGORITHMIC_MODEL_STAMP

    # =========================================================================
    # 5. Full League Slate Review Generator
    # =========================================================================

    def build_slate_llm_system_prompt(self) -> str:
        """System instructions for the league-wide night-in-review sports editor."""
        return (
            "You are an executive NHL sports editor writing 'The NHL Night in Review'.\n"
            "Your task is to write a headline and compelling 3-paragraph executive overview summarizing the entire evening across the league.\n"
            "Highlight the collective goal scoring, standouts from the Three Stars, walk-off overtime thrillers, and key surprises.\n\n"
            "CRITICAL FORMAT MANDATE:\n"
            "1. Line 1 MUST be the executive headline formatted exactly as: HEADLINE: <Executive headline under 15 words>\n"
            "2. Follow immediately with the 3 paragraphs separated by blank lines.\n"
            "3. Do NOT include any chain-of-thought, reasoning monologue, planning notes, or intro markdown headings. Start directly with Line 1.\n"
            "4. Stick strictly to the audited numbers provided. Do not hallucinate.\n"
            "5. STRICT POINTS VS. GOALS MANDATE: Do not confuse points with goals. If a player recorded assists but zero goals (e.g. 0G, 3A), state that they recorded assists or points, but never claim they scored a goal or 'factored into goals'."
        )

    def build_slate_llm_user_prompt(self, slate: AnalyzedSlate) -> str:
        """Step-by-step plain-language facts for the entire slate."""
        stars_text = "\n".join([
            f"- Star #{s.star}: {s.name} ({s.team_abbrev} - {s.position}) -> {s.stat_summary}"
            for s in slate.slate_three_stars
        ])
        games_text = "\n".join([
            f"- {g.away_team.team_abbrev} {g.away_team.score} @ {g.home_team.team_abbrev} {g.home_team.score} ({'Final' if g.period_type == 'REG' else ('Final/OT' if g.period_type == 'OT' else 'Final/SO')}): {g.headline}"
            for g in slate.games
        ])
        notables_text = "\n".join([
            f"- {format_milestone_badge(n.category, n.title, n.description)}"
            for n in slate.slate_notables[:5]
        ])

        return (
            f"Write a headline and 3-paragraph executive overview of the NHL action on {slate.date}.\n\n"
            f"MACRO METRICS:\n"
            f"- Total Games: {slate.total_games}\n"
            f"- Total Goals: {slate.total_goals} (League average: {slate.avg_goals_per_game:.1f} per game)\n"
            f"- Home Record: {slate.home_wins} Wins vs {slate.away_wins} Road Wins\n"
            f"- Overtime/Shootouts: {slate.ot_so_games} matches\n\n"
            f"THREE STARS OF THE NIGHT:\n{stars_text}\n\n"
            f"KEY NOTABLES & MILESTONES:\n{notables_text}\n\n"
            f"MATCHUPS REVIEW:\n{games_text}\n\n"
            "FINAL DIRECTIVE: Start immediately with HEADLINE: <headline> on the first line, followed by exactly 3 polished paragraphs without preamble or reasoning."
        )

    def _generate_slate_narrative_with_llm(self, slate: AnalyzedSlate, client: InferenceClient) -> str:
        """Invokes Meta-Llama-3.2-3B-Instruct to draft the slate overview."""
        system_prompt = self.build_slate_llm_system_prompt()
        user_prompt = self.build_slate_llm_user_prompt(slate)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = client.chat.completions.create(
            messages=messages,
            max_tokens=1200,
            temperature=0.7,
        )
        content = response.choices[0].message.content or ""
        cleaned = clean_llm_narrative(content)
        return cleaned.strip()

    def _generate_slate_narrative_with_custom_endpoint(self, slate: AnalyzedSlate, endpoint_url: str) -> str:
        """Invokes custom remote endpoint for slate review."""
        system_prompt = self.build_slate_llm_system_prompt()
        user_prompt = self.build_slate_llm_user_prompt(slate)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        url = f"{endpoint_url.rstrip('/')}/chat/completions"
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY") or self.token or "none"
        headers = {"Authorization": f"Bearer {api_key}"}
        payload = {
            "model": self.llm_model_id,
            "messages": messages,
            "max_tokens": 1200,
            "temperature": 0.3,
        }

        resp = requests.post(url, headers=headers, json=payload, timeout=60.0)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        content = choice["message"].get("content") or ""
        cleaned = clean_llm_narrative(content)
        return cleaned.strip()

    def _generate_slate_narrative_with_lm_studio(self, slate: AnalyzedSlate, model_name: str) -> str:
        """Invokes local LM Studio instance for slate review without leaking reasoning."""
        system_prompt = self.build_slate_llm_system_prompt()
        user_prompt = self.build_slate_llm_user_prompt(slate)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        url = f"{self.lm_studio_url}/chat/completions"
        payload = {
            "model": model_name,
            "messages": messages,
            "max_tokens": 1200,
            "temperature": 0.3,
        }

        resp = requests.post(url, json=payload, timeout=180.0)
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        content = choice["message"].get("content") or ""
        # NEVER fall back to reasoning_content
        cleaned = clean_llm_narrative(content)
        return cleaned.strip()

    def build_slate_narrative_algorithmic(self, slate: AnalyzedSlate) -> str:
        """Algorithmic executive summary across the full evening slate."""
        p1 = (
            f"A high-octane {slate.total_games}-game slate across the National Hockey League on {slate.date} "
            f"produced a combined **{slate.total_goals} goals**, averaging {slate.avg_goals_per_game:.1f} markers per contest. "
            f"Road clubs captured the territorial advantage, claiming {slate.away_wins} victories compared to {slate.home_wins} for home squads. "
            f"Extra time was required in {slate.ot_so_games} matchups to determine a victor."
        )

        p2 = ""
        if slate.slate_three_stars:
            star1 = slate.slate_three_stars[0]
            p2 = (
                f"Headline individual honours belonged to **{star1.name}** ({star1.team_abbrev}), who earned the #1 Star of the Night "
                f"behind a dynamic performance ({star1.stat_summary}). "
            )
            if len(slate.slate_three_stars) > 1:
                star2 = slate.slate_three_stars[1]
                p2 += f"Not far behind was **{star2.name}** ({star2.team_abbrev}), earning second-star recognition ({star2.stat_summary})."

        p3 = ""
        if slate.slate_notables:
            top_notable = slate.slate_notables[0]
            p3 = f"Dramatic finishes punctuated the schedule, highlighted by {top_notable.description}"

        return "\n\n".join(filter(None, [p1, p2, p3]))

    def build_slate_narrative(
        self,
        slate: AnalyzedSlate,
        force_mode: Optional[str] = None,
    ) -> Tuple[str, str]:
        """Dispatches slate narrative generation to LM Studio, Hugging Face Space, or Algorithmic engine."""
        mode = force_mode or self.model_mode
        is_local = is_local_streamlit()
        default_h = getattr(slate, "executive_headline", "") or f"NHL Night in Review: {slate.total_games} Games, {slate.total_goals} Goals on {slate.date}"

        # 1. Local LM Studio Server
        if is_local and mode in ("lm-studio", "local-llm", "local", "auto"):
            is_healthy, detected_model, alarm_err = self.check_lm_studio_health()
            if is_healthy and detected_model:
                try:
                    logger.info("Generating slate overview via local LM Studio model (%s)...", detected_model)
                    raw_output = self._generate_slate_narrative_with_lm_studio(slate, detected_model)
                    llm_headline, narrative = extract_headline_and_narrative(raw_output, default_h)
                    if self._verify_slate_narrative_facts(narrative, slate):
                        slate.executive_headline = llm_headline
                        return narrative, f"lm-studio/{detected_model}"

                    logger.info("LM Studio first attempt failed slate verification checks. Retrying once with focused temperature (0.2)...")
                    raw_retry = self._generate_slate_narrative_with_lm_studio(slate, detected_model)
                    retry_h, retry_narrative = extract_headline_and_narrative(raw_retry, default_h)
                    if self._verify_slate_narrative_facts(retry_narrative, slate):
                        slate.executive_headline = retry_h
                        return retry_narrative, f"lm-studio/{detected_model}"

                    logger.warning("LM Studio output failed slate verification checks after retry. Using algorithmic fallback.")
                except Exception as e:
                    self._trigger_alarm(f"LM Studio slate generation error: {e}")
            elif mode in ("lm-studio", "local-llm", "local"):
                logger.warning("LM Studio unavailable for slate. Using algorithmic fallback.")

        # 2. Remote / Hugging Face Space Runtime
        if not is_local:
            is_space_up, space_name, space_note = self.check_hf_space_health()
            logger.info("HF Space Status [%s]: %s (local_streamlit=False)", space_name, space_note)

            # 2a. Priority: In-process ZeroGPU inside Hugging Face Space
            if self._can_use_zerogpu():
                target_model = resolve_zerogpu_model(self.zerogpu_model_id)
                try:
                    logger.info("Generating slate overview via Space ZeroGPU (%s)...", target_model)
                    raw_output = self._generate_slate_narrative_with_zerogpu(slate, target_model)
                    llm_headline, narrative = extract_headline_and_narrative(raw_output, default_h)
                    if self._verify_slate_narrative_facts(narrative, slate):
                        slate.executive_headline = llm_headline
                        return narrative.strip(), f"hf-space/{target_model}"
                    logger.warning("ZeroGPU slate output failed verification. Retrying once with focused temperature (0.2)...")
                    raw_retry = self._generate_slate_narrative_with_zerogpu(slate, target_model, temperature=0.2)
                    retry_h, retry_n = extract_headline_and_narrative(raw_retry, default_h)
                    if self._verify_slate_narrative_facts(retry_n, slate):
                        slate.executive_headline = retry_h
                        return retry_n.strip(), f"hf-space/{target_model}"
                except Exception as e:
                    logger.warning("Space ZeroGPU slate generation failed: %s", e)

            custom_endpoint = self.hf_llm_endpoint or os.environ.get("HF_LLM_URL") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL")
            if custom_endpoint:
                try:
                    logger.info("Attempting slate generation via remote endpoint (%s)...", custom_endpoint)
                    raw_output = self._generate_slate_narrative_with_custom_endpoint(slate, custom_endpoint)
                    llm_headline, narrative = extract_headline_and_narrative(raw_output, default_h)
                    if self._verify_slate_narrative_facts(narrative, slate):
                        slate.executive_headline = llm_headline
                        return narrative, f"remote-endpoint/{self.llm_model_id}"
                except Exception as e:
                    logger.warning("Remote endpoint slate generation failed: %s", e)

            if mode in ("auto", "llm", "router-llm"):
                client = self._get_inference_client()
                if client:
                    try:
                        logger.info("Attempting slate LLM generation via %s...", self.llm_model_id)
                        raw_output = self._generate_slate_narrative_with_llm(slate, client)
                        llm_headline, narrative = extract_headline_and_narrative(raw_output, default_h)
                        if self._verify_slate_narrative_facts(narrative, slate):
                            slate.executive_headline = llm_headline
                            return narrative.strip(), self.llm_model_id
                        logger.warning("Router LLM failed slate factual checks. Using algorithmic fallback.")
                    except Exception as e:
                        err_str = str(e)
                        if "402" in err_str or "Payment Required" in err_str or "credits" in err_str.lower():
                            logger.warning("Hugging Face Router LLM requires pre-paid credits (HTTP 402). Falling back to Algorithmic Guardrail Engine.")
                        else:
                            logger.warning("Slate LLM generation failed: %s. Using algorithmic fallback.", e)

        # 3. Explicit Remote Router LLM in local mode if requested
        elif mode in ("llm", "router-llm"):
            client = self._get_inference_client()
            if client:
                try:
                    logger.info("Attempting slate LLM generation via %s...", self.llm_model_id)
                    raw_output = self._generate_slate_narrative_with_llm(slate, client)
                    llm_headline, narrative = extract_headline_and_narrative(raw_output, default_h)
                    if self._verify_slate_narrative_facts(narrative, slate):
                        slate.executive_headline = llm_headline
                        return narrative.strip(), self.llm_model_id
                    logger.warning("Router LLM failed slate factual checks. Using algorithmic fallback.")
                except Exception as e:
                    logger.warning("Slate LLM generation failed: %s. Using algorithmic fallback.", e)

        # 4. Algorithmic Guardrail Engine (Always guaranteed to succeed)
        if not getattr(slate, "executive_headline", ""):
            slate.executive_headline = default_h
        narrative = self.build_slate_narrative_algorithmic(slate)
        if not is_local:
            target_model = resolve_zerogpu_model(self.zerogpu_model_id)
            return narrative.strip(), f"hf-space/{target_model}"
        return narrative.strip(), ALGORITHMIC_MODEL_STAMP

    # =========================================================================
    # 6. Publication Markdown Report Compilers
    # =========================================================================

    def format_narrative_engine_badge(self, model_stamp: str) -> str:
        """Returns clean human-readable badge for the active narrative engine."""
        m = str(model_stamp).lower()
        if "lm-studio" in m:
            clean_name = model_stamp.replace("lm-studio/", "").split("/")[-1]
            return f"LM Studio ({clean_name})"
        elif "llama" in m:
            return "Meta-Llama-3.2-3B (Router)"
        elif "qwen" in m or "local-cpu" in m:
            return f"Local LLM ({model_stamp})"
        else:
            return "Algorithmic Engine"

    def generate_single_game_markdown(self, game: AnalyzedGame) -> str:
        """Render a single game into a publication-ready markdown document with all tables."""
        headline = game.headline or self.build_game_headline(game)
        narrative = game.recap_narrative
        if not narrative:
            narrative, model_stamp = self.build_game_narrative(game)
            game.recap_narrative = narrative
            game.generator_model = model_stamp

        model_display = self.format_narrative_engine_badge(game.generator_model)
        dec = "Final" if game.period_type == "REG" else ("Final/OT" if game.period_type == "OT" else "Final/SO")

        md = []
        md.append(f"# 🏒 {headline}")
        md.append(f"**Date:** {game.game_date} | **Venue:** {game.venue} | **Final:** {game.away_team.team_abbrev} {game.away_team.score} @ {game.home_team.team_abbrev} {game.home_team.score} ({dec})")
        md.append(f"`🛡️ Data Quality: Reconciled & Audited` | `🤖 Narrative Generator: {model_display}`")
        md.append("---")
        md.append("## 📰 Game Recap")
        md.append(narrative)
        md.append("")

        # Three Stars of the Game
        if game.three_stars:
            md.append("## ⭐ Three Stars of the Game")
            star_emojis = ["🥇", "🥈", "🥉"]
            for idx, star in enumerate(game.three_stars):
                badge = star_emojis[idx] if idx < len(star_emojis) else "⭐"
                md.append(f"- {badge} **#{star.star}. {star.name}** ({star.team_abbrev} - {star.position}) — *{star.stat_summary}*")
            md.append("")

        # Goaltender Duel Table
        w_g = game.winning_goalie_profile
        l_g = game.losing_goalie_profile
        if w_g or l_g:
            md.append("## 🥅 Goaltending Performance")
            md.append("| Goaltender | Team | Decision | Saves / Shots | SV% | Rating | Assessment |")
            md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :--- |")
            for g in [w_g, l_g]:
                if g:
                    r_clean = str(g.performance_rating).replace("_", " ").title()
                    md.append(f"| **{g.name}** | {g.team_abbrev} | {g.decision} | {g.saves}/{g.shots_against} | {g.save_pct*100:.1f}% | {r_clean} | {g.assessment} |")
            md.append("")

        # Defensive Effort & Physicality Table
        w_def = game.home_defense_profile if game.winner.team_abbrev == game.home_team.team_abbrev else game.away_defense_profile
        l_def = game.away_defense_profile if game.winner.team_abbrev == game.home_team.team_abbrev else game.home_defense_profile
        if w_def and l_def:
            w_blk = str(w_def.blocks_rating).replace("_", " ").title()
            l_blk = str(l_def.blocks_rating).replace("_", " ").title()
            w_hit = str(w_def.hits_rating).replace("_", " ").title()
            l_hit = str(l_def.hits_rating).replace("_", " ").title()
            w_sup = str(w_def.shot_suppression).replace("_", " ").title()
            l_sup = str(l_def.shot_suppression).replace("_", " ").title()
            md.append("## 🛡️ Defensive Effort & Physicality Comparison")
            md.append("| Metric | " + game.winner.team_abbrev + " (Winner) | " + game.loser.team_abbrev + " (Loser) | Rating / Advantage |")
            md.append("| :--- | :---: | :---: | :--- |")
            md.append(f"| **Blocked Shots** | {w_def.blocked_shots} ({w_blk}) | {l_def.blocked_shots} ({l_blk}) | {'+' + str(w_def.blocked_shots - l_def.blocked_shots) + ' ' + game.winner.team_abbrev if w_def.blocked_shots >= l_def.blocked_shots else '+' + str(l_def.blocked_shots - w_def.blocked_shots) + ' ' + game.loser.team_abbrev} |")
            md.append(f"| **Body Checks (Hits)** | {w_def.hits} ({w_hit}) | {l_def.hits} ({l_hit}) | {'+' + str(w_def.hits - l_def.hits) + ' ' + game.winner.team_abbrev if w_def.hits >= l_def.hits else '+' + str(l_def.hits - w_def.hits) + ' ' + game.loser.team_abbrev} |")
            md.append(f"| **Giveaways / Takeaways** | {w_def.giveaways} / {w_def.takeaways} (Diff {w_def.turnover_diff:+d}) | {l_def.giveaways} / {l_def.takeaways} (Diff {l_def.turnover_diff:+d}) | Turnover edge: {'+' if w_def.turnover_diff >= l_def.turnover_diff else ''}{w_def.turnover_diff - l_def.turnover_diff} |")
            md.append(f"| **Shots Allowed** | {w_def.shots_allowed} ({w_sup}) | {l_def.shots_allowed} ({l_sup}) | Shot suppression: {game.winner.team_abbrev if w_def.shots_allowed <= l_def.shots_allowed else game.loser.team_abbrev} |")
            md.append("")

        # Season Standings & Pythagorean Analytics Table
        w_ctx = game.home_season_context if game.winner.team_abbrev == game.home_team.team_abbrev else game.away_season_context
        l_ctx = game.away_season_context if game.winner.team_abbrev == game.home_team.team_abbrev else game.home_season_context
        if w_ctx and l_ctx:
            md.append("## 📈 Season Standings & Pythagorean Analytics")
            md.append("| Team | GP | Record | PTS | PTS% | GF | GA | Goal Diff | Pyth Exp % | Luck/Variance | Stage |")
            md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
            for ctx in [w_ctx, l_ctx]:
                stg = str(ctx.season_stage).replace("_", " ").title()
                md.append(f"| **{ctx.team_abbrev}** | {ctx.games_played} | {ctx.wins}-{ctx.losses}-{ctx.ot_losses} | {ctx.points} | {ctx.points_pct*100:.1f}% | {ctx.goals_for} | {ctx.goals_against} | {ctx.goal_diff:+d} | {ctx.pythagorean_pct*100:.1f}% | `{ctx.pythagorean_diff*100:+.1f}%` | {stg} |")
            md.append("")

        # Standout Milestones & Highlights
        if getattr(game, "notables", None):
            md.append("## ⚡ Standout Milestones & Highlights")
            for n in game.notables:
                md.append(f"- {format_milestone_badge(n.category, n.title, n.description)}")
            md.append("")

        return "\n".join(md)

    def generate_slate_markdown(self, slate: AnalyzedSlate) -> str:
        """Render a full day's slate into an executive summary markdown document."""
        model_display = self.format_narrative_engine_badge(getattr(slate, "generator_model", ""))

        md = []
        md.append(f"# 🏒 NHL Slate Night in Review — {slate.date}")
        md.append(f"**Total Games:** {slate.total_games} | **Total Goals:** {slate.total_goals} (Avg {slate.avg_goals_per_game:.1f}) | **OT/SO Thrillers:** {slate.ot_so_games}")
        md.append(f"**Outcomes:** {slate.home_wins} Home Wins — {slate.away_wins} Road Victories")
        md.append(f"`🛡️ Data Quality: Audited` | `🤖 Editorial Generator: {model_display}`")
        md.append("---")

        if slate.executive_summary:
            md.append("## 🌟 Executive Slate Summary")
            md.append(slate.executive_summary)
            md.append("")

        # League Three Stars of the Night
        if slate.slate_three_stars:
            md.append("## ⭐ League Three Stars of the Night")
            star_emojis = ["🥇", "🥈", "🥉"]
            for idx, star in enumerate(slate.slate_three_stars):
                badge = star_emojis[idx] if idx < len(star_emojis) else "⭐"
                md.append(f"- {badge} **#{star.star} Star: {star.name}** ({star.team_abbrev} - {star.position}) — *{star.stat_summary}*")
            md.append("")

        # Top Performers Table
        if slate.top_performers:
            md.append("## 🏆 Top Skaters Across the League")
            md.append("| Player | Team | Pos | G | A | PTS | +/- | SOG | TOI |")
            md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
            for p in slate.top_performers:
                md.append(f"| **{p['name']}** | {p['team']} | {p['pos']} | {p['goals']} | {p['assists']} | **{p['points']}** | {p['plus_minus']:+d} | {p['sog']} | {p['toi']} |")
            md.append("")

        # Notable Highlights Across the League
        if slate.slate_notables:
            md.append("## ⚡ Standout Milestones & Dramatic Finishes")
            for n in slate.slate_notables[:8]:
                md.append(f"- {format_milestone_badge(n.category, n.title, n.description)}")
            md.append("")

        # Game by Game Scores
        md.append("## 📋 Final Scores & Game Links")
        md.append("| Away | Score | Home | Score | Dec | Key Storyline |")
        md.append("| :---: | :---: | :---: | :---: | :---: | :--- |")
        for g in slate.games:
            dec = "Final" if g.period_type == "REG" else ("Final/OT" if g.period_type == "OT" else "Final/SO")
            md.append(f"| {g.away_team.team_abbrev} | {g.away_team.score} | {g.home_team.team_abbrev} | {g.home_team.score} | `{dec}` | {g.headline} |")
        md.append("")

        return "\n".join(md)
