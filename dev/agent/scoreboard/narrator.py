"""narrator.py — Unified Agent Zamboni Sports Journalism & Narrative Recap Engine.

Synthesizes structured game and slate analyses into polished, broadcast-grade editorial recaps,
unified with `dev/zamboni/recap_engine.py`:
1. Context Engineering: Static Prefix-Cached System Prompts & Compact State Projection.
2. Strategic Deterministic Reflection Critic with Context Pruning via Diffing on retry.
3. Verified Game Scripts, xG/GSAx Shot Quality, 3F/2D Shift Chemistry & L5 Rolling Form.
4. Multi-Tier Compute Routing: Local LM Studio (`http://127.0.0.1:1234/v1`) -> Space ZeroGPU -> Deterministic Guardrail Fallback.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import requests

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from huggingface_hub import InferenceClient
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

try:
    from .models import AnalyzedGame, AnalyzedSlate
    from .env_utils import is_local_streamlit, discover_hf_token, discover_model_name
except (ImportError, ValueError):
    from models import AnalyzedGame, AnalyzedSlate
    from env_utils import is_local_streamlit, discover_hf_token, discover_model_name

from dev.zamboni.recap_engine import (  # noqa: E402,F401
    ALGORITHMIC_MODEL_STAMP,
    NHL_TEAM_NAMES,
    ZAMBONI_GAME_RECAP_SYSTEM_PROMPT,
    ZAMBONI_SLATE_REVIEW_SYSTEM_PROMPT,
    build_projected_game_prompt,
    build_projected_slate_prompt,
    build_zamboni_game_headline,
    build_zamboni_slate_headline,
    build_zamboni_slate_narrative_from_records,
    clean_llm_narrative,
    expand_first_mention_abbreviations,
    extract_headline_and_narrative,
    format_milestone_badge,
    verify_game_narrative_facts,
    verify_slate_narrative_facts,
)

logger = logging.getLogger("NarratorEngine")

DEFAULT_LLM_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
DEFAULT_LM_STUDIO_URL = "http://127.0.0.1:1234/v1"
DEFAULT_HF_SPACE_URL = "https://jjerlee-rinklytics.hf.space"
DEFAULT_ZEROGPU_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-FP8"


def resolve_zerogpu_model(model_override: Optional[str] = None) -> str:
    """Resolves active Space ZeroGPU model from user public Space variables, secrets, .env, or defaults."""
    if model_override and str(model_override).strip():
        return str(model_override).strip()
    return discover_model_name(DEFAULT_ZEROGPU_MODEL)


def _has_native_mamba_ssm() -> bool:
    """Checks whether compiled CUDA/Triton mamba_ssm kernels are installed in the current environment."""
    try:
        import mamba_ssm.ops.triton.layernorm_gated  # noqa: F401
        return True
    except Exception:
        return False


def _resolve_zerogpu_weights_repo(model_id: Optional[str] = None) -> Tuple[str, str]:
    """Maps configured Space model identifier to a transformers-compatible safetensors checkpoint."""
    display_model = resolve_zerogpu_model(model_id)
    low = display_model.lower()

    if "nemotron-3-nano-4b" in low:
        if _has_native_mamba_ssm():
            return "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16", display_model
        return "Qwen/Qwen2.5-3B-Instruct", display_model

    if low.endswith("-fp8") and "nemotron" in low:
        return "Qwen/Qwen2.5-3B-Instruct", display_model

    return display_model, display_model


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
_ZEROGPU_SNAPSHOT_DIRS: dict = {}
_ZEROGPU_STARTUP_REPORTED: bool = False


def _pre_download_zerogpu_model(model_id: Optional[str] = None) -> str:
    """Pre-downloads model weights on CPU outside @spaces.GPU so ZeroGPU quota is never spent on network I/O."""
    global _ZEROGPU_SNAPSHOT_DIRS
    weights_repo, _ = _resolve_zerogpu_weights_repo(model_id)
    if weights_repo in _ZEROGPU_SNAPSHOT_DIRS and os.path.isdir(_ZEROGPU_SNAPSHOT_DIRS[weights_repo]):
        return _ZEROGPU_SNAPSHOT_DIRS[weights_repo]

    if not HF_HUB_AVAILABLE:
        return weights_repo

    try:
        from huggingface_hub import snapshot_download
        token = discover_hf_token()
        logger.info("Pre-caching model snapshot on CPU for ZeroGPU: %s...", weights_repo)
        local_dir = snapshot_download(
            repo_id=weights_repo,
            token=token,
            ignore_patterns=["*.msgpack", "*.h5", "consolidated*"],
        )
        _ZEROGPU_SNAPSHOT_DIRS[weights_repo] = local_dir
        return local_dir
    except Exception as e:
        logger.warning("Snapshot pre-download warning for %s: %s", weights_repo, e)
        return weights_repo


def _prepare_zerogpu_messages(messages: list, weights_repo: str) -> list:
    """Prepares chat messages for fast non-reasoning inference on Nemotron / Instruct models."""
    prepared = []
    is_nemotron = "nemotron" in weights_repo.lower()
    for idx, msg in enumerate(messages):
        role = msg.get("role", "user")
        content = str(msg.get("content", ""))
        if idx == 0 and role == "system" and is_nemotron and not content.lower().startswith("detailed thinking off"):
            content = f"detailed thinking off\n\n{content}"
        prepared.append({"role": role, "content": content})
    return prepared


def _get_zerogpu_pipeline(model_id: Optional[str] = None, local_model_dir: Optional[str] = None):
    """Lazy loader for in-process ZeroGPU model on Hugging Face Spaces with dynamic model caching."""
    global _ZEROGPU_CACHE
    weights_repo, display_model = _resolve_zerogpu_weights_repo(model_id)
    load_source = local_model_dir or _ZEROGPU_SNAPSHOT_DIRS.get(weights_repo) or weights_repo

    if weights_repo not in _ZEROGPU_CACHE:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        logger.info(
            "Loading ZeroGPU model '%s' (configured: '%s') from '%s' onto %s...",
            weights_repo,
            display_model,
            load_source,
            device,
        )
        tokenizer = AutoTokenizer.from_pretrained(load_source, trust_remote_code=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            load_source,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
        model.eval()
        _ZEROGPU_CACHE[weights_repo] = (model, tokenizer, weights_repo, display_model)
    return _ZEROGPU_CACHE[weights_repo]


def _run_single_generation(
    model,
    tokenizer,
    weights_repo: str,
    messages: list,
    max_new_tokens: int = 1150,
    temperature: float = 0.35,
) -> str:
    """Executes a single prompt generation on an already-loaded ZeroGPU model."""
    import torch

    prepped_messages = _prepare_zerogpu_messages(messages, weights_repo)
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt = tokenizer.apply_chat_template(
                prepped_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except (TypeError, Exception):
            prompt = tokenizer.apply_chat_template(
                prepped_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in prepped_messages) + "\nassistant:"

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.08,
            do_sample=(temperature > 0.05),
            pad_token_id=pad_id,
            use_cache=True,
        )
    gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True)


@spaces.GPU(duration=120)
def _zerogpu_infer(
    messages: list,
    model_id: Optional[str] = None,
    max_new_tokens: int = 1150,
    temperature: float = 0.35,
    local_model_dir: Optional[str] = None,
) -> str:
    """Runs single generation inside Hugging Face Spaces ZeroGPU allocation."""
    model, tokenizer, weights_repo, _ = _get_zerogpu_pipeline(model_id, local_model_dir=local_model_dir)
    return _run_single_generation(
        model=model,
        tokenizer=tokenizer,
        weights_repo=weights_repo,
        messages=messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


@spaces.GPU(duration=120)
def _zerogpu_infer_batch(
    batch_messages: list,
    model_id: Optional[str] = None,
    max_new_tokens: int = 1150,
    temperature: float = 0.35,
    local_model_dir: Optional[str] = None,
) -> List[str]:
    """Runs a batch of prompts inside a single Hugging Face Spaces ZeroGPU allocation without reloading weights."""
    model, tokenizer, weights_repo, _ = _get_zerogpu_pipeline(model_id, local_model_dir=local_model_dir)
    outputs: List[str] = []
    for messages in batch_messages:
        text = _run_single_generation(
            model=model,
            tokenizer=tokenizer,
            weights_repo=weights_repo,
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        outputs.append(text)
    return outputs


def ensure_zerogpu_startup_report() -> bool:
    """Registers all decorated @spaces.GPU functions with the ZeroGPU device daemon."""
    global _ZEROGPU_STARTUP_REPORTED
    if _ZEROGPU_STARTUP_REPORTED:
        return True
    if not SPACES_AVAILABLE:
        return False
    try:
        from spaces.zero import client as zero_client
        zero_client.startup_report()
        _ZEROGPU_STARTUP_REPORTED = True
        logger.info("Successfully registered @spaces.GPU functions via spaces.zero.client.startup_report().")
        return True
    except Exception as e:
        logger.debug("ZeroGPU startup_report note: %s", e)
        return False


def _call_zerogpu_with_auth(gpu_fn, *args, **kwargs):
    """Invokes a @spaces.GPU function after ensuring startup_report and Gradio auth context."""
    ensure_zerogpu_startup_report()
    try:
        try:
            return gpu_fn(*args, **kwargs)
        except Exception as first_err:
            err_msg = str(first_err)
            hf_token = discover_hf_token()
            if hf_token:
                try:
                    from gradio.context import LocalContext

                    class _ZeroAuthHeaders(dict):
                        pass

                    class _ZeroAuthRequest:
                        pass

                    headers = _ZeroAuthHeaders({"x-ip-token": hf_token, " authorization": f"Bearer {hf_token}"})
                    headers.__dict__["x-ip-token"] = hf_token
                    req = _ZeroAuthRequest()
                    req.headers = headers
                    req.client = type("_Client", (), {"host": "127.0.0.1"})()

                    prev_req = LocalContext.request.get(None)
                    LocalContext.request.set(req)
                    try:
                        return gpu_fn(*args, **kwargs)
                    finally:
                        LocalContext.request.set(prev_req)
                except Exception as second_err:
                    logger.warning("ZeroGPU fallback auth attempt failed (%s); original error: %s", second_err, err_msg)
            raise first_err
    finally:
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


if SPACES_AVAILABLE and (os.environ.get("SPACES_ZERO_GPU") or os.environ.get("SPACE_ID")):
    ensure_zerogpu_startup_report()


class NarratorEngine:
    """Unified Agent Zamboni narrative engine for NHL game recaps and league slate reviews."""

    def __init__(
        self,
        model_mode: str = "auto",
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

        env_space_host = os.environ.get("SPACE_HOST")
        derived_space_url = f"https://{env_space_host}" if env_space_host else DEFAULT_HF_SPACE_URL
        self.hf_space_url = os.environ.get("HF_SPACE_HOST") or os.environ.get("HF_SPACE_URL") or hf_space_url or derived_space_url
        self.hf_llm_endpoint = os.environ.get("HF_LLM_URL")
        self.zerogpu_model_id = resolve_zerogpu_model()

        self.token = discover_hf_token(token)
        self._inference_client = None
        self._active_lm_studio_model = None
        self.last_zerogpu_error: Optional[str] = None

    def _get_inference_client(self) -> Optional[InferenceClient]:
        if self._inference_client is not None:
            return self._inference_client
        if not HF_HUB_AVAILABLE:
            return None
        token = discover_hf_token(self.token)
        if not token:
            return None
        try:
            self._inference_client = InferenceClient(model=self.llm_model_id, token=token, timeout=45.0)
            return self._inference_client
        except Exception as e:
            logger.warning("Failed to initialize Hugging Face InferenceClient: %s", e)
            return None

    def check_lm_studio_health(self) -> Tuple[bool, Optional[str], Optional[str]]:
        """Verify connectivity to the local LM Studio instance at http://127.0.0.1:1234."""
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
                resp = requests.get(f"{base_url}/models", timeout=3.0)
                if resp.status_code == 200:
                    self.lm_studio_url = base_url
                    data = resp.json()
                    raw_models = [m.get("id") for m in data.get("data", []) if m.get("id")]
                    chat_models = [m for m in raw_models if "embed" not in m.lower()]
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
        space_url = self.hf_space_url.rstrip("/")
        try:
            resp = requests.get(space_url, timeout=5.0)
            if resp.status_code in (200, 301, 302):
                return True, "jjerlee/rinklytics", "Space host reachable"
            return False, "jjerlee/rinklytics", f"Space returned HTTP status {resp.status_code}"
        except Exception as e:
            return False, "jjerlee/rinklytics", f"Unable to reach Space at {space_url}: {e}"

    def _trigger_alarm(self, reason: str):
        alarm_banner = (
            "\n" + "🚨" * 38 + "\n"
            f"🚨 [ALARM: LM STUDIO UNREACHABLE]\n"
            f"🚨 Details: {reason}\n"
            f"🚨 Action: Please verify LM Studio is running and the local server is started at {self.lm_studio_url}\n"
            f"🚨 Safeguard: Falling back to Algorithmic Guardrail Engine (by Zamboni).\n"
            + "🚨" * 38 + "\n"
        )
        logger.error(alarm_banner)
        sys.stderr.write(alarm_banner)
        sys.stderr.flush()

    def _verify_game_narrative_facts(self, narrative: str, game: AnalyzedGame, headline: str = "") -> bool:
        ok, reasons = verify_game_narrative_facts(narrative, game, headline=headline)
        if not ok:
            logger.warning("Zamboni Game Critic rejected narrative: %s", "; ".join(reasons))
        return ok

    def _verify_slate_narrative_facts(self, narrative: str, slate: AnalyzedSlate, headline: str = "") -> bool:
        ok, reasons = verify_slate_narrative_facts(narrative, slate, headline=headline)
        if not ok:
            logger.warning("Zamboni Slate Critic rejected overview: %s", "; ".join(reasons))
        return ok

    # =========================================================================
    # 1. Headline & Prompt Construction (Delegated to Zamboni Recap Engine)
    # =========================================================================

    def build_game_headline(self, game: AnalyzedGame) -> str:
        return build_zamboni_game_headline(game)

    def build_game_llm_system_prompt(self) -> str:
        return ZAMBONI_GAME_RECAP_SYSTEM_PROMPT

    def build_game_llm_user_prompt(self, game: AnalyzedGame, retry_violations: Optional[List[str]] = None) -> str:
        base_prompt = build_projected_game_prompt(game)
        if retry_violations:
            diff_note = (
                "\n\n[CRITICAL CORRECTION DIRECTIVE — FIX THESE VIOLATIONS FROM PRIOR DRAFT]:\n"
                + "\n".join(f"- {v}" for v in retry_violations)
            )
            return base_prompt + diff_note
        return base_prompt

    def build_slate_llm_system_prompt(self) -> str:
        return ZAMBONI_SLATE_REVIEW_SYSTEM_PROMPT

    def build_slate_llm_user_prompt(self, slate: AnalyzedSlate, retry_violations: Optional[List[str]] = None) -> str:
        base_prompt = build_projected_slate_prompt(slate)
        if retry_violations:
            diff_note = (
                "\n\n[CRITICAL CORRECTION DIRECTIVE — FIX THESE VIOLATIONS FROM PRIOR DRAFT]:\n"
                + "\n".join(f"- {v}" for v in retry_violations)
            )
            return base_prompt + diff_note
        return base_prompt

    # =========================================================================
    # 2. LLM Transport Helpers (LM Studio / ZeroGPU / Custom / Router)
    # =========================================================================

    def _can_use_zerogpu(self) -> bool:
        if SPACES_AVAILABLE:
            return True
        return bool(os.environ.get("SPACES_ZERO_GPU") or os.environ.get("SPACE_ID"))

    def _generate_game_narrative_with_lm_studio(
        self,
        game: AnalyzedGame,
        model_name: str,
        temperature: float = 0.30,
        max_tokens: int = 1600,
        retry_violations: Optional[List[str]] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": self.build_game_llm_system_prompt()},
            {"role": "user", "content": self.build_game_llm_user_prompt(game, retry_violations=retry_violations)},
        ]
        resp = requests.post(
            f"{self.lm_studio_url}/chat/completions",
            json={"model": model_name, "messages": messages, "max_tokens": max_tokens, "temperature": temperature},
            timeout=180.0,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"].get("content") or ""
        return clean_llm_narrative(content).strip()

    def _generate_game_narrative_with_zerogpu(
        self,
        game: AnalyzedGame,
        model_id: Optional[str] = None,
        temperature: float = 0.30,
        max_tokens: int = 1150,
        retry_violations: Optional[List[str]] = None,
    ) -> str:
        target_model = resolve_zerogpu_model(model_id or self.zerogpu_model_id)
        local_dir = _pre_download_zerogpu_model(target_model)
        messages = [
            {"role": "system", "content": self.build_game_llm_system_prompt()},
            {"role": "user", "content": self.build_game_llm_user_prompt(game, retry_violations=retry_violations)},
        ]
        raw = _call_zerogpu_with_auth(
            _zerogpu_infer,
            messages,
            model_id=target_model,
            max_new_tokens=max_tokens,
            temperature=temperature,
            local_model_dir=local_dir,
        )
        return clean_llm_narrative(raw).strip()

    def _generate_game_narrative_with_custom_endpoint(self, game: AnalyzedGame, endpoint_url: str) -> str:
        messages = [
            {"role": "system", "content": self.build_game_llm_system_prompt()},
            {"role": "user", "content": self.build_game_llm_user_prompt(game)},
        ]
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY") or self.token or "none"
        resp = requests.post(
            f"{endpoint_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": self.llm_model_id, "messages": messages, "max_tokens": 1500, "temperature": 0.30},
            timeout=60.0,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"].get("content") or ""
        return clean_llm_narrative(content).strip()

    def _generate_game_narrative_with_llm(self, game: AnalyzedGame, client: InferenceClient) -> str:
        messages = [
            {"role": "system", "content": self.build_game_llm_system_prompt()},
            {"role": "user", "content": self.build_game_llm_user_prompt(game)},
        ]
        response = client.chat.completions.create(messages=messages, max_tokens=1500, temperature=0.45)
        content = response.choices[0].message.content or ""
        return clean_llm_narrative(content).strip()

    def _generate_slate_narrative_with_lm_studio(
        self,
        slate: AnalyzedSlate,
        model_name: str,
        temperature: float = 0.25,
        retry_violations: Optional[List[str]] = None,
    ) -> str:
        messages = [
            {"role": "system", "content": self.build_slate_llm_system_prompt()},
            {"role": "user", "content": self.build_slate_llm_user_prompt(slate, retry_violations=retry_violations)},
        ]
        resp = requests.post(
            f"{self.lm_studio_url}/chat/completions",
            json={"model": model_name, "messages": messages, "max_tokens": 1100, "temperature": temperature},
            timeout=180.0,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"].get("content") or ""
        return clean_llm_narrative(content).strip()

    def _generate_slate_narrative_with_zerogpu(
        self,
        slate: AnalyzedSlate,
        model_id: Optional[str] = None,
        temperature: float = 0.25,
        max_tokens: int = 950,
        retry_violations: Optional[List[str]] = None,
    ) -> str:
        target_model = resolve_zerogpu_model(model_id or self.zerogpu_model_id)
        local_dir = _pre_download_zerogpu_model(target_model)
        messages = [
            {"role": "system", "content": self.build_slate_llm_system_prompt()},
            {"role": "user", "content": self.build_slate_llm_user_prompt(slate, retry_violations=retry_violations)},
        ]
        raw = _call_zerogpu_with_auth(
            _zerogpu_infer,
            messages,
            model_id=target_model,
            max_new_tokens=max_tokens,
            temperature=temperature,
            local_model_dir=local_dir,
        )
        return clean_llm_narrative(raw).strip()

    def _generate_slate_narrative_with_custom_endpoint(self, slate: AnalyzedSlate, endpoint_url: str) -> str:
        messages = [
            {"role": "system", "content": self.build_slate_llm_system_prompt()},
            {"role": "user", "content": self.build_slate_llm_user_prompt(slate)},
        ]
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("GROQ_API_KEY") or self.token or "none"
        resp = requests.post(
            f"{endpoint_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": self.llm_model_id, "messages": messages, "max_tokens": 1100, "temperature": 0.25},
            timeout=60.0,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"].get("content") or ""
        return clean_llm_narrative(content).strip()

    def _generate_slate_narrative_with_llm(self, slate: AnalyzedSlate, client: InferenceClient) -> str:
        messages = [
            {"role": "system", "content": self.build_slate_llm_system_prompt()},
            {"role": "user", "content": self.build_slate_llm_user_prompt(slate)},
        ]
        response = client.chat.completions.create(messages=messages, max_tokens=1100, temperature=0.45)
        content = response.choices[0].message.content or ""
        return clean_llm_narrative(content).strip()

    def populate_games_narratives_zerogpu_batch(
        self,
        games: List[AnalyzedGame],
        batch_size: int = 4,
    ) -> bool:
        """Synthesizes narratives for multiple games in batched ZeroGPU allocations."""
        if not games or not self._can_use_zerogpu():
            return False

        target_model = resolve_zerogpu_model(self.zerogpu_model_id)
        model_stamp = f"hf-space/{target_model}"
        local_dir = _pre_download_zerogpu_model(target_model)
        system_prompt = self.build_game_llm_system_prompt()

        pending_games: List[AnalyzedGame] = []
        pending_messages: List[list] = []
        for g in games:
            if not g.headline:
                g.headline = self.build_game_headline(g)
            u_prompt = self.build_game_llm_user_prompt(g)
            pending_games.append(g)
            pending_messages.append([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": u_prompt},
            ])

        any_succeeded = False
        retry_games: List[AnalyzedGame] = []
        retry_messages: List[list] = []

        for start_idx in range(0, len(pending_games), batch_size):
            chunk_games = pending_games[start_idx : start_idx + batch_size]
            chunk_msgs = pending_messages[start_idx : start_idx + batch_size]
            try:
                raw_outputs = _call_zerogpu_with_auth(
                    _zerogpu_infer_batch,
                    chunk_msgs,
                    model_id=target_model,
                    max_new_tokens=1150,
                    temperature=0.30,
                    local_model_dir=local_dir,
                )
                for g, msg, raw_out in zip(chunk_games, chunk_msgs, raw_outputs):
                    default_h = g.headline or self.build_game_headline(g)
                    llm_h, narr = extract_headline_and_narrative(clean_llm_narrative(raw_out), default_h, g.period_type)
                    ok, violations = verify_game_narrative_facts(narr, g, headline=llm_h)
                    if ok:
                        g.headline = llm_h
                        g.recap_narrative = narr.strip()
                        g.generator_model = model_stamp
                        g.full_report_markdown = self.generate_single_game_markdown(g)
                        any_succeeded = True
                    else:
                        retry_games.append(g)
                        retry_messages.append([
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": self.build_game_llm_user_prompt(g, retry_violations=violations)},
                        ])
            except Exception as e:
                self.last_zerogpu_error = f"{type(e).__name__}: {e}"
                logger.warning("ZeroGPU batch chunk failed: %s", e)
                for g, msg in zip(chunk_games, chunk_msgs):
                    retry_games.append(g)
                    retry_messages.append(msg)

        if retry_games and any_succeeded:
            for start_idx in range(0, len(retry_games), batch_size):
                r_games = retry_games[start_idx : start_idx + batch_size]
                r_msgs = retry_messages[start_idx : start_idx + batch_size]
                try:
                    raw_retries = _call_zerogpu_with_auth(
                        _zerogpu_infer_batch,
                        r_msgs,
                        model_id=target_model,
                        max_new_tokens=1150,
                        temperature=0.20,
                        local_model_dir=local_dir,
                    )
                    for g, raw_out in zip(r_games, raw_retries):
                        default_h = g.headline or self.build_game_headline(g)
                        llm_h, narr = extract_headline_and_narrative(clean_llm_narrative(raw_out), default_h, g.period_type)
                        if self._verify_game_narrative_facts(narr, g, headline=llm_h):
                            g.headline = llm_h
                            g.recap_narrative = narr.strip()
                            g.generator_model = model_stamp
                            g.full_report_markdown = self.generate_single_game_markdown(g)
                            any_succeeded = True
                except Exception as e:
                    logger.warning("ZeroGPU retry batch failed: %s", e)

        for g in games:
            if not getattr(g, "recap_narrative", ""):
                narr, stamp = self.build_game_narrative(g)
                g.recap_narrative = narr
                g.generator_model = stamp
                g.full_report_markdown = self.generate_single_game_markdown(g)

        return any_succeeded

    # =========================================================================
    # 3. Deterministic Algorithmic NLG Engine (by Zamboni)
    # =========================================================================

    def build_game_narrative_algorithmic(self, game: AnalyzedGame) -> str:
        """Deterministic 5-paragraph generator incorporating Line Shifts, xG/GSAx, L5 Trends, and Schedule Context."""
        paragraphs = []
        winner = game.winner
        loser = game.loser
        w_score = winner.score
        l_score = loser.score
        flow = game.period_flow
        w_is_home = (winner.team_abbrev == game.home_team.team_abbrev)
        w_tr = game.home_shift_trend_profile if w_is_home else game.away_shift_trend_profile
        l_tr = game.away_shift_trend_profile if w_is_home else game.home_shift_trend_profile

        ot_clause = f" in {game.period_type}" if game.period_type in ("OT", "SO") else ""
        star1 = game.three_stars[0] if game.three_stars else None

        venue_context = "on home ice" if w_is_home else "on the road"
        if w_tr:
            if not w_is_home and w_tr.consec_road_games >= 2:
                venue_context = f"on the road in game #{w_tr.consec_road_games} of their current trip"
            if w_tr.is_back_to_back:
                venue_context += " while navigating the second night of a back-to-back"
            elif w_tr.rest_advantage < 0:
                venue_context += f" despite a rest disadvantage ({w_tr.rest_days}d vs. {w_tr.opponent_rest_days}d)"

        script_desc = ""
        if flow:
            if flow.game_script_type == "WIRE_TO_WIRE":
                script_desc = f"The {winner.team_name} dictated terms from the opening period and never trailed, executing a composed wire-to-wire performance. "
            elif flow.game_script_type == "CONTROLLED_LATE_PUSH":
                script_desc = f"The {winner.team_name} built a multi-goal cushion through forty minutes before withstanding a late third-period push from the {loser.team_name}. "
            elif flow.game_script_type == "COMEBACK_WIN":
                script_desc = f"Trailing entering the final frame, the {winner.team_name} mounted a decisive third-period comeback to flip the script. "
            elif flow.game_script_type == "TRUE_THRILLER":
                script_desc = f"In a tight-checking battle featuring {flow.ties_count} ties and {flow.lead_changes} lead changes, the {winner.team_name} delivered the decisive late counter-punch. "
            elif flow.game_script_type == "WALK_OFF_OT_SO":
                script_desc = f"After sixty minutes of regulation left the clubs deadlocked, the {winner.team_name} secured the extra point in sudden-death fashion. "

        lede = (
            f"The **{winner.team_name}** defeated the **{loser.team_name}** {w_score}–{l_score}{ot_clause} {venue_context} "
            f"at {game.venue}. {script_desc}"
        )
        if star1:
            lede += f"**{star1.name}** ({star1.team_abbrev}) headlined the evening as the #1 Star ({star1.stat_summary})."
        paragraphs.append(lede.strip())

        # Paragraph 2: Scoring Progression
        if game.scoring_plays:
            first_goal = game.scoring_plays[0]
            last_goal = game.scoring_plays[-1]
            p_desc_low = first_goal.period_descriptor.lower()
            open_clause = (
                f"Scoring opened in {p_desc_low}"
                if p_desc_low in ("overtime", "shootout")
                else f"Scoring opened in the {p_desc_low}"
            )
            prog_text = (
                f"{open_clause} when **{first_goal.goal_scorer}** ({first_goal.scoring_team}) broke the ice at "
                f"{first_goal.time_in_period} ({first_goal.strength}). "
            )
            if flow:
                prog_text += (
                    f"The teams reached the first intermission at **{flow.score_after_p1}** and carried a "
                    f"**{flow.score_after_p2}** scoreline into the second intermission. "
                )
            if len(game.scoring_plays) > 2:
                middle_scorers = [sp.goal_scorer for sp in game.scoring_plays[1:-1]]
                unique_mid = list(dict.fromkeys(middle_scorers))[:4]
                if unique_mid:
                    prog_text += f"Intermediate tallies from {', '.join(unique_mid)} shaped the middle stretches of the contest. "
            if game.period_type in ("OT", "SO"):
                prog_text += (
                    f"With regulation unable to separate the clubs, **{last_goal.goal_scorer}** ({last_goal.scoring_team}) "
                    f"buried the walk-off winner at {last_goal.time_in_period} of {last_goal.period_descriptor}."
                )
            else:
                prog_text += (
                    f"**{last_goal.goal_scorer}** ({last_goal.scoring_team}) sealed the {w_score}–{l_score} final at "
                    f"{last_goal.time_in_period} of {last_goal.period_descriptor}."
                )
            paragraphs.append(prog_text.strip())

        # Paragraph 3: 5v5 Line Shift Chemistry & L5 Line Trends
        shift_parts = []
        if w_tr and w_tr.has_shift_data:
            if w_tr.new_lines_deployed_count > 0:
                shift_parts.append(
                    f"At five-on-five, {winner.team_abbrev} deployed {w_tr.new_lines_deployed_count} newly assembled or reshuffled forward combinations "
                    f"({w_tr.line_continuity_pct:.1f}% line continuity over the last five games (L5))."
                )
            else:
                shift_parts.append(
                    f"Five-on-five deployment saw {winner.team_abbrev} lean on established chemistry ({w_tr.line_continuity_pct:.1f}% of forward Time on Ice (TOI) played by regular L5 trios)."
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

        # Paragraph 4: Goaltending GSAx, Shot Quality (xG) & Defensive Sacrifice
        w_goalie = game.winning_goalie_profile
        l_goalie = game.losing_goalie_profile
        w_def = game.home_defense_profile if w_is_home else game.away_defense_profile
        l_def = game.away_defense_profile if w_is_home else game.home_defense_profile

        def_parts = []
        if w_goalie:
            def_parts.append(f"In the crease, **{w_goalie.name}** ({winner.team_abbrev}) {w_goalie.assessment[0].lower() + w_goalie.assessment[1:]}")
        if l_goalie and l_goalie.name != getattr(w_goalie, "name", ""):
            def_parts.append(f"Opposite him, **{l_goalie.name}** ({loser.team_abbrev}) {l_goalie.assessment[0].lower() + l_goalie.assessment[1:]}")
        if winner.xg > 0 or loser.xg > 0:
            def_parts.append(
                f"Expected Goals (xG) telemetry favored {winner.team_abbrev if winner.xg >= loser.xg else loser.team_abbrev}, "
                f"with {winner.team_abbrev} generating **{winner.xg:.2f} xG** ({winner.xg_pct:.1f}% share, {winner.hdcf} high-danger chances) "
                f"compared to {loser.team_abbrev}'s **{loser.xg:.2f} xG** ({loser.xg_pct:.1f}% share, {loser.hdcf} high-danger chances)."
            )
        if w_def and l_def:
            def_parts.append(
                f"Defensively, {winner.team_abbrev} recorded **{w_def.blocked_shots}** blocked shots and **{w_def.hits}** hits "
                f"({w_def.turnover_diff:+d} turnover differential), while {loser.team_abbrev} logged **{l_def.blocked_shots}** blocks "
                f"and **{l_def.hits}** hits ({l_def.turnover_diff:+d} turnover differential)."
            )
        pp_text = []
        if winner.power_play_conv and winner.power_play_conv != "0/0":
            pp_text.append(f"{winner.team_abbrev} went {winner.power_play_conv} ({winner.power_play_pct:.1f}%) on the power play")
        if loser.power_play_conv and loser.power_play_conv != "0/0":
            pp_text.append(f"{loser.team_abbrev} finished {loser.power_play_conv} with the man advantage")
        if pp_text:
            def_parts.append("On special teams, " + " while ".join(pp_text) + ".")
        paragraphs.append(" ".join(def_parts).strip())

        # Paragraph 5: Season Standings, Pythagorean Expectation & L5 Rolling Form
        w_ctx = game.home_season_context if w_is_home else game.away_season_context
        l_ctx = game.away_season_context if w_is_home else game.home_season_context
        if w_ctx and l_ctx:
            rec_line = (
                f"{w_ctx.stage_narrative} The victory moves the **{winner.team_name}** to **{w_ctx.wins}–{w_ctx.losses}–{w_ctx.ot_losses}** "
                f"({w_ctx.points} points, {w_ctx.points_pct*100:.1f}% points share, {w_ctx.pythagorean_pct*100:.1f}% Pythagorean expectation), "
                f"while the **{loser.team_name}** stand at **{l_ctx.wins}–{l_ctx.losses}–{l_ctx.ot_losses}** "
                f"({l_ctx.points} points, {l_ctx.points_pct*100:.1f}% points share, {l_ctx.pythagorean_pct*100:.1f}% Pythagorean expectation). "
            )
            l5_notes = []
            if w_tr and w_tr.l5_games > 0:
                l5_notes.append(
                    f"{winner.team_abbrev} entered at {w_tr.l5_record} over their L5 ({w_tr.l5_cf_pct:.1f}% L5 CF%, {w_tr.l5_gf_per_game:.2f} GF/G) "
                    f"and posted a {w_tr.game_cf_pct:.1f}% CF% ({w_tr.game_vs_l5_cf_delta:+.1f}% vs. L5)"
                )
            if l_tr and l_tr.l5_games > 0:
                l5_notes.append(
                    f"{loser.team_abbrev} ({l_tr.l5_record} in L5, {l_tr.l5_cf_pct:.1f}% L5 CF%) finished at {l_tr.game_cf_pct:.1f}% CF% ({l_tr.game_vs_l5_cf_delta:+.1f}% vs. L5)"
                )
            l5_sentence = ("; ".join(l5_notes) + ".") if l5_notes else ""
            paragraphs.append(f"{rec_line}{l5_sentence}".strip())

        return expand_first_mention_abbreviations("\n\n".join(paragraphs))

    def build_slate_narrative_algorithmic(self, slate: AnalyzedSlate) -> str:
        """Deterministic 3-paragraph League Slate Media Review."""
        stars_dicts = [
            {
                "star_num": s.star,
                "name": s.name,
                "team": s.team_abbrev,
                "position": s.position,
                "stat_summary": s.stat_summary,
            }
            for s in getattr(slate, "slate_three_stars", [])
        ]
        game_dicts = [
            {
                "away_team": g.away_team.team_abbrev,
                "away_score": g.away_team.score,
                "home_team": g.home_team.team_abbrev,
                "home_score": g.home_team.score,
                "period_ending": g.period_type,
                "game_script_type": g.period_flow.game_script_type if g.period_flow else "REGULATION",
                "headline": g.headline or self.build_game_headline(g),
            }
            for g in getattr(slate, "games", [])
        ]
        ms_dicts = [
            {"category": m.category, "title": m.title, "description": m.description}
            for m in getattr(slate, "slate_notables", [])
        ]
        return build_zamboni_slate_narrative_from_records(
            date_str=slate.date,
            total_games=slate.total_games,
            total_goals=slate.total_goals,
            avg_goals_per_game=slate.avg_goals_per_game,
            home_wins=slate.home_wins,
            away_wins=slate.away_wins,
            ot_so_games=slate.ot_so_games,
            three_stars=stars_dicts,
            game_records=game_dicts,
            milestones=ms_dicts,
        )

    # =========================================================================
    # 4. Unified Narrative Dispatchers (with Context-Pruned Reflection Retry)
    # =========================================================================

    def build_game_narrative(
        self,
        game: AnalyzedGame,
        force_mode: Optional[str] = None,
    ) -> Tuple[str, str]:
        mode = force_mode or self.model_mode
        is_local = is_local_streamlit()
        default_h = game.headline or self.build_game_headline(game)

        # 1. Local LM Studio Server
        if is_local and mode in ("lm-studio", "local-llm", "local", "auto"):
            is_healthy, detected_model, _ = self.check_lm_studio_health()
            if is_healthy and detected_model:
                try:
                    raw_output = self._generate_game_narrative_with_lm_studio(game, detected_model)
                    llm_headline, narrative = extract_headline_and_narrative(raw_output, default_h, game.period_type)
                    ok, violations = verify_game_narrative_facts(narrative, game, headline=llm_headline)
                    if ok:
                        game.headline = llm_headline
                        return narrative, f"lm-studio/{detected_model}"

                    logger.info("LM Studio draft had violations (%s); retrying with Context-Pruned Diff...", "; ".join(violations))
                    raw_retry = self._generate_game_narrative_with_lm_studio(
                        game, detected_model, temperature=0.15, max_tokens=1650, retry_violations=violations
                    )
                    retry_h, retry_narrative = extract_headline_and_narrative(raw_retry, default_h, game.period_type)
                    if self._verify_game_narrative_facts(retry_narrative, game, headline=retry_h):
                        game.headline = retry_h
                        return retry_narrative, f"lm-studio/{detected_model}"
                except Exception as e:
                    self._trigger_alarm(f"LM Studio generation error: {e}")

        # 2. Remote Hugging Face Space Runtime / ZeroGPU
        if not is_local:
            if self._can_use_zerogpu():
                target_model = resolve_zerogpu_model(self.zerogpu_model_id)
                try:
                    raw_output = self._generate_game_narrative_with_zerogpu(game, target_model)
                    llm_headline, narrative = extract_headline_and_narrative(raw_output, default_h, game.period_type)
                    ok, violations = verify_game_narrative_facts(narrative, game, headline=llm_headline)
                    if ok:
                        game.headline = llm_headline
                        return narrative.strip(), f"hf-space/{target_model}"
                    raw_retry = self._generate_game_narrative_with_zerogpu(
                        game, target_model, temperature=0.15, retry_violations=violations
                    )
                    retry_h, retry_n = extract_headline_and_narrative(raw_retry, default_h, game.period_type)
                    if self._verify_game_narrative_facts(retry_n, game, headline=retry_h):
                        game.headline = retry_h
                        return retry_n.strip(), f"hf-space/{target_model}"
                except Exception as e:
                    self.last_zerogpu_error = f"{type(e).__name__}: {e}"
                    logger.warning("Space ZeroGPU narrative generation failed: %s", e)

            custom_endpoint = self.hf_llm_endpoint or os.environ.get("HF_LLM_URL") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL")
            if custom_endpoint:
                try:
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
                        raw_output = self._generate_game_narrative_with_llm(game, client)
                        llm_headline, narrative = extract_headline_and_narrative(raw_output, default_h, game.period_type)
                        if self._verify_game_narrative_facts(narrative, game, headline=llm_headline):
                            game.headline = llm_headline
                            return narrative.strip(), self.llm_model_id
                    except Exception as e:
                        logger.warning("Router LLM inference call failed: %s", e)

        # 3. Deterministic Algorithmic Guardrail Engine (by Zamboni)
        if not game.headline:
            game.headline = self.build_game_headline(game)
        narrative = self.build_game_narrative_algorithmic(game)
        return narrative.strip(), ALGORITHMIC_MODEL_STAMP

    def build_slate_narrative(
        self,
        slate: AnalyzedSlate,
        force_mode: Optional[str] = None,
    ) -> Tuple[str, str]:
        mode = force_mode or self.model_mode
        is_local = is_local_streamlit()
        stars_dicts = [
            {"star_num": s.star, "name": s.name, "team": s.team_abbrev, "position": s.position, "stat_summary": s.stat_summary}
            for s in getattr(slate, "slate_three_stars", [])
        ]
        default_h = build_zamboni_slate_headline(
            date_str=slate.date,
            total_games=slate.total_games,
            total_goals=slate.total_goals,
            home_wins=slate.home_wins,
            away_wins=slate.away_wins,
            ot_so_games=slate.ot_so_games,
            three_stars=stars_dicts,
        )
        slate.executive_headline = default_h

        # 1. Local LM Studio Server
        if is_local and mode in ("lm-studio", "local-llm", "local", "auto"):
            is_healthy, detected_model, _ = self.check_lm_studio_health()
            if is_healthy and detected_model:
                try:
                    raw_output = self._generate_slate_narrative_with_lm_studio(slate, detected_model)
                    llm_headline, narrative = extract_headline_and_narrative(raw_output, default_h)
                    ok, violations = verify_slate_narrative_facts(narrative, slate, headline=llm_headline)
                    if ok:
                        slate.executive_headline = default_h if str(slate.total_games) not in llm_headline else llm_headline
                        return narrative, f"lm-studio/{detected_model}"

                    raw_retry = self._generate_slate_narrative_with_lm_studio(
                        slate, detected_model, temperature=0.15, retry_violations=violations
                    )
                    retry_h, retry_narrative = extract_headline_and_narrative(raw_retry, default_h)
                    if self._verify_slate_narrative_facts(retry_narrative, slate, headline=retry_h):
                        slate.executive_headline = default_h if str(slate.total_games) not in retry_h else retry_h
                        return retry_narrative, f"lm-studio/{detected_model}"
                except Exception as e:
                    self._trigger_alarm(f"LM Studio slate generation error: {e}")

        # 2. Remote Hugging Face Space Runtime / ZeroGPU
        if not is_local:
            if self._can_use_zerogpu():
                target_model = resolve_zerogpu_model(self.zerogpu_model_id)
                try:
                    raw_output = self._generate_slate_narrative_with_zerogpu(slate, target_model)
                    llm_headline, narrative = extract_headline_and_narrative(raw_output, default_h)
                    ok, violations = verify_slate_narrative_facts(narrative, slate, headline=llm_headline)
                    if ok:
                        slate.executive_headline = default_h if str(slate.total_games) not in llm_headline else llm_headline
                        return narrative.strip(), f"hf-space/{target_model}"
                    raw_retry = self._generate_slate_narrative_with_zerogpu(
                        slate, target_model, temperature=0.15, retry_violations=violations
                    )
                    retry_h, retry_n = extract_headline_and_narrative(raw_retry, default_h)
                    if self._verify_slate_narrative_facts(retry_n, slate, headline=retry_h):
                        slate.executive_headline = default_h if str(slate.total_games) not in retry_h else retry_h
                        return retry_n.strip(), f"hf-space/{target_model}"
                except Exception as e:
                    self.last_zerogpu_error = f"{type(e).__name__}: {e}"
                    logger.warning("Space ZeroGPU slate generation failed: %s", e)

            custom_endpoint = self.hf_llm_endpoint or os.environ.get("HF_LLM_URL") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("LLM_BASE_URL")
            if custom_endpoint:
                try:
                    raw_output = self._generate_slate_narrative_with_custom_endpoint(slate, custom_endpoint)
                    llm_headline, narrative = extract_headline_and_narrative(raw_output, default_h)
                    if self._verify_slate_narrative_facts(narrative, slate, headline=llm_headline):
                        slate.executive_headline = default_h if str(slate.total_games) not in llm_headline else llm_headline
                        return narrative, f"remote-endpoint/{self.llm_model_id}"
                except Exception as e:
                    logger.warning("Remote endpoint slate generation failed: %s", e)

        # 3. Deterministic Guardrail Engine (by Zamboni)
        slate.executive_headline = default_h
        narrative = self.build_slate_narrative_algorithmic(slate)
        return narrative.strip(), ALGORITHMIC_MODEL_STAMP

    # =========================================================================
    # 5. Publication Markdown Report Compilers
    # =========================================================================

    def format_narrative_engine_badge(self, model_stamp: str) -> str:
        m = str(model_stamp).lower()
        if "hf-space" in m or "zerogpu" in m:
            clean_name = model_stamp.replace("hf-space/", "").replace("zerogpu/", "").strip()
            return f"Zamboni • Hugging Face Space ({clean_name})"
        elif "lm-studio" in m:
            clean_name = model_stamp.replace("lm-studio/", "").split("/")[-1]
            return f"Zamboni • LM Studio ({clean_name})"
        elif "llama" in m:
            return "Zamboni • Meta-Llama-3.2-3B (Router)"
        else:
            return "Zamboni • Algorithmic Guardrail Engine"

    def generate_single_game_markdown(self, game: AnalyzedGame) -> str:
        headline = game.headline or self.build_game_headline(game)
        narrative = game.recap_narrative
        if not narrative:
            narrative, model_stamp = self.build_game_narrative(game)
            game.recap_narrative = narrative
            game.generator_model = model_stamp

        model_display = self.format_narrative_engine_badge(game.generator_model)
        dec = "Final" if game.period_type == "REG" else ("Final/OT" if game.period_type == "OT" else "Final/SO")
        script_key = getattr(getattr(game, "period_flow", None), "game_script_type", getattr(game, "game_script_type", "REGULATION"))
        script_label = str(script_key).replace("_", " ").title()

        md = [
            f"# 🏒 {headline}",
            f"**Date:** {game.game_date} | **Venue:** {game.venue} | **Final:** {game.away_team.team_abbrev} {game.away_team.score} @ {game.home_team.team_abbrev} {game.home_team.score} ({dec})",
            f"`🏒 Game Script: {script_label}`",
            f"`🤖 Narrative Generator: {model_display}`",
            "`🛡️ Data Quality: Reconciled & Audited by Zamboni`",
            "---",
            "## 📰 Game Recap (by Zamboni)",
            narrative,
            "",
        ]

        if game.three_stars:
            md.append("## ⭐ Three Stars of the Game")
            star_emojis = ["🥇", "🥈", "🥉"]
            for idx, star in enumerate(game.three_stars):
                badge = star_emojis[idx] if idx < len(star_emojis) else "⭐"
                md.append(f"- {badge} **#{star.star}. {star.name}** ({star.team_abbrev} - {star.position}) — *{star.stat_summary}*")
            md.append("")

        w_g = game.winning_goalie_profile
        l_g = game.losing_goalie_profile
        if w_g or l_g:
            md.append("## 🥅 Goaltending Performance")
            md.append("| Goaltender | Team | Decision | Saves / Shots | SV% | GSAx | Rating | Assessment |")
            md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")
            for g in [w_g, l_g]:
                if g:
                    r_clean = str(g.performance_rating).replace("_", " ").title()
                    md.append(f"| **{g.name}** | {g.team_abbrev} | {g.decision} | {g.saves}/{g.shots_against} | {g.save_pct*100:.1f}% | {g.gsax:+.2f} | {r_clean} | {g.assessment} |")
            md.append("")

        return "\n".join(md)

    def generate_slate_markdown(self, slate: AnalyzedSlate) -> str:
        model_display = self.format_narrative_engine_badge(getattr(slate, "generator_model", ""))
        day_title = getattr(slate, "executive_headline", "") or f"NHL Slate Night in Review — {slate.date}"

        md = [
            f"# 🏒 {day_title}",
            f"**Total Games:** {slate.total_games} | **Total Goals:** {slate.total_goals} (Avg {slate.avg_goals_per_game:.1f}) | **OT/SO Thrillers:** {slate.ot_so_games}",
            f"**Outcomes:** {slate.home_wins} Home Wins — {slate.away_wins} Road Victories",
            f"`🤖 Editorial Generator: {model_display}`",
            "`🛡️ Data Quality: Reconciled & Audited by Zamboni`",
            "---",
        ]

        if slate.executive_summary:
            md.append("## 🌟 Executive Slate Summary (by Zamboni)")
            md.append(slate.executive_summary)
            md.append("")

        if slate.slate_three_stars:
            md.append("## ⭐ League Three Stars of the Night")
            star_emojis = ["🥇", "🥈", "🥉"]
            for idx, star in enumerate(slate.slate_three_stars):
                badge = star_emojis[idx] if idx < len(star_emojis) else "⭐"
                md.append(f"- {badge} **#{star.star} Star: {star.name}** ({star.team_abbrev} - {star.position}) — *{star.stat_summary}*")
            md.append("")

        md.append("## 📋 Final Scores & Game Storylines")
        md.append("| Away | Score | Home | Score | Dec | Key Storyline |")
        md.append("| :---: | :---: | :---: | :---: | :---: | :--- |")
        for g in slate.games:
            dec = "Final" if g.period_type == "REG" else ("Final/OT" if g.period_type == "OT" else "Final/SO")
            md.append(f"| {g.away_team.team_abbrev} | {g.away_team.score} | {g.home_team.team_abbrev} | {g.home_team.score} | `{dec}` | {g.headline} |")
        md.append("")

        return "\n".join(md)
