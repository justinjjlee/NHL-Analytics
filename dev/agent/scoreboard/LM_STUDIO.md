# LM Studio Integration & Narrative Generation Architecture

This document describes the orchestration and guardrails for generating NHL sports journalism recaps using **LM Studio** and the **Algorithmic Guardrail Engine**.

---

## 1. Architecture Overview

```
                      +---------------------------------------+
                      |       ScoreboardAgent Ingestion       |
                      |   (NHL Boxscore, Landing, Play-by-Play)|
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |         GameAnalyzer Audits           |
                      |  - Period-by-Period Game Scripts      |
                      |  - Goaltending Duel Assessments       |
                      |  - Defensive Effort (Blocks/Hits/TKA) |
                      |  - Pythagorean Win Expectation Math   |
                      +-------------------+-------------------+
                                          |
                         +----------------+----------------+
                         |                                 |
                         v                                 v
        [LM Studio Local Server: 127.0.0.1:1234]     [LM Studio Offline / Unreachable]
        - OpenAI-compatible REST API                      - Prominent ALARM emitted
        - Zero API fees & GPU acceleration                - Deterministic Algorithmic NLG
        - Dynamic model auto-detection                    - 100% Mathematically Reconciled
                         |                                 |
                         +----------------+----------------+
                                          |
                                          v
                      +---------------------------------------+
                      |      Factual Verification Guardrail   |
                      |  - Audits winner team name            |
                      |  - Validates non-hockey term absence  |
                      |  - Checks score/decision alignment    |
                      +-------------------+-------------------+
                                          |
                                          v
                      +---------------------------------------+
                      |     Hugging Face Dataset Storage      |
                      |        `jjerlee/nhl-analytics`        |
                      +---------------------------------------+
```

---

## 2. LM Studio Server Configuration

- **Host**: `http://127.0.0.1:1234`
- **Base Endpoint**: `http://127.0.0.1:1234/v1`
- **Supported Models**: Auto-detected via `GET /v1/models` (e.g., `nvidia/nemotron-3-nano-4b`).
- **Dependencies**: Uses standard Python `requests` and `json`. No heavy PyTorch or HuggingFace Transformers dependencies are required.

---

## 3. Alarm & Fail-Safe Behavior

1. **Active Alarm**:
   When `--mode lm-studio` or `--mode auto` cannot reach `http://127.0.0.1:1234`, an alarm is immediately logged to `stderr` and the logger:
   ```
   🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
   🚨 [ALARM: LM STUDIO UNREACHABLE]
   🚨 Details: Unable to reach LM Studio at http://127.0.0.1:1234/v1
   🚨 Action: Please verify LM Studio is running and the local server is started at http://127.0.0.1:1234/v1
   🚨 Safeguard: Falling back to Algorithmic Guardrail Engine to guarantee verified box-score recaps.
   🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
   ```

2. **Streamlit UI Banner**:
   In the Streamlit application, users are alerted if the local LM Studio instance is offline while reviewing recaps.

3. **Deterministic Algorithmic Fallback**:
   Execution never crashes. The system immediately serves the audited, deterministic recap created by `build_game_narrative_algorithmic()`.

---

## 4. CLI Execution Examples

```bash
# 1. Analyze single game using LM Studio with recap printed to console
python dev/agent/scoreboard/cli.py --game-id 2026020090 --mode lm-studio --print-recap

# 2. Analyze daily slate with auto-fallback and sync to Hugging Face dataset
python dev/agent/scoreboard/cli.py --date 2026-10-13 --record-hf

# 3. Target specific model loaded in LM Studio
python dev/agent/scoreboard/cli.py --game-id 2026020090 --mode lm-studio --lm-model "nvidia/nemotron-3-nano-4b" --print-recap
```
