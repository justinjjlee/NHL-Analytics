# 🏒 NHL Scoreboard Agent Framework

An autonomous, multi-layered sports analytics agent framework designed to review recent NHL games, extract official box scores and team performance sheets, verify statistical integrity via mathematical guardrails, analyze player milestones and goaltending duels, and collectively synthesize editorial recaps and notable news stories across an entire evening of hockey.

Persistent record-keeping is managed directly within the Hugging Face dataset lakehouse (`jjerlee/nhl-analytics`), powering seamless, zero-latency retrieval and on-demand generation in the Streamlit application ([`app/views/game_recaps.py`](file:///Users/jjlee/Documents/Github/NHL-Analytics/app/views/game_recaps.py)).

---

## 🏛️ Architecture & System Overview

The agent framework lives in `dev/agent/scoreboard/` and operates across five decoupled layers:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Official NHL API (v1)                           │
│     /score/{date}   •   /gamecenter/{id}/landing   •   /boxscore       │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │ HTTPS Streaming
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│               Data Ingestion Layer (client.py & models.py)             │
│   • Multi-endpoint aggregation (landing, boxscore, right-rail)         │
│   • Strongly-typed domain models for skaters, goalies, teams, plays    │
│   • Built-in memory cache & exponential backoff retry policies         │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│          Analytical Guardrails & Evaluation Engine (analyzer.py)       │
│   • Stats Reconciliation: Goal/SOG cross-math audit & verification     │
│   • Period Flow Audit: Verifies real game scripts (no false thrillers) │
│   • Goalie Profiles: Brick Wall, Steady, Vulnerable & composure grade  │
│   • Defensive Sacrifice: Blocks, hits, turnovers, shot suppression     │
│   • Season Context: GP sample-size stages & Pythagorean win% (exp 2.15)│
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│         Narrative & Journalism Synthesis Engine (narrator.py)          │
│   • Dual Mode: Meta-Llama-3.2-3B (Serverless LLM) + Algorithmic Engine │
│   • Plain-Language Step-by-Step Prompt Fact Sheets (Anti-Hallucination)│
│   • 4-Paragraph Publication Articles: period progression, defense, pyth│
│   • Full Markdown Reports: goalie tables, defensive grids, standings   │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│      Persistent Lakehouse Store & Cloud Sync (dataset_store.py)        │
│   • Hugging Face Dataset: jjerlee/nhl-analytics (scoreboard/*.parquet) │
│   • Idempotent Execution: instant lookup if recorded; on-demand compile│
│   • Auto Cloud Sync: background commit via huggingface_hub & HF_TOKEN  │
└──────────────────────────────────┬─────────────────────────────────────┘
                                   │
                                   ▼
┌────────────────────────────────────────────────────────────────────────┐
│           Streamlit Interactive UI (app/views/game_recaps.py)          │
│   • Dual-mode historical date selector (calendar + pre-compiled slates)│
│   • Verified game script badges & Three Stars podium                   │
│   • Dual Tabs: Executive Breakdown Cards & Full Scouting Report (.md)  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📂 File Structure

| File | Purpose |
| :--- | :--- |
| [**`models.py`**](models.py) | Strongly-typed dataclasses for `AnalyzedGame`, `AnalyzedSlate`, `PeriodFlowAnalysis`, `GoaliePerformanceProfile`, `DefensiveEffortProfile`, `TeamSeasonContext`, and `StatsValidationReport`. |
| [**`client.py`**](client.py) | Modern HTTP client interfacing with NHL API endpoints (`api-web.nhle.com/v1`), including automated date detection and retry resilience. |
| [**`analyzer.py`**](analyzer.py) | Statistical analysis engine with guardrails: data reconciliation, verified period flow scripts, defensive sacrifice profiles, goalie grades, and cumulative Pythagorean expectations. |
| [**`narrator.py`**](narrator.py) | Sports journalism synthesis engine crafting expressive game headlines, 4-paragraph analytical recaps, and publication markdown reports. |
| [**`dataset_store.py`**](dataset_store.py) | Hugging Face lakehouse record-keeping manager for `slate_recaps.parquet` and `game_recaps.parquet` with idempotent querying and cloud sync. |
| [**`agent.py`**](agent.py) | Primary `ScoreboardAgent` orchestrator tying ingestion, analysis, narrative generation, and report persistence. |
| [**`cli.py`**](cli.py) | Command-line interface with rich flag options (`--latest`, `--date`, `--game-id`, `--print-recap`, `--llm-prompt`). |
| [**`example_run.py`**](example_run.py) | Standalone demonstration script showing full slate analysis and single-game deep-dives on real NHL game data. |
| [**`reports/`**](reports/) | Directory storing local markdown copies of generated slate reviews and game reports. |

---

## 🛡️ Analytical Guardrails & Quality Controls

To ensure absolute statistical integrity and avoid generic or factually misleading reporting, the agent applies strict guardrails before any narrative is compiled:

### 1. Stats Cross-Reconciliation (`StatsValidationReport`)
- Verifies that `Away Goals + Home Goals` matches the period-by-period goal sum.
- Confirms that total team SOG equals the sum of individual skater shots and matches goaltender `saves + goals allowed`.
- Reconciles individual goal and assist points with final scoring sheets.

### 2. Period Flow Audit & Verified Game Scripts (`PeriodFlowAnalysis`)
Instead of blindly describing a game as a "thriller" based purely on the final goal differential, the agent traces the lead trajectory across all periods:
- **`WIRE_TO_WIRE`**: Winning team led early and never relinquished the lead.
- **`CONTROLLED_LATE_PUSH`**: Leader held a multi-goal edge into P3; opponent scored late but leader weathered the attack.
- **`COMEBACK_WIN`**: Winning team trailed entering P3 and scored unanswered goals to win.
- **`TRUE_THRILLER`**: Featured multiple lead changes and ties throughout regulation.
- **`WALK_OFF_OT_SO`**: Deadlocked after 60 minutes, decided in overtime or shootout.
- **`BLOWOUT`**: Decided by 4+ goals with sustained territorial domination.

### 3. In-Depth Goaltending Performance (`GoaliePerformanceProfile`)
Netminders are evaluated on volume, workload, and rebound composure:
- **`BRICK_WALL`**: Faced high-volume traffic (≥ 35 SOG) with ≥ .925 save percentage.
- **`STEADY`**: Efficient, calm puck absorption behind solid defensive structure.
- **`UNDER_PRESSURE`**: Faced high shot counts or heavy odd-man rush volume.
- **`VULNERABLE`**: Conceded soft goals or struggled to control rebound trajectories.

### 4. Defensive Sacrifice & Physicality Grid (`DefensiveEffortProfile`)
- **Blocked Shots**: Evaluated as `SACRIFICIAL` (≥ 18 blocks), `DISCIPLINED` (12–17), or `POROUS` (< 12).
- **Physical Force**: Graded on body checks (hits), identifying blue-collar forechecks.
- **Turnover Margin**: Audits giveaways vs. takeaways to determine defensive zone puck security.
- **Shot Suppression**: Evaluates defensive shell efficiency and interior slot denial.

### 5. Games Played ($GP$) Sensitivity & Pythagorean Expectations (`TeamSeasonContext`)
- **Sample-Size Context**:
  - `EARLY_SEASON` ($GP < 25$): Flags small-sample variance; line combinations remain in calibration.
  - `MID_SEASON` ($25 \le GP \le 50$): Territorial possession metrics have stabilized.
  - `PLAYOFF_RACE` ($GP > 50$): Direct implications for wild-card races and divisional seeding.
- **Pythagorean Expectation**:
  $$\text{Pythagorean Win \%} = \frac{\text{GF}^{2.15}}{\text{GF}^{2.15} + \text{GA}^{2.15}}$$
  Measures whether a team is outperforming or underperforming its underlying goal differential (identifying luck and variance).

---

## 🦙 Hybrid Narrative Engine: Meta-Llama-3.2-3B + Algorithmic Guardrails

The narration engine (`narrator.py`) provides two complementary generation approaches:

```
┌────────────────────────────────────────────────────────────────────────┐
│               Data Ingestion & Verification (analyzer.py)              │
│       • Mathematical reconciliation & cross-stats audit                │
│       • Verified game scripts, goalie grades & defense profiles        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│           Plain-Language Step-by-Step Prompt Fact Sheet                │
│   • Step 1: Lede & Tone (Scoreline, heroics, verified script)          │
│   • Step 2: Scoring Progression & Timing (P1 ice-breaker, P2 cushion)  │
│   • Step 3: Goaltending Duel & Defensive Sacrifice (Blocks, hits, TOs) │
│   • Step 4: Standings & Pythagorean Analytics (GF/GA variance)         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
           ┌────────────────────────┴────────────────────────┐
           │ Mode: 'llm' or 'auto' (HF_TOKEN present)        │ Mode: 'algorithmic' or Fallback
           ▼                                                 ▼
┌──────────────────────────────────────┐     ┌──────────────────────────────────────┐
│  Hugging Face Serverless LLM Client  │     │   Deterministic Algorithmic Engine   │
│  Model: meta-llama/Llama-3.2-3B      │     │   • Instant execution (< 0.05s)      │
│  • Creative journalistic prose       │     │   • Zero hallucination risk          │
│  • Grounded strictly in prompt facts │     │   • 100% mathematical fidelity       │
└──────────────────┬───────────────────┘     └──────────────────┬───────────────────┘
                   │                                            │
                   └─────────────────────┬──────────────────────┘
                                         ▼
┌────────────────────────────────────────────────────────────────────────┐
│                Publication Report & Parquet Serialization              │
│  • Stamps generator model ('meta-llama/Llama-3.2-3B' vs 'algorithmic') │
│  • Renders dual tabs in Streamlit UI with live engine badges           │
└────────────────────────────────────────────────────────────────────────┘
```

### Engine Selection Modes
- **`mode="auto"` (Default):** Checks for `HF_TOKEN`. If available, drafts creative sports journalism using `meta-llama/Llama-3.2-3B-Instruct` via Hugging Face `InferenceClient`. If the API is offline, rate-limited, or unauthenticated, it seamlessly falls back to the deterministic algorithmic engine without throwing errors.
- **`mode="llm"`:** Directs all narration requests to `meta-llama/Llama-3.2-3B-Instruct`.
- **`mode="algorithmic"`:** Bypasses LLM APIs entirely for instant, zero-cost, deterministic Natural Language Generation.

---

## ☁️ Hugging Face Lakehouse Record-Keeping

All generated texts and analytical tables are stored in the user's Hugging Face dataset [`jjerlee/nhl-analytics`](https://huggingface.co/datasets/jjerlee/nhl-analytics):

```
jjerlee/nhl-analytics/
├── data/
│   └── games.parquet                   # Historical game box scores (2013–present)
└── insights/
    └── scoreboard/
        ├── slate_recaps.parquet        # Nightly summaries, 3 stars, macro stats
        └── game_recaps.parquet         # Game recaps, verified scripts, goalie/defense/pyth data
```

### Idempotent Generation Policy
- When a user views a date in the Streamlit app, the store checks if recaps for that date already exist in `jjerlee/nhl-analytics` under `insights/scoreboard/`.
- **If Recorded:** Returns records instantly (< 0.05 seconds) with zero compute cost.
- **If Unrecorded:** Triggers the Scoreboard Agent on demand, verifies stats, compiles markdown reports, writes new Parquet rows, and pushes the updated dataset directly to Hugging Face via `HF_TOKEN`.

> [!IMPORTANT]
> **Data Storage Isolation:** Generated data (`game_recaps.parquet` and `slate_recaps.parquet`) is stored **strictly in the Hugging Face dataset under `insights/scoreboard/`**, never committed to this Git repository. Transient local caching is directed to `~/.cache/rinklytics/insights/scoreboard/` outside the workspace.

---

## 🔄 Orchestration & Trigger Mechanisms

How is the agent triggered and scheduled?

### 1. On-Demand via the Streamlit App (`rinklytics` Space)
- **Mechanism:** Lazy evaluation triggered directly by users navigating the app.
- **Workflow:** When a user opens the **Game Recaps** tab and chooses an unrecorded date, the app executes `agent.analyze_slate(date)` in the background, pushes the new rows to the Hugging Face dataset via `HF_TOKEN`, and immediately presents the recap in the UI.

### 2. Scheduled Nightly Automation via GitHub Actions
- **Mechanism:** Pre-compilation workflow running in GitHub Actions ([`.github/workflows/scoreboard_agent.yml`](file:///.github/workflows/scoreboard_agent.yml)).
- **Schedule:** Triggers daily at **07:00 UTC (2:00 AM Eastern / 11:00 PM Pacific)** once all West Coast games have concluded.
- **Command Executed:**
  ```bash
  python dev/agent/scoreboard/cli.py --latest --record-hf --no-save
  ```
- **Environment:** Authenticated via the repository secret `HF_TOKEN`.
- **Advantage:** Users waking up in the morning experience instant page loads (< 0.05s) because the nightly recap was already pre-compiled and pushed to Hugging Face overnight.

### 3. Manual / Ad-Hoc CLI Trigger
- **Command:**
  ```bash
  python dev/agent/scoreboard/cli.py --date 2026-10-13 --record-hf --no-save
  ```

---

## 💻 Streamlit UI Integration (`app/views/game_recaps.py`)

The user-facing dashboard is located in the navigation directly following the **Scoreboard** tab:
1. **Interactive Date Selector**:
   - Calendar date picker supporting any date from 2013 to the present.
   - Pre-compiled slates archive dropdown for instant one-click browsing.
   - Quick-jump buttons (`Oct 13, 2026`, `Yesterday`, `Today`).
2. **Slate KPI Banner & Podium**:
   - Total games, total goals, goal average, home vs. road win splits, overtime count.
   - League Three Stars of the Night podium with medals (🥇, 🥈, 🥉).
   - Expandable full league slate scouting and media review.
3. **Dual-Tab Game Expander Cards**:
   - `📊 Executive Breakdown & Tables`: Game narrative, Goaltending Performance table, Defensive Sacrifice & Physicality grid, Season Standings & Pythagorean Analytics table, and detected milestones.
   - `📄 Full Scouting & Media Report (.md)`: Renders the full publication-ready markdown article directly.

---

## 📰 Sample Publication Report (Showcase Example)

Below is an authentic, full sample report generated by the framework for the **New Jersey Devils at Detroit Red Wings** matchup on **October 13, 2026**:

```markdown
# 🏒 J. Hughes Nets Overtime Winner as New Jersey Devils Edge Detroit Red Wings 5-4
**Date:** 2026-10-13 | **Venue:** Little Caesars Arena | **Final:** NJD 5 @ DET 4 (OT)
`🛡️ Data Quality Guardrail: Reconciled & Verified Against Official NHL Feeds`
---
## 📰 Game Recap
The **New Jersey Devils** earned a 5–4 victory in OT against the **Detroit Red Wings** at Little Caesars Arena. After sixty minutes proved insufficient to separate the clubs, New Jersey Devils seized the extra point in sudden-death fashion. Leading the charge was **J. Hughes**, who delivered an impactful showing to anchor the result.

Action opened in the reg period when **E. Rodrigues** (NJD) broke the ice at 06:07 (EV). Entering the second intermission, the scoreboard read **NJD 4 - DET 3**. The clubs combined for 9 total goals across the tilt. With regulation ending deadlocked, the decisive strike belonged to **J. Hughes** at 01:19 of OT.

Between the pipes, **J. Allen** struggled to contain dangerous rebounds, conceding 4 goals on 27 shots. At the opposite end, **J. Gibson** faced high-volume traffic (37 sog), turning aside 32 pucks. Defensive commitment was a key differentiator: NJD sacrificed for **13** blocked shots and laid **13** hits, compared to DET's **17** blocks and **25** checks. Careless puck management hampered DET, who conceded 19 giveaways. Special teams saw NJD went 1/1 (100.0%) on the man advantage; DET capitalized 0/1 on power plays.

Early-season baseline (6 GP). Small-sample variance dominates; line combinations and defensive pairings remain in calibration. With the result, the **New Jersey Devils** move to 3–3–0 (6 pts, 50.0%), while the **Detroit Red Wings** stand at 1–3–2 (4 pts, 33.3%).  NJD's underlying goal differential suggests a Pythagorean expected win rate of 33.5% (outperforming by 16.5%).

## ⭐ Three Stars of the Game
- 🥇 **#1. J. Hughes** (NJD - C) — *1G, 2A (3 pts)*
- 🥈 **#2. A. DeBrincat** (DET - R) — *2G, 1A (3 pts)*
- 🥉 **#3. J. Bratt** (NJD - L) — *1G, 2A (3 pts)*

## 🥅 Goaltending Performance
| Goaltender | Team | Decision | Saves / Shots | SV% | Rating | Assessment |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **J. Allen** | NJD | W | 23/27 | 85.2% | `VULNERABLE` | Struggled to contain dangerous rebounds, conceding 4 goals on 27 shots. |
| **J. Gibson** | DET | O | 32/37 | 86.5% | `UNDER_PRESSURE` | Faced high-volume traffic (37 SOG), turning aside 32 pucks. |

## 🛡️ Defensive Effort & Physicality Comparison
| Metric | NJD (Winner) | DET (Loser) | Rating / Advantage |
| :--- | :---: | :---: | :--- |
| **Blocked Shots** | 13 (`DISCIPLINED`) | 17 (`DISCIPLINED`) | +4 DET |
| **Body Checks (Hits)** | 13 (`CONTAINED`) | 25 (`PHYSICAL_FORCE`) | +12 DET |
| **Giveaways / Takeaways** | 14 / 7 (Diff -7) | 19 / 4 (Diff -15) | Turnover edge: +8 |
| **Shots Allowed** | 27 (`BALANCED`) | 37 (`UNDER_SIEGE`) | Shot suppression: NJD |

## 📈 Season Standings & Pythagorean Analytics
| Team | GP | Record | PTS | PTS% | GF | GA | Goal Diff | Pyth Exp % | Luck/Variance | Stage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **NJD** | 6 | 3-3-0 | 6 | 50.0% | 16 | 22 | -6 | 33.5% | `+16.5%` | EARLY_SEASON |
| **DET** | 6 | 1-3-2 | 4 | 33.3% | 14 | 20 | -6 | 31.7% | `+1.6%` | EARLY_SEASON |
```

---

## 🚀 CLI Quick Start

```bash
# 1. Analyze the latest completed NHL slate
python dev/agent/scoreboard/cli.py --latest

# 2. Analyze a specific date (e.g. October 13, 2026)
python dev/agent/scoreboard/cli.py --date 2026-10-13

# 3. Single game deep-dive with recap output
python dev/agent/scoreboard/cli.py --game-id 2026020090 --print-recap
```
