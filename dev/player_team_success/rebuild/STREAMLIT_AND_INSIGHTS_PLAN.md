# Rebuild Delivery: Markdown Insights Architecture & Interactive Streamlit Dashboard Plan

## Comprehensive Blueprint for Programmatic Markdown Reporting, Historical Franchise Autopsies, and Interactive Streamlit Visualizations (2000–2025)

---

> [!IMPORTANT]
> ### Purpose of This Blueprint
> This document details the **execution plan for communicating and visualizing the empirical and econometric findings** of the 25-Year NHL Franchise Rebuild Study. It establishes:
> 1. **Automated Markdown Insights Architecture**: The programmatic pipeline that synthesizes econometric model outputs into human-readable executive summaries, 32 individual franchise retrospective dossiers, failure autopsies, and historical era transition reports.
> 2. **Interactive Streamlit Dashboard (`app/views/rebuild_tracker.py`)**: A production-grade, highly engaging visual interface within Rinklytics that enables users to explore 25 years of rebuild history, inspect why specific rebuilds succeeded or failed, analyze team-by-team regime trends, and simulate counterfactual front-office rebuild strategies in real time.
> 3. **Bilingual & Production Standards**: Compliance with Rinklytics design conventions (EN/FR i18n, Plotly responsive themes, Hugging Face Lakehouse Parquet caching, and local Docker testing on port 1901).
> 4. **Post-Hoc Retrospective Historical Study Mandate (`Inaction`, `Rebuild`, `Peak`)**: Every generated Markdown dossier and every Streamlit dashboard tab prominently emphasizes that this research is a **post-hoc, retrospective historical evaluation (2000–2025)** grounded in time-series regime-switching state-space econometrics (`Inaction`, `Rebuild`, `Peak`), and is **in no way indicative or predictive of future franchise behaviors or trends**. Its insights and policy elasticities speak instead to general economic laws—explaining why specific front-office decisions have historically proven **effective or tragically ineffective** in moving a franchise from the `Inaction` and `Rebuild` regimes into the `Peak` regime.
>
> **Companion Research Documentation Suite**:
> - [**Project Vision & Fundamental Questions (README.md)**](file:///Users/jjlee/Documents/github/nhl-analytics/dev/player_team_success/rebuild/README.md)
> - [**Core Definitions, Ontology & Algorithmic Conditions (DEFINITION.md)**](file:///Users/jjlee/Documents/github/nhl-analytics/dev/player_team_success/rebuild/DEFINITION.md)
> - [**Data Architecture & Panel Engineering (METHODOLOGY_DATA.md)**](file:///Users/jjlee/Documents/github/nhl-analytics/dev/player_team_success/rebuild/METHODOLOGY_DATA.md)
> - [**Econometric Structural Model & Robustness Protocol (METHODOLOGY_MODEL.md)**](file:///Users/jjlee/Documents/github/nhl-analytics/dev/player_team_success/rebuild/METHODOLOGY_MODEL.md)
> - [**Academic Delivery Charter & Exhibit Specifications (METHODOLOGY_DELIVERY.md)**](file:///Users/jjlee/Documents/github/nhl-analytics/dev/player_team_success/rebuild/METHODOLOGY_DELIVERY.md)

---

## Table of Contents

1. [Executive Delivery Workflow & System Architecture](#1-executive-delivery-workflow--system-architecture)
2. [Part I: Programmatic Markdown Insights & Dossier Generation](#2-part-i-programmatic-markdown-insights--dossier-generation)
   - [2.1 Generated Insights Directory Structure](#21-generated-insights-directory-structure)
   - [2.2 Master Executive Summary (`EXECUTIVE_SUMMARY.md`)](#22-master-executive-summary-executive_summarymd)
   - [2.3 Franchise Retrospective Dossiers (`insights/franchises/{TEAM}.md`)](#23-franchise-retrospective-dossiers-insightsfranchisesteammd)
   - [2.4 Forensic Failure Autopsies (`AUTOPSY_FAILED_REBUILDS.md`)](#24-forensic-failure-autopsies-autopsy_failed_rebuildsmd)
   - [2.5 Historical Era Transitions & Regime Shifts (`REGIME_TRENDS_AND_ERA_SHIFTS.md`)](#25-historical-era-transitions--regime-shifts-regime_trends_and_era_shiftsmd)
   - [2.6 Automation Script Specification (`generate_rebuild_insights.py`)](#26-automation-script-specification-generate_rebuild_insightspy)
3. [Part II: Interactive Streamlit Dashboard Blueprint (`rebuild_tracker.py`)](#3-part-ii-interactive-streamlit-dashboard-blueprint-rebuild_trackerpy)
   - [3.1 Application Layout, Styling & Navigation Integration](#31-application-layout-styling--navigation-integration)
   - [3.2 Tab 1: The 25-Year Franchise Regime Matrix & Interactive Timeline](#32-tab-1-the-25-year-franchise-regime-matrix--interactive-timeline)
   - [3.3 Tab 2: Succeeded vs. Failed Rebuild Scorecard & Quadrant (The Simple Method)](#33-tab-2-succeeded-vs-failed-rebuild-scorecard--quadrant-the-simple-method)
   - [3.4 Tab 3: "Why Did It Succeed or Fail?" Forensic Decomposer](#34-tab-3-why-did-it-succeed-or-fail-forensic-decomposer)
   - [3.5 Tab 4: Franchise Longitudinal Retrospective Dossier](#35-tab-4-franchise-longitudinal-retrospective-dossier)
   - [3.6 Tab 5: Macro Trends, CBA Regimes & Era Evolution](#36-tab-5-macro-trends-cba-regimes--era-evolution)
   - [3.7 Tab 6: Interactive Rebuild Strategy Simulator (What-If Counterfactual Tool)](#37-tab-6-interactive-rebuild-strategy-simulator-what-if-counterfactual-tool)
4. [Part III: Data Contracts, JSON Caching & Hugging Face Lakehouse Sync](#4-part-iii-data-contracts-json-caching--hugging-face-lakehouse-sync)
5. [Part IV: Implementation Roadmap & Verification Checklist](#5-part-iv-implementation-roadmap--verification-checklist)

---

## 1. Executive Delivery Workflow & System Architecture

The research program's output pipeline operates as a dual delivery engine: transforming raw panel datasets and econometric Markov Chain Monte Carlo ($\text{MCMC}$) posterior distributions into both **persistent, static Markdown knowledge dossiers** for deep reading and an **interactive, dynamic Streamlit web application** for visual exploration.

```mermaid
flowchart TD
    subgraph DataEngine ["Data & Econometric Engine"]
        P1["panel_team_season.parquet"]
        P2["panel_player_season_core.parquet"]
        P3["panel_rebuild_episodes.parquet"]
        M1["State-Space SEM Posteriors (T, B, Z)"]
        M2["Survival Hazard Models (Cox, Fine-Gray)"]
        M3["MomentumSSM & Hit KPIs"]
    end

    subgraph Generator ["Programmatic Insights Generator (src/generate_rebuild_insights.py)"]
        G1["Franchise Episode Classifier & Evaluator"]
        G2["Causal Attribution & Shapley Decomposer"]
        G3["Era Trend & Regime Transition Aggregator"]
        G4["Markdown Templating & Exhibit Formatter"]
    end

    subgraph MarkdownDossiers ["Static Markdown Insights Suite (results/insights/)"]
        MD1["EXECUTIVE_SUMMARY.md<br>(The 1-Page Front Office Brief)"]
        MD2["AUTOPSY_FAILED_REBUILDS.md<br>(Forensic Breakdown of Purgatory)"]
        MD3["CASE_STUDIES_CHAMPIONS.md<br>(Cup Winners & Final Four Blueprints)"]
        MD4["REGIME_TRENDS_AND_ERA_SHIFTS.md<br>(25-Year Macro Evolution)"]
        MD5["franchises/{TEAM}.md<br>(32 Individual Franchise Dossiers)"]
    end

    subgraph StreamlitApp ["Interactive Streamlit Dashboard (app/views/rebuild_tracker.py)"]
        T1["Tab 1: 32x25 Regime Matrix & Timeline"]
        T2["Tab 2: Succeeded vs Failed Quadrant (Simple Method)"]
        T3["Tab 3: 'Why Did It Succeed/Fail?' Decomposer"]
        T4["Tab 4: Franchise Retrospective Dossier"]
        T5["Tab 5: 4-Era Macro Trends & Shifts"]
        T6["Tab 6: Front-Office Rebuild Simulator"]
    end

    subgraph Lakehouse ["Hugging Face Lakehouse Sync (jjerlee/nhl-analytics)"]
        HF1["insights/rebuild/panel_summary.parquet"]
        HF2["insights/rebuild/episodes_annotated.parquet"]
        HF3["insights/rebuild/rebuild_manifest.json"]
    end

    DataEngine --> Generator
    Generator --> MarkdownDossiers
    Generator --> Lakehouse
    Lakehouse --> StreamlitApp
    DataEngine -. Local Fallback .-> StreamlitApp
```

---

## 2. Part I: Programmatic Markdown Insights & Dossier Generation

### 2.1 Generated Insights Directory Structure

All analytical summaries, franchise dossiers, and failure autopsies will be generated programmatically by `src/generate_rebuild_insights.py` and saved under `dev/player_team_success/rebuild/results/insights/`:

```
dev/player_team_success/rebuild/results/insights/
├── EXECUTIVE_SUMMARY.md               # C-suite briefing: Core findings, rulebooks, and key stats
├── AUTOPSY_FAILED_REBUILDS.md         # Forensic dissection of the 12 most instructive failed rebuilds
├── CASE_STUDIES_CHAMPIONS.md          # Deconstruction of all 25 Stanley Cup winners & Final Four finalists
├── REGIME_TRENDS_AND_ERA_SHIFTS.md    # 25-year macro trends across the 4 CBA & economic eras
├── rebuild_manifest.json              # Machine-readable JSON summary for dashboard hydration
└── franchises/                        # 32 individual franchise longitudinal dossiers
    ├── ANA.md                         # Mighty Ducks / Anaheim Ducks (2000–2025)
    ├── ARI.md                         # Phoenix / Arizona Coyotes / Utah Hockey Club
    ├── BOS.md                         # Boston Bruins (Perpetual Retool Case Study)
    ├── BUF.md                         # Buffalo Sabres (The Decade-Long Purgatory Autopsy)
    ├── CAR.md                         # Carolina Hurricanes (From Drought to Analytic Contender)
    ├── CHI.md                         # Chicago Blackhawks (The Era 2 Tank Dynasty & 2023 Reset)
    ├── COL.md                         # Colorado Avalanche (The 2016–17 Valley to 2022 Peak)
    ├── EDM.md                         # Edmonton Oilers (The 1-OA Lottery Era vs Modern Era)
    ├── FLA.md                         # Florida Panthers (Masterclass in Pathway B Retooling)
    ├── LAK.md                         # Los Angeles Kings (Cup Windows & Dual Rebuild Waves)
    ├── PIT.md                         # Pittsburgh Penguins (The Archetypal Tank Dynasty)
    ├── TBL.md                         # Tampa Bay Lightning (Hybrid Sourcing & Tactical Physicality)
    ├── VGK.md                         # Vegas Golden Knights (Zero-Tank Contention Blueprint)
    ├── WSH.md                         # Washington Capitals (Ovechkin Era Longevity)
    └── ... [all 32 NHL franchises]
```

---

### 2.2 Master Executive Summary (`EXECUTIVE_SUMMARY.md`)

The `EXECUTIVE_SUMMARY.md` serves as the authoritative, 10-minute briefing designed for NHL Team Presidents, General Managers, and sports analytics directors. It synthesizes the academic econometrics into unambiguous strategic takeaways:

#### Required Sections & Visualizations:
1. **Epistemological Scope Banner (Post-Hoc Retrospective Historical Evaluation)**:
   - Explicit header callout stating that all regime classifications (`Inaction`, `Rebuild`, `Peak`) and multi-season rebuild evaluations are retrospective historical measurements of 2000–2025, not deterministic forecasts of future seasons—while distilling the general economic mechanisms that make front-office decisions **effective or tragically ineffective** in moving from `Inaction` and `Rebuild` to `Peak`.
2. **The Three Macro Equilibrium Regimes (`Inaction`, `Rebuild`, `Peak`) & Executive Bottom Line**:
   - **League-Wide 25-Year Regime Occupancy**:
     - **`Inaction` Regime** (*Unsuccessful seasons with no major structural changes to improve & nothing to show for it, or stalled Purgatory*): $35.4\%$ of all franchise-seasons.
     - **`Rebuild` Regime** (*Unsuccessful/transitional seasons actively managing & in the post-process of major structural changes hoping to reach `Peak`*): $24.2\%$ of all franchise-seasons.
     - **`Peak` Regime** (*Successful seasons operating in sustained playoff, Final Four, or Stanley Cup contention*): $40.4\%$ of all franchise-seasons.
   - **Total Rebuild Episodes Analyzed** ($N = 68$ across 2000–2025) & **The True Odds of Transitioning `Rebuild` $\to$ `Peak`**:
     - **Tier 1 (Sustained Playoff Core $\ge 3$ yrs)**: $44.1\%$ of episodes.
     - **Tier 2 (Conference Final / Final Four)**: $26.5\%$ of episodes.
     - **Tier 3 (Stanley Cup Champion)**: $14.7\%$ of episodes.
     - **Relapse into `Inaction` / Rebuild Purgatory ($\ge 5$ yrs Stagnant)**: $33.8\%$ of episodes.
   - Average Velocity: Median time to Tier 2 `Peak` Contention is $5.4\text{ seasons}$ for teardowns vs. $3.1\text{ seasons}$ for aggressive retools.
3. **The "Great Rebuild Reversal" (Era 2 vs. Era 3/4)**:
   - **Pre-2015 (Era 2)**: $60\%$ of Stanley Cup champions were anchored by top-3 drafted superstars acquired via multi-year teardowns (Pittsburgh, Chicago, Los Angeles).
   - **Post-2015 (Eras 3 & 4)**: $70\%$ of champions and finalists were built via **Hybrid Retooling** (St. Louis, Tampa Bay, Vegas, Florida). The draft-to-championship pipeline compressed due to post-ELC second contract inflation and draft pick variance.
4. **The Five Deadly Sins of Failed Transitions (`Rebuild` $\to$ `Inaction` Collapse)**:
   - *Sin 1: The Youth Overload Trap* (Exceeding $30\%$ ice-time share to `0–3 yr` ELC skaters without veteran shelter, inducing negative cultural momentum $-\kappa_{\text{tank}}$).
   - *Sin 2: The RFA Cap-Squeeze* (Letting a teardown exceed 4 years, causing early top picks to exit ELCs and demand $\$8\text{M}–\$11\text{M}$ before supporting depth arrives).
   - *Sin 3: Positional Capital Misallocation* (Centering rebuilds on wingers or volatile goaltending rather than anchoring top-pair defensemen and #1 centers).
   - *Sin 4: The Reactive Hit Illusion* (Accumulating defensive-zone chasing hits while bleeding puck possession, rather than proactive forecheck hits in the offensive zone).
   - *Sin 5: The Panic Trade Collapse* (GM trading future 1st-round draft capital to save employment during Year 3 or 4 of an incomplete rebuild).
5. **The Modern Front-Office Rebuild Rulebook**:
   - 10 actionable decision rules summarizing empirical findings (`RULE-1` to `RULE-10`) for transitioning out of `Inaction` and `Rebuild` into `Peak`.

---

### 2.3 Franchise Retrospective Dossiers (`insights/franchises/{TEAM}.md`)

Each of the 32 franchise dossiers provides a 25-year chronological breakdown formatted with GitHub markdown tables, mermaid timelines, and retrospective three-regime occupancy cards (`Inaction`, `Rebuild`, `Peak`):

```markdown
# Franchise Dossier: Buffalo Sabres (BUF) — 2000 to 2025
> **Retrospective Historical Note**: This dossier is a post-hoc historical evaluation of 2000–2025 and is not a deterministic forecast of future franchise performance.

## 1. Executive Rebuild & Three-Regime Occupancy Summary
- **25-Year Macro Equilibrium Regime Distribution (`Inaction`, `Rebuild`, `Peak`)**:
  - **Peak Regime**: 20.0% (5 seasons: 2000–01, 2005–07, 2009–11)
  - **Rebuild Regime**: 32.0% (8 seasons of active teardown & youth incubation)
  - **Inaction Regime**: 48.0% (12 seasons of passive slump & chronic Purgatory stagnation)
  - **Predominant Historical Regime**: `Inaction` (48.0% occupancy; 0.0% Rebuild -> Peak conversion rate since 2011)
- **Total Rebuild Episodes**: 2 (`EP-BUF-01`: 2013–14 to 2020–21; `EP-BUF-02`: 2021–22 to Present).
- **Highest Contention Tier Reached**: Tier 2 (2006 & 2007 Conference Finals); None since 2011.
- **Total Draft Capital Consumed (EDV)**: 1,420 points (Rank #1 in NHL over 2013–2023).
- **Draft Efficiency Realization (κ_draft)**: 0.582 (League Avg: 1.000 — 41.8% value destruction).

## 2. 25-Year Regime Timeline (2000–2025)
```mermaid
gantt
    title Buffalo Sabres Macro & Operational Regimes (2000–2025)
    dateFormat YYYY
    section Regimes
    Peak (Briere/Drury Contender)       :done,    r0, 2005, 2007
    Inaction (Post-Drury Slump)         :active,  r1, 2008, 2013
    Rebuild (Scorched-Earth Teardown)   :crit,    r2, 2013, 2015
    Rebuild (Eichel/Reinhart Incubation):crit,    r3, 2015, 2018
    Inaction (Rebuild Purgatory Trap)   :crit,    r5, 2018, 2021
    Rebuild (Dahlin/Power Second Reset) :active,  r4, 2021, 2025
```

## 3. Episode Autopsy: Why the 2013–2021 Rebuild Collapsed into Inaction
- **Initiation Metric**: ALI = 0.842 (Aggressive liquidation of Miller, Vanek, Pominville, Campbell).
- **Peak Youth Share**: 41.2% in 2015–16 (Exceeded the 30% Youth Overload threshold).
- **Veteran Insulation Ratio (VIR)**: 0.28 (Severely under-insulated; League benchmark >= 0.70).
- **Root Causes of Tragically Ineffective Transition (Rebuild -> Inaction)**:
  1. Contract Desynchronization: Eichel (\$10\text{M}) and Reinhart (\$5.2\text{M}) exited ELCs before defensive infrastructure matured.
  2. D-Zone Hit Chasing: High hit volume driven by defensive-zone pinning (PFHS = 28.4% vs League Avg 41.2%).
  3. Asset Arbitrage Loss: Traded prime assets at suppressed valuations (O'Reilly trade yield: -0.42 SD talent deficit).
```

---

### 2.4 Forensic Failure Autopsies (`AUTOPSY_FAILED_REBUILDS.md`)

This document provides a forensic post-mortem on the 12 most severe failed rebuild episodes in modern NHL history:
1. **Buffalo Sabres (2013–2021)**: The archetypal multi-lottery scorched-earth failure.
2. **Edmonton Oilers (2009–2019)**: Four 1st-overall picks trapped in cultural attrition until McDavid/Draisaitl maturity.
3. **Arizona Coyotes (2014–2021)**: Perpetual liquidation, low veteran insulation, and arena/ownership instability.
4. **Ottawa Senators (2018–2024)**: Post-2017 teardown, high youth talent, but delayed emergence due to blue-line depth deficits.
5. **Detroit Red Wings (2017–Present)**: The prolonged incubation curve post-25-year playoff streak.
6. **Philadelphia Flyers (2014–2022)**: The half-measure trap (alternating between retooling and slumping without committing).
7. **Vancouver Canucks (2014–2021)**: The accelerated retool error (overpaying bottom-six free agents to fast-track youth).
8. **Montreal Canadiens (2012–2021)**: Goalie-dependent veneer masking systemic structural scoring deficiencies.
9. **Toronto Maple Leafs (2005–2014)**: The pre-Shanahan trade-away-draft-picks era.
10. **New Jersey Devils (2013–2021)**: The post-Parise/Kovalchuk transition and the secondary Hischier/Hughes reset.
11. **Calgary Flames (2010–2014)**: Delaying the inevitable teardown of the Iginla/Kipper core.
12. **New York Islanders (2007–2014)**: The Tavares early era and arena/rebuilding frictions.

For each episode, the report details:
- **Timeline & Transition Dates**.
- **Quantified Failure Metrics**: Realized Draft Value vs. Expected, Youth Overload Index, RFA Cap Squeeze Loss, and Origin Trade Arbitrage Score.
- **Counterfactual Simulation**: "What if the franchise had adopted a 2018 Florida Panthers Hybrid Retool strategy instead?"

---

### 2.5 Historical Era Transitions & Regime Shifts (`REGIME_TRENDS_AND_ERA_SHIFTS.md`)

Documents the macroeconomic and tactical changes across the 25-year sample:
- **Era 1 (2000–01 to 2003–04: Pre-Cap Era)**: Uncapped payrolls, veteran stacking, low draft pick liquidity, slow rebuild velocity.
- **Era 2 (2005–06 to 2012–13: Early Salary Cap & Golden Age of Tanking)**: Introduction of the hard cap (\$39\text{M} to \$64\text{M}), high ELC surplus value, low second-contract salaries, Pittsburgh/Chicago/LA dynasty blueprints.
- **Era 3 (2013–14 to 2019–20: Analytic Speed, Skill & ELC Compression)**: Introduction of the 2013 CBA, elimination of 10+ year back-diving contracts, RFA stars demanding maximum cap percentage off ELCs, draft pick depreciation.
- **Era 4 (2020–21 to 2024–25: Flat-Cap, Cap-Squeeze & Asset Arbitrage)**: COVID-induced flat cap (\$81.5\text{M} for 4 years), cap space as a weapon, prime-age trade acquisitions superseding raw draft tanking (Vegas, Florida, Carolina).

---

### 2.6 Automation Script Specification (`generate_rebuild_insights.py`)

A standalone script in `dev/player_team_success/rebuild/src/generate_rebuild_insights.py` will execute the generation pipeline using `uv run python`:

```python
"""
generate_rebuild_insights.py
Generates Markdown Dossiers, Failure Autopsies, and Executive Briefs from Rebuild Panels.
"""
import os
import json
import duckdb
import pandas as pd
import numpy as np

def generate_all_insights():
    # 1. Load Parquet Panels
    # 2. Compute Franchise Retrospective Metrics
    # 3. Classify Historical Episodes into Tiers (Succeeded vs Purgatory)
    # 4. Generate EXECUTIVE_SUMMARY.md
    # 5. Generate 32 Franchise Markdown Dossiers
    # 6. Generate AUTOPSY_FAILED_REBUILDS.md
    # 7. Generate REGIME_TRENDS_AND_ERA_SHIFTS.md
    # 8. Export rebuild_manifest.json for Streamlit hydration
    pass

if __name__ == "__main__":
    generate_all_insights()
```

---

## 3. Part II: Interactive Streamlit Dashboard Blueprint (`rebuild_tracker.py`)

### 3.1 Application Layout, Styling & Navigation Integration

- **File Path**: [`app/views/rebuild_tracker.py`](file:///Users/jjlee/Documents/github/nhl-analytics/app/views/rebuild_tracker.py)
- **Navigation Registration** in [`app/app.py`](file:///Users/jjlee/Documents/github/nhl-analytics/app/app.py):
  ```python
  REBUILD_TRACKER = st.Page(
      page="views/rebuild_tracker.py", 
      title=t("nav_rebuild_tracker"), 
      url_path="rebuild-tracker"
  )
  # Registered under the "Teams" navigation section:
  # pages[t("nav_team")] = [SCOREBOARD, GAME_RECAPS, TEAM_MAP, TEAM_SEASON_STATS, REBUILD_TRACKER]
  ```
- **Page Layout**: Wide layout (`layout="wide"`) with custom Rinklytics styling:
  - **Persistent Top Epistemological Banner (`st.info` / Custom Banner)**: Prominently informs users that this dashboard presents a **post-hoc, retrospective historical study (2000–2025)** embedding a three-state Markov-switching equilibrium model (`Inaction`, `Rebuild`, `Peak`). It is **not** a predictive forecast of future franchise behaviors, but an empirical evaluation of which historical decisions were **effective or tragically ineffective** in moving teams from `Inaction` and `Rebuild` into `Peak`.
  - Metric cards with subtle drop shadows, custom KPI status tags (`Peak Regime`, `Rebuild Regime`, `Inaction Regime`, `Succeeded`, `Purgatory Trap`).
  - Bilingual string dictionary (`EN` and `FR`) using `i18n.t()`.
  - Responsive Plotly interactive charts with unified hover templates, team colors, and clean dark/light rink styling.

---

### 3.2 Tab 1: The 25-Year Franchise Regime Matrix, Three-Regime Equilibrium Toggle (`Inaction`, `Rebuild`, `Peak`) & Team Predominance Audit

**Purpose**: Provide an immediate, panoramic bird's-eye view of every franchise's operational state across all 25 seasons (2000–01 to 2024–25) and quantify which of the three macro regimes (`Inaction`, `Rebuild`, `Peak`) each franchise predominantly occupied.

#### Visual & Interactive Components:
1. **Regime Granularity Toggle**:
   - **View Mode A (Default — Three Macro Econometric Equilibrium Regimes)**:
     - 🟩 **`Peak` Regime ($M_{j, t} = \text{Peak}$)**: Successful seasons operating in sustained playoff, Final Four, or Stanley Cup contention (`R0_CONTENDER` & sustained `R4_EMERGENCE`).
     - 🟧 **`Rebuild` Regime ($M_{j, t} = \text{Rebuild}$)**: Unsuccessful/transitional seasons where the franchise is actively managing and in the ongoing/post-process of major structural changes hoping to reach `Peak` (`R2_TEARDOWN`, `R3_INCUBATION`, transitional `R4`).
     - ⬛ **`Inaction` Regime ($M_{j, t} = \text{Inaction}$)**: Unsuccessful seasons where the team either makes no major changes to improve and has nothing to show for it (`R1_SLUMP`) or is trapped in chronic Purgatory stagnation (`R5_PURGATORY`).
   - **View Mode B (Granular Operational Sub-States `R0`–`R5`)**:
     - `R0_CONTENDER` (`#1b7837`), `R1_SLUMP` (`#e69f00`), `R2_TEARDOWN` (`#b2182b`), `R3_INCUBATION` (`#d6604d`), `R4_EMERGENCE` (`#2166ac`), `R5_PURGATORY` (`#252525`).
2. **The 32x25 Regime Heatmap + Right-Margin Franchise Predominance Bar Chart**:
   - **X-Axis**: Seasons ($2000–01$ to $2024–25$).
   - **Y-Axis**: 32 NHL Franchises (Sortable by `Peak` Occupancy %, `Inaction` Occupancy %, Stanley Cup count, or Division).
   - **Right-Margin Horizontal Stacked Bar**: Directly displays for each franchise $j$ its retrospective 25-year regime breakdown $\left(\widehat{\text{Share}}_j(\text{Peak}), \, \widehat{\text{Share}}_j(\text{Rebuild}), \, \widehat{\text{Share}}_j(\text{Inaction})\right)$ and badges its **Predominant Historical Regime** (`Mostly Peak`, `Mostly Rebuild`, or `Mostly Inaction`).
3. **Interactive Hover Card**:
   - Hovering over any cell displays: Team Logo, Season, Macro Regime (`Inaction` / `Rebuild` / `Peak`), Granular Sub-State (`R0`–`R5`), Smoothed Regime Probability $\Pr(M_{j, t} \mid \mathcal{Y}_{j, 1:T})$, Record, Points %, Core Continuity ($\text{CCM}$), and Asset Liquidation Index ($\text{ALI}$).
4. **Click-to-Filter Synchronization**:
   - Clicking any franchise row automatically updates Tabs 3 and 4 to display that team's deep-dive dossier.

```
+---------------------------------------------------------------------------------------------------------+
| [Mode: ● 3 Macro Regimes (Inaction / Rebuild / Peak)  ○ 6 Sub-States (R0-R5)]  [Sort: Peak % | Cups]    |
|                                                                                                         |
|  32x25 REGIME HEATMAP (2000–01 to 2024–25)                 25-YR REGIME OCCUPANCY & PREDOMINANT REGIME  |
|  [BOS] 🟩🟩🟧🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟨🟨🟩🟩🟩🟩🟩🟩🟩🟩🟩   [76% Peak | 8% Reb | 16% Inact] -> PEAK     |
|  [PIT] 🟩🟨🟥🟥🟧🟧🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟨🟨🟨   [64% Peak | 20% Reb| 16% Inact] -> PEAK     |
|  [TBL] 🟨🟨🟩🟩🟩🟨🟥🟧🟧🟩🟨🟧🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩   [60% Peak | 24% Reb| 16% Inact] -> PEAK     |
|  [BUF] 🟩🟨🟨🟨🟩🟩🟨🟨🟨🟩🟩🟨🟨🟥🟥🟧🟧🟧⬛⬛⬛🟧🟧🟧🟨   [20% Peak | 32% Reb| 48% Inact] -> INACTION |
|  ...                                                                                                    |
|  [Legend: 🟩 Peak (Contending)   🟧 Rebuild (Active Restructuring)   🟨/⬛ Inaction (Slump / Purgatory)]|
+---------------------------------------------------------------------------------------------------------+
```

---

### 3.3 Tab 2: Succeeded vs. Failed Rebuild Scorecard & Quadrant (The Simple Method)

**Purpose**: Provide users with a dead-simple, highly intuitive method to see which rebuilds succeeded, which failed, and how long they took.

#### Visual & Interactive Components:
1. **Executive KPI Ribbon**:
   - `Total Episodes Identified`: $68$
   - `Succeeded (Tier 1/2/3)`: $30$ ($44.1\%$)
   - `Trapped in Purgatory`: $23$ ($33.8\%$)
   - `Median Duration to Contention`: $5.4\text{ Seasons}$
2. **The "Rebuild Outcome Quadrant" (Interactive Scatter Plot)**:
   - **X-Axis**: *Rebuild Duration* (Years spent in Teardown + Incubation before playoff return, $1\text{ to }10\text{ years}$).
   - **Y-Axis**: *Peak Contention Lift* ($\Delta \text{Points}\%$ over pre-rebuild baseline or Playoff Rounds Won during peak 3-year window).
   - **Bubble Size**: Total Draft Capital Invested ($\sum \text{EDV}$ of top-60 picks).
   - **Bubble Color**:
     - 🟢 **Succeeded — Tier 3 (Stanley Cup Winner)**: e.g. Pittsburgh (2002–06), Chicago (2004–08), Colorado (2016–19), Tampa Bay (2010–14).
     - 🔵 **Succeeded — Tier 2 (Final Four / Conference Final)**: e.g. Carolina (2014–19), Ottawa (2000–03), NY Rangers (2018–22).
     - 🟡 **Succeeded — Tier 1 (Consistent Playoff Team $\ge 3$ yrs)**: e.g. Toronto (2014–17), New Jersey (2018–23).
     - 🔴 **Failed / Purgatory Trap**: e.g. Buffalo (2013–21), Edmonton (2009–15), Arizona (2014–21), Detroit (2017–23).
3. **Quadrant Annotations**:
   - *Top-Left: "Rapid Turnaround / Masterclass Retool"* (Quick turnaround $\le 3$ yrs, high contention lift).
   - *Top-Right: "Long-Payoff Tank Dynasty"* (5–7 yrs teardown, sustained dynasty peak).
   - *Bottom-Right: "Rebuild Purgatory / Capital Sink"* ($>5$ yrs, low lift, trapped).
   - *Bottom-Left: "Failed Accelerated Retool"* (Cut rebuild short, immediate plateau).
4. **Searchable & Filterable Summary Table**:
   - Table columns: `Franchise`, `Episode Dates`, `Rebuild Archetype` (Draft-Centric vs. Retool vs. Hybrid), `Years in Incubation`, `Outcome Tier`, `Core Sourced From Draft %`, `Peak Result`.
   - Export to CSV / Markdown button.

```
+---------------------------------------------------------------------------------------------------------+
| REBUILD OUTCOME QUADRANT: Duration vs. Peak Contention Lift                                             |
|                                                                                                         |
| Peak Contention Lift (Playoff Rounds / Pts Δ)                                                           |
|    ▲                                                                                                    |
|    │       [TBL 10-14] 🟢            [CHI 04-09] 🟢                                                     |
|    │   [FLA 18-21] 🟢                [COL 16-19] 🟢                                                     |
|    │                                 [PIT 02-06] 🟢                                                     |
|    │   [NYR 18-22] 🔵                                                                                   |
|    │   [TOR 14-17] 🟡               [CAR 14-19] 🔵                                                     |
|    │ ───────────────────────────────────────────────────────────────────                                |
|    │   [VAN 14-17] 🔴                                  [EDM 09-16] 🔴                                   |
|    │                                                   [ARI 14-21] 🔴                                   |
|    │                                                   [BUF 13-21] 🔴 (10 Yrs, 0 Playoffs)              |
|    └───────────────────────────────────────────────────────────────────► Duration (Years)               |
|         1 yr            3 yrs            5 yrs            7 yrs            10 yrs                       |
+---------------------------------------------------------------------------------------------------------+
```

---

### 3.4 Tab 3: "Why Did It Succeed or Fail?" Forensic Decomposer

**Purpose**: Allow users to select any specific rebuild episode (or compare two contrasting rebuilds, e.g. *2016–2022 Colorado Avalanche* vs. *2013–2021 Buffalo Sabres*) and visually deconstruct the root causal mechanisms.

#### Visual & Interactive Components:
1. **Episode Selector**:
   - Compare Episode A (e.g. `COL: 2016–2022 [Succeeded - Tier 3]`) vs. Episode B (e.g. `BUF: 2013–2021 [Failed - Purgatory]`).
2. **The 5-Pillar Rebuild Radar / Spider Chart**:
   - Visualizes the 5 core strategic pillars on a 0–100 percentile scale against League Champions vs. Purgatory average:
     - **Pillar 1: Draft Capital Yield ($\kappa_{\text{draft}}$)**: Realized performance vs. draft pick expectation.
     - **Pillar 2: Core Continuity Metric ($\text{CCM}$)**: Retaining top-3 forwards and top-2 defensemen.
     - **Pillar 3: Veteran Insulation Ratio ($\text{VIR}$)**: Shielding rookie ELCs ($0–3$ yrs) with peak veterans ($7–9$ yrs).
     - **Pillar 4: Trade Arbitrage Yield ($\psi_2$)**: Acquiring undervalued prime skaters from distressed teams.
     - **Pillar 5: Forecheck Physicality & Momentum ($\text{PFHS} \times \text{SPSI}$)**: Offensive zone hits driving turnover conversions vs. defensive chasing.
3. **Causal Attribution Waterfall (Shapley / Parameter Contribution Bar Chart)**:
   - Quantifies exactly how much each variable contributed to that team's success or failure (in terms of estimated Contention Probability $\Delta P$ from the econometric model):
     - $+18.4\%$ from High-End Center/Defense Core Selection.
     - $+14.2\%$ from Masterstroke Prime Trade (e.g. Devon Toews trade arbitrage).
     - $-12.6\%$ from Defensive Zone Hit Chasing.
     - $-19.1\%$ from Contract Desynchronization / Youth Overload.
4. **Executive Verdict & Post-Mortem Card**:
   - Dynamic narrative card explaining the strategic lesson of that rebuild episode.

```
+---------------------------------------------------------------------------------------------------------+
| SELECT EPISODE: [COL (2016–2022) - Stanley Cup]  vs.  [BUF (2013–2021) - Purgatory Trap]                |
|                                                                                                         |
| 5-PILLAR REBUILD RADAR COMPARISON                 SHAPLEY CAUSAL ATTRIBUTION (POINTS LIFT)              |
|              Draft Yield                                                                                |
|                 100                               Center/D Core Lift:  [========+] +14.2 pts            |
|                  /\                               Trade Arbitrage (ψ): [======+]   +9.8 pts             |
|   Trade (ψ)     /  \     Core Retention           Veteran Insulation:  [====+]     +6.1 pts             |
|        \       /COL \       /                     Youth Overload Pen:  [---]       -2.4 pts             |
|         \     /      \     /                      D-Zone Hit Chasing:  [-]         -0.8 pts             |
|          \   /   BUF  \   /                                                                             |
|           \ /__________\ /                        NET REBUILD LIFT:    +26.9 Points Percentage          |
|    Tactical Hits      Veteran Insulation                                                                |
+---------------------------------------------------------------------------------------------------------+
```

---

### 3.5 Tab 4: Franchise Longitudinal Retrospective Dossier

**Purpose**: A dedicated interactive franchise portal where users can select any of the 32 NHL teams and explore their complete 25-year history.

#### Visual & Interactive Components:
1. **Franchise Selector & Header**:
   - Displays Team Logo, All-Time Record (2000–2025), Stanley Cups Won, Conference Finals Made, Total Rebuilds Attempted.
2. **Triple-Track Longitudinal Timeline (Plotly Synchronized Figure)**:
   - **Track 1 (Performance)**: Points Percentage (line chart) with Playoff Appearances (points) and Stanley Cup wins (gold star markers).
   - **Track 2 (Regime Banner)**: Color-coded operational regime ribbon indicating exactly when the team was Contending, Slumping, Tearing Down, Incubating, Emerging, or in Purgatory.
   - **Track 3 (Roster Sourcing Streamgraph)**: Percentage of team ice time (TOI) sourced from:
     - 🔵 *In-House Drafted Core* (Pathway A)
     - 🟢 *Acquired Prime Trade Core* (Pathway B)
     - 🟡 *Free Agent Signings*
     - 🔴 *Rookie ELC Players ($0–3$ yrs tenure)*
     - 🟣 *Peak Veterans ($7–9$ yrs tenure)*
3. **Core Player Tree & Roster Architecture**:
   - Interactive table/network showing the pillars of each era: how they were acquired (Draft, Trade, FA), their age during contention, their peak $\text{xGF}\%$, and their contract surplus value.
4. **GM Tenure & Major Transaction Callout Cards**:
   - Vertical timeline showing General Manager tenures and landmark trades that accelerated or sabotaged the rebuild.

---

### 3.6 Tab 5: Macro Trends, CBA Regimes & Era Evolution

**Purpose**: Show past trends by team and explain how the structural economics of rebuilding transformed across the four distinct historical eras.

#### Visual & Interactive Components:
1. **Era Selector**:
   - `All Eras (2000–2025)` | `Era 1: Pre-Cap (2000–04)` | `Era 2: Early Cap / Tank Dynasties (2005–13)` | `Era 3: Analytic Speed & Compression (2013–20)` | `Era 4: Flat-Cap & Arbitrage (2020–25)`.
2. **Key Longitudinal Trendlines**:
   - **Trend 1: The Collapse of the Rebuild Half-Life**: Shows how the average time an ELC star remains on an undervalued contract shrank from $6.2\text{ years}$ (Era 2) to $3.0\text{ years}$ (Era 4).
   - **Trend 2: Sourcing of Champions (Draft vs. Trade)**: Stacked area chart showing that while 2008–2015 champions were $70\%$ draft-built, 2018–2024 champions derived $>45\%$ of their core minutes from prime-age trades.
   - **Trend 3: The Rise of Rebuild Purgatory**: Line graph illustrating how the probability of getting stuck in Purgatory for $\ge 5$ years rose from $18\%$ in Era 2 to $42\%$ in Era 4.
3. **Team-by-Team Era Comparison Matrix**:
   - Table showing how every franchise performed across each of the 4 eras (Points %, Playoff Berths, Rebuild Episodes).

---

### 3.7 Tab 6: Interactive Rebuild Strategy Simulator (What-If Counterfactual Tool)

**Purpose**: An engaging "Front-Office Sandbox" allowing users to act as an NHL General Manager and test different rebuild strategies using our estimated structural econometric parameters.

#### Interactive Controls (User Inputs):
- **Rebuild Pathway**: *Aggressive Scorched-Earth Teardown* vs. *Hybrid Retool* vs. *Pure Trade Arbitrage*.
- **Youth Ice-Time Share ($s^{B_1}$)**: Slider ($10\%$ to $50\%$).
- **Veteran Insulation Ratio ($\text{VIR}$)**: Slider ($0.20$ to $1.20$).
- **Draft Pick Focus**: Slider (Targeting Top-5 picks vs. Deep draft capital volume).
- **Tactical Forecheck Hit Focus**: Slider (Passive D-zone containment vs. Aggressive O-zone forecheck hits).
- **Market Size & Ownership Mandate**: Dropdown (Low Revenue / High Leeway vs. High Revenue Original Six / Zero Patience).

#### Real-Time Counterfactual Outputs:
- 📈 **Predicted Probability of Sustained Contention ($P(\text{Tier 1})$ & $P(\text{Tier 2})$)** with $95\%$ credible intervals.
- ⏱️ **Expected Timeline to Contention** (e.g. $4.2\text{ seasons}$).
- ⚠️ **Purgatory Risk Meter**: Probability of stalling out for $\ge 5$ years (flags when Youth Overload $> 30\%$ triggers the penalty).
- 💡 **Executive Advice Box**: Automated feedback explaining the economic trade-offs of the user's selected configuration.

---

## 4. Part III: Data Contracts, JSON Caching & Hugging Face Lakehouse Sync

To ensure near-instantaneous load times in Streamlit and avoid recomputing heavy econometric models on every user interaction, all processed metrics are serialized into structured, lightweight artifacts:

### 4.1 Master JSON Summary Manifest (`rebuild_manifest.json`)

Stored under `dev/player_team_success/rebuild/results/insights/rebuild_manifest.json` and mirrored to Hugging Face:

```json
{
  "generated_at": "2026-11-06T14:40:00Z",
  "study_period": "2000-01 to 2024-25",
  "epistemological_notice": "Post-hoc retrospective historical evaluation of 2000-2025. Not a deterministic forecast of future franchise behavior; structural parameters quantify historical effectiveness of managerial transitions from Inaction and Rebuild to Peak.",
  "total_seasons": 25,
  "total_franchises": 32,
  "kpis": {
    "total_episodes": 68,
    "succeeded_tier1": 30,
    "succeeded_tier2": 18,
    "succeeded_tier3": 10,
    "purgatory_count": 23,
    "median_rebuild_years": 5.4,
    "league_share_peak": 0.404,
    "league_share_rebuild": 0.242,
    "league_share_inaction": 0.354
  },
  "franchise_regime_occupancy": {
    "BOS": {"share_peak": 0.76, "share_rebuild": 0.08, "share_inaction": 0.16, "predominant_regime": "Peak"},
    "PIT": {"share_peak": 0.64, "share_rebuild": 0.20, "share_inaction": 0.16, "predominant_regime": "Peak"},
    "TBL": {"share_peak": 0.60, "share_rebuild": 0.24, "share_inaction": 0.16, "predominant_regime": "Peak"},
    "COL": {"share_peak": 0.56, "share_rebuild": 0.28, "share_inaction": 0.16, "predominant_regime": "Peak"},
    "BUF": {"share_peak": 0.20, "share_rebuild": 0.32, "share_inaction": 0.48, "predominant_regime": "Inaction"}
  },
  "eras": {
    "Era 1": {"name": "Pre-Cap Wild West", "years": "2000-2004", "success_rate": 0.38},
    "Era 2": {"name": "Early Cap Tank Dynasties", "years": "2005-2013", "success_rate": 0.54},
    "Era 3": {"name": "Speed & ELC Compression", "years": "2013-2020", "success_rate": 0.41},
    "Era 4": {"name": "Flat-Cap Asset Arbitrage", "years": "2020-2025", "success_rate": 0.36}
  },
  "episodes": [
    {
      "episode_id": "EP-COL-02",
      "team": "COL",
      "start_season": "2016-17",
      "end_season": "2021-22",
      "duration": 5,
      "archetype": "Hybrid Retool",
      "outcome_tier": "Tier 3 (Stanley Cup)",
      "edv_invested": 612.4,
      "youth_overload_peak": 0.28,
      "vir_average": 0.74,
      "trade_arbitrage_score": 0.88,
      "verdict": "Masterclass in pairing top drafted center (MacKinnon) with prime trade arbitrage (Toews, Kadri) and veteran insulation."
    }
  ]
}
```

### 4.2 Streamlit Data Hydration Protocol

In [`app/views/rebuild_tracker.py`](file:///Users/jjlee/Documents/github/nhl-analytics/app/views/rebuild_tracker.py):
1. **Primary Source**: Hugging Face Datasets (`jjerlee/nhl-analytics` via `load_hf_dataset()`) for fast ZeroGPU / cloud container execution.
2. **Local Fallback**: Local Parquet files in `dev/player_team_success/rebuild/results/` or `latest/rebuild/`.
3. **Caching**: `@st.cache_data(ttl=3600)` on all data loader functions.

---

## 5. Part IV: Implementation Roadmap & Verification Checklist

| Phase | Milestone / Task | Target Script / Output File | Status |
| :--- | :--- | :--- | :---: |
| **Phase 1** | **Insights Generator Pipeline** | `src/generate_rebuild_insights.py` | Planned |
| **Phase 2** | **Executive Summary & Autopsies** | `results/insights/EXECUTIVE_SUMMARY.md`<br>`results/insights/AUTOPSY_FAILED_REBUILDS.md`<br>`results/insights/REGIME_TRENDS_AND_ERA_SHIFTS.md` | Planned |
| **Phase 3** | **32 Franchise Dossiers** | `results/insights/franchises/{TEAM}.md` | Planned |
| **Phase 4** | **Streamlit Rebuild Tracker View** | `app/views/rebuild_tracker.py` | Planned |
| **Phase 5** | **Navigation & i18n Registration** | `app/app.py` & `app/i18n.py` | Planned |
| **Phase 6** | **Local Docker Verification** | Port 1901 (`docker compose up -d --build`) | Mandatory |
| **Phase 7** | **Hugging Face Space Sync** | Deploy without LFS bloat (< 5 MB) | Mandatory |

---

### Strict Development Rules Compliance:
- **No Git Commit Rule**: All files remain uncommitted in the local working directory.
- **UV Execution**: All Python testing executed with `uv run python`.
- **Local Docker Testing**: Verify at `http://localhost:1901` before finalizing.
