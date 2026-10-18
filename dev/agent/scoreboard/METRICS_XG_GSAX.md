# Expected Goals ($xG$) & Goals Saved Above Expected ($GSAx$) — Measurement, Empirical Validation & Model-Based Estimation Roadmap

## 1. Executive Overview & Canonical Architecture

This document defines the mathematical formulation, empirical calibration, data pipeline architecture, and future model-based estimation roadmap for **Expected Goals ($xG$)**, **Individual Expected Goals ($ixG$)**, **High-Danger Chances ($HDCF$)**, and **Goals Saved Above Expected ($GSAx$)** across **NHL Analytics (Rinklytics)**.

### Canonical Source Code Locations
- **Core Measurement & Estimation Engine**: [`src/data/apinhle/function/procs_xg.py`](../../src/data/apinhle/function/procs_xg.py)
  - Houses all shot-level coordinate geometry, Dual-Stage Conditional Logistic Calibration, game-level team & player aggregation, season-level rolling/finalized aggregation, and historical panel enrichment.
- **Standalone & Pipeline CLI Runner**: [`src/data/apinhle/data_proc_xg.py`](../../src/data/apinhle/data_proc_xg.py)
  - Enriches `latest/box/{year}_box.csv`, `latest/box/{year}_box_team.csv`, `latest/box/{year}_box_gameStats.csv`, `latest/team/season/{year}_team_season.csv`, `latest/box/{year}_box_player.csv`, and `latest/player/{year}_player.csv`.
- **Automated Ingestion Hooks**:
  - [`src/data/apinhle/data_proc_gameSummary.py`](../../src/data/apinhle/data_proc_gameSummary.py) (enriches team box, game stats, and team season KPIs)
  - [`src/data/apinhle/data_pull_player.py`](../../src/data/apinhle/data_pull_player.py) (enriches player game logs and rolling/season-ending player tables)
  - [`.github/workflows/proc_pull_box.yml`](../../.github/workflows/proc_pull_box.yml) (daily GitHub Actions workflow)
- **Scoreboard Agent Re-Export**: [`dev/agent/scoreboard/metrics_xg.py`](metrics_xg.py)
  - Cleanly re-exports all functions and constants from `src.data.apinhle.function.procs_xg` for backward compatibility with [`ScoreboardAgent`](scoreboard_agent.py) and [`game_recaps.py`](../../app/views/game_recaps.py).

---

## 2. Methodological Evaluation & Dual-Stage Conditional Calibration

### 2.1 The Fundamental Econometric Distinction: Unblocked Attempts (Fenwick) vs. Shots on Goal (SOG)
An unblocked shot attempt ($F \in \{\text{goal}, \text{shot-on-goal}, \text{missed-shot}\}$) passes through two sequential stochastic stages:
1. **On-FrameAccuracy Stage**: Whether the unblocked attempt hits the net frame ($SOG = 1$) vs. misses wide or hits the crossbar/post ($SOG = 0$, ~34.8% of NHL Fenwick attempts). Because a missed shot cannot result in a goal ($P(G=1 \mid SOG=0) = 0$), the unconditional Fenwick conversion probability is the product:
   $$P(G = 1 \mid F = 1, \mathbf{x}) = P(SOG = 1 \mid F = 1, \mathbf{x}) \cdot P(G = 1 \mid SOG = 1, \mathbf{x})$$
2. **Goaltender Save Stage**: Conditional on the shot reaching the net frame ($SOG = 1$) against a non-empty net ($EN = 0$), the goaltender must prevent a goal against.

#### Why Single-Stage Calibration Fails
If a single probability model is applied simultaneously to all Fenwick attempts (for team $xG$ / skater $ixG$) and restricted to Shots on Goal (for goaltender $xGA$ / $GSAx$):
- Calibrating to **SOG conversion rates (~10.21%)** causes total Fenwick $xG$ to **over-predict league goals by +22.8%** (because missed shots are assigned on-goal conversion probabilities).
- Conversely, calibrating to **Fenwick conversion rates (~6.82%)** causes goaltender on-target $xGA$ to **under-predict non-empty-net goals against by -20.8%**, forcing league-wide $\sum GSAx \ll 0$ (`-1,726.5` goals across a season instead of `0.0`).

### 2.2 Dual-Stage Conditional Logistic Specification
To guarantee zero-sum calibration across both offensive shot creation and goaltending evaluation, [`procs_xg.py`](../../src/data/apinhle/function/procs_xg.py) implements a **Dual-Stage Conditional Logistic Model**:

#### Spatial Coordinate Normalization & Effective Crease Distance
Given NHL API coordinates $(x, y)$ in feet (rink $[-100, 100] \times [-42.5, 42.5]$, goal line at $|x| = 89\text{ ft}$):
1. Normalize attacking direction using `homeTeamDefendingSide` and `zoneCode` (`O`, `N`, `D`) so the target net is at $(89, 0)$:
   $$d = \sqrt{(89 - x_{\text{norm}})^2 + y^2}, \qquad \theta = \arctan2(|y|, \max(89 - x_{\text{norm}}, 1.0)) \cdot \frac{180}{\pi}$$
2. **Crease-Jam Clamping ($d_{\text{eff}} = \max(d, 10.5\text{ ft})$)**: Empirical NHL conversion rates peak in the inner slot ($8\text{–}15\text{ ft}$) and plateau or slightly decline inside $0\text{–}7\text{ ft}$ due to goaltender pad seals and jammed net-front scrambles. Clamping effective distance at $10.5\text{ ft}$ prevents linear log-odds extrapolation from inflating jam-shot probabilities.
3. **Mid-Slot Release Corridor Indicator ($\mathbb{I}_{\text{slot}}$)**: High-velocity wrist/snap releases from the mid-slot ($20 < d \le 35\text{ ft}$, $\theta \le 50^\circ$) benefit from pre-shot lateral movement and defensive screens, receiving a dedicated corridor log-odds bonus.

#### Stage 1: Fenwick Expected Goals ($xG_F$) — Team $xG$ & Skater $ixG$
Applied to all unblocked shot attempts (`shot-on-goal`, `missed-shot`, `goal`):
$$\eta_F(\mathbf{x}) = \beta_{0,F} + \beta_{d,F}\, d_{\text{eff}} + \beta_{\theta,F}\, \theta + \beta_{\text{slot},F}\, \mathbb{I}_{\text{slot}} + \gamma_{\text{type}}(\text{shotType}) + \delta_{\text{reb},F}\, \mathbb{I}_{\text{reb}} + \delta_{\text{rush},F}\, \mathbb{I}_{\text{rush}} + \delta_{\text{EN},F}\, \mathbb{I}_{\text{EN}}$$
$$xG_F = \sigma(\eta_F(\mathbf{x})) = \frac{1}{1 + \exp(-\eta_F(\mathbf{x}))}$$
- Parameters: $\beta_{0,F} = -1.25$, $\beta_{d,F} = -0.0395\text{ ft}^{-1}$, $\beta_{\theta,F} = -0.0135\text{ deg}^{-1}$, $\beta_{\text{slot},F} = +0.36$, $\delta_{\text{reb},F} = +0.68$ ($\Delta t \le 3\text{s}$), $\delta_{\text{rush},F} = +0.35$ ($\Delta t \le 4\text{s}$ from N/D zone), $\delta_{\text{EN},F} = +3.35$.

#### Stage 2: Conditional On-Goal Expected Goals Against ($xGA_{\text{SOG}}$) — Goaltender $GSAx$
Applied exclusively to non-empty-net Shots on Goal (`shot-on-goal`, `goal` with $\mathbb{I}_{\text{EN}} = 0$):
$$\eta_{\text{SOG}}(\mathbf{x}) = \beta_{0,\text{SOG}} + \beta_{d,\text{SOG}}\, d_{\text{eff}} + \beta_{\theta,\text{SOG}}\, \theta + \beta_{\text{slot},\text{SOG}}\, \mathbb{I}_{\text{slot}} + \gamma_{\text{type}}(\text{shotType}) + \delta_{\text{reb},\text{SOG}}\, \mathbb{I}_{\text{reb}} + \delta_{\text{rush},\text{SOG}}\, \mathbb{I}_{\text{rush}}$$
$$xGA_{\text{SOG}} = \sigma(\eta_{\text{SOG}}(\mathbf{x})) = \frac{1}{1 + \exp(-\eta_{\text{SOG}}(\mathbf{x}))}$$
- Parameters: $\beta_{0,\text{SOG}} = -0.82$, $\beta_{d,\text{SOG}} = -0.0385\text{ ft}^{-1}$, $\beta_{\theta,\text{SOG}} = -0.0125\text{ deg}^{-1}$, $\beta_{\text{slot},\text{SOG}} = +0.38$, $\delta_{\text{reb},\text{SOG}} = +0.62$, $\delta_{\text{rush},\text{SOG}} = +0.32$.
- **Goaltender GSAx**:
  $$\text{GSAx} = \sum_{i \in \text{SOG}_{\text{non-EN}}} xGA_{\text{SOG}, i} - GA_{\text{non-EN}}$$

### 2.3 Multi-Season Empirical Validation (`2023–2025` NHL Regular Seasons)
Validated across $N = 363,752$ regular-season unblocked shot attempts ($241,107$ non-empty-net SOG):

| Metric / Danger Zone | 2023–24 | 2024–25 | 2025–26 | Calibration Target |
| :--- | :---: | :---: | :---: | :--- |
| **Fenwick Calibration Ratio ($\sum xG_F / \sum G$)** | `0.971` | `0.992` | `0.999` | $1.000$ (Zero Bias) |
| **Goalie On-Goal Calibration ($\sum xGA_{\text{SOG}} / \sum GA$)** | `0.974` | `0.978` | `0.996` | $1.000$ ($\sum \text{GSAx} \approx 0$) |
| **League-Wide $\sum \text{GSAx}$ (Goals)** | `-201.3` | `-166.9` | `-29.7` | $\approx 0.0$ across 1,312 GP |
| **Stage 1 Discrimination (ROC-AUC / Brier)** | `0.7204` / `0.0628` | `0.7131` / `0.0626` | `0.7166` / `0.0621` | Public PBP benchmark: `0.71–0.73` |
| **High-Danger Fenwick ($d \le 25\text{ ft}, \theta \le 40^\circ$)** | `13.83%` pred vs `13.84%` act (2025) | — | — | Exact slot calibration |
| **Low-Danger Fenwick ($d > 40\text{ ft}$ or $\theta > 50^\circ$)** | `2.65%` pred vs `2.72%` act (2025) | — | — | Exact perimeter calibration |
| **High-Danger On-Goal ($xGA_{\text{SOG}}$)** | `20.34%` pred vs `20.16%` act (2025) | — | — | Exact goalie slot save benchmark |

---

## 3. Future Opportunities: Model-Based Estimation Roadmap

While the current Dual-Stage Conditional Logistic engine provides a fast, zero-dependency, out-of-sample calibrated parametric measurement layer, transitioning to **non-parametric and hierarchical statistical models** offers major opportunities for research in `dev/` and production estimation:

### 3.1 Generalized Additive Models (GAMs) with Bivariate Spatial Tensor Splines
- **Opportunity**: Replace linear log-odds terms ($\beta_d d + \beta_\theta \theta + \beta_{\text{slot}} \mathbb{I}_{\text{slot}}$) with a smooth 2D thin-plate regression spline or tensor product smooth $f(x_{\text{norm}}, y)$ estimated via penalized likelihood (`pygam` or `statsmodels` B-splines):
  $$\text{logit}(P(G_i = 1)) = \alpha + f(x_i, y_i) + \sum_k \beta_k z_{ik}$$
- **Advantage**: Automatically captures non-linear spatial contours around the goal line, behind-the-net wrap-around geometry, and asymmetric strong-side/off-wing shooting lanes without step-function indicator boundaries.

### 3.2 Gradient Boosted Decision Trees (LightGBM / CatBoost) with Monotonicity & Isotonic Calibration
- **Opportunity**: Train separate non-parametric tree ensembles for **5v5 Even Strength**, **Power Play (5v4 / 5v3)**, **Shorthanded (4v5)**, and **Empty Net** manpower states using:
  - Continuous spatial coordinates $(x, y, d, \theta)$ and prior-event kinematics ($\Delta x, \Delta y, \Delta t$, implied puck transfer velocity $v = \sqrt{\Delta x^2 + \Delta y^2}/\Delta t$).
  - Score-state and period interaction terms (`score_diff`, `period`, `time_remaining`) to control for score-effects.
  - Monotonic constraints ($\partial \hat{p} / \partial d \le 0$ for $d \ge 12\text{ ft}$) combined with out-of-fold **Isotonic Regression** or **Beta Calibration**.
- **Advantage**: Captures higher-order interactions (e.g., cross-slot royal-road passes prior to one-timers vs. unscreened perimeter slap shots) and typically improves ROC-AUC from `~0.72` to `~0.76–0.78`.

### 3.3 Hierarchical Bayesian Mixed-Effects Models (Shooter & Goaltender Talent Shrinkage)
- **Opportunity**: Estimate a Generalized Linear Mixed Model (GLMM) or hierarchical Bayesian logistic model with crossed random intercepts for shooter finishing talent ($u_{\text{shooter}[i]} \sim \mathcal{N}(0, \sigma_s^2)$) and goaltender shot-stopping talent ($v_{\text{goalie}[i]} \sim \mathcal{N}(0, \sigma_g^2)$):
  $$\text{logit}(P(G_i = 1 \mid \text{SOG}_i = 1)) = \mathbf{x}_i^\top \boldsymbol{\beta} + u_{\text{shooter}[i]} - v_{\text{goalie}[i]}$$
- **Advantage**:
  1. **Empirical Bayes Shrinkage (ebGSAx & ebFinishing)**: Small-sample call-ups (e.g., a goalie with 3 starts or a rookie with 15 shots) are shrunk toward the league mean proportional to sample size, eliminating small-sample leaderboard noise.
  2. **Shooter-Adjusted GSAx**: Credits goaltenders who face elite finishers (e.g., Matthews, Draisaitl, Bedard) where the true threat exceeds location-only $xGA_{\text{SOG}}$.

### 3.4 Sequence-Based & NHL EDGE Optical Tracking Models
- **Opportunity**: Incorporate multi-event possession chains (zone entry type $\to$ pass sequence $\to$ shot attempt) using Recurrent/Transformer sequence encoders or Markov Possession Value models (Expected Threat / $xT$), and integrate NHL EDGE optical tracking summaries (shot velocity, pre-shot lateral goalie displacement, defensive screen density).

### 3.5 Regularized Adjusted Plus-Minus ($xG\text{-RAPM}$) Using Shift Data
- **Opportunity**: Combine shot-level $xG_F$ with shift-level on-ice stints (`src/data/apinhle/data_pull_lines.py`) in a Ridge-penalized regression ($\ell_2$ penalty $\lambda$) to isolate individual skaters' **5v5 Offensive $xGF/60$ Impact** and **Defensive $xGA/60$ Suppression** controlling for teammates, opponents, and zone starts.

---

## 4. Data Pipeline & Hugging Face `measurement/` Architecture

### 4.1 Local `latest/` Enriched Files
Running `uv run python src/data/apinhle/data_proc_xg.py` enriches:
1. **`latest/box/{year}_box.csv`**: Adds `xg_for`, `xg_against`, `xg_diff`, `xg_pct`, `hdcf_for`, `hdcf_against`, `hdcf_pct`, `mdcf_for`, `mdcf_against`, `ldcf_for`, `ldcf_against`, `gsax_for`, `gsax_against`.
2. **`latest/box/{year}_box_team.csv`**: Adds per-team-game `xg`, `xga`, `xg_diff`, `xg_pct`, `hdcf`, `hdca`, `hdcf_pct`, `mdcf`, `mdca`, `ldcf`, `ldca`, `gsax`, `xsave_pct`, `dsave_pct`.
3. **`latest/box/{year}_box_gameStats.csv`**: Appends game-level `xg`, `xg_against`, `xg_pct`, `hdcf`, `hdcf_pct`, and `gsax` rows to the long-format game stats table.
4. **`latest/team/season/{year}_team_season.csv`**: Adds season-level `xg_for`, `xg_against`, `xg_lvl`, `xg_pct`, `kpi_xg`, `hdcf_for`, `hdcf_against`, `hdcf_lvl`, `hdcf_pct`, `kpi_hdcf`, `gsax`, and `kpi_gsax`.
5. **`latest/box/{year}_box_player.csv`**: Adds per-game skater (`ixg`, `goals_above_expected`, `fenwick_attempts`, `hd_shots`, `hd_goals`, `md_shots`, `ld_shots`, `rebound_shots`) and goalie (`xga`, `gsax`, `xsave_pct`, `dsave_pct`, `hd_sa`, `hd_ga`, `hd_save_pct`) columns.
6. **`latest/player/{year}_player.csv`**: Adds season-to-date (rolling for active season, finalized for completed seasons) skater (`ixg`, `ixg_per_game`, `goals_above_expected`, `fenwick_attempts`, `hd_shots`, `hd_goals`, `md_shots`, `ld_shots`, `rebound_shots`) and goalie (`xga`, `gsax`, `xsave_pct`, `dsave_pct`, `hd_sa`, `hd_ga`, `hd_save_pct`) aggregates.

### 4.2 Hugging Face Dataset (`jjerlee/nhl-analytics`) `measurement/` Tables
Synchronized via `uv run python deploy/huggingface/sync_dataset.py --target all`:
- **`measurement/player/player_season.parquet`**:
  - Unified player-season table across all seasons (`2013–2026`).
  - Contains `season_status` (`in_progress` for the active rolling season; `completed` for finalized seasons) and `is_season_final` (`bool`).
  - Includes all traditional NHL stats plus `ixg`, `ixg_per_game`, `goals_above_expected`, `fenwick_attempts`, `hd_shots`, `hd_goals`, `xga`, `gsax`, `xsave_pct`, `dsave_pct`, `hd_sa`, `hd_ga`, and `hd_save_pct`.
- **`measurement/player/player_game.parquet`**:
  - Unified player-game panel across all seasons with **both per-game metrics and cumulative rolling season-to-date columns**:
    - Skaters: `gp_to_date`, `goals_to_date`, `assists_to_date`, `points_to_date`, `sog_to_date`, `ixg_to_date`, `g_minus_ixg_to_date`, `hd_shots_to_date`, `fenwick_to_date`.
    - Goalies: `ga_to_date`, `sa_to_date`, `saves_to_date`, `xga_to_date`, `gsax_to_date`, `sv_pct_to_date`, `xsv_pct_to_date`, `dsv_pct_to_date`.
    - Includes `is_season_end` (`True` on the player's final appearance of that season, matching their season-ending total).
- **`measurement/box/team_season.parquet`** & **`measurement/box/game_stats.parquet`**:
  - Enriched with team-level `xg_for`, `xg_against`, `xg_lvl`, `xg_pct`, `kpi_xg`, `hdcf_pct`, `kpi_hdcf`, `gsax`, and `kpi_gsax`.
