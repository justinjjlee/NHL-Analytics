"""
knowledge_base.py — Canonical Rinklytics Knowledge Corpus & Calculation Compendium for Agent Zamboni.

Contains structured knowledge chunks covering all application pages in `app/views/`,
all analytical calculation methodologies (`procs_xg.py`, `procs_boxscore.py`, `player_stats.py`,
`rebuild_tracker.py`, `hit_impact.py`, `blocking_shots.py`, `draft_analysis.py`, `map_utils.py`,
`dev/agent/scoreboard/`), and all Hugging Face Dataset tables (`jjerlee/nhl-analytics`).
"""

from __future__ import annotations

from typing import List
from .models import KnowledgeChunk


def get_canonical_knowledge_chunks() -> List[KnowledgeChunk]:
    """Returns the complete, curated knowledge base of Rinklytics methodologies and page references."""
    return [
        # 1. Expected Goals (xG), Skater ixG, and Goalie GSAx (Dual-Stage Logistic Model)
        KnowledgeChunk(
            chunk_id="kb_xg_gsax_dual_stage",
            title="Expected Goals (xG, ixG) & Goaltender GSAx — Dual-Stage Conditional Logistic Model",
            domain="xg_gsax",
            page_view="app/views/skater_stats.py | app/views/goalie_stats.py | app/views/team_season_stats.py | app/views/game_recaps.py",
            source_files=[
                "src/data/apinhle/function/procs_xg.py",
                "src/data/apinhle/data_proc_xg.py",
                "dev/agent/scoreboard/METRICS_XG_GSAX.md",
            ],
            hf_dataset_tables=[
                "measurement/player/player_season.parquet",
                "measurement/player/player_game.parquet",
                "measurement/box/team_season.parquet",
                "measurement/box/game_stats.parquet",
            ],
            formulas_latex=[
                r"d = \sqrt{(89 - x_{\text{norm}})^2 + y^2}, \quad d_{\text{eff}} = \max(d, 10.5\text{ ft}), \quad \theta = \arctan2(|y|, \max(89 - x_{\text{norm}}, 1.0)) \cdot \frac{180}{\pi}",
                r"xG_F = \sigma\left(-1.25 - 0.0395\,d_{\text{eff}} - 0.0135\,\theta + 0.36\,\mathbb{I}_{\text{slot}} + \gamma_{\text{type}} + 0.68\,\mathbb{I}_{\text{reb}} + 0.35\,\mathbb{I}_{\text{rush}} + 3.35\,\mathbb{I}_{\text{EN}}\right)",
                r"xGA_{\text{SOG}} = \sigma\left(-0.82 - 0.0385\,d_{\text{eff}} - 0.0125\,\theta + 0.38\,\mathbb{I}_{\text{slot}} + \gamma_{\text{type}} + 0.62\,\mathbb{I}_{\text{reb}} + 0.32\,\mathbb{I}_{\text{rush}}\right)",
                r"\text{GSAx} = \sum_{i \in \text{SOG}_{\text{non-EN}}} xGA_{\text{SOG}, i} - GA_{\text{non-EN}}, \quad xSV\% = 1 - \frac{xGA_{\text{SOG}}}{SA}, \quad dSV\% = SV\% - xSV\%",
            ],
            summary=(
                "Rinklytics computes Expected Goals (xG/ixG) and Goals Saved Above Expected (GSAx) using a "
                "Dual-Stage Conditional Logistic Model in `src/data/apinhle/function/procs_xg.py` to guarantee "
                "zero-sum league calibration across both unblocked Fenwick attempts (~6.82% conversion) and "
                "non-empty-net Shots on Goal (~10.21% conversion)."
            ),
            methodology_markdown=(
                "### Dual-Stage Conditional Logistic xG & GSAx Methodology\n"
                "1. **Why Dual-Stage Calibration is Required**: Roughly **34.8%** of NHL unblocked shot attempts (Fenwick: `shot-on-goal`, `missed-shot`, `goal`) miss the net frame. "
                "A single model calibrated to Fenwick under-predicts goaltender on-target $xGA$ by **-20.8%**, whereas calibrating to SOG over-predicts team Fenwick $xG$ by **+22.8%**.\n"
                "2. **Spatial Geometry & Crease-Jam Clamping**:\n"
                "   - Coordinates $(x, y)$ are normalized toward the target net at $(89, 0)$.\n"
                "   - **Crease-Jam Clamping ($d_{\\text{eff}} = \\max(d, 10.5\\text{ ft})$)**: Empirical conversion peaks in the inner slot ($8\\text{–}15\\text{ ft}$) and plateaus inside $0\\text{–}7\\text{ ft}$ due to pad seals and jammed scrambles.\n"
                "   - **Mid-Slot Release Corridor ($\\mathbb{I}_{\\text{slot}}$)**: Shots from $20 < d \\le 35\\text{ ft}$ and $\\theta \\le 50^\\circ$ receive a $+0.36$ (Stage 1) / $+0.38$ (Stage 2) log-odds bonus.\n"
                "   - **Danger Zones**: **High-Danger (HD)** is $d \\le 25\\text{ ft}, \\theta \\le 40^\\circ$ (or rebound $d \\le 30\\text{ ft}$); **Medium-Danger (MD)** is $d \\le 40\\text{ ft}, \\theta \\le 50^\\circ$ (non-HD); **Low-Danger (LD)** is perimeter ($d > 40\\text{ ft}$ or $\\theta > 50^\\circ$).\n"
                "3. **Stage 1 (Team $xG$ & Skater $ixG$)**: Applied to all Fenwick attempts ($\beta_0=-1.25, \\beta_d=-0.0395, \\beta_\\theta=-0.0135$). Skater **Goals Above Expected** is $G - ixG$.\n"
                "4. **Stage 2 (Goalie $xGA_{\\text{SOG}}$ & $GSAx$)**: Applied exclusively to non-empty-net Shots on Goal ($\\beta_0=-0.82, \\beta_d=-0.0385, \\beta_\\theta=-0.0125$). **GSAx** = $xGA_{\\text{SOG}} - GA_{\\text{non-EN}}$."
            ),
            keywords=[
                "xg", "ixg", "xga", "gsax", "expected goals", "goals saved above expected",
                "dual-stage", "logistic", "crease-jam", "clamping", "high-danger", "hdcf",
                "xsave_pct", "dsave_pct", "xsv%", "dsv%", "fenwick", "procs_xg", "slot", "angle", "distance",
            ],
        ),

        # 2. Skater Stats & Skater Player Performance Index (PPI)
        KnowledgeChunk(
            chunk_id="kb_skater_stats_ppi",
            title="Skater Statistics & Skater Player Performance Index (PPI)",
            domain="player_ppi",
            page_view="app/views/skater_stats.py",
            source_files=[
                "app/views/skater_stats.py",
                "app/exe/player_stats.py",
                "app/exe/data_loader.py",
            ],
            hf_dataset_tables=[
                "measurement/player/player_season.parquet",
                "measurement/player/player_game.parquet",
            ],
            formulas_latex=[
                r"z_{i, c} = \frac{x_{i, c} - \mu_c}{\sigma_c}, \quad \text{PPI}_{\text{skater}, i} = \frac{1}{9}\sum_{c \in \mathcal{M}_S} z_{i, c}",
                r"\mathcal{M}_S = \left\{\frac{\text{PTS}}{\text{GP}}, \frac{\text{G}}{\text{GP}}, \frac{\text{A}}{\text{GP}}, +/-, \frac{\text{SOG}}{\text{GP}}, \text{Sh\%}, \frac{\text{TOI}}{\text{GP}}, \frac{ixG}{\text{GP}}, (\text{G} - ixG)\right\}",
            ],
            summary=(
                "The **Skater Statistics** page (`app/views/skater_stats.py`) evaluates NHL forwards (`C`, `L`, `R`) "
                "and defensemen (`D`) using traditional box metrics, spatial Stage-1 Fenwick `ixG`, `G - ixG`, "
                "`hd_shots`, weekly game logs, and the composite **Skater Player Performance Index (PPI)**."
            ),
            methodology_markdown=(
                "### Skater Player Performance Index (PPI) & Profile Methodology\n"
                "1. **Eligibility Filter**: Regular season skaters (`idx_season_type == 2`) with at least **20 Games Played (`min_games = 20`)** in the selected season.\n"
                "2. **9-Dimension Z-Score Composite (`compute_skater_ppi` in `app/exe/player_stats.py`)**:\n"
                "   - Standardizes 9 metrics against the single-season peer mean ($\\mu_c$) and standard deviation ($\\sigma_c$):\n"
                "     1. Points per Game (`points_per_game`)\n"
                "     2. Goals per Game (`goals_per_game`)\n"
                "     3. Assists per Game (`assists_per_game`)\n"
                "     4. Plus/Minus (`plusMinus`)\n"
                "     5. Shots on Goal per Game (`shots_per_game`)\n"
                "     6. Shooting Percentage (`shootingPctg`)\n"
                "     7. Average Time on Ice per Game in seconds (`avgTimeOnIcePerGame`)\n"
                "     8. Individual Expected Goals per Game (`ixg_per_game`)\n"
                "     9. Goals Above Expected (`goals_above_expected` = $G - ixG$)\n"
                "3. **Interpretation**: A Skater PPI of `0.000` is league average; `+1.000` is one standard deviation above the NHL peer group across all 9 offensive, volume, and finishing dimensions."
            ),
            keywords=[
                "skater", "skaters", "ppi", "player performance index", "skater_stats", "z-score",
                "points_per_game", "goals_above_expected", "ixg_per_game", "hd_shots", "player_season",
                "player_game", "forward", "defenseman", "connor bedard", "leaderboard",
            ],
        ),

        # 3. Goalie Stats & Goalie Player Performance Index (PPI)
        KnowledgeChunk(
            chunk_id="kb_goalie_stats_ppi",
            title="Goalie Statistics & Goaltender Player Performance Index (PPI)",
            domain="goalie_ppi",
            page_view="app/views/goalie_stats.py",
            source_files=[
                "app/views/goalie_stats.py",
                "app/exe/player_stats.py",
                "src/data/apinhle/function/procs_xg.py",
            ],
            hf_dataset_tables=[
                "measurement/player/player_season.parquet",
                "measurement/player/player_game.parquet",
            ],
            formulas_latex=[
                r"\text{gaa\_inv} = -\text{GAA}, \quad \text{PPI}_{\text{goalie}, i} = \frac{1}{7}\sum_{c \in \mathcal{M}_G} \frac{x_{i,c} - \mu_c}{\sigma_c}",
                r"\mathcal{M}_G = \left\{\text{SV\%},\; -\text{GAA},\; \text{Wins},\; \text{Shutouts},\; \frac{\text{Saves}}{\text{GS}},\; \text{GSAx},\; dSV\%\right\}",
            ],
            summary=(
                "The **Goalie Statistics** page (`app/views/goalie_stats.py`) evaluates NHL goaltenders using "
                "Stage-2 conditional on-goal `xGA`, `GSAx`, Expected Save % (`xsave_pct`), Save % Above Expected "
                "(`dsave_pct`), High-Danger Save % (`hd_save_pct`), and the 7-metric **Goalie PPI**."
            ),
            methodology_markdown=(
                "### Goaltender PPI & Crease Analytics Methodology\n"
                "1. **Eligibility Filter**: Regular season goaltenders (`positionCode == 'G'`) with at least **15 Games Played (`min_games = 15`)** in the selected season.\n"
                "2. **7-Dimension Z-Score Composite (`compute_goalie_ppi` in `app/exe/player_stats.py`)**:\n"
                "   - Standardizes and averages z-scores across 7 workload and shot-stopping dimensions:\n"
                "     1. Save Percentage (`savePercentage`)\n"
                "     2. Inverted Goals Against Average (`gaa_inv = -goalsAgainstAverage` so lower GAA yields higher z-score)\n"
                "     3. Total Wins (`wins`)\n"
                "     4. Shutouts (`shutouts`)\n"
                "     5. Saves per Game Started (`saves_per_game` = $\\text{Saves} / \\text{GS}$)\n"
                "     6. Goals Saved Above Expected (`gsax` = $xGA_{\\text{SOG}} - GA_{\\text{non-EN}}$)\n"
                "     7. Save Percentage Above Expected (`dsave_pct` = $SV\\% - xSV\\%$)\n"
                "3. **Why Both Traditional & Spatial Metrics Are Combined**: Combining workload/results (`Wins`, `Saves/GS`, `SO`) with shot-quality-adjusted metrics (`GSAx`, `dSV%`) rewards workhorse starters who outperform expected shot danger."
            ),
            keywords=[
                "goalie", "goalies", "goaltender", "goalie_stats", "goalie ppi", "gsax",
                "xga", "xsave_pct", "dsave_pct", "hd_save_pct", "save percentage", "gaa",
                "shutouts", "jakub dobes", "vasilevskiy", "hellebuyck",
            ],
        ),

        # 4. Team Season Stats: RPE, RPI, Pairwise Win Rank, Streaks & KPI Normalization
        KnowledgeChunk(
            chunk_id="kb_team_season_kpis",
            title="Team Season Statistics — Pythagorean Expectation (RPE), RPI, Common-Opponent Pairwise Rank & KPI Normalization",
            domain="team_season",
            page_view="app/views/team_season_stats.py",
            source_files=[
                "app/views/team_season_stats.py",
                "src/data/apinhle/function/procs_boxscore.py",
                "src/data/apinhle/data_proc_gameSummary.py",
            ],
            hf_dataset_tables=[
                "measurement/box/team_season.parquet",
                "measurement/box/game_stats.parquet",
                "data/games.parquet",
            ],
            formulas_latex=[
                r"\text{RPE} = \frac{\text{GF}^2}{\text{GF}^2 + \text{GA}^2}, \quad \text{rwin\_fittedPE} = (0.809623 \cdot \text{RPE} + 0.042901) \times 82",
                r"\text{RPI} = 0.25 \cdot \text{WP}_{\text{own}} + 0.50 \cdot \text{WP}_{\text{opponents}} + 0.25 \cdot \text{WP}_{\text{common opponents}}",
                r"\text{h2h\_win}_{i,j} = \mathbb{I}\left(\frac{W_{i,\mathcal{O}_{i,j}}}{GP_{i,\mathcal{O}_{i,j}}} > \frac{W_{j,\mathcal{O}_{i,j}}}{GP_{j,\mathcal{O}_{i,j}}}\right), \quad \text{kpinorm}(x) = \frac{x - \min(x)}{\max(x) - \min(x)}",
            ],
            summary=(
                "The **Team Season Stats** page (`app/views/team_season_stats.py`) benchmarks all 32 NHL franchises "
                "using Standings (`W`, `L`, `OTL`, `PTS`), Rolling Pythagorean Expectation (`rpe`), Rating Percentage "
                "Index (`rpi`), Strength of Record via Common-Opponent Pairwise Win Rank (`pairwise_win`), Win/Hot "
                "Streaks, and Min-Max $[0,1]$ normalized KPIs (`kpi_pe`, `kpi_pairwise`, `kpi_wp`, `kpi_xg`, `kpi_hdcf`, `kpi_gsax`)."
            ),
            methodology_markdown=(
                "### Team Season Success Measurements (`procs_boxscore.py`)\n"
                "1. **Rolling Pythagorean Expectation (`rpe`)**: Computed as $GF^2 / (GF^2 + GA^2)$ from cumulative goals for and against (`rpe_10` tracks a 10-game moving window). `rwin_fittedPE` projects 82-game wins using OLS coefficients calibrated on 2010–2020 NHL data ($\\beta=0.809623, \\alpha=0.042901$).\n"
                "2. **Rating Percentage Index (`rpi`)**: Schedule-adjusted win index weighting 25% own win percentage (`wp_own`), 50% opponents' cumulative win percentage (`wp_ow`), and 25% common-opponent win percentage (`wp_oow`).\n"
                "3. **Strength of Record — Common-Opponent Pairwise Win Rank (`pairwise_win`)**:\n"
                "   - For every pair of teams $(i, j)$, isolates their common opponents (`df_gen_h2h_common`).\n"
                "   - Awards `h2h_win = 1` if team $i$'s win rate against those common opponents strictly exceeds team $j$'s win rate.\n"
                "   - Ranks teams $1 \\dots 32$ by total pairwise victories (`pairwise_win = 1` is best in the NHL).\n"
                "4. **Win Streaks & Hot Streaks**:\n"
                "   - `idx_WinStreak`: 3 consecutive wins.\n"
                "   - `idx_WinStreak_expected`: 3-game win streak with $\\ge 2$ wins against lower-RPI opponents.\n"
                "   - `idx_HotStreak`: 3-game win streak featuring $\\ge 1$ win against a top-quartile opponent ($\\text{RPI}_{\\text{against}} > Q_{0.75}$).\n"
                "5. **KPI $[0,1]$ Normalization (`kpinorm`)**: Rescales every metric to $[0, 1]$ across the 32 teams so radar/bar comparisons (`kpi_pe`, `kpi_pairwise`, `kpi_wp`, `kpi_xg`, `kpi_hdcf`, `kpi_gsax`) share a unified scale where `1.000` is the league leader."
            ),
            keywords=[
                "team season", "team_season_stats", "rpe", "pythagorean", "pythagorean expectation",
                "rwin_fittedpe", "rpi", "rating percentage index", "pairwise", "pairwise_win",
                "common opponent", "strength of record", "kpinorm", "kpi_pe", "kpi_pairwise",
                "kpi_wp", "kpi_xg", "kpi_hdcf", "kpi_gsax", "hot streak", "win streak", "standings",
            ],
        ),

        # 5. Franchise Rebuild Tracker & Equilibrium Regimes (2000-2025)
        KnowledgeChunk(
            chunk_id="kb_rebuild_tracker",
            title="Franchise Rebuild Tracker & Equilibrium Regimes (2000–2025) — Markov States, Cohorts & GM Policy Levers",
            domain="rebuild",
            page_view="app/views/rebuild_tracker.py",
            source_files=[
                "app/views/rebuild_tracker.py",
                "dev/player_team_success/rebuild/",
            ],
            hf_dataset_tables=[
                "dev/player_team_success/rebuild/data/processed/panel_franchise_season.csv",
                "dev/player_team_success/rebuild/results/tables/",
            ],
            formulas_latex=[
                r"H_{\text{exp}} = -\sum_{k=1}^4 s_{B_k} \ln(s_{B_k}), \quad \sum_{k=1}^4 s_{B_k} = 1, \quad B_1\text{ (0–3 yr)}, B_2\text{ (4–6 yr)}, B_3\text{ (7–9 yr)}, B_4\text{ (10+ yr)}",
                r"\text{Net Cap Surplus (\$M)} = \sum_{p \in \text{Roster}} \left(\text{MRP}_{\text{est}, p} - \text{CapHit}_p\right)",
            ],
            summary=(
                "The **Franchise Rebuild Tracker** (`app/views/rebuild_tracker.py`) presents a 25-year econometric study "
                "(2000–2025) classifying all 32 NHL franchises into 3 Markov-switching macro regimes (`Inaction`, `Rebuild`, `Peak`), "
                "6 granular sub-states (`R0_CONTENDER` to `R5_PURGATORY`), and 3 strategic archetypes (`SCORCHED_EARTH_DRAFT`, "
                "`HYBRID_REBUILD`, `COMPOSITIONAL_RETOOL`), including a Montréal vs. Chicago comparative case study and a GM Simulator."
            ),
            methodology_markdown=(
                "### Franchise Rebuilds & Equilibrium Regimes Methodology (2000–2025)\n"
                "1. **Macro Regimes & Granular Sub-States (`R0–R5`)**:\n"
                "   - `R0_CONTENDER` (**Peak**): Sustained playoff & Stanley Cup contention.\n"
                "   - `R1_SLUMP` (**Inaction**): Mild decline / ”mushy middle” without committing to asset liquidation.\n"
                "   - `R2_TEARDOWN` (**Rebuild**): Active veteran sell-off (high `ali_score`).\n"
                "   - `R3_INCUBATION` (**Rebuild**): Developing youth core with rising entry-level TOI (`B1`).\n"
                "   - `R4_EMERGENCE` (**Rebuild -> Peak**): Playoff breakout driven by prime bridge (`B2`) & peak core (`B3`).\n"
                "   - `R5_PURGATORY` (**Failed Rebuild**): Stalled in rebuild for $\\ge 5$ years without achieving Tier-1 playoff core status.\n"
                "2. **3 Rebuild Archetypes**:\n"
                "   - `SCORCHED_EARTH_DRAFT`: Pure draft tanking (strips veteran insulation, high purgatory risk if uninsulated).\n"
                "   - `HYBRID_REBUILD`: Combines high draft capital with prime bridge trades (`B2` 4–6 yr players like Kirby Dach / Alex Newhook).\n"
                "   - `COMPOSITIONAL_RETOOL`: On-the-fly retool preserving the veteran core.\n"
                "3. **Key Econometric Variables**:\n"
                "   - **Experience Buckets (`B1–B4`)**: Share of team TOI played by `B1` (ELC 0–3 yrs), `B2` (Prime Bridge 4–6 yrs), `B3` (Peak Core 7–9 yrs), and `B4` (Veterans 10+ yrs).\n"
                "   - **ALI (`ali_score`)**: Asset Liquidation Index (veteran sell-off intensity).\n"
                "   - **YIS (`yis_score`)**: Youth Ice-Time Share.\n"
                "   - **RCI (`rci_score`) & CCM (`ccm_to_prior_year`)**: Roster Churn Index & Core Continuity Metric.\n"
                "   - **VIR (`vet_insulation_ratio`) & Shannon Entropy (`shannon_exp_entropy`)**: Measures whether young ELC players are properly insulated by experienced cohorts.\n"
                "   - **SPSI (`spsi_style_synergy`)**: Skill $\\times$ Physicality Synergy (interaction of 5v5 $xGF\\%$ and offensive-zone forecheck hit share `ozone_hit_pct`).\n"
                "   - **Net Cap Surplus (`surplus_value_total`)**: On-ice Market Marginal Revenue Product (`mrp_est`) minus actual salary cap hit (`cap_hit`)."
            ),
            keywords=[
                "rebuild", "rebuilds", "rebuild tracker", "rebuild_tracker", "markov", "regime",
                "inaction", "teardown", "incubation", "emergence", "purgatory", "contender",
                "r0", "r1", "r2", "r3", "r4", "r5", "scorched earth", "hybrid rebuild", "retool",
                "ali", "yis", "rci", "vir", "ccm", "spsi", "shannon", "entropy", "surplus",
                "cap hit", "montreal", "chicago", "gm lever", "counterfactual", "b1", "b2", "b3", "b4",
            ],
        ),

        # 6. Hit on Impact: Sequential EDA & Spatial Ice Analytics
        KnowledgeChunk(
            chunk_id="kb_hit_impact",
            title="Hit on Impact — Sequential Event Chains (+1 to +4), 30s Goal Windows & Spatial Rink Grid Analytics",
            domain="hits",
            page_view="app/views/hit_impact.py",
            source_files=[
                "app/views/hit_impact.py",
                "app/exe/rink_utils.py",
                "app/exe/data_loader.py",
            ],
            hf_dataset_tables=[
                "measurement/action/hits/hit_zone_summary_kpi.parquet",
                "measurement/action/hits/tabulated_hit_sequences.parquet",
                "measurement/action/hits/hit_goal_window_30s.parquet",
                "measurement/action/hits/rink_grid_hit_sequences.parquet",
                "measurement/action/hits/team_season_hit_kpi.parquet",
                "measurement/action/hits/team_season_spatial_hits.parquet",
                "measurement/action/hits/hit_zone_summary.parquet",
                "measurement/action/hits/hit_next_events.parquet",
                "measurement/action/hits/hit_sequence_transitions.parquet",
            ],
            formulas_latex=[
                r"\text{Puck Win Rate (Location } g\text{)} = \frac{\text{Hits in } g \text{ followed by Hitting Team Possession/Shot}}{\text{Total Hits in } g}",
                r"\text{Net 30s Shots} = \text{Shots For}_{(t, t+30\text{s}]} - \text{Shots Against}_{(t, t+30\text{s}]}",
            ],
            summary=(
                "The **Hit on Impact** page (`app/views/hit_impact.py`) analyzes play-by-play event chains "
                "(`Event +1` to `Event +4`) and 30-second downstream shot/goal windows after every NHL bodycheck, "
                "mapping spatial $(x,y)$ rink grid tiles to Puck Retention Win Rates and goal trajectories."
            ),
            methodology_markdown=(
                "### Hit on Impact Sequential & Spatial Methodology\n"
                "1. **Downstream Event Chain Tracking (`Event +1` to `Event +4`)**:\n"
                "   - Every recorded NHL `hit` event is linked to the next 4 consecutive play-by-play events in the same period.\n"
                "   - Evaluates whether the **hitting team** (`For`) or the **hit team** (`Against`) owns the subsequent action (shot on goal, takeaway, giveaway, stoppage, goal).\n"
                "2. **30-Second Shot & Goal Window (`hit_goal_window_30s.parquet`)**:\n"
                "   - Measures all shots and goals occurring within $\\Delta t \\le 30\\text{ seconds}$ after a hit, computing **Net 30s Shots** and **Goals For vs. Goals Against**.\n"
                "3. **Spatial $(x, y)$ Rink Grid (`rink_grid_hit_sequences.parquet`)**:\n"
                "   - Discretizes the $200 \\times 85\\text{ ft}$ ice surface into spatial tiles.\n"
                "   - Clicking any hotspot tile inspects local hit volume (% of league hits), **Puck Win Rate** (and delta vs. league average), immediate shots for/conceded, and 30s goal vectors.\n"
                "4. **Key Empirical Takeaway**: **Offensive-Zone Forecheck Hits** along the end-boards generate positive puck recovery and high-danger slot shots, whereas high volumes of **Defensive-Zone Hits** often indicate a team chasing the puck without possession."
            ),
            keywords=[
                "hit", "hits", "hitting", "hit on impact", "hit_impact", "bodycheck", "puck win rate",
                "puck retention", "30s window", "30-second", "event +1", "sequence", "forecheck",
                "rink_grid_hit_sequences", "hit_zone_summary_kpi", "team_season_hit_kpi", "spatial",
            ],
        ),

        # 7. Blocking Shots: Sequential Decision Science & Sankey Markov Flow
        KnowledgeChunk(
            chunk_id="kb_blocking_shots",
            title="Blocking Shots — Downstream Possession Recovery & Sankey Sequence Analysis",
            domain="blocking_shots",
            page_view="app/views/blocking_shots.py",
            source_files=[
                "app/views/blocking_shots.py",
                "dev/decision_science/plays_blockingShots/",
            ],
            hf_dataset_tables=[
                "dev/decision_science/plays_blockingShots/result/table - sequence blocked shot stats.csv",
            ],
            formulas_latex=[
                r"P(E_{t+1} \mid \text{Blocked Shot}, \text{Zone}_{t-1}) = \frac{N(\text{Zone}_{t-1} \to \text{Owner}_{t+1} \to \text{Group}_{t+1} \to \text{Event}_{t+1})}{\sum N(\text{Blocked Shots})}",
            ],
            summary=(
                "The **Blocking Shots** page (`app/views/blocking_shots.py`) models the sequential decision science "
                "of shot blocking using a 4-layer Sankey transition diagram to test whether blocking shots recovers "
                "possession (`Own action`) or allows the attacking team to sustain pressure (`Opponent action`)."
            ),
            methodology_markdown=(
                "### Shot Blocking Sequential Transition Methodology (`blocking_shots.py`)\n"
                "1. **4-Layer Sankey Markov Chain (`table - sequence blocked shot stats.csv`)**:\n"
                "   - **Layer 1 (`blocker_team_lag1_now`)**: Prior tactical context (`Defense`, `Neutral`, `Offense`).\n"
                "   - **Layer 2 (`eventOwnerTeam_sequence`)**: Who controls the next play-by-play event after the block (`Own action` vs. `Opponent action`).\n"
                "   - **Layer 3 (`group`)**: Macro event class (`Shot`, `Possession`, `Stoppage`, `Penalty`, `Goal`).\n"
                "   - **Layer 4 (`typeDescKey`)**: Specific NHL event (`shot-on-goal`, `blocked-shot`, `missed-shot`, `hit`, `giveaway`, `takeaway`, `stoppage`, `faceoff`, `goal`).\n"
                "2. **Decision Science Insight**: Evaluates the hidden cost of heavy shot-blocking—when blocked shots deflect back to the shooting team (`Opponent action -> Shot`), the defending five-man unit remains trapped in the defensive zone."
            ),
            keywords=[
                "block", "blocked", "blocking", "blocking shots", "blocked shots", "blocking_shots",
                "sankey", "own action", "opponent action", "blocker_team_lag1_now", "sequence",
            ],
        ),

        # 8. Scoreboard & Game Recaps (Scoreboard Agent Infrastructure)
        KnowledgeChunk(
            chunk_id="kb_scoreboard_and_recaps",
            title="Scoreboard, Closing Moneyline Odds & Autonomous Game Recaps (Scoreboard Agent)",
            domain="scoreboard_recaps",
            page_view="app/views/scoreboard.py | app/views/game_recaps.py",
            source_files=[
                "app/views/scoreboard.py",
                "app/views/game_recaps.py",
                "app/exe/team_stats.py",
                "dev/agent/scoreboard/agent.py",
                "dev/agent/scoreboard/analyzer.py",
                "dev/agent/scoreboard/narrator.py",
                "dev/agent/scoreboard/AGENT_INFRASTRUCTURE.md",
            ],
            hf_dataset_tables=[
                "data/games.parquet",
                "data/odds.parquet",
                "insights/scoreboard/game_recaps.parquet",
                "insights/scoreboard/slate_recaps.parquet",
            ],
            formulas_latex=[
                r"\text{Turnover Ratio} = \frac{\text{Takeaways}}{\text{Takeaways} + \text{Giveaways}}, \quad \text{Pyth Win \%} = \frac{\text{GF}^{2.15}}{\text{GF}^{2.15} + \text{GA}^{2.15}}",
                r"P_{\text{implied}}(M) = \begin{cases} \frac{-M}{-M + 100} & \text{if } M < 0 \\ \frac{100}{M + 100} & \text{if } M > 0 \end{cases}",
            ],
            summary=(
                "The **Scoreboard** (`scoreboard.py`) and **Game Recaps** (`game_recaps.py`) views combine official "
                "NHL game results, 2-way Moneyline closing odds (`data/odds.parquet`), and the autonomous "
                "**Scoreboard Agent** (`dev/agent/scoreboard/`) which synthesizes fact-verified journalistic recaps."
            ),
            methodology_markdown=(
                "### Scoreboard, Odds & Autonomous Game Recaps Methodology\n"
                "1. **Scoreboard & Moneyline Odds (`scoreboard.py`, `data_loader.py`)**:\n"
                "   - Streams `data/games.parquet` and `data/odds.parquet` filtered to `odds_description == 'MONEY_LINE_2_WAY'` (prioritizing `US` then `CAN` markets).\n"
                "2. **10-Step Feature Hydration (`GameAnalyzer` in `dev/agent/scoreboard/analyzer.py`)**:\n"
                "   - **7 Game Script Classifications**: `REGULATION`, `COMEBACK` (overcame multi-goal deficit), `WIRE_TO_WIRE` (led start to finish), `BLOWOUT` ($\\ge 4$ goal margin), `DEFENSIVE_BATTLE` ($\\le 3$ total goals + high blocks), `OVERTIME_THRILLER`, `SHOOTOUT_DUEL`.\n"
                "   - **Goaltender Composure Tiers**: `ELITE_STEAL`, `SOLID_ANCHOR`, `SHAKY_SURVIVAL`, `REPLACED`.\n"
                "   - **Takeaway Taxonomy**: Strictly separates `playmaker` ($\\ge 3$ pts with 0 goals), `offensive_catalyst` ($\\ge 3$ pts with goals & assists), and `multi_goal`/`hat_trick` so assists are never conflated with goals.\n"
                "3. **Factual Verification Guardrails & Model Precedence (`narrator.py`, `dataset_store.py`)**:\n"
                "   - Reconciles exact score, winner/loser direction, OT/SO endings, and scorer names against official boxscore telemetry. Retries at $T=0.2$ on drift before falling back to `algorithmic-guardrails-v1`.\n"
                "   - Enforces **Model Priority Hierarchy** in `insights/scoreboard/`: Tier-2 Neural LLM recaps overwrite Tier-1 Algorithmic recaps and are protected against accidental downgrades."
            ),
            keywords=[
                "scoreboard", "game recaps", "game_recaps", "recap", "recaps", "odds", "moneyline",
                "game script", "comeback", "wire_to_wire", "blowout", "defensive_battle", "goalie composure",
                "narrator", "scoreboardagent", "game_recaps.parquet", "slate_recaps.parquet",
            ],
        ),

        # 9. Magnifiques Patineuses (Amateur Draft Lifecycles & 95th Percentile Exceptional Seasons)
        KnowledgeChunk(
            chunk_id="kb_draft_magnifiques",
            title="Magnifiques Patineuses — 1st-Round Amateur Draft Lifecycles & 95th-Percentile Exceptional Skaters",
            domain="draft",
            page_view="app/views/magnifiques_patineuses.py",
            source_files=[
                "app/views/magnifiques_patineuses.py",
                "app/exe/draft_analysis.py",
                "dev/player/player_draft/data/dataproc.py",
            ],
            hf_dataset_tables=[
                "dev/player/player_draft/draftee_lifecycle/results/multi_exceptional_players.csv",
                "dev/player/player_draft/draftee_lifecycle/results/exceptional_seasons_details.csv",
                "dev/player/player_draft/data/merged_draft_player_stats.csv",
                "dev/player/player_draft/data/all_drafted_players_chronicle.csv",
            ],
            formulas_latex=[
                r"\text{Exceptional Season}_{i, t} = \mathbb{I}\left(\text{PTS}_{i, t} \ge Q_{0.95}(\text{PTS}_{\text{Draft Peer Cohort}, t})\right)",
                r"\text{year\_inNHL}_{i, t} = \text{dense\_rank}(\text{Season}_t \text{ within Player } i)",
            ],
            summary=(
                "The **Magnifiques Patineuses** page (`app/views/magnifiques_patineuses.py` & `app/exe/draft_analysis.py`) "
                "tracks the career lifecycle of 1st-round NHL draft picks, identifying **Exceptional Skaters** who record "
                "multiple seasons above the **95th percentile** in points production, as well as **Not-So-Magnificent** high-pick attrition."
            ),
            methodology_markdown=(
                "### Amateur Draft Lifecycle & Exceptional Skaters Methodology (`draft_analysis.py`)\n"
                "1. **Tab 1 — First-Round Offensive Stars (`render_exceptional_players_analysis`)**:\n"
                "   - Evaluates 1st-round NHL draftees (`round == 1`, `leagueAbbrev == 'NHL'`, regular season `gameTypeId == 2`).\n"
                "   - Defines an **Exceptional Season** as a season in which a skater scores at or above the **95th percentile** in points compared to their draft peer cohort.\n"
                "   - Tracks `exceptional_seasons` count vs. total NHL seasons played (`season`), career progression by `year_inNHL` (Years in the NHL), and points-per-game trajectories.\n"
                "2. **Tab 2 — Not-So-Magnificent (`render_not_so_magnificent_analysis`)**:\n"
                "   - Analyzes 1st-round draft picks who fell short of NHL longevity expectations, examining games-played survival curves and draft-slot bust rates."
            ),
            keywords=[
                "draft", "drafted", "draftee", "magnifiques patineuses", "magnifiques_patineuses",
                "exceptional", "exceptional players", "exceptional seasons", "95th percentile",
                "first round", "1st round", "not so magnificent", "overall pick", "year_innhl", "draft_analysis",
            ],
        ),

        # 10. Teams Map & Inter-Arena Geodesic Travel Distance Matrix
        KnowledgeChunk(
            chunk_id="kb_teams_map_travel",
            title="Teams Map — Arena Geographic Coordinates & Inter-City Geodesic Travel Distances",
            domain="map_travel",
            page_view="app/views/teams_map.py",
            source_files=[
                "app/views/teams_map.py",
                "app/exe/map_utils.py",
                "app/exe/data_loader.py",
            ],
            hf_dataset_tables=[
                "app/data/teamlist_locations_distance_meters.csv",
                "app/data/teamlist.csv",
            ],
            formulas_latex=[
                r"d_{\text{km}} = \frac{d_{\text{meters}}}{1000}, \quad d_{\text{miles}} = d_{\text{meters}} \times 0.000621371",
            ],
            summary=(
                "The **Teams Map** page (`app/views/teams_map.py` & `app/exe/map_utils.py`) renders an interactive Folium "
                "map of all NHL arenas and computes exact pairwise geodesic travel distances (`distance_meters`) between "
                "any selected Home and Away franchise."
            ),
            methodology_markdown=(
                "### NHL Arena Geography & Travel Distance Methodology (`map_utils.py`)\n"
                "1. **Coordinate & Distance Matrix**: Joins `app/data/teamlist.csv` with `app/data/teamlist_locations_distance_meters.csv` containing exact latitude/longitude coordinates (`lat_home`, `long_home`, `lat_away`, `long_away`) and pre-computed geodesic distances in meters for every NHL team pair.\n"
                "2. **Interactive Visualization**: Renders custom team markers and geodesic flight arcs in Folium (`st_folium`), reporting travel distance in kilometers and miles."
            ),
            keywords=[
                "map", "teams map", "teams_map", "distance", "travel", "arena", "location",
                "latitude", "longitude", "kilometers", "miles", "folium", "map_utils",
            ],
        ),

        # 11. Platform Architecture, Data-as-Code Pipelines & Agent Zamboni System
        KnowledgeChunk(
            chunk_id="kb_platform_and_zamboni",
            title="Rinklytics Platform Architecture, Hugging Face Lakehouse (`jjerlee/nhl-analytics`) & Agent Zamboni System",
            domain="platform_architecture",
            page_view="app/views/home.py | app/views/zamboni.py",
            source_files=[
                "app/app.py",
                "app/views/home.py",
                "app/views/zamboni.py",
                "app/exe/data_loader.py",
                "deploy/huggingface/sync_dataset.py",
                "deploy/huggingface/spaces_rinklytics.py",
                "dev/zamboni/ZAMBONI_ARCHITECTURE_DESIGN.md",
            ],
            hf_dataset_tables=[
                "data/games.parquet",
                "data/odds.parquet",
                "data/player_box.parquet",
                "measurement/player/player_season.parquet",
                "measurement/player/player_game.parquet",
                "measurement/box/team_season.parquet",
                "measurement/box/game_stats.parquet",
                "measurement/action/hits/*.parquet",
                "insights/scoreboard/*.parquet",
                "insights/zamboni/*.parquet",
            ],
            formulas_latex=[],
            summary=(
                "**Rinklytics (NHL Analytics)** is an open-source hockey analytics platform combining automated daily "
                "NHL API ingestion (`api-web.nhle.com/v1`), a longitudinal Parquet Lakehouse on Hugging Face Datasets "
                "(`jjerlee/nhl-analytics`), a ZeroGPU Streamlit Space (`jjerlee/rinklytics`), and **Agent Zamboni**."
            ),
            methodology_markdown=(
                "### Rinklytics Data-as-Code & Agent Zamboni Infrastructure\n"
                "1. **Automated Daily Pipelines (`.github/workflows/`)**:\n"
                "   - **09:00 UTC Daily**: Pulls completed box scores (`data_pull_box.py`), play-by-play (`data_pull_plays.py`), player stats (`data_pull_player.py`), team KPIs (`data_proc_gameSummary.py`), and Dual-Stage xG/GSAx (`data_proc_xg.py`).\n"
                "   - **20:00 UTC Daily**: Captures closing US & CAN moneyline odds (`data_pull_box_odds.py`).\n"
                "   - **10:00 UTC Sync**: Deduplicates and pushes Parquet panels to [`hf://datasets/jjerlee/nhl-analytics`](https://huggingface.co/datasets/jjerlee/nhl-analytics).\n"
                "2. **Agent Zamboni Architecture (`dev/zamboni/`)**:\n"
                "   - **Hockey-Only Guardrail**: 3-layer gate rejecting non-hockey queries immediately.\n"
                "   - **Global HF Rate Limit**: Unlimited locally (`is_local_streamlit() == True`); strictly throttled to **1 response / 1.0 second across all internet users** on Hugging Face Spaces (`is_local_streamlit() == False`).\n"
                "   - **Compute-First Synthesis**: Executes deterministic DuckDB/Pandas skills over `jjerlee/nhl-analytics` + Hybrid RAG over methodology docs, then synthesizes via local **LM Studio** (`http://127.0.0.1:1234/v1`), cloud **ZeroGPU** (`nvidia/Nemotron-Mini-4B-Instruct`), or `zamboni-guardrails-v1`.\n"
                "   - **LangGraph & Label Studio Ready**: Built on a 6-node state graph with Label Studio HITL export/import (`label_studio_bridge.py`) to curate golden Q&A pairs in `insights/zamboni/golden_qa.parquet`."
            ),
            keywords=[
                "rinklytics", "zamboni", "gordie", "zambia", "architecture", "platform", "home",
                "huggingface", "hugging face", "jjerlee/nhl-analytics", "data_loader", "parquet",
                "lakehouse", "lm studio", "zerogpu", "rate limit", "guardrail", "label studio", "langgraph",
            ],
        ),
    ]


KNOWLEDGE_CHUNKS_FR = {
    "kb_xg_gsax_dual_stage": {
        "title": "Buts attendus (xG, ixG) et GSAx des gardiens — Modèle logistique conditionnel à deux étapes",
        "summary": (
            "Rinklytics calcule les buts attendus (xG/ixG) et les buts sauvés au-dessus de la moyenne attendue (GSAx) "
            "à l'aide d'un modèle logistique conditionnel à deux étapes dans `src/data/apinhle/function/procs_xg.py` "
            "afin de garantir un calibrage à somme nulle sur l'ensemble de la ligue, couvrant à la fois les tirs Fenwick non bloqués "
            "(~6,82 % de conversion) et les tirs cadrés hors filet désert (~10,21 % de conversion)."
        ),
    },
    "kb_skater_stats_ppi": {
        "title": "Indice de performance des patineurs (PPI) — Modèle multi-facteurs en Z-Scores",
        "summary": (
            "Rinklytics synthétise l'impact individuel des patineurs en combinant la production offensive réelle (buts, passes), "
            "la création offensive sous-jacente (ixG, tirs à haut danger) et l'efficacité différentielle au sein d'un "
            "score Z standardisé par minute de temps de glace (TOI)."
        ),
    },
    "kb_goalie_stats_ppi": {
        "title": "Indice de performance des gardiens (PPI) et GSAx — Évaluation indépendante de la défense",
        "summary": (
            "Mesure l'efficacité réelle des gardiens indépendamment de la qualité de la structure défensive devant eux, "
            "en calculant la différence cumulative entre les buts attendus cadrés (xGA_SOG) et les buts réels accordés (GA)."
        ),
    },
    "kb_team_season_kpis": {
        "title": "Statistiques d'équipe — Espérance pythagoricienne (RPE), RPI et rang par pairs",
        "summary": (
            "Évalue la véritable force d'une franchise au-delà du simple total de points grâce à l'espérance pythagoricienne "
            "adaptée aux prolongations (exposant 2,15), à l'indice de force d'opposition (RPI) et au classement par victoires directes "
            "face à des adversaires communs."
        ),
    },
    "kb_rebuild_tracker": {
        "title": "Reconstructions de franchises et régimes d'équilibre (2000–2026) — Modèle Markovien",
        "summary": (
            "Étude économétrique rétrospective sur 25 saisons classant chaque saison en trois macro-régimes (`Peak`, `Rebuild`, `Inaction`) "
            "et six sous-états (`R0` à `R5`), isolant l'impact du démantèlement (`ALI`), de l'injection jeunesse (`YIS`), "
            "du ratio d'encadrement vétéran (`VIR`) et de la synergie physique (`SPSI`)."
        ),
    },
    "kb_hit_impact": {
        "title": "Impact des mises en échec — Chaînes séquentielles (+1 à +4) et fenêtres de 30 secondes",
        "summary": (
            "Analyse séquentielle démontrant que les mises en échec défensives sont des actions d'endiguement (taux de conservation "
            "adverse > 61 %) alors que les mises en échec en échec avant offensif augmentent drastiquement les chances de revirement et de but."
        ),
    },
    "kb_blocking_shots": {
        "title": "Valeur des tirs bloqués — Diagramme de Sankey et chaînes de possession",
        "summary": (
            "Modélise les tirs bloqués comme des bifurcations stochastiques montrant le taux de récupération de rondelle par la défense "
            "vs la poursuite de l'attaque adverse en zone défensive."
        ),
    },
    "kb_scoreboard_recaps": {
        "title": "Résumés de matchs et analyses de soirées — Architecture AGENT ZAMBONI",
        "summary": (
            "Génération de résumés de matchs de calibre journalistique et synthèses de soirées de la LNH vérifiés à 100 % par critique "
            "déterministe contre les feuilles de match, la chronologie des périodes et les profils de gardiens de relève."
        ),
    },
    "kb_draft_magnifiques": {
        "title": "Analyse du repêchage et saisons exceptionnelles — Modélisation de survie",
        "summary": (
            "Analyse du développement des espoirs repêchés et identification des saisons exceptionnelles au 95e percentile de la ligue."
        ),
    },
    "kb_teams_map_travel": {
        "title": "Emplacements des équipes et distances de déplacement — Carte interactive Folium",
        "summary": (
            "Calcul des distances de déplacement géodésiques le long du calendrier de la LNH, impacts de la fatigue et des matchs consécutifs."
        ),
    },
    "kb_platform_and_zamboni": {
        "title": "Architecture de la plateforme Rinklytics et AGENT ZAMBONI",
        "summary": (
            "Système d'agent conversationnel spécialisé dans l'analytique du hockey sur glace, doté d'un filtre de domaine strict, "
            "d'une confidentialité par session éphémère, d'une limitation de débit globale et d'une vérification déterministe."
        ),
    },
}
