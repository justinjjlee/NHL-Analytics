"""
Franchise Rebuilds & Equilibrium Regimes (2000-2025)
====================================================
Interactive Streamlit dashboard presenting the 25-year retrospective study of NHL
franchise rebuilds, Markov-switching macro regimes ('Inaction', 'Rebuild', 'Peak'),
succeeded vs. failed rebuild quadrant diagnostics, causal decomposer, 32-franchise
retrospective dossiers, 5-part Medium blog series (Montréal vs. Chicago), and a
front-office counterfactual strategy simulator.

Designed with plain-English terminology guides, spacious full-width stacked charts
(no cramped side-by-side subplots), human-readable table headers, and zero clipped text.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent.parent
REBUILD_DIR = REPO_ROOT / "dev" / "player_team_success" / "rebuild"
PROCESSED_DIR = REBUILD_DIR / "data" / "processed"
INSIGHTS_DIR = REBUILD_DIR / "results" / "insights"
TABLES_DIR = REBUILD_DIR / "results" / "tables"

STATE_DISPLAY = {
    "R0_CONTENDER": "R0: Contender (Peak Contention)",
    "R1_SLUMP": "R1: Slump (Mild Inaction)",
    "R2_TEARDOWN": "R2: Teardown (Veteran Sell-Off)",
    "R3_INCUBATION": "R3: Incubation (Developing Core)",
    "R4_EMERGENCE": "R4: Emergence (Playoff Breakout)",
    "R5_PURGATORY": "R5: Purgatory (Stalled 5+ Years)",
}

STATE_DISPLAY_FR = {
    "R0_CONTENDER": "R0 : Prétendant (Pleine compétitivité)",
    "R1_SLUMP": "R1 : Creux (Inaction légère)",
    "R2_TEARDOWN": "R2 : Démantèlement (Vente de vétérans)",
    "R3_INCUBATION": "R3 : Incubation (Noyau en développement)",
    "R4_EMERGENCE": "R4 : Émergence (Percée en séries)",
    "R5_PURGATORY": "R5 : Purgatoire (Bloqué 5+ ans)",
}

STATE_BADGE = {
    "R0_CONTENDER": "R0",
    "R1_SLUMP": "R1",
    "R2_TEARDOWN": "R2",
    "R3_INCUBATION": "R3",
    "R4_EMERGENCE": "R4",
    "R5_PURGATORY": "R5",
}

ARCHETYPE_PLAIN = {
    "SCORCHED_EARTH_DRAFT": "Pure Draft Tanking ('Scorched Earth')",
    "HYBRID_REBUILD": "Hybrid Rebuild (Draft + Prime Trades)",
    "COMPOSITIONAL_RETOOL": "On-the-Fly Retool (Preserves Veteran Core)",
}

ARCHETYPE_PLAIN_FR = {
    "SCORCHED_EARTH_DRAFT": "Repêchage intensif (« Terre brûlée »)",
    "HYBRID_REBUILD": "Reconstruction hybride (Repêchage + Échanges clés)",
    "COMPOSITIONAL_RETOOL": "Réaménagement à la volée (Conserve les vétérans)",
}

ERA_PLAIN = {
    "ERA_1_PRE_CAP": "Era 1: Pre-Salary Cap (2000–2004)",
    "ERA_2_EARLY_CAP": "Era 2: Early Salary Cap (2005–2012)",
    "ERA_3_MODERN_CBA": "Era 3: Modern CBA (2013–2019)",
    "ERA_4_FLAT_CAP": "Era 4: Flat & Expanding Cap (2020–2025)",
}

ERA_PLAIN_FR = {
    "ERA_1_PRE_CAP": "Ère 1 : Avant le plafond salarial (2000–2004)",
    "ERA_2_EARLY_CAP": "Ère 2 : Début du plafond salarial (2005–2012)",
    "ERA_3_MODERN_CBA": "Ère 3 : Convention collective moderne (2013–2019)",
    "ERA_4_FLAT_CAP": "Ère 4 : Plafond fixe et en hausse (2020–2026)",
}

OUTCOME_TIER_FR = {
    "Tier 3: Stanley Cup Champion": "Palier 3 : Champion Coupe Stanley",
    "Tier 2: Conference Finalist (Final Four)": "Palier 2 : Finaliste de conférence (Final Four)",
    "Tier 1: Sustained Playoff Core": "Palier 1 : Noyau régulier en séries",
    "Failed: Rebuild Purgatory Trap (>=5 Yrs)": "Échec : Piège du purgatoire (>=5 ans)",
    "In-Progress / Incomplete Cycle": "En cours / Cycle incomplet",
}

BLOG_TABLE_COLUMN_RENAMES = {
    "franchise_id": "Franchise",
    "franchise_name": "Team Name",
    "season": "Season Start Year",
    "season_id": "NHL Season ID",
    "pts_pct": "Points %",
    "pred_pts_pct": "Model-Predicted Points %",
    "residual_pts_pct": "Actual Minus Predicted Points %",
    "pyth_win_pct": "Pythagorean Expected Win %",
    "xg_share_5v5": "5-on-5 Expected Goals Share (xGF%)",
    "playoff_games_won": "Playoff Game Wins",
    "total_playoff_games": "Total Playoff Game Wins",
    "playoff_rounds_won": "Playoff Series Won",
    "total_playoff_rounds": "Total Playoff Series Won",
    "macro_regime": "Macro Regime (Inaction / Rebuild / Peak)",
    "rebuild_regime": "Granular Sub-State (R0–R5)",
    "ali_score": "Veteran Sell-Off Score (ALI)",
    "yis_score": "Youth Ice-Time Share (YIS)",
    "rci_score": "Year-to-Year Roster Churn (RCI)",
    "s_b1_elc_0_3": "Entry-Level Youth Ice Time (0–3 Yrs, B1)",
    "s_b2_prime_4_6": "Prime Bridge Ice Time (4–6 Yrs, B2)",
    "s_b3_peak_7_9": "Peak Core Ice Time (7–9 Yrs, B3)",
    "s_b4_vet_10_plus": "Greybeard Veteran Ice Time (10+ Yrs, B4)",
    "prime_bridge_b2_b3": "Combined Prime Core Ice Time (4–9 Yrs, B2+B3)",
    "shannon_exp_entropy": "Roster Experience Balance (Shannon Entropy)",
    "vet_insulation_ratio": "Veteran Insulation Ratio (VIR)",
    "ccm_to_prior_year": "Year-Over-Year Core Continuity (CCM)",
    "pathway_a_draft_core_share": "Homegrown Draft Core Share",
    "pathway_b_trade_core_share": "External Trade / Free-Agent Core Share",
    "ozone_hit_pct": "Offensive-Zone Forecheck Hit Share (%)",
    "spsi_style_synergy": "Skill × Physicality Synergy Score (SPSI)",
    "surplus_value_total": "Roster Net Cap Surplus ($M)",
    "pts_pct_2019": "2019/20 Points %",
    "trough_pts_pct": "Rebuild Bottom Points %",
    "trough_season": "Rebuild Bottom Year",
    "pts_pct_2024": "2024/25 Points %",
    "pts_pct_2025": "2025/26 Points %",
    "pts_pct_latest": "Latest Points %",
    "rebound_from_trough": "Points % Gain From Bottom",
    "net_change_from_2019": "Net Points % Change From 2019",
    "net_change_2019_2024": "Net Points % Change (2019–2025)",
    "mean_ali_2019_2024": "Avg Veteran Sell-Off Score (ALI)",
    "mean_yis_2019_2024": "Avg Youth Ice-Time Share (YIS)",
    "mean_elc_share": "Avg Entry-Level Youth TOI Share (B1)",
    "mean_bridge_b2_share": "Avg Prime Bridge TOI Share (B2)",
    "inaction_seasons_since_2019": "Inaction Seasons (Since 2019)",
    "rebuild_seasons_since_2019": "Rebuild Seasons (Since 2019)",
    "peak_seasons_since_2019": "Peak Seasons (Since 2019)",
    "state_latest": "Latest Sub-State",
    "macro_latest": "Latest Macro Regime",
    "state_2024": "2024/25 Sub-State",
    "macro_2024": "2024/25 Macro Regime",
    "year_k": "Rebuild Year (Years Elapsed)",
    "at_risk_entering_year": "Rebuilds Still Active Entering Year",
    "new_tier1_breakouts": "New Playoff Breakouts (Tier 1)",
    "cum_prob_tier1_playoff_core": "Cumulative Odds: Playoff Core (Tier 1)",
    "cum_prob_tier2_final_four": "Cumulative Odds: Conference Finals (Tier 2)",
    "cum_prob_tier3_stanley_cup": "Cumulative Odds: Stanley Cup (Tier 3)",
    "cum_prob_purgatory_failure": "Cumulative Odds: Rebuild Purgatory (Failed)",
    "duration_bucket": "Rebuild Duration Group",
    "n_episodes": "Number of Rebuilds",
    "tier1_playoff_core_rate": "Playoff Contention Rate (Tier 1)",
    "tier2_final_four_rate": "Conference Finals Rate (Tier 2)",
    "tier3_cup_rate": "Stanley Cup Rate (Tier 3)",
    "purgatory_failure_rate": "Purgatory Failure Rate",
    "mean_edv_invested": "Avg Draft Capital Invested (EDV)",
    "mean_net_surplus_m": "Avg Cap Surplus Generated ($M)",
    "primary_archetype": "Rebuild Strategy Type",
    "mean_duration_yrs": "Avg Rebuild Length (Years)",
    "median_duration_yrs": "Median Rebuild Length (Years)",
    "draft_core_share": "Homegrown Draft Core Share",
    "trade_ufa_core_share": "External Trade / Free-Agent Core Share",
    "tier1_playoff_rate": "Playoff Contention Rate (Tier 1)",
    "player_name": "Player Name",
    "pos_group": "Position",
    "year_inNHL": "NHL Seasons Played",
    "exp_bucket": "Experience Cluster (B1–B4)",
    "sourcing_pathway": "How Player Was Acquired",
    "draft_overall": "Overall Draft Pick #",
    "draft_franchise": "Drafting Team",
    "gamesPlayed": "Games Played",
    "points": "Points",
    "toi_per_game": "Ice Time / Game (Mins)",
    "cap_hit": "Salary Cap Hit ($M)",
    "mrp_est": "On-Ice Market Value ($M)",
    "surplus_value_est": "Net Cap Surplus ($M)",
    "parameter": "Model Variable",
    "beta_est": "Estimated Effect (Beta)",
    "robust_se_hc1": "Robust Standard Error",
    "t_stat": "t-Statistic",
    "priority_rank": "Priority Rank",
    "gm_lever_name": "GM Policy Lever & Target",
    "quirky_rule_name": "Quirky GM Rule Name",
    "parameter_shift_tested": "Parameter Range Tested (10th to 90th Pct)",
    "low_setting_peak_odds_pct": "10th-Pct Mistake Breakout Odds (%)",
    "high_setting_peak_odds_pct": "90th-Pct Optimized Breakout Odds (%)",
    "net_playoff_breakout_swing_pct": "Net Playoff Breakout Swing (+Pct Pts)",
    "net_purgatory_risk_drop_pct": "Purgatory Risk Change (Pct Pts)",
    "expected_pts_pct_gain": "Expected Points % Lift",
    "expected_wait_years_saved": "Rebuild Wait Years Saved",
    "sobol_variance_share_pct": "Sobol Global Variance Share (%)",
    "step_order": "Checklist Order",
    "checklist_step": "GM Priority Checklist Step",
    "gm_action_description": "Concrete Front-Office Action",
    "playoff_breakout_odds_pct": "5-Yr Playoff Breakout Odds (%)",
    "marginal_breakout_lift_pct": "Marginal Breakout Lift (+Pct Pts)",
    "final_four_odds_pct": "5-Yr Conference Finals Odds (%)",
    "purgatory_failure_risk_pct": "Purgatory Failure Risk (%)",
    "expected_points_pct": "Expected Points %",
    "expected_standings_points_82gm": "Expected 82-Game Points",
    "expected_wait_years": "Expected Wait Time (Years)",
    "franchise_case": "Franchise Case Study (2024/25)",
    "checklist_scorecard": "6-Step Checklist Grade",
    "simulated_playoff_breakout_odds_pct": "Simulated 5-Yr Breakout Odds (%)",
    "simulated_purgatory_risk_pct": "Simulated Purgatory Risk (%)",
    "simulated_pts_pct": "Simulated Points %",
    "actual_2024_pts_pct": "Actual 2024/25 Points %",
    "simulated_wait_years": "Simulated Wait Time (Years)",
    "exact_shapley_lift_pct": "Exact Shapley Lift (+Pct Pts)",
    "shapley_share_pct": "Shapley Attribution Share (%)",
    "sobol_first_order_pct": "Sobol First-Order Share (Si, %)",
    "sobol_total_order_pct": "Sobol Total-Order Share (STi, %)",
    "sobol_interaction_pct": "Non-Linear Interaction Share (%)",
    "sobol_95ci": "Sobol Total-Order 95% Bootstrap CI",
    "breakout_90pct_mc_band": "90% Monte Carlo Band (10k Rollouts)",
    "execution_sequence": "Multi-Year GM Execution Sequence (Steps 1 to 6)",
    "trajectory_5yr_breakout_odds_pct": "5-Yr Trajectory Breakout Odds (%)",
    "trajectory_5yr_purgatory_exposure_pct": "5-Yr Purgatory Exposure (%)",
    "trajectory_mean_pts_pct": "Trajectory Mean Points %",
    "sequence_rank_of_720": "Sequence Rank (Out of 6! = 720)",
    "computation_module": "Computational Sensitivity & Validation Module",
    "sample_or_grid_size": "Simulation / Resample Scale",
    "primary_diagnostic": "Primary Econometric Diagnostic",
    "validation_result": "Out-of-Sample / Global Validation Result",
}

BLOG_TABLE_COLUMN_RENAMES_FR = {
    "franchise_id": "Franchise",
    "franchise_name": "Nom de l'équipe",
    "season": "Année de début",
    "season_id": "ID Saison LNH",
    "pts_pct": "% de points",
    "pred_pts_pct": "% de points prédit par le modèle",
    "residual_pts_pct": "Différence (% réel - prédit)",
    "pyth_win_pct": "% de victoires pythagoricien",
    "xg_share_5v5": "Part des buts attendus à 5v5 (xGF%)",
    "playoff_games_won": "Matchs de séries gagnés",
    "total_playoff_games": "Total matchs de séries gagnés",
    "playoff_rounds_won": "Séries éliminatoires remportées",
    "total_playoff_rounds": "Total séries remportées",
    "macro_regime": "Macro-régime (Inaction / Rebuild / Peak)",
    "rebuild_regime": "Sous-état opérationnel (R0–R5)",
    "ali_score": "Intensité de liquidation (ALI)",
    "yis_score": "Part de temps de glace jeunesse (YIS)",
    "rci_score": "Renouvellement d'effectif (RCI)",
    "s_b1_elc_0_3": "Temps de glace recrues (0–3 ans, B1)",
    "s_b2_prime_4_6": "Temps de glace passerelle (4–6 ans, B2)",
    "s_b3_peak_7_9": "Temps de glace sommet (7–9 ans, B3)",
    "s_b4_vet_10_plus": "Temps de glace vétérans (10+ ans, B4)",
    "prime_bridge_b2_b3": "Noyau d'âge mûr combiné (4–9 ans, B2+B3)",
    "shannon_exp_entropy": "Équilibre d'expérience (Entropie de Shannon)",
    "vet_insulation_ratio": "Ratio d'encadrement vétéran (VIR)",
    "ccm_to_prior_year": "Continuité du noyau (CCM)",
    "pathway_a_draft_core_share": "Part du noyau issu du repêchage",
    "pathway_b_trade_core_share": "Part du noyau issu d'échanges / autonomes",
    "ozone_hit_pct": "Mises en échec en zone offensive (%)",
    "spsi_style_synergy": "Synergie habileté × jeu physique (SPSI)",
    "surplus_value_total": "Surplus salarial net de l'effectif ($M)",
    "pts_pct_2019": "% de points 2019/20",
    "trough_pts_pct": "% de points au creux de reconstruction",
    "trough_season": "Année du creux",
    "pts_pct_2024": "% de points 2024/25",
    "pts_pct_2025": "% de points 2025/26",
    "pts_pct_latest": "% de points le plus récent",
    "rebound_from_trough": "Gain de % de points depuis le creux",
    "net_change_from_2019": "Évolution nette depuis 2019",
    "net_change_2019_2024": "Variation nette 2019–2025 (% de points)",
    "mean_ali_2019_2024": "Moyenne liquidation vétérans (ALI)",
    "mean_yis_2019_2024": "Moyenne temps de glace jeunesse (YIS)",
    "mean_elc_share": "Moyenne temps de glace recrues (B1)",
    "mean_bridge_b2_share": "Moyenne temps de glace passerelle (B2)",
    "inaction_seasons_since_2019": "Saisons en inaction (depuis 2019)",
    "rebuild_seasons_since_2019": "Saisons en reconstruction (depuis 2019)",
    "peak_seasons_since_2019": "Saisons au sommet (depuis 2019)",
    "state_latest": "Dernier sous-état",
    "macro_latest": "Dernier macro-régime",
    "state_2024": "Sous-état 2024/25",
    "macro_2024": "Macro-régime 2024/25",
    "year_k": "Année de reconstruction",
    "at_risk_entering_year": "Reconstructions actives au début de l'année",
    "new_tier1_breakouts": "Nouvelles percées en séries (Palier 1)",
    "cum_prob_tier1_playoff_core": "Chances cumulées : Séries (Palier 1)",
    "cum_prob_tier2_final_four": "Chances cumulées : Finale de conférence (Palier 2)",
    "cum_prob_tier3_stanley_cup": "Chances cumulées : Coupe Stanley (Palier 3)",
    "cum_prob_purgatory_failure": "Chances cumulées : Purgatoire (Échec)",
    "duration_bucket": "Groupe de durée",
    "n_episodes": "Nombre de reconstructions",
    "tier1_playoff_core_rate": "Taux de présence en séries (Palier 1)",
    "tier2_final_four_rate": "Taux de finale de conférence (Palier 2)",
    "tier3_cup_rate": "Taux de Coupe Stanley (Palier 3)",
    "purgatory_failure_rate": "Taux d'échec en purgatoire",
    "mean_edv_invested": "Capital de repêchage investi (EDV)",
    "mean_net_surplus_m": "Surplus salarial généré ($M)",
    "primary_archetype": "Stratégie de reconstruction",
    "mean_duration_yrs": "Durée moyenne (années)",
    "median_duration_yrs": "Durée médiane (années)",
    "draft_core_share": "Part du noyau issu du repêchage",
    "trade_ufa_core_share": "Part du noyau issu d'échanges / UFA",
    "tier1_playoff_rate": "Taux de qualification en séries (Palier 1)",
    "player_name": "Nom du joueur",
    "pos_group": "Position",
    "year_inNHL": "Saisons jouées dans la LNH",
    "exp_bucket": "Cohorte d'expérience (B1–B4)",
    "sourcing_pathway": "Méthode d'acquisition",
    "draft_overall": "Rang de sélection au repêchage",
    "draft_franchise": "Équipe d'origine au repêchage",
    "gamesPlayed": "Matchs joués",
    "points": "Points",
    "toi_per_game": "Temps de glace / match (min)",
    "cap_hit": "Impact sous le plafond salarial ($M)",
    "mrp_est": "Valeur marchande sur glace ($M)",
    "surplus_value_est": "Surplus salarial net ($M)",
    "parameter": "Variable du modèle",
    "beta_est": "Effet estimé (Beta)",
    "robust_se_hc1": "Erreur-type robuste",
    "t_stat": "Statistique t",
    "priority_rank": "Rang de priorité",
    "gm_lever_name": "Levier stratégique du DG",
    "quirky_rule_name": "Règle mnémotechnique du DG",
    "parameter_shift_tested": "Plage testée (10e au 90e percentile)",
    "low_setting_peak_odds_pct": "Chances de qualification au réglage faible (%)",
    "high_setting_peak_odds_pct": "Chances de qualification au réglage optimisé (%)",
    "net_playoff_breakout_swing_pct": "Variation nette des chances de qualification (+pts %)",
    "net_purgatory_risk_drop_pct": "Variation du risque de purgatoire (pts %)",
    "expected_pts_pct_gain": "Gain attendu en % de points",
    "expected_wait_years_saved": "Années d'attente économisées",
    "sobol_variance_share_pct": "Part de variance globale Sobol (%)",
    "step_order": "Ordre de la liste de contrôle",
    "checklist_step": "Étape de priorité du DG",
    "gm_action_description": "Action concrète de gestion",
    "playoff_breakout_odds_pct": "Chances de qualification sur 5 ans (%)",
    "marginal_breakout_lift_pct": "Gain marginal de qualification (+pts %)",
    "final_four_odds_pct": "Chances de finale de conférence sur 5 ans (%)",
    "purgatory_failure_risk_pct": "Risque d'échec en purgatoire (%)",
    "expected_points_pct": "% de points attendu",
    "expected_standings_points_82gm": "Points attendus sur 82 matchs",
    "expected_wait_years": "Temps d'attente attendu (années)",
    "franchise_case": "Étude de cas (2024/25)",
    "checklist_scorecard": "Note sur la liste de contrôle",
    "simulated_playoff_breakout_odds_pct": "Chances simulées de qualification sur 5 ans (%)",
    "simulated_purgatory_risk_pct": "Risque simulé de purgatoire (%)",
    "simulated_pts_pct": "% de points simulé",
    "actual_2024_pts_pct": "% de points réel 2024/25",
    "simulated_wait_years": "Temps d'attente simulé (années)",
    "exact_shapley_lift_pct": "Gain Shapley exact (+pts %)",
    "shapley_share_pct": "Part d'attribution Shapley (%)",
    "sobol_first_order_pct": "Part de premier ordre Sobol (Si, %)",
    "sobol_total_order_pct": "Part d'ordre total Sobol (STi, %)",
    "sobol_interaction_pct": "Part d'interaction non-linéaire (%)",
    "sobol_95ci": "Intervalle de confiance 95% Sobol",
    "breakout_90pct_mc_band": "Bande Monte Carlo 90% (10k simulations)",
    "execution_sequence": "Séquence d'exécution pluriannuelle du DG (Étapes 1 à 6)",
    "trajectory_5yr_breakout_odds_pct": "Chances de percée sur trajectoire 5 ans (%)",
    "trajectory_5yr_purgatory_exposure_pct": "Exposition au purgatoire sur 5 ans (%)",
    "trajectory_mean_pts_pct": "% de points moyen sur la trajectoire",
    "sequence_rank_of_720": "Rang de la séquence (sur 6! = 720)",
    "computation_module": "Module de calcul et de validation",
    "sample_or_grid_size": "Échelle de simulation / grille",
    "primary_diagnostic": "Diagnostic économétrique principal",
    "validation_result": "Résultat de validation globale",
}


def format_season_label(s: int) -> str:
    """
    Format integer start year (e.g. 2000 -> '2000/01').
    IMPORTANT: Never use 'YYYY-MM' hyphen format (e.g. '2000-01'..'2011-12') because
    Plotly.js auto-detects 'YYYY-01'..'YYYY-12' as ISO calendar dates (Jan..Dec) and
    truncates seasons >= '2012-13' (month 13 is invalid)! Using '2000/01' + xaxis
    type='category' guarantees all 26 seasons (2000/01 to 2025/26) render cleanly.
    """
    s_int = int(s)
    return f"{s_int}/{str(s_int + 1)[2:]}"


@st.cache_data(ttl=3600)
def load_rebuild_datasets():
    ts_path = PROCESSED_DIR / "panel_team_season.parquet"
    ep_path = PROCESSED_DIR / "panel_rebuild_episodes.parquet"
    occ_path = PROCESSED_DIR / "panel_franchise_regime_occupancy.parquet"
    man_path = INSIGHTS_DIR / "rebuild_manifest.json"

    df_ts = pd.read_parquet(ts_path) if ts_path.exists() else pd.DataFrame()
    df_ep = pd.read_parquet(ep_path) if ep_path.exists() else pd.DataFrame()
    df_occ = pd.read_parquet(occ_path) if occ_path.exists() else pd.DataFrame()
    manifest = json.loads(man_path.read_text(encoding="utf-8")) if man_path.exists() else {}
    return df_ts, df_ep, df_occ, manifest


def style_plotly_high_contrast(fig: go.Figure) -> go.Figure:
    """
    Enforces a consistent, high-contrast light canvas (#FFFFFF / #F8FAFC) with dark slate
    typography (#0F172A) across all axes, legends, colorbars, subplot titles, annotations,
    and polar/spider layers so charts never suffer from white-on-white text in Dark Mode.
    """
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#F8FAFC",
        font=dict(color="#0F172A", size=11.5, family="Inter, DejaVu Sans, sans-serif"),
        title_font=dict(color="#0F172A", size=13.5),
        legend=dict(
            bgcolor="#F8FAFC",
            bordercolor="#CBD5E1",
            borderwidth=1,
            font=dict(color="#0F172A", size=10.5),
        ),
        hoverlabel=dict(
            bgcolor="#0F172A",
            bordercolor="#334155",
            font=dict(color="#F8FAFC", size=11.5),
        ),
    )
    fig.update_xaxes(
        title_font=dict(color="#0F172A", size=11.5),
        tickfont=dict(color="#0F172A", size=10.5),
        gridcolor="#E2E8F0",
        linecolor="#64748B",
        zerolinecolor="#94A3B8",
        automargin=True,
    )
    fig.update_yaxes(
        title_font=dict(color="#0F172A", size=11.5),
        tickfont=dict(color="#0F172A", size=10.5),
        gridcolor="#E2E8F0",
        linecolor="#64748B",
        zerolinecolor="#94A3B8",
        automargin=True,
    )
    if fig.layout.annotations:
        for ann in fig.layout.annotations:
            ann.font = dict(color="#0F172A", size=11)
    return fig


def render_terminology_guide(lang: str = "EN"):
    """Renders a well-spaced, plain-English or French reference guide so every term and acronym is clear."""
    is_fr = lang == "FR"
    exp_title = (
        "📖 Guide des régimes, cohortes d'expérience et métriques du modèle (Cliquer pour ouvrir)"
        if is_fr
        else "📖 Plain-English Guide to All Regimes, Roster Experience Clusters & Model Metrics (Click to Expand)"
    )
    with st.expander(exp_title, expanded=False):
        if is_fr:
            st.markdown(
                """
                ### 1. Les 3 macro-régimes et les 6 sous-états opérationnels
                Notre modèle de séries chronologiques classe chaque saison d'équipe de la LNH (2000/01 à 2025/26) en **trois macro-régimes** et **six sous-états opérationnels** :
                - 🟩 **Régime `Peak` (Compétitivité et séries — 45,8 % des saisons)** :
                  - **`R0 : Prétendant`** : Équipe établie des séries (`% de points >= 0,550` ou remportant des séries avec un noyau mature).
                  - **`R4 : Émergence` (Percée durable)** : Équipe en reconstruction accédant aux séries ou au seuil de `0,550+`.
                - 🟧 **Régime `Rebuild` (Reconstruction active — 28,8 % des saisons)** :
                  - **`R2 : Démantèlement`** : La direction échange ses vétérans contre des choix de repêchage et des espoirs tout en descendant au classement.
                  - **`R3 : Incubation`** : L'équipe cesse de vendre et donne un temps de glace important aux jeunes joueurs pour qu'ils grandissent ensemble.
                - ⬛ **Régime `Inaction` (Stagnation ou purgatoire — 25,4 % des saisons)** :
                  - **`R1 : Creux`** : Saison médiocre sans séries où la direction n'entreprend **aucun changement majeur** (ni reconstruction, ni prétendant).
                  - **`R5 : Purgatoire`** : Reconstruction qui s'étire sur **5 années consécutives ou plus** sans percer.

                ---

                ### 2. Les 4 cohortes d'expérience des joueurs (Structure d'âge de l'effectif)
                Au lieu d'analyser uniquement la moyenne d'âge brute, nous divisons le temps de glace total (`TOI`) en **quatre paliers d'expérience dans la LNH** :
                - **`Cohorte B1 — Jeunesse recrue (0 à 3 ans d'expérience LNH)`** : Joueurs sous contrat d'entrée (ELC). Fort potentiel, mais au-delà de **30 % du temps de glace de l'équipe** sans vétérans d'élite, la structure défensive s'effondre (*le seuil critique de surcharge jeunesse à 30 %*).
                - **`Cohorte B2 — Passerelle d'âge mûr (4 à 6 ans d'expérience LNH)`** : Joueurs de 22 à 25 ans sous leur deuxième contrat (ex. Nick Suzuki, Cole Caufield). **Le moteur de croissance le plus déterminant du hockey moderne.**
                - **`Cohorte B3 — Vétérans au sommet (7 à 9 ans d'expérience LNH)`** : Vétérans complets de 26 à 29 ans qui stabilisent les confrontations difficiles.
                - **`Cohorte B4 — Vétérans d'expérience (10+ ans d'expérience LNH)`** : Vétérans aguerris de 30 ans et plus. Utiles avec modération, mais dépasser 40 % de temps de glace avec des joueurs vieillissants tout en alignant des recrues crée un effectif polarisé sans noyau passerelle.

                ---

                ### 3. Dictionnaire vulgarisé des métriques du modèle
                | Nom de la métrique | Code | Signification vulgarisée et interprétation |
                | :--- | :--- | :--- |
                | **Intensité de liquidation des vétérans** | `ALI` | **Asset Liquidation Intensity** : Part du temps de glace des vétérans de la saison précédente (`4+ ans`) échangée ou partie. Plus de `30 %` signale un démantèlement délibéré. |
                | **Part de temps de glace jeunesse** | `YIS` | **Youth Influx Score** : Part du temps de glace total joué par de jeunes patineurs (`0 à 3 ans`) et recrues. |
                | **Renouvellement d'effectif** | `RCI` | **Roster Churn Index** : Pourcentage des minutes de temps de glace remplacées par de nouveaux joueurs par rapport à la saison précédente. |
                | **Continuité du noyau** | `CCM` | **Core Continuity Metric** : Part des 12 joueurs clés conservée d'une saison à l'autre (`100 %` = stabilité totale du noyau). |
                | **Ratio d'encadrement vétéran** | `VIR` | Ratio entre le **temps de glace des vétérans clés (`4 à 9 ans`)** et celui de la **jeunesse recrue (`0 à 3 ans`)**. Des valeurs `>= 1,25` signifient que les jeunes sont bien encadrés. |
                | **Capital de repêchage investi** | `EDV` | Valeur historique totale des choix de repêchage investis par l'équipe durant son cycle de reconstruction. |
                | **Part des mises en échec en zone O** | `% Échec avant zone O` | Pourcentage des mises en échec portées en **zone offensive** en échec avant (pour provoquer des revirements) plutôt que de poursuivre la rondelle en zone défensive. |
                | **Synergie habileté × jeu physique** | `SPSI` | Score combinant les **buts attendus à 5v5 (`xGF%`)** et les **mises en échec offensives proactives**. Les scores élevés (`>= 0,50`) sont fortement corrélés aux victoires en séries. |
                """
            )
        else:
            st.markdown(
                """
                ### 1. The 3 Macro Regimes & 6 Operational Sub-States
                Our time-series model classifies every NHL team-season (2000/01 through 2025/26) into **three overarching regimes** and **six specific team states**:
                - 🟩 **`Peak` Regime (Successful Contention — 45.8% of seasons)**:
                  - **`R0: Contender`**: Established playoff powerhouse (`Points % >= .550` or winning playoff series with a mature core).
                  - **`R4: Emergence` (Sustained Breakout)**: A rebuilding team that has officially graduated into the playoffs or `.550+` contention.
                - 🟧 **`Rebuild` Regime (Active Reconstruction — 28.8% of seasons)**:
                  - **`R2: Teardown`**: The front office is actively trading away veterans for draft picks and prospects while bottoming out in the standings.
                  - **`R3: Incubation`**: The team has stopped selling veterans and is giving heavy NHL minutes to young players so they can grow together.
                - ⬛ **`Inaction` Regime (Treading Water or Stuck in Purgatory — 25.4% of seasons)**:
                  - **`R1: Slump`**: A mediocre, non-playoff season where management makes **no major changes**—neither contending nor rebuilding.
                  - **`R5: Purgatory`**: A rebuild that has dragged on for **5 or more consecutive non-contending years** without breaking through.

                ---

                ### 2. The 4 Player Experience Clusters (How Roster Age Is Measured)
                Instead of looking only at average roster age, we divide every team's total ice time (`TOI`) into **four NHL experience tiers**:
                - **`Cluster B1 — Entry-Level Youth (0–3 Years NHL Experience)`**: Rookies and players on Entry-Level Contracts (ELCs). High upside, but if they play **more than 30% of team ice time** without veteran help, defensive structure collapses (*the 30% Youth Overload Cliff*).
                - **`Cluster B2 — Prime Bridge Core (4–6 Years NHL Experience)`**: Players aged roughly 22–25 on their second contract (e.g., Nick Suzuki, Cole Caufield). **This is the single most important growth engine in modern hockey.**
                - **`Cluster B3 — Peak Core Veterans (7–9 Years NHL Experience)`**: Prime two-way veterans aged roughly 26–29 who stabilize tough matchups.
                - **`Cluster B4 — Greybeard Anchors (10+ Years NHL Experience)`**: Seasoned 30+ year-old veterans. Helpful in moderation, but filling >40% of ice time with aging stopgaps while playing 19-year-olds creates a *"Dumbbell Roster"* with no 23–26 year-old bridge core.

                ---

                ### 3. Plain-English Dictionary of Model Metrics
                | Metric Name | Short Code | Plain-English Meaning & How to Read It |
                | :--- | :--- | :--- |
                | **Veteran Sell-Off Score** | `ALI` | **Asset Liquidation Intensity**: Share of last season's veteran ice time (`4+ yrs experience`) traded away or departed. Above `30%` signals a deliberate front-office teardown. |
                | **Youth Ice-Time Share** | `YIS` | **Youth Influx Score**: Share of total team ice time played by young skaters (`0–3 yrs experience`) plus rookie debuts. |
                | **Year-to-Year Roster Churn** | `RCI` | **Roster Churn Index**: Percentage of a team's total ice-time minutes replaced by different players compared to the previous season. |
                | **Core Continuity** | `CCM` | **Core Continuity Metric**: Share of a team's top-12 core players retained from the prior season (`100%` = zero core turnover). |
                | **Veteran Insulation Ratio** | `VIR` | Ratio of **Prime/Peak Veteran ice time (`4–9 yrs experience`)** to **Entry-Level Youth ice time (`0–3 yrs`)**. Values `>= 1.25` mean young players are properly sheltered by prime veterans. |
                | **Expected Draft Value** | `EDV` | Total historical point value of all NHL Draft picks invested by a team during its rebuild window (higher = more lottery picks spent). |
                | **Offensive-Zone Hit Share** | `O-Zone Hit %` | Percentage of a team's physical hits delivered in the **offensive zone** on the forecheck (forcing turnovers) rather than chasing the puck in their own defensive zone. |
                | **Skill × Physicality Synergy** | `SPSI` | Combined score measuring whether a team pairs **5-on-5 Expected Goals Skill (`xGF%`)** with **proactive offensive-zone forecheck hitting**. High scores (`>= 0.50`) strongly predict playoff series wins. |
                """
            )


def render_page():
    lang = st.session_state.get("lang", "EN")
    is_fr = lang == "FR"
    df_ts, df_ep, df_occ, manifest = load_rebuild_datasets()

    if df_ts.empty or df_ep.empty or df_occ.empty:
        err_msg = (
            "Données analytiques de reconstruction introuvables. Veuillez exécuter "
            "`uv run python dev/player_team_success/rebuild/src/data/build_rebuild_panels.py`."
            if is_fr
            else "Rebuild analytical panels not found. Please run "
            "`uv run python dev/player_team_success/rebuild/src/data/build_rebuild_panels.py`."
        )
        st.error(err_msg)
        return

    # Localization mappings
    state_display_local = STATE_DISPLAY_FR if is_fr else STATE_DISPLAY
    archetype_plain_local = ARCHETYPE_PLAIN_FR if is_fr else ARCHETYPE_PLAIN
    era_plain_local = ERA_PLAIN_FR if is_fr else ERA_PLAIN

    # Ensure clean human-readable columns exist across df_ts and df_ep
    df_ts = df_ts.sort_values(["franchise_id", "season"]).copy()
    df_ts["season_label"] = [format_season_label(s) for s in df_ts["season"]]
    df_ts["substate_clean"] = df_ts["rebuild_regime"].map(state_display_local).fillna(df_ts["rebuild_regime"])

    df_ep = df_ep.copy()
    df_ep["archetype_plain"] = df_ep["primary_archetype"].map(archetype_plain_local).fillna(df_ep["primary_archetype"])
    df_ep["era_plain"] = df_ep["era_code"].map(era_plain_local).fillna(df_ep["era_code"])

    # Responsive CSS styling
    st.markdown(
        """
        <style>
        div[data-testid="stMainBlockContainer"],
        .main .block-container {
            padding-left: 1.5rem !important;
            padding-right: 1.5rem !important;
        }
        div[data-testid="stMetric"] {
            background-color: #F8FAFC !important;
            border: 1.5px solid #CBD5E1 !important;
            border-radius: 10px !important;
            padding: 12px 14px !important;
            margin-bottom: 12px !important;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.06) !important;
            overflow: visible !important;
            box-sizing: border-box !important;
        }
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] > div,
        div[data-testid="stMetricLabel"] p,
        div[data-testid="stMetricLabel"] span {
            color: #334155 !important;
            font-weight: 600 !important;
            font-size: 0.90rem !important;
            line-height: 1.3 !important;
            white-space: normal !important;
            word-break: break-word !important;
        }
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] > div {
            color: #0F172A !important;
            font-weight: 800 !important;
            font-size: 1.25rem !important;
            line-height: 1.25 !important;
            margin-top: 4px !important;
            margin-bottom: 4px !important;
            white-space: normal !important;
            word-break: break-word !important;
        }
        div[data-testid="stMetricDelta"],
        div[data-testid="stMetricDelta"] > div {
            font-weight: 600 !important;
            font-size: 0.82rem !important;
            line-height: 1.3 !important;
            white-space: normal !important;
            word-break: break-word !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Title & Persistent Retrospective Epistemological Notice Banner
    if lang == "FR":
        st.title("🏒 Reconstructions des franchises et régimes d'équilibre (2000–2026)")
        st.caption(
            "Analyse économétrique rétrospective sur 25 saisons (2000/01 à 2025/26) des cycles de reconstruction, des régimes "
            "d'équilibre (« Inaction », « Rebuild », « Peak »), de la structure d'âge et du style de jeu physique."
        )
        st.info(
            "📌 **Note méthodologique (Étude historique rétrospective post-hoc)** : "
            "Cette analyse évalue de manière rétrospective les 25 saisons complétées 2000/01 à 2025/26 de la LNH à l'aide d'un modèle "
            "à changement de régime (« Inaction », « Rebuild », « Peak »). **Elle ne constitue en aucun cas une prédiction "
            "du comportement futur des équipes.** Ses résultats expliquent plutôt pourquoi certaines décisions "
            "stratégiques ont historiquement été efficaces — ou inefficaces — pour faire passer une franchise vers le régime « Peak »."
        )
    else:
        st.title("🏒 Franchise Rebuilds & Equilibrium Regimes (2000/01–2025/26)")
        st.caption(
            "25-Season Retrospective Study of NHL Franchise Rebuilds, Equilibrium Regimes "
            "(`Inaction`, `Rebuild`, `Peak`), Roster Experience Structure, and Forechecking Style Synergy."
        )
        st.info(
            "📌 **Important Context (Post-Hoc Retrospective Historical Study)**: "
            "This study looks back at 25 completed NHL seasons (**2000/01 through 2025/26**, `N = 764` team-seasons) using an econometric "
            "state-space regime model (`Inaction`, `Rebuild`, `Peak`). **It is a historical audit, not a crystal-ball prediction "
            "of future team behavior.** Instead, its measurements show which front-office decisions have historically been "
            "**effective or tragically ineffective** in moving an organization out of `Inaction` and `Rebuild` into `Peak` contention."
        )

    # Always-accessible Plain-English Terminology & Metric Guide right below the header
    render_terminology_guide(lang=lang)

    tab_labels = (
        [
            "1️⃣ 25-Yr Regime Matrix",
            "2️⃣ Succeeded vs. Failed Rebuilds",
            "3️⃣ Why Did It Succeed or Fail?",
            "4️⃣ Franchise Dossiers (32 Teams)",
            "5️⃣ Era Trends & Youth/Hit Synergy",
            "6️⃣ GM Rebuild Simulator",
            "7️⃣ 6-Part Blog Series (MTL vs. CHI)",
        ]
        if lang == "EN"
        else [
            "1️⃣ Matrice des régimes (25 ans)",
            "2️⃣ Succès vs Échecs",
            "3️⃣ Pourquoi succès ou échec ?",
            "4️⃣ Dossiers des 32 franchises",
            "5️⃣ Tendances par ère et synergie",
            "6️⃣ Simulateur stratégique DG",
            "7️⃣ Série de 6 articles (MTL vs CHI)",
        ]
    )

    t1, t2, t3, t4, t5, t6, t7 = st.tabs(tab_labels)

    # =========================================================================
    # TAB 1: 25-Year Franchise Regime Matrix & Three-Regime Predominance Audit
    # =========================================================================
    with t1:
        st.subheader(
            "1. Matrice des régimes des franchises (25 ans : Inaction, Rebuild, Peak)"
            if is_fr
            else "1. 25-Season Franchise Regime Map (Inaction, Rebuild, Peak)"
        )
        if is_fr:
            st.markdown(
                """
                **Comment lire ce graphique** : Chaque ligne représente l'une des 32 franchises de la LNH (selon son identifiant canonique le plus récent, par ex. **Utah Hockey Club `UTA`** pour la lignée `PHX`/`ARI`/`UTA`) et chaque colonne est une saison LNH complétée de **`2000/01` à `2025/26`**.
                - 🟩 **Vert (Régime « Peak » / Sommet — 46,1 % des saisons-équipes)** : Équipes prétendantes remportant des séries éliminatoires ou affichant un % de points `>= .550` (`R0 : Prétendant` et `R4 : Émergence` soutenue).
                - 🟧 **Orange (Régime « Rebuild » / Reconstruction — 28,5 % des saisons-équipes)** : Équipes échangeant activement des vétérans contre des choix au repêchage (`R2 : Démantèlement`) ou développant un jeune noyau (`R3 : Incubation`).
                - ⬛ **Gris ardoise (Régime « Inaction » — 25,4 % des saisons-équipes)** : Saisons infructueuses où l'équipe **ne fait aucun changement majeur** (`R1 : Affaissement`) ou **reste coincée 5+ ans dans une reconstruction ratée** (`R5 : Purgatoire`).
                """
            )
        else:
            st.markdown(
                """
                **How to read this chart**: Every row is one of the 32 NHL franchises (using the latest franchise label as the canonical ID, such as **Utah Hockey Club `UTA`** for the `PHX`/`ARI`/`UTA` lineage) and every column is a completed NHL season from **`2000/01` through `2025/26`**.
                - 🟩 **Green (`Peak` Regime — 46.1% of all team-seasons)**: Contending teams winning playoff series or posting a `.550+` points percentage (`R0: Contender` and sustained `R4: Emergence`).
                - 🟧 **Orange (`Rebuild` Regime — 28.5% of all team-seasons)**: Teams actively trading away veterans for draft capital (`R2: Teardown`) or developing a young core (`R3: Incubation`).
                - ⬛ **Slate Grey (`Inaction` Regime — 25.4% of all team-seasons)**: Unsuccessful seasons where a team either makes **no major changes to improve** (`R1: Slump`) or gets **stuck for 5+ years in a failed rebuild** (`R5: Purgatory`).
                """
            )

        col_ctrl1, col_ctrl2 = st.columns([1.3, 1.0])
        with col_ctrl1:
            view_mode_opts = (
                [
                    "3 Macro-régimes (Inaction / Rebuild / Peak)",
                    "6 Sous-états détaillés (R0 Prétendant à R5 Purgatoire)",
                ]
                if is_fr
                else [
                    "3 Macro Regimes (Inaction / Rebuild / Peak)",
                    "6 Detailed Sub-States (R0 Contender to R5 Purgatory)",
                ]
            )
            view_mode = st.radio(
                "Niveau de détail de la carte :" if is_fr else "Choose Level of Detail on the Map:",
                view_mode_opts,
                horizontal=True,
            )
        with col_ctrl2:
            sort_opts = (
                [
                    "Temps passé au sommet / Peak (%)",
                    "Temps passé en inaction / purgatoire (%)",
                    "Temps passé en reconstruction (%)",
                    "Coupes Stanley remportées (2000–2025)",
                    "Alphabétique (A à Z)",
                ]
                if is_fr
                else [
                    "Most Time Spent in Peak Contention (%)",
                    "Most Time Spent in Inaction / Purgatory (%)",
                    "Most Time Spent in Active Rebuild (%)",
                    "Stanley Cups Won (2000–2025)",
                    "Alphabetical (A to Z)",
                ]
            )
            sort_mode = st.selectbox(
                "Trier les franchises par :" if is_fr else "Sort Franchises From Top to Bottom By:",
                sort_opts,
            )

        df_occ_sorted = df_occ.copy()
        if sort_mode in ("Most Time Spent in Peak Contention (%)", "Temps passé au sommet / Peak (%)"):
            df_occ_sorted = df_occ_sorted.sort_values(["share_peak", "stanley_cups_won"], ascending=[True, True])
        elif sort_mode in ("Most Time Spent in Inaction / Purgatory (%)", "Temps passé en inaction / purgatoire (%)"):
            df_occ_sorted = df_occ_sorted.sort_values(["share_inaction", "share_rebuild"], ascending=[True, True])
        elif sort_mode in ("Most Time Spent in Active Rebuild (%)", "Temps passé en reconstruction (%)"):
            df_occ_sorted = df_occ_sorted.sort_values(["share_rebuild", "share_peak"], ascending=[True, True])
        elif sort_mode in ("Stanley Cups Won (2000–2025)", "Coupes Stanley remportées (2000–2025)"):
            df_occ_sorted = df_occ_sorted.sort_values(["stanley_cups_won", "share_peak"], ascending=[True, True])
        else:
            df_occ_sorted = df_occ_sorted.sort_values("franchise_id", ascending=False)

        teams_ord = df_occ_sorted["franchise_id"].tolist()
        seasons_ord = sorted(df_ts["season"].unique())
        season_labels = [format_season_label(s) for s in seasons_ord]

        if "3 Macro" in view_mode:
            code_map = {"Inaction": 0, "Rebuild": 1, "Peak": 2}
            colorscale = [
                [0.0, "#334155"],
                [0.33, "#334155"],
                [0.34, "#EA580C"],
                [0.66, "#EA580C"],
                [0.67, "#15803D"],
                [1.0, "#15803D"],
            ]
            col_field = "macro_regime"
            zmin, zmax = 0, 2
            cbar_vals = [0.33, 1.0, 1.67]
            cbar_text = (
                ["Inaction", "Reconstruction", "Sommet (Peak)"]
                if is_fr
                else ["Inaction", "Rebuild", "Peak"]
            )
        else:
            code_map = {
                "R5_PURGATORY": 0,
                "R1_SLUMP": 1,
                "R2_TEARDOWN": 2,
                "R3_INCUBATION": 3,
                "R4_EMERGENCE": 4,
                "R0_CONTENDER": 5,
            }
            colorscale = [
                [0.0, "#1E293B"],
                [0.16, "#1E293B"],
                [0.17, "#64748B"],
                [0.33, "#64748B"],
                [0.34, "#B91C1C"],
                [0.50, "#B91C1C"],
                [0.51, "#EA580C"],
                [0.66, "#EA580C"],
                [0.67, "#2563EB"],
                [0.83, "#2563EB"],
                [0.84, "#15803D"],
                [1.0, "#15803D"],
            ]
            col_field = "rebuild_regime"
            zmin, zmax = 0, 5
            cbar_vals = [0.4, 1.25, 2.1, 2.9, 3.75, 4.6]
            cbar_text = (
                [
                    "R5 : Purgatoire",
                    "R1 : Affaissement",
                    "R2 : Démantèlement",
                    "R3 : Incubation",
                    "R4 : Émergence",
                    "R0 : Prétendant",
                ]
                if is_fr
                else [
                    "R5: Purgatory",
                    "R1: Slump",
                    "R2: Teardown",
                    "R3: Incubation",
                    "R4: Emergence",
                    "R0: Contender",
                ]
            )

        z_mat = np.full((len(teams_ord), len(seasons_ord)), np.nan)
        hover_mat = np.empty((len(teams_ord), len(seasons_ord)), dtype=object)
        cup_x, cup_y = [], []

        for i, f_id in enumerate(teams_ord):
            sub = df_ts[df_ts["franchise_id"] == f_id].set_index("season")
            for j, s in enumerate(seasons_ord):
                s_lbl = season_labels[j]
                if s in sub.index:
                    r = sub.loc[s]
                    z_mat[i, j] = code_map.get(r[col_field], 0)
                    cup_str = (
                        (" 🏆 CHAMPION COUPE STANLEY" if is_fr else " 🏆 STANLEY CUP CHAMPION")
                        if r["won_stanley_cup"]
                        else ((" ⭐ Finaliste de conférence" if is_fr else " ⭐ Conference Finalist") if r["reached_final_four"] else "")
                    )
                    if is_fr:
                        hover_mat[i, j] = (
                            f"<b>{r['franchise_name']} ({f_id} / Code : {r['raw_team_abbrev']}) — {s_lbl}</b>{cup_str}<br>"
                            f"Macro-régime : <b>{r['macro_regime']}</b> | Sous-état : <b>{state_display_local.get(r['rebuild_regime'], r['rebuild_regime'])}</b><br>"
                            f"% de points : <b>{r['pts_pct']:.3f}</b> | Victoires séries : <b>{int(r.get('playoff_games_won', 0))}</b> | Séries gagnées : <b>{int(r['playoff_rounds_won'])}</b><br>"
                            f"Liquidation vétérans (ALI) : {100*r['ali_score']:.1f}% | Temps jeunesse (YIS) : {100*r['yis_score']:.1f}% | Churn effectif : {100*r['rci_score']:.1f}%<br>"
                            f"Recrues 0–3 ans : {100*r['s_b1_elc_0_3']:.1f}% | Passerelle 4–6 ans : {100*r['s_b2_prime_4_6']:.1f}% | Échec avant zone offensive : {r['ozone_hit_pct']:.1f}%"
                        )
                    else:
                        hover_mat[i, j] = (
                            f"<b>{r['franchise_name']} ({f_id} / Active Code: {r['raw_team_abbrev']}) — {s_lbl}</b>{cup_str}<br>"
                            f"Macro Regime: <b>{r['macro_regime']}</b> | Specific State: <b>{STATE_DISPLAY.get(r['rebuild_regime'], r['rebuild_regime'])}</b><br>"
                            f"Regular-Season Points %: <b>{r['pts_pct']:.3f}</b> | Playoff Game Wins: <b>{int(r.get('playoff_games_won', 0))}</b> | Playoff Series Won: <b>{int(r['playoff_rounds_won'])}</b><br>"
                            f"Veteran Sell-Off (ALI): {100*r['ali_score']:.1f}% | Youth Ice Time (YIS): {100*r['yis_score']:.1f}% | Roster Churn: {100*r['rci_score']:.1f}%<br>"
                            f"Entry-Level Youth TOI (0–3 yrs): {100*r['s_b1_elc_0_3']:.1f}% | Prime Bridge TOI (4–6 yrs): {100*r['s_b2_prime_4_6']:.1f}% | O-Zone Forecheck Hits: {r['ozone_hit_pct']:.1f}%"
                        )
                    if r["won_stanley_cup"]:
                        cup_x.append(s_lbl)
                        cup_y.append(f_id)
                else:
                    hover_mat[i, j] = (
                        f"<b>{f_id} ({s_lbl})</b> : Franchise d'expansion non active"
                        if is_fr
                        else f"<b>{f_id} ({s_lbl})</b>: Expansion Franchise Not Yet Active in NHL"
                    )

        fig_hm = go.Figure()
        fig_hm.add_trace(
            go.Heatmap(
                z=z_mat,
                x=season_labels,
                y=teams_ord,
                text=hover_mat,
                hoverinfo="text",
                colorscale=colorscale,
                zmin=zmin,
                zmax=zmax,
                showscale=True,
                colorbar=dict(
                    title=dict(text="Régime" if is_fr else "Regime State", font=dict(color="#0F172A", size=12)),
                    tickvals=cbar_vals,
                    ticktext=cbar_text,
                    tickfont=dict(color="#0F172A", size=11),
                    len=0.75,
                    thickness=18,
                ),
                xgap=2.0,
                ygap=2.0,
            )
        )
        if cup_x:
            fig_hm.add_trace(
                go.Scatter(
                    x=cup_x,
                    y=cup_y,
                    mode="markers+text",
                    text=["🏆"] * len(cup_x),
                    textposition="middle center",
                    marker=dict(size=12, color="rgba(0,0,0,0)"),
                    hoverinfo="skip",
                    name="Champion Coupe Stanley (🏆)" if is_fr else "Stanley Cup Champion (🏆)",
                )
            )
        fig_hm.update_layout(
            height=900,
            margin=dict(l=65, r=35, t=65, b=115),
            title=dict(
                text="Chronologie des régimes des franchises (2000–2026)" if is_fr else "25-Season Franchise Regime Timeline",
                font=dict(size=14, color="#0F172A"),
                y=0.98,
                x=0.0,
                xanchor="left",
            ),
            xaxis=dict(
                title="Saison LNH" if is_fr else "NHL Season",
                type="category",
                categoryorder="array",
                categoryarray=season_labels,
                tickmode="array",
                tickvals=season_labels,
                ticktext=season_labels,
                tickangle=-35,
                side="bottom",
            ),
            yaxis=dict(
                title="Franchise LNH" if is_fr else "NHL Franchise",
                type="category",
                categoryorder="array",
                categoryarray=teams_ord,
            ),
            legend=dict(orientation="h", yanchor="top", y=-0.11, xanchor="center", x=0.5),
        )
        style_plotly_high_contrast(fig_hm)
        st.caption(
            "💡 Survolez n'importe quelle case de saison pour afficher les métriques complètes."
            if is_fr
            else "💡 Hover over any season block to inspect detailed underlying metrics."
        )
        st.plotly_chart(fig_hm, use_container_width=True, theme=None)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown(
            "#### Dans quel régime chaque franchise a-t-elle passé le plus d'années ? (2000–2026)"
            if is_fr
            else "#### Which Regime Did Each Franchise Spend the Most Years In? (2000/01–2025/26)"
        )
        st.caption(
            "Ce graphique à barres horizontales cumule les 25 saisons de chaque équipe en parts de temps passé au "
            "**Sommet / Peak (Vert)**, en **Reconstruction active (Orange)**, et en **Inaction / Purgatoire (Gris ardoise)**."
            if is_fr
            else "This horizontal bar chart adds up each team's 25-season history into percentage shares of time spent in "
            "**Peak Contention (Green)**, **Active Rebuild (Orange)**, and **Inaction / Purgatory (Slate Grey)**."
        )
        fig_occ = go.Figure()
        fig_occ.add_trace(
            go.Bar(
                y=df_occ_sorted["franchise_id"],
                x=(df_occ_sorted["share_peak"] * 100).round(1),
                name="Sommet / Peak (% des saisons)" if is_fr else "Peak Contention (% of Seasons)",
                orientation="h",
                marker_color="#15803D",
                text=[f"{v:.0f}%" if v >= 12 else "" for v in (df_occ_sorted["share_peak"] * 100)],
                textposition="inside",
                insidetextfont=dict(color="#FFFFFF", size=11),
            )
        )
        fig_occ.add_trace(
            go.Bar(
                y=df_occ_sorted["franchise_id"],
                x=(df_occ_sorted["share_rebuild"] * 100).round(1),
                name="Reconstruction active (% des saisons)" if is_fr else "Active Rebuild (% of Seasons)",
                orientation="h",
                marker_color="#EA580C",
                text=[f"{v:.0f}%" if v >= 12 else "" for v in (df_occ_sorted["share_rebuild"] * 100)],
                textposition="inside",
                insidetextfont=dict(color="#FFFFFF", size=11),
            )
        )
        fig_occ.add_trace(
            go.Bar(
                y=df_occ_sorted["franchise_id"],
                x=(df_occ_sorted["share_inaction"] * 100).round(1),
                name="Inaction / Purgatoire (% des saisons)" if is_fr else "Inaction / Stalled Purgatory (% of Seasons)",
                orientation="h",
                marker_color="#334155",
                text=[f"{v:.0f}%" if v >= 12 else "" for v in (df_occ_sorted["share_inaction"] * 100)],
                textposition="inside",
                insidetextfont=dict(color="#FFFFFF", size=11),
            )
        )
        fig_occ.update_layout(
            barmode="stack",
            height=840,
            margin=dict(l=65, r=35, t=60, b=105),
            title=dict(
                text="Part historique passée en Sommet vs Reconstruction vs Inaction (%)"
                if is_fr
                else "Share of 25-Season History Spent in Peak vs. Rebuild vs. Inaction (%)",
                font=dict(size=14, color="#0F172A"),
                y=0.98,
                x=0.0,
                xanchor="left",
            ),
            xaxis=dict(
                title="Pourcentage des saisons analysées (2000/01 à 2025/26, %)" if is_fr else "Percentage of Measured NHL Seasons (2000/01 to 2025/26, %)",
                range=[0, 100],
            ),
            yaxis=dict(
                title="Franchise LNH" if is_fr else "NHL Franchise",
                type="category",
                categoryorder="array",
                categoryarray=teams_ord,
            ),
            legend=dict(orientation="h", yanchor="top", y=-0.09, xanchor="center", x=0.5),
        )
        style_plotly_high_contrast(fig_occ)
        st.plotly_chart(fig_occ, use_container_width=True, theme=None)

        st.markdown(
            "#### Tableau récapitulatif des régimes des 32 franchises (2000–2026)"
            if is_fr
            else "#### Full 32-Team Historical Regime Summary Table (2000/01–2025/26)"
        )
        po_by_team = (
            df_ts.groupby("franchise_id")[["playoff_games_won", "playoff_rounds_won"]]
            .sum()
            .reindex(df_occ["franchise_id"])
            .fillna(0)
            .astype(int)
        )
        if is_fr:
            disp_occ = pd.DataFrame(
                {
                    "Franchise": df_occ["franchise_id"].values,
                    "Nom de l'équipe": df_occ["franchise_name"].values,
                    "Saisons analysées": df_occ["seasons_measured"].values,
                    "Régime prédominant": df_occ["predominant_regime"].values,
                    "Sommet / Peak (%)": (df_occ["share_peak"] * 100).round(1).astype(str).values + " %",
                    "Reconstruction (%)": (df_occ["share_rebuild"] * 100).round(1).astype(str).values + " %",
                    "Inaction / Purgatoire (%)": (df_occ["share_inaction"] * 100).round(1).astype(str).values + " %",
                    "Coupes Stanley": df_occ["stanley_cups_won"].values,
                    "Finales de conférence": df_occ["final_fours_reached"].values,
                    "Participations séries": df_occ["playoff_berths"].values,
                    "Séries remportées": po_by_team["playoff_rounds_won"].values,
                    "Victoires de séries": po_by_team["playoff_games_won"].values,
                }
            )
        else:
            disp_occ = pd.DataFrame(
                {
                    "Franchise": df_occ["franchise_id"].values,
                    "Team Name": df_occ["franchise_name"].values,
                    "Seasons Tracked": df_occ["seasons_measured"].values,
                    "Most Common Regime": df_occ["predominant_regime"].values,
                    "Peak Contention (%)": (df_occ["share_peak"] * 100).round(1).astype(str).values + "%",
                    "Active Rebuild (%)": (df_occ["share_rebuild"] * 100).round(1).astype(str).values + "%",
                    "Inaction / Purgatory (%)": (df_occ["share_inaction"] * 100).round(1).astype(str).values + "%",
                    "Stanley Cups Won": df_occ["stanley_cups_won"].values,
                    "Conference Finals": df_occ["final_fours_reached"].values,
                    "Playoff Berths": df_occ["playoff_berths"].values,
                    "Playoff Series Won": po_by_team["playoff_rounds_won"].values,
                    "Playoff Game Wins": po_by_team["playoff_games_won"].values,
                }
            )
        st.dataframe(disp_occ, use_container_width=True, hide_index=True)
        st.dataframe(disp_occ, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 2: Succeeded vs. Failed Rebuild Scorecard & Quadrant (Simple View)
    # =========================================================================
    with t2:
        st.subheader(
            "2. Reconstructions réussies vs échouées : durée et sommet atteint"
            if is_fr
            else "2. Succeeded vs. Failed Rebuilds: Duration & Peak Attainment"
        )
        kpis = manifest.get("kpis", {})
        n_comp_k = kpis.get("completed_episodes", 82)
        n_tot_k = kpis.get("total_episodes", 97)
        r_t1 = 100 * kpis.get("tier1_rate", 0.7073)
        r_t2 = 100 * kpis.get("tier2_rate", 0.5488)
        r_t3 = 100 * kpis.get("tier3_rate", 0.1463)
        r_purg = 100 * kpis.get("purgatory_rate", 0.2927)

        if is_fr:
            st.markdown(
                f"""
                **Guide des 4 paliers d'issue de reconstruction** :
                - 🏆 **Palier 3 — Champion Coupe Stanley (`{r_t3:.1f} %` des cycles complétés)** : Le noyau reconstruit a remporté au moins une Coupe Stanley en 8 ans.
                - 🔵 **Palier 2 — Finaliste de conférence (`{r_t2:.1f} %` des cycles complétés)** : Le noyau reconstruit a atteint le carré d'as de la LNH (finales de conférence) ou affiché 3+ saisons au-dessus d'un % de points de `.600`.
                - 🟠 **Palier 1 — Noyau régulier en séries (`{r_t1:.1f} %` des cycles complétés)** : L'équipe a participé aux séries lors d'au moins 2 saisons sur 3 (ou remporté une ronde avec un % de points `>= .560`) sans rechuter dans la loterie.
                - 🔴 **Échec — Piège du purgatoire (`{r_purg:.1f} %` des cycles complétés)** : L'équipe a passé **5 années consécutives ou plus** dans les bas-fonds du classement sans jamais atteindre le noyau de séries du Palier 1.
                """
            )
        else:
            st.markdown(
                f"""
                **Plain-English Guide to the 4 Rebuild Outcome Tiers**:
                - 🏆 **Tier 3 — Stanley Cup Champion (`{r_t3:.1f}%` of completed rebuilds)**: The rebuilt core won at least one Stanley Cup within 8 years.
                - 🔵 **Tier 2 — Conference Finalist (`{r_t2:.1f}%` of completed rebuilds)**: The rebuilt core reached the NHL Final Four (Conference Finals) or posted 3+ seasons above a `.600` points percentage.
                - 🟠 **Tier 1 — Sustained Playoff Core (`{r_t1:.1f}%` of completed rebuilds)**: The team made the playoffs in at least 2 of 3 seasons (or won a playoff round with `.560+` Points %) without collapsing back into the lottery.
                - 🔴 **Failed — Rebuild Purgatory Trap (`{r_purg:.1f}%` of completed rebuilds)**: The team spent **5 or more consecutive years** at the bottom of the standings without achieving even Tier 1 playoff contention.
                """
            )

        r1c1, r1c2 = st.columns(2)
        if is_fr:
            r1c1.metric(
                "Reconstructions historiques complétées (2000–2026)",
                f"{n_comp_k} cycles",
                f"{n_tot_k} au total ({n_tot_k - n_comp_k} encore en cours)",
            )
            r1c2.metric(
                "Succès Palier 1 : Noyau régulier en séries",
                f"{r_t1:.1f} % ({kpis.get('succeeded_tier1_count', 58)} sur {n_comp_k})",
                "Présence durable en séries",
            )
        else:
            r1c1.metric(
                "Completed Historical Rebuilds (2000–2026)",
                f"{n_comp_k} Rebuilds",
                f"{n_tot_k} total ({n_tot_k - n_comp_k} still in progress)",
            )
            r1c2.metric(
                "Tier 1 Success: Reached Playoff Core",
                f"{r_t1:.1f}% ({kpis.get('succeeded_tier1_count', 58)} of {n_comp_k})",
                "Sustained Playoff Contention",
            )
        r2c1, r2c2 = st.columns(2)
        if is_fr:
            r2c1.metric(
                "Succès Palier 2 : Finaliste de conférence",
                f"{r_t2:.1f} % ({kpis.get('succeeded_tier2_count', 45)} sur {n_comp_k})",
                "Carré d'as de la LNH",
            )
            r2c2.metric(
                "Succès Palier 3 : Champion Coupe Stanley",
                f"{r_t3:.1f} % ({kpis.get('succeeded_tier3_count', 12)} sur {n_comp_k})",
                "Championnat en moins de 8 ans",
            )
        else:
            r2c1.metric(
                "Tier 2 Success: Reached Conference Finals",
                f"{r_t2:.1f}% ({kpis.get('succeeded_tier2_count', 45)} of {n_comp_k})",
                "NHL Final Four Contender",
            )
            r2c2.metric(
                "Tier 3 Success: Won the Stanley Cup",
                f"{r_t3:.1f}% ({kpis.get('succeeded_tier3_count', 12)} of {n_comp_k})",
                "Championship Core Within 8 Years",
            )
        r3c1, r3c2 = st.columns(2)
        if is_fr:
            r3c1.metric(
                "Échec : Piège du purgatoire (5+ ans)",
                f"{r_purg:.1f} % ({kpis.get('purgatory_count', 24)} sur {n_comp_k})",
                "5+ ans sans jamais percer en séries",
                delta_color="inverse",
            )
            r3c2.metric(
                "Fenêtre optimale d'émergence",
                "Années 3 à 4",
                "81,0 % des reconstructions réussies percent d'ici l'an 4",
            )
        else:
            r3c1.metric(
                "Failed: Trapped in Rebuild Purgatory (5+ Years)",
                f"{r_purg:.1f}% ({kpis.get('purgatory_count', 24)} of {n_comp_k})",
                "Stalled 5+ Years Without Playoff Core",
                delta_color="inverse",
            )
            r3c2.metric(
                "Optimal Rebuild Breakout Window",
                "Years 3 to 4",
                "81.0% of Successful Rebuilds Break Out by Year 4",
            )
        st.markdown("<br>", unsafe_allow_html=True)

        fc1, fc2 = st.columns(2)
        with fc1:
            era_options = sorted(df_ep["era_plain"].unique())
            era_filter = st.multiselect(
                "Filtrer par ère LNH :" if is_fr else "Filter by NHL Era:",
                options=era_options,
                default=era_options,
            )
        with fc2:
            arch_options = sorted(df_ep["archetype_plain"].unique())
            arch_filter = st.multiselect(
                "Filtrer par stratégie de reconstruction :" if is_fr else "Filter by Front-Office Rebuild Strategy:",
                options=arch_options,
                default=arch_options,
            )

        sub_ep = df_ep[
            df_ep["era_plain"].isin(era_filter) & df_ep["archetype_plain"].isin(arch_filter)
        ].copy()

        rng = np.random.default_rng(42)
        sub_ep["duration_display"] = sub_ep["duration_years"] + rng.uniform(-0.18, 0.18, size=len(sub_ep))
        sub_ep["pts_lift_pct"] = (sub_ep["contention_pts_lift"] * 100).round(1)
        sub_ep["outcome_tier_display"] = (
            sub_ep["outcome_tier_label"].map(OUTCOME_TIER_FR).fillna(sub_ep["outcome_tier_label"])
            if is_fr
            else sub_ep["outcome_tier_label"]
        )

        # Show non-overlapping labels on key benchmark episodes; full details appear on hover for every dot
        landmark_ids = {
            "EP-PIT-2001",
            "EP-CHI-2002",
            "EP-CHI-2017",
            "EP-MTL-2021",
            "EP-TBL-2007",
            "EP-COL-2010",
            "EP-FLA-2010",
            "EP-BUF-2011",
            "EP-EDM-2006",
            "EP-DET-2016",
            "EP-OTT-2017",
            "EP-TOR-2013",
            "EP-UTA-2012",
        }
        sub_ep["label_clean"] = [
            f"{r.franchise_id} ({r.t_start})" if r.episode_id in landmark_ids else ""
            for r in sub_ep.itertuples()
        ]

        if is_fr:
            color_map_ep = {
                "Palier 3 : Champion Coupe Stanley": "#15803D",
                "Palier 2 : Finaliste de conférence (Final Four)": "#2563EB",
                "Palier 1 : Noyau régulier en séries": "#D97706",
                "Échec : Piège du purgatoire (>=5 ans)": "#DC2626",
                "En cours / Cycle incomplet": "#64748B",
            }
        else:
            color_map_ep = {
                "Tier 3: Stanley Cup Champion": "#15803D",
                "Tier 2: Conference Finalist (Final Four)": "#2563EB",
                "Tier 1: Sustained Playoff Core": "#D97706",
                "Failed: Rebuild Purgatory Trap (>=5 Yrs)": "#DC2626",
                "In-Progress / Incomplete Cycle": "#64748B",
            }

        fig_quad = px.scatter(
            sub_ep,
            x="duration_display",
            y="pts_lift_pct",
            size="edv_invested",
            color="outcome_tier_display",
            color_discrete_map=color_map_ep,
            hover_name="episode_id",
            text="label_clean",
            hover_data={
                "franchise_name": True,
                "t_start": True,
                "t_resolution": True,
                "archetype_plain": True,
                "duration_years": True,
                "duration_display": False,
                "peak_pts_pct": ":.3f",
                "trough_pts_pct": ":.3f",
                "pts_lift_pct": ":.1f",
                "max_playoff_rounds_won": True,
                "edv_invested": ":.0f",
                "label_clean": False,
            },
            labels={
                "duration_display": "Durée (années)" if is_fr else "Rebuild Duration (Years)",
                "pts_lift_pct": "Progression (+Pts %)" if is_fr else "Standings Improvement (+Pts %)",
                "outcome_tier_display": "Issue finale" if is_fr else "Final Outcome",
                "archetype_plain": "Stratégie" if is_fr else "Strategy Type",
                "duration_years": "Durée (années)" if is_fr else "Duration (Years)",
                "t_start": "Année début" if is_fr else "Start Year",
                "t_resolution": "Année fin" if is_fr else "End Year",
                "edv_invested": "Capital repêchage (EDV)" if is_fr else "Draft Capital (EDV)",
            },
            title="Carte des résultats : Durée vs Progression" if is_fr else "Rebuild Duration vs. Standings Jump",
        )
        fig_quad.update_traces(textposition="top center", textfont=dict(color="#0F172A", size=10.5))
        fig_quad.add_vline(
            x=4.5,
            line_dash="dash",
            line_color="#475569",
            annotation_text="Seuil critique de l'an 5" if is_fr else "Year 5 Danger Cliff",
            annotation_position="top right",
            annotation_font_color="#0F172A",
        )
        fig_quad.add_hline(
            y=14.0,
            line_dash="dash",
            line_color="#475569",
            annotation_text="Accès aux séries (+14 Pts %)" if is_fr else "Playoff Breakout Jump (+14 Pts %)",
            annotation_position="bottom right",
            annotation_font_color="#0F172A",
        )
        fig_quad.update_layout(
            height=680,
            margin=dict(l=65, r=40, t=65, b=125),
            title=dict(
                text="Carte des résultats : Durée vs Progression" if is_fr else "Rebuild Duration vs. Standings Jump",
                font=dict(size=14, color="#0F172A"),
                y=0.98,
                x=0.0,
                xanchor="left",
            ),
            xaxis=dict(
                title="Durée de la reconstruction (années)" if is_fr else "Rebuild Duration (Years)",
                tickmode="linear",
                tick0=1,
                dtick=1,
                range=[0.4, 9.3],
            ),
            yaxis=dict(
                title="Progression depuis le creux (+Pts %)" if is_fr else "Standings Improvement (+Percentage Points)",
            ),
            legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5),
        )
        style_plotly_high_contrast(fig_quad)
        st.caption(
            "💡 Chaque point représente un cycle de reconstruction. La taille du cercle correspond au capital de repêchage investi (EDV)."
            if is_fr
            else "💡 Each bubble represents a franchise rebuild episode. Bubble size corresponds to draft capital invested (EDV)."
        )
        st.plotly_chart(fig_quad, use_container_width=True, theme=None)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            "#### Tableau complet des reconstructions et réoutillages historiques (2000–2026)"
            if is_fr
            else "#### Complete Historical Table of NHL Rebuilds & Retools (2000–2025)"
        )
        st.caption(
            "Toutes les colonnes sont rédigées en clair pour comparer les années de début, durées, stratégies et résultats finaux en séries."
            if is_fr
            else "Every column below is written in plain English so you can compare start years, durations, strategies, and playoff results."
        )
        sub_ep_sorted = sub_ep.sort_values(["t_start", "franchise_id"], ascending=[False, True])
        if is_fr:
            disp_ep = pd.DataFrame(
                {
                    "ID Épisode": sub_ep_sorted["episode_id"],
                    "Franchise": sub_ep_sorted["franchise_id"],
                    "Nom de l'équipe": sub_ep_sorted["franchise_name"],
                    "Début": sub_ep_sorted["t_start"],
                    "Fin": sub_ep_sorted["t_resolution"],
                    "Durée (ans)": sub_ep_sorted["duration_years"],
                    "Stratégie DG": sub_ep_sorted["archetype_plain"],
                    "Résultat final": sub_ep_sorted["outcome_tier_display"],
                    "Séries remportées": sub_ep_sorted["max_playoff_rounds_won"],
                    "Victoires matchs séries": sub_ep_sorted.get("max_playoff_games_won", sub_ep_sorted["max_playoff_rounds_won"] * 4),
                    "Noyau repêchage (%)": (sub_ep_sorted["pathway_a_draft_share"] * 100).round(1).astype(str) + " %",
                    "Noyau échanges/UFA (%)": (sub_ep_sorted["pathway_b_trade_share"] * 100).round(1).astype(str) + " %",
                    "Encadrement vétéran (VIR)": sub_ep_sorted["mean_vet_insulation_ratio"].round(2),
                    "Synergie Habileté × Physique": sub_ep_sorted["mean_spsi_style_synergy"].round(2),
                }
            )
        else:
            disp_ep = pd.DataFrame(
                {
                    "Episode ID": sub_ep_sorted["episode_id"],
                    "Franchise": sub_ep_sorted["franchise_id"],
                    "Team Name": sub_ep_sorted["franchise_name"],
                    "Start Year": sub_ep_sorted["t_start"],
                    "End Year": sub_ep_sorted["t_resolution"],
                    "Length (Yrs)": sub_ep_sorted["duration_years"],
                    "Rebuild Strategy": sub_ep_sorted["archetype_plain"],
                    "Final Outcome": sub_ep_sorted["outcome_tier_label"],
                    "Playoff Series Won": sub_ep_sorted["max_playoff_rounds_won"],
                    "Playoff Game Wins (Peak Yr)": sub_ep_sorted.get("max_playoff_games_won", sub_ep_sorted["max_playoff_rounds_won"] * 4),
                    "Draft Core (%)": (sub_ep_sorted["pathway_a_draft_share"] * 100).round(1).astype(str) + "%",
                    "Trade/UFA Core (%)": (sub_ep_sorted["pathway_b_trade_share"] * 100).round(1).astype(str) + "%",
                    "Veteran Insulation (VIR)": sub_ep_sorted["mean_vet_insulation_ratio"].round(2),
                    "Skill × Hit Synergy": sub_ep_sorted["mean_spsi_style_synergy"].round(2),
                }
            )
        st.dataframe(disp_ep, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 3: "Why Did It Succeed or Fail?" Forensic Decomposer (Spacious Vertical Layout)
    # =========================================================================
    with t3:
        st.subheader(
            "3. Comparaison face-à-face : pourquoi un succès ou un piège ?"
            if is_fr
            else "3. Head-to-Head Rebuild Comparison: Why Did One Succeed While Another Stalled?"
        )
        if is_fr:
            st.markdown(
                """
                Choisissez deux reconstructions ci-dessous (par défaut : **Reconstruction hybride de Montréal 2021–2025** vs **Cycle de purgatoire de Chicago 2017–2024**) pour comparer leurs **5 piliers d'architecture d'effectif** :
                1. **Colonne Centres & Défenseurs** : Part du temps de jeu clé occupée par les défenseurs de premier duo et les deux premiers centres (les postes les plus rares).
                2. **Acquisitions échanges d'âge mûr** : Part des minutes clés acquises par échange de joueurs de 21 à 25 ans (ex. Kirby Dach, Alex Newhook, Sam Reinhart) plutôt que d'attendre uniquement des choix de 18 ans.
                3. **Continuité annuelle de l'effectif (`CCM`)** : Maintien du noyau d'une saison à l'autre favorisant la chimie sur la glace.
                4. **Encadrement vétéran de la jeunesse (`VIR`)** : Entourer les jeunes recrues (`0–3 ans d'expérience`) de vétérans dans leur force de l'âge (`4–9 ans d'expérience`) pour leur éviter des duels destructeurs.
                5. **Synergie Habileté × Échec avant physique (`SPSI`)** : Combiner la qualité des tirs à 5c5 (`xGF%`) avec un échec avant physique et agressif en zone offensive pour gagner les batailles de rondelle.
                """
            )
        else:
            st.markdown(
                """
                Pick any two rebuilds below (default: **Montréal's 2021–2025 Hybrid Rebuild** vs. **Chicago's 2017–2024 Purgatory Cycle**) to compare their **5 Roster Building Blocks**:
                1. **Center & Defense Core Spine**: Share of core ice time played by top-pair defensemen and top-two centers (the hardest positions to acquire).
                2. **Prime Trade Arbitrage**: Share of core minutes acquired by trading for 21–25 year-old players (like Kirby Dach, Alex Newhook, Sam Reinhart) instead of waiting only on 18-year-old draft picks.
                3. **Year-Over-Year Core Continuity (`CCM`)**: Keeping the core group together year after year so linemates build chemistry.
                4. **Veteran Insulation (`VIR`)**: Surrounding young rookies (`0–3 yrs experience`) with prime-age veterans (`4–9 yrs experience`) so rookies aren't crushed by tough matchups.
                5. **Skill × Forecheck Hit Synergy (`SPSI`)**: Combining 5-on-5 Expected Goals (`xGF%`) with aggressive offensive-zone forecheck hitting to win puck battles.
                """
            )

        outcome_col = df_ep["outcome_tier_label"].map(OUTCOME_TIER_FR).fillna(df_ep["outcome_tier_label"]) if is_fr else df_ep["outcome_tier_label"]
        ep_options = (
            df_ep["episode_id"]
            + " ("
            + df_ep["t_start"].astype(str)
            + "–"
            + df_ep["t_resolution"].astype(str)
            + ") — "
            + df_ep["franchise_name"]
            + " ["
            + outcome_col
            + "]"
        )
        ep_list = ep_options.tolist()
        default_a = next((i for i, s in enumerate(ep_list) if "EP-MTL-2021" in s or "EP-COL-2010" in s), 0)
        default_b = next((i for i, s in enumerate(ep_list) if "EP-CHI-2017" in s or "EP-BUF-2011" in s), min(1, len(ep_list) - 1))

        sc1, sc2 = st.columns(2)
        with sc1:
            sel_a = st.selectbox(
                "Sélectionner l'épisode A (Vert) :" if is_fr else "Select Rebuild Episode A (Green):",
                ep_list,
                index=default_a,
            )
        with sc2:
            sel_b = st.selectbox(
                "Sélectionner l'épisode B (Rouge) :" if is_fr else "Select Rebuild Episode B (Red):",
                ep_list,
                index=default_b,
            )

        row_a = df_ep.iloc[ep_list.index(sel_a)]
        row_b = df_ep.iloc[ep_list.index(sel_b)]

        def compute_radar_scores(r):
            s_cd = min(100.0, max(15.0, float(r["center_def_core_share"]) * 125.0))
            s_tr = min(100.0, max(15.0, float(r["pathway_b_trade_share"]) * 135.0))
            s_ccm = min(100.0, max(15.0, float(r["mean_ccm"]) * 110.0))
            s_vir = min(
                100.0,
                max(
                    15.0,
                    float(r["mean_vet_insulation_ratio"]) * 45.0
                    - float(r["mean_youth_overload_excess"]) * 600.0,
                ),
            )
            s_sty = min(100.0, max(15.0, float(r["mean_spsi_style_synergy"]) * 140.0))
            return [round(s_cd, 1), round(s_tr, 1), round(s_ccm, 1), round(s_vir, 1), round(s_sty, 1)]

        cats = (
            [
                "Colonne Centres & Défenseurs",
                "Acquisitions échanges d'âge mûr",
                "Continuité du noyau clé",
                "Encadrement vétéran de la jeunesse",
                "Synergie Habileté × Échec avant",
            ]
            if is_fr
            else [
                "Center & Defense Core Spine",
                "Prime Trade Acquisitions",
                "Year-to-Year Core Continuity",
                "Veteran Insulation of Youth",
                "Skill × Forecheck Hit Synergy",
            ]
        )
        vals_a = compute_radar_scores(row_a)
        vals_b = compute_radar_scores(row_b)

        tier_a = OUTCOME_TIER_FR.get(row_a['outcome_tier_label'], row_a['outcome_tier_label']) if is_fr else row_a['outcome_tier_label']
        tier_b = OUTCOME_TIER_FR.get(row_b['outcome_tier_label'], row_b['outcome_tier_label']) if is_fr else row_b['outcome_tier_label']

        # Full-Width Chart 1: 5-Pillar Radar Profile (High-Contrast Spider Layers)
        fig_rad = go.Figure()
        fig_rad.add_trace(
            go.Scatterpolar(
                r=vals_a + [vals_a[0]],
                theta=cats + [cats[0]],
                mode="lines+markers",
                fill="toself",
                fillcolor="rgba(21, 128, 61, 0.22)",
                name=f"{row_a['franchise_name']} ({row_a['t_start']}–{row_a['t_resolution']}): {tier_a}",
                line=dict(color="#15803D", width=3.2),
                marker=dict(size=8, color="#15803D"),
            )
        )
        fig_rad.add_trace(
            go.Scatterpolar(
                r=vals_b + [vals_b[0]],
                theta=cats + [cats[0]],
                mode="lines+markers",
                fill="toself",
                fillcolor="rgba(220, 38, 38, 0.20)",
                name=f"{row_b['franchise_name']} ({row_b['t_start']}–{row_b['t_resolution']}): {tier_b}",
                line=dict(color="#DC2626", width=3.2),
                marker=dict(size=8, color="#DC2626"),
            )
        )
        style_plotly_high_contrast(fig_rad)
        fig_rad.update_layout(
            polar=dict(
                bgcolor="#F8FAFC",
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickvals=[20, 40, 60, 80, 100],
                    tickfont=dict(color="#0F172A", size=11.5),
                    gridcolor="#94A3B8",
                    linecolor="#475569",
                    layer="above traces",
                ),
                angularaxis=dict(
                    tickfont=dict(color="#0F172A", size=12.5),
                    gridcolor="#94A3B8",
                    linecolor="#475569",
                ),
            ),
            title=dict(
                text="Profil d'architecture en 5 piliers (Score de 0 à 100)"
                if is_fr
                else "5-Pillar Rebuild Architecture Profile (0 to 100 Scale)",
                font=dict(color="#0F172A", size=14),
                y=0.97,
                x=0.0,
                xanchor="left",
            ),
            height=600,
            margin=dict(l=110, r=110, t=75, b=115),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.12,
                xanchor="center",
                x=0.5,
                bgcolor="#F8FAFC",
                bordercolor="#CBD5E1",
                borderwidth=1,
                font=dict(color="#0F172A", size=11.5),
            ),
        )
        st.plotly_chart(fig_rad, use_container_width=True, theme=None)
        st.markdown("<br>", unsafe_allow_html=True)

        # Full-Width Chart 2: Horizontal Bar Attribution (Spacious Full Width)
        factors = (
            [
                "1. Colonne Centres & Défenseurs (vs moy. 55%)",
                "2. Part du noyau acquise par échange (vs moy. 35%)",
                "3. Encadrement vétéran des jeunes (vs moy. 0.85)",
                "4. Prévention surcharge jeunesse (Temps recrues <= 30%)",
                "5. Synergie habileté × échec avant offensif (vs moy. 0.35)",
            ]
            if is_fr
            else [
                "1. Center & Defenseman Core Spine (vs. 55% Avg)",
                "2. External Prime Trade Core Share (vs. 35% Avg)",
                "3. Veteran Insulation of Young Skaters (vs. 0.85 Avg)",
                "4. Avoiding Youth Overload Penalty (Keeping ELC Youth <= 30%)",
                "5. Offensive Skill × Forecheck Hit Synergy (vs. 0.35 Avg)",
            ]
        )
        contrib_a = [
            round((row_a["center_def_core_share"] - 0.55) * 28.0, 2),
            round((row_a["pathway_b_trade_share"] - 0.35) * 24.0, 2),
            round((row_a["mean_vet_insulation_ratio"] - 0.85) * 8.5, 2),
            round(-row_a["mean_youth_overload_excess"] * 180.0, 2),
            round((row_a["mean_spsi_style_synergy"] - 0.35) * 26.0, 2),
        ]
        contrib_b = [
            round((row_b["center_def_core_share"] - 0.55) * 28.0, 2),
            round((row_b["pathway_b_trade_share"] - 0.35) * 24.0, 2),
            round((row_b["mean_vet_insulation_ratio"] - 0.85) * 8.5, 2),
            round(-row_b["mean_youth_overload_excess"] * 180.0, 2),
            round((row_b["mean_spsi_style_synergy"] - 0.35) * 26.0, 2),
        ]
        fig_bar = go.Figure()
        fig_bar.add_trace(
            go.Bar(
                y=factors,
                x=contrib_a,
                orientation="h",
                name=f"{row_a['franchise_name']} ({row_a['t_start']}–{row_a['t_resolution']})",
                marker_color="#15803D",
                text=[f"{v:+.1f} pts%" for v in contrib_a],
                textposition="outside",
                outsidetextfont=dict(color="#0F172A", size=11),
            )
        )
        fig_bar.add_trace(
            go.Bar(
                y=factors,
                x=contrib_b,
                orientation="h",
                name=f"{row_b['franchise_name']} ({row_b['t_start']}–{row_b['t_resolution']})",
                marker_color="#DC2626",
                text=[f"{v:+.1f} pts%" for v in contrib_b],
                textposition="outside",
                outsidetextfont=dict(color="#0F172A", size=11),
            )
        )
        fig_bar.update_layout(
            barmode="group",
            height=540,
            title=dict(
                text="Attribution des facteurs de succès en séries (Impact +/- Pts %)"
                if is_fr
                else "Breakout Odds Factor Attribution (Percentage Points vs. Baseline)",
                font=dict(size=14, color="#0F172A"),
                y=0.97,
                x=0.0,
                xanchor="left",
            ),
            xaxis=dict(
                title="Impact sur les chances d'accès aux séries (+/- Pts %)"
                if is_fr
                else "Impact on Playoff Breakout Probability (Percentage Points vs. Baseline)",
                range=[-18, 22],
            ),
            yaxis=dict(type="category"),
            margin=dict(l=55, r=65, t=65, b=115),
            legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="center", x=0.5),
        )
        style_plotly_high_contrast(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True, theme=None)
        st.markdown("<br>", unsafe_allow_html=True)

        # Side-by-Side Plain-English Summary Table of the Two Selected Episodes
        st.markdown(
            "#### Données mesurées côte à côte pour les deux reconstructions sélectionnées"
            if is_fr
            else "#### Side-by-Side Measured Numbers for Your Two Selected Rebuilds"
        )
        if is_fr:
            comp_metric_names = [
                "Franchise et fenêtre de reconstruction",
                "Stratégie de direction générale",
                "Résultat historique final",
                "Nombre total d'années en reconstruction",
                "Creux au classement (% de points)",
                "Sommet atteint après la reconstruction (% de points)",
                "Séries éliminatoires remportées",
                "Matchs de séries gagnés au sommet",
                "Part du noyau issue du repêchage (%)",
                "Part du noyau issue d'échanges / arbitrage salarial (%)",
                "Part du temps de jeu clé (Centres + Défenseurs, %)",
                "Ratio d'encadrement vétéran (Vétérans d'âge mûr ÷ Recrues)",
                "Continuité annuelle du noyau (%)",
                "Score de synergie Habileté × Échec avant (Échelle 0–100)",
            ]
        else:
            comp_metric_names = [
                "Franchise & Rebuild Window",
                "Front-Office Strategy Type",
                "Final Historical Outcome",
                "Total Years Spent in Rebuild",
                "Lowest Standings Point (Rebuild Bottom Points %)",
                "Peak Standings Point After Rebuild (Points %)",
                "Playoff Series Won by Rebuilt Core",
                "Playoff Game Wins in Peak Run",
                "Share of Core Built via Draft (Homegrown %)",
                "Share of Core Built via Trades / Cap Arbitrage (%)",
                "Center & Defenseman Share of Core Ice Time (%)",
                "Veteran Insulation Ratio (Prime Vets ÷ ELC Youth)",
                "Year-Over-Year Core Continuity (%)",
                "Skill × Forecheck Hit Synergy Score (0–100 Scale)",
            ]
        comp_df = pd.DataFrame(
            {
                "Métrique" if is_fr else "Metric (Plain-English Meaning)": comp_metric_names,
                f"{row_a['franchise_id']} ({row_a['t_start']}–{row_a['t_resolution']})": [
                    f"{row_a['franchise_name']} ({row_a['t_start']}–{row_a['t_resolution']})",
                    row_a["archetype_plain"],
                    tier_a,
                    f"{int(row_a['duration_years'])} ans" if is_fr else f"{int(row_a['duration_years'])} Years",
                    f"{row_a['trough_pts_pct']:.3f}",
                    f"{row_a['peak_pts_pct']:.3f}",
                    f"{int(row_a['max_playoff_rounds_won'])} séries" if is_fr else f"{int(row_a['max_playoff_rounds_won'])} Series Won",
                    f"{int(row_a.get('max_playoff_games_won', 0))} victoires" if is_fr else f"{int(row_a.get('max_playoff_games_won', 0))} Game Wins",
                    f"{100*row_a['pathway_a_draft_share']:.1f}%",
                    f"{100*row_a['pathway_b_trade_share']:.1f}%",
                    f"{100*row_a['center_def_core_share']:.1f}%",
                    f"{row_a['mean_vet_insulation_ratio']:.2f}",
                    f"{100*row_a['mean_ccm']:.1f}%",
                    f"{100*row_a['mean_spsi_style_synergy']:.1f}",
                ],
                f"{row_b['franchise_id']} ({row_b['t_start']}–{row_b['t_resolution']})": [
                    f"{row_b['franchise_name']} ({row_b['t_start']}–{row_b['t_resolution']})",
                    row_b["archetype_plain"],
                    tier_b,
                    f"{int(row_b['duration_years'])} ans" if is_fr else f"{int(row_b['duration_years'])} Years",
                    f"{row_b['trough_pts_pct']:.3f}",
                    f"{row_b['peak_pts_pct']:.3f}",
                    f"{int(row_b['max_playoff_rounds_won'])} séries" if is_fr else f"{int(row_b['max_playoff_rounds_won'])} Series Won",
                    f"{int(row_b.get('max_playoff_games_won', 0))} victoires" if is_fr else f"{int(row_b.get('max_playoff_games_won', 0))} Game Wins",
                    f"{100*row_b['pathway_a_draft_share']:.1f}%",
                    f"{100*row_b['pathway_b_trade_share']:.1f}%",
                    f"{100*row_b['center_def_core_share']:.1f}%",
                    f"{row_b['mean_vet_insulation_ratio']:.2f}",
                    f"{100*row_b['mean_ccm']:.1f}%",
                    f"{100*row_b['mean_spsi_style_synergy']:.1f}",
                ],
            }
        )
        st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 4: Franchise Longitudinal Retrospective Dossier (32 Teams)
    # =========================================================================
    with t4:
        st.subheader(
            "4. Dossier rétrospectif sur 25 ans par franchise (2000–2026)"
            if is_fr
            else "4. Single-Franchise 25-Year Deep Dive (2000/01 through 2025/26)"
        )
        if is_fr:
            st.markdown(
                """
                Sélectionnez l'une des 32 franchises de la LNH ci-dessous pour analyser son **historique complet sur 25 saisons (`2000/01` à `2025/26`)** sur trois pistes synchronisées :
                - **Piste 1 (Haut)** : % de points en saison régulière comparé au repère de prétendant aux séries (`.550`).
                - **Piste 2 (Milieu)** : Classification du régime chaque saison (**Vert = Peak**, **Orange = Rebuild**, **Gris = Inaction**) avec son sous-état (`R0` à `R5`).
                - **Piste 3 (Bas)** : Répartition du temps de jeu de l'équipe entre les **4 cohortes d'expérience LNH** (`B1 : 0–3 ans recrues`, `B2 : 4–6 ans passerelle`, `B3 : 7–9 ans sommet`, et `B4 : 10+ ans vétérans`).
                """
            )
        else:
            st.markdown(
                """
                Select any of the 32 NHL franchises below to inspect its complete **25-year history (`2000/01` to `2025/26`)** across three synchronized tracks:
                - **Track 1 (Top)**: Regular-season Points Percentage (`Pts %`) relative to the `.550` playoff contender benchmark.
                - **Track 2 (Middle)**: Which regime the team was in each season (**Green = Peak**, **Orange = Rebuild**, **Grey = Inaction**) labeled with its sub-state (`R0` to `R5`).
                - **Track 3 (Bottom)**: How the team's total ice time was divided across the **4 NHL Experience Clusters** (`0–3 yrs Entry-Level Youth`, `4–6 yrs Prime Bridge`, `7–9 yrs Peak Core`, and `10+ yrs Veterans`).
                """
            )
        team_choices = [
            f"{r['franchise_id']} — {r['franchise_name']} ({'Régime prédominant :' if is_fr else 'Most Common Regime:'} {r['predominant_regime']})"
            for _, r in df_occ.iterrows()
        ]
        default_team_idx = next((i for i, c in enumerate(team_choices) if c.startswith("MTL —")), 0)
        sel_team_str = st.selectbox(
            "Choisir une franchise LNH à inspecter :" if is_fr else "Choose an NHL Franchise to Inspect:",
            team_choices,
            index=default_team_idx,
        )
        sel_fid = sel_team_str.split(" — ")[0].strip()

        occ_t = df_occ[df_occ["franchise_id"] == sel_fid].iloc[0]
        ts_t = df_ts[df_ts["franchise_id"] == sel_fid].sort_values("season").copy()

        tm1, tm2 = st.columns(2)
        if is_fr:
            tm1.metric(
                "Régime historique prédominant",
                occ_t["predominant_regime"],
                f"{int(occ_t['seasons_measured'])} saisons complétées suivies (2000/01–2025/26)",
            )
            tm2.metric(
                "Temps passé au sommet (Peak)",
                f"{100*occ_t['share_peak']:.1f} % ({int(occ_t['peak_seasons'])} saisons)",
                "Équilibre de prétendant aux séries",
            )
        else:
            tm1.metric(
                "Most Common Historical Regime",
                occ_t["predominant_regime"],
                f"{int(occ_t['seasons_measured'])} completed seasons tracked (2000/01–2025/26)",
            )
            tm2.metric(
                "Time Spent in Peak Contention",
                f"{100*occ_t['share_peak']:.1f}% ({int(occ_t['peak_seasons'])} seasons)",
                "Playoff Contender Equilibrium",
            )
        tm3, tm4 = st.columns(2)
        if is_fr:
            tm3.metric(
                "Temps passé en reconstruction active",
                f"{100*occ_t['share_rebuild']:.1f} % ({int(occ_t['rebuild_seasons'])} saisons)",
                "Démantèlement et incubation jeunesse",
            )
            tm4.metric(
                "Temps passé en inaction / purgatoire",
                f"{100*occ_t['share_inaction']:.1f} % ({int(occ_t['inaction_seasons'])} saisons)",
                "Affaissement passif ou purgatoire 5+ ans",
            )
        else:
            tm3.metric(
                "Time Spent in Active Rebuild",
                f"{100*occ_t['share_rebuild']:.1f}% ({int(occ_t['rebuild_seasons'])} seasons)",
                "Teardown & Youth Incubation",
            )
            tm4.metric(
                "Time Spent in Inaction / Purgatory",
                f"{100*occ_t['share_inaction']:.1f}% ({int(occ_t['inaction_seasons'])} seasons)",
                "Unchanged Slump or Stalled 5+ Yr Rebuild",
            )
        tm5, tm6 = st.columns(2)
        if is_fr:
            tm5.metric(
                "Coupes Stanley remportées (2000–2026)",
                f"{int(occ_t['stanley_cups_won'])} Coupes 🏆",
                f"{int(occ_t['final_fours_reached'])} Finales de conférence ⭐",
            )
            tm6.metric(
                "Participations et victoires en séries",
                f"{int(occ_t['playoff_berths'])} participations aux séries",
                f"{int(ts_t['playoff_games_won'].sum())} matchs gagnés ({int(ts_t['playoff_rounds_won'].sum())} séries)",
            )
        else:
            tm5.metric(
                "Stanley Cups Won (2000/01–2025/26)",
                f"{int(occ_t['stanley_cups_won'])} Cups 🏆",
                f"{int(occ_t['final_fours_reached'])} Conference Finals Reached ⭐",
            )
            tm6.metric(
                "Total Playoff Berths & Postseason Wins",
                f"{int(occ_t['playoff_berths'])} Playoff Berths",
                f"{int(ts_t['playoff_games_won'].sum())} Playoff Game Wins ({int(ts_t['playoff_rounds_won'].sum())} Series Won)",
            )
        st.markdown("<br>", unsafe_allow_html=True)

        s_lbls = [format_season_label(s) for s in ts_t["season"]]

        # Spacious Synchronized 3-Track Franchise Timeline
        fig_tr = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.10,
            row_heights=[0.42, 0.16, 0.42],
            subplot_titles=(
                f"Piste 1 : {occ_t['franchise_name']} ({sel_fid}) — % de points"
                if is_fr
                else f"Track 1: {occ_t['franchise_name']} ({sel_fid}) — Regular-Season Points %",
                "Piste 2 : Régime par saison (Vert = Peak · Orange = Rebuild · Gris = Inaction)"
                if is_fr
                else "Track 2: Season-by-Season Regime Classification (Green = Peak · Orange = Rebuild · Slate = Inaction)",
                "Piste 3 : Part du temps de jeu par cohorte d'expérience (B1 à B4)"
                if is_fr
                else "Track 3: Share of Team Ice Time by Player NHL Experience (0–3 Yrs Youth vs. 4–6 Yrs Bridge vs. 7+ Yrs Veterans)",
            ),
        )

        # Track 1: Points % + Contention Threshold
        fig_tr.add_trace(
            go.Scatter(
                x=s_lbls,
                y=ts_t["pts_pct"],
                mode="lines+markers",
                name="% de points" if is_fr else "Regular-Season Points %",
                line=dict(color="#1D4ED8", width=3.4),
                marker=dict(size=8.5),
                text=[
                    (
                        f"<b>{s}</b> : % de points = {p:.3f} | Victoires matchs : {int(gw)} | Séries gagnées : {int(w)} | État : {st_name}"
                        if is_fr
                        else f"<b>{s}</b>: Points % = {p:.3f} | Playoff Game Wins: {int(gw)} | Playoff Series Won: {int(w)} | State: {st_name}"
                    )
                    for s, p, gw, w, st_name in zip(
                        s_lbls,
                        ts_t["pts_pct"],
                        ts_t["playoff_games_won"],
                        ts_t["playoff_rounds_won"],
                        ts_t["substate_clean"],
                    )
                ],
                hoverinfo="text",
            ),
            row=1,
            col=1,
        )
        fig_tr.add_hline(
            y=0.550,
            line_dash="dash",
            line_color="#15803D",
            annotation_text="Repère prétendant aux séries (.550)" if is_fr else "Playoff Contender Benchmark (.550 Points %)",
            annotation_position="top left",
            row=1,
            col=1,
        )

        # Track 2: Regime Strip Bar with Sub-State Code
        reg_colors = [
            {"Peak": "#15803D", "Rebuild": "#EA580C", "Inaction": "#334155"}.get(m, "#64748B")
            for m in ts_t["macro_regime"]
        ]
        badge_texts = [STATE_BADGE.get(st_code, st_code[:2]) for st_code in ts_t["rebuild_regime"]]
        hover_strip = [
            (
                f"<b>{s}</b> : Régime {m} — {state_display_local.get(st_code, st_code)} (% de points : {p:.3f})"
                if is_fr
                else f"<b>{s}</b>: {m} Regime — {STATE_DISPLAY.get(st_code, st_code)} (Points %: {p:.3f})"
            )
            for s, m, st_code, p in zip(
                s_lbls, ts_t["macro_regime"], ts_t["rebuild_regime"], ts_t["pts_pct"]
            )
        ]
        fig_tr.add_trace(
            go.Bar(
                x=s_lbls,
                y=[1] * len(s_lbls),
                marker_color=reg_colors,
                text=badge_texts,
                textposition="inside",
                insidetextanchor="middle",
                textfont=dict(color="#FFFFFF", size=10.5),
                hovertext=hover_strip,
                hoverinfo="text",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig_tr.update_yaxes(visible=False, range=[0, 1], row=2, col=1)

        # Track 3: Experience Tenure Streamgraph
        fig_tr.add_trace(
            go.Scatter(
                x=s_lbls,
                y=(ts_t["s_b1_elc_0_3"] * 100).round(1),
                stackgroup="one",
                name="Cohorte B1 : 0–3 ans (Jeunes recrues)" if is_fr else "Cluster B1: 0–3 Yrs Experience (Entry-Level Youth)",
                line=dict(color="#DC2626"),
            ),
            row=3,
            col=1,
        )
        fig_tr.add_trace(
            go.Scatter(
                x=s_lbls,
                y=(ts_t["s_b2_prime_4_6"] * 100).round(1),
                stackgroup="one",
                name="Cohorte B2 : 4–6 ans (Noyau passerelle)" if is_fr else "Cluster B2: 4–6 Yrs Experience (Prime Bridge Core)",
                line=dict(color="#EA580C"),
            ),
            row=3,
            col=1,
        )
        fig_tr.add_trace(
            go.Scatter(
                x=s_lbls,
                y=(ts_t["s_b3_peak_7_9"] * 100).round(1),
                stackgroup="one",
                name="Cohorte B3 : 7–9 ans (Vétérans sommet)" if is_fr else "Cluster B3: 7–9 Yrs Experience (Peak Core Veterans)",
                line=dict(color="#16A34A"),
            ),
            row=3,
            col=1,
        )
        fig_tr.add_trace(
            go.Scatter(
                x=s_lbls,
                y=(ts_t["s_b4_vet_10_plus"] * 100).round(1),
                stackgroup="one",
                name="Cohorte B4 : 10+ ans (Vétérans aguerris)" if is_fr else "Cluster B4: 10+ Yrs Experience (Greybeard Veterans)",
                line=dict(color="#2563EB"),
            ),
            row=3,
            col=1,
        )

        for r_idx in (1, 2, 3):
            fig_tr.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=s_lbls,
                tickmode="array",
                tickvals=s_lbls,
                ticktext=s_lbls,
                tickangle=-35,
                showticklabels=(r_idx == 3),
                row=r_idx,
                col=1,
            )

        fig_tr.update_yaxes(title_text="% de points" if is_fr else "Points %", row=1, col=1)
        fig_tr.update_yaxes(title_text="Part du temps de jeu (%)" if is_fr else "Share of Ice Time (%)", range=[0, 100], row=3, col=1)
        fig_tr.update_layout(
            height=880,
            margin=dict(l=65, r=35, t=65, b=120),
            legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5),
        )
        style_plotly_high_contrast(fig_tr)
        st.plotly_chart(fig_tr, use_container_width=True, theme=None)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            f"#### Tableau saison par saison pour {occ_t['franchise_name']} (`{sel_fid}`)"
            if is_fr
            else f"#### Season-by-Season Table for {occ_t['franchise_name']} (`{sel_fid}`)"
        )
        if is_fr:
            disp_ts_team = pd.DataFrame(
                {
                    "Saison": s_lbls,
                    "Code actif": ts_t["raw_team_abbrev"],
                    "Macro-régime": ts_t["macro_regime"],
                    "Sous-état (R0–R5)": ts_t["substate_clean"],
                    "% de points": ts_t["pts_pct"].round(3),
                    "Accès séries": np.where(ts_t["playoff_berth"], "Oui", "Non"),
                    "Victoires matchs": ts_t["playoff_games_won"].astype(int),
                    "Séries remportées": ts_t["playoff_rounds_won"].astype(int),
                    "Buts attendus 5c5 (xGF%)": (ts_t["xg_share_5v5"] * 100).round(1).astype(str) + " %",
                    "Temps recrues (0–3 ans)": (ts_t["s_b1_elc_0_3"] * 100).round(1).astype(str) + " %",
                    "Temps passerelle (4–6 ans)": (ts_t["s_b2_prime_4_6"] * 100).round(1).astype(str) + " %",
                    "Temps vétérans (7+ ans)": ((ts_t["s_b3_peak_7_9"] + ts_t["s_b4_vet_10_plus"]) * 100).round(1).astype(str) + " %",
                    "Échec avant zone offensive (%)": ts_t["ozone_hit_pct"].round(1).astype(str) + " %",
                }
            )
        else:
            disp_ts_team = pd.DataFrame(
                {
                    "Season": s_lbls,
                    "Active Code": ts_t["raw_team_abbrev"],
                    "Macro Regime": ts_t["macro_regime"],
                    "Specific State (R0–R5)": ts_t["substate_clean"],
                    "Points %": ts_t["pts_pct"].round(3),
                    "Playoff Berth": np.where(ts_t["playoff_berth"], "Yes", "No"),
                    "Playoff Game Wins": ts_t["playoff_games_won"].astype(int),
                    "Playoff Series Won": ts_t["playoff_rounds_won"].astype(int),
                    "5v5 Expected Goals (xGF%)": (ts_t["xg_share_5v5"] * 100).round(1).astype(str) + "%",
                    "Entry-Level Youth TOI (0–3 Yrs)": (ts_t["s_b1_elc_0_3"] * 100).round(1).astype(str) + "%",
                    "Prime Bridge TOI (4–6 Yrs)": (ts_t["s_b2_prime_4_6"] * 100).round(1).astype(str) + "%",
                    "Peak & Vet TOI (7+ Yrs)": ((ts_t["s_b3_peak_7_9"] + ts_t["s_b4_vet_10_plus"]) * 100).round(1).astype(str) + "%",
                    "O-Zone Forecheck Hits (%)": ts_t["ozone_hit_pct"].round(1).astype(str) + "%",
                }
            )
        st.dataframe(disp_ts_team, use_container_width=True, hide_index=True)

        dossier_file = INSIGHTS_DIR / "franchises" / f"{sel_fid}.md"
        if dossier_file.exists():
            exp_label = (
                f"📄 Lire le rapport narratif historique pour {occ_t['franchise_name']} (`{sel_fid}.md`)"
                if is_fr
                else f"📄 Read Written Historical Narrative Dossier for {occ_t['franchise_name']} (`{sel_fid}.md`)"
            )
            with st.expander(exp_label, expanded=False):
                st.markdown(dossier_file.read_text(encoding="utf-8"))

    # =========================================================================
    # TAB 5: Era Trends, Age/Experience Complementarity & Hit Synergy (Spacious Stacked)
    # =========================================================================
    with t5:
        st.subheader(
            "5. Facteurs clés : Le seuil de surcharge jeunesse à 30 % et la synergie physique"
            if is_fr
            else "5. Key Drivers: The 30% Youth Overload Cliff & Forecheck Hit Synergy"
        )
        if is_fr:
            st.markdown(
                """
                Les deux graphiques ci-dessous mettent en lumière deux des découvertes les plus déterminantes de notre étude sur les **764 saisons-équipes de la LNH (2000/01 à 2025/26)** :
                1. **Graphique 5A — Le seuil critique de surcharge jeunesse à 30 %** : Allouer aux jeunes recrues (`0–3 ans d'expérience`) jusqu'à **25 %–30 % du temps de jeu** est vertueux lorsqu'elles sont encadrées par des vétérans d'âge mûr (`Points verts`). Dès que les recrues dépassent **30 % du temps de jeu** sans encadrement vétéran suffisant (`Points rouges`), la structure défensive s'effondre et les équipes s'enfoncent dans le purgatoire de la loterie.
                2. **Graphique 5B — Habileté offensive (`xGF%`) × Échec avant physique** : L'habileté pure ne suffit pas au printemps, et frapper sans possession signifie simplement courir après la rondelle. Les équipes du **quadrant supérieur droit** — combinant **50 %+ de buts attendus** avec **un échec avant offensif très agressif** — dominent les séries éliminatoires (`cercles verts plus volumineux`).
                """
            )
        else:
            st.markdown(
                """
                Rather than cramming charts side-by-side, the two full-width charts below illustrate two of our study's most important findings across all **764 NHL team-seasons (2000/01–2025/26)**:
                1. **Chart 5A — The 30% Entry-Level Youth Overload Cliff**: Giving young players (`0–3 years NHL experience`) up to **25%–30% of team ice time** is healthy when they are sheltered by prime veterans (`Green dots`). Once rookies exceed **30% of team ice time** without veteran insulation (`Red dots`), defensive structure collapses and teams sink into the lottery.
                2. **Chart 5B — Offensive Skill (`xGF%`) × Forecheck Physicality**: Skill alone doesn't win in the spring, and hitting alone just means you never have the puck. Teams in the **top-right quadrant**—combining **50%+ Expected Goals Share** with **high offensive-zone forecheck hitting**—dominate playoff series (`bigger green circles`).
                """
            )
        df_ts_plot = df_ts.copy()
        df_ts_plot["elc_toi_pct"] = (df_ts_plot["s_b1_elc_0_3"] * 100).round(1)
        df_ts_plot["xgf_pct"] = (df_ts_plot["xg_share_5v5"] * 100).round(1)
        df_ts_plot["pfhs_pct"] = (df_ts_plot["proactive_forecheck_share"] * 100).round(1)
        df_ts_plot["bubble_sz"] = np.maximum(1, df_ts_plot["playoff_rounds_won"] + 1) * 4

        grp_insulated = "Jeunesse bien encadrée (Vétérans >= 1,25x Recrues)" if is_fr else "Well-Insulated Youth (Prime Vets >= 1.25x ELC Youth Ice Time)"
        grp_uninsulated = "Jeunesse non encadrée (Vétérans < 1,25x Recrues)" if is_fr else "Uninsulated Youth (Prime Vets < 1.25x ELC Youth Ice Time)"
        df_ts_plot["Veteran Insulation Group"] = np.where(
            df_ts_plot["vet_insulation_ratio"] >= 1.25,
            grp_insulated,
            grp_uninsulated,
        )

        color_map_age = {
            grp_insulated: "#15803D",
            grp_uninsulated: "#DC2626",
        }

        # Full-Width Chart 5A: Age/Experience Complementarity
        fig_age = px.scatter(
            df_ts_plot,
            x="elc_toi_pct",
            y="pts_pct",
            color="Veteran Insulation Group",
            color_discrete_map=color_map_age,
            hover_name="franchise_id",
            hover_data={
                "season_label": True,
                "macro_regime": True,
                "substate_clean": True,
                "elc_toi_pct": ":.1f",
                "vet_insulation_ratio": ":.2f",
                "pts_pct": ":.3f",
            },
            labels={
                "elc_toi_pct": "Temps recrues 0–3 ans (%)" if is_fr else "Share of Ice Time (0–3 Yrs, %)",
                "pts_pct": "% de points" if is_fr else "Points %",
                "season_label": "Saison" if is_fr else "Season",
                "substate_clean": "État du régime" if is_fr else "Regime State",
                "vet_insulation_ratio": "Ratio VIR" if is_fr else "VIR Ratio",
            },
            title="Le seuil critique de surcharge jeunesse à 30 %" if is_fr else "The 30% Youth Overload Cliff",
        )
        fig_age.add_vline(
            x=30.0,
            line_dash="dash",
            line_color="#334155",
            annotation_text="Seuil critique jeunesse à 30 %" if is_fr else "30% Youth Overload Cliff",
            annotation_position="top right",
        )
        fig_age.add_hline(
            y=0.550,
            line_dash="dot",
            line_color="#15803D",
            annotation_text="Seuil prétendant aux séries (.550)" if is_fr else "Playoff Contender Benchmark (.550 Points %)",
            annotation_position="bottom left",
        )
        fig_age.update_layout(
            height=580,
            margin=dict(l=65, r=40, t=65, b=115),
            title=dict(
                text="Le seuil critique de surcharge jeunesse à 30 %" if is_fr else "The 30% Youth Overload Cliff",
                font=dict(size=14, color="#0F172A"),
                y=0.98,
                x=0.0,
                xanchor="left",
            ),
            xaxis=dict(
                title="Part du temps de jeu total jouée par les recrues (0–3 ans, %)"
                if is_fr
                else "Share of Total Team Ice Time Played by Entry-Level Youth (0–3 Years, %)",
            ),
            yaxis=dict(
                title="% de points en saison régulière" if is_fr else "Regular-Season Points Percentage (Pts %)",
            ),
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        )
        style_plotly_high_contrast(fig_age)
        st.caption(
            "💡 Les points verts représentent les équipes dont les recrues sont encadrées par des vétérans (VIR >= 1,25)."
            if is_fr
            else "💡 Green dots show teams sheltering rookies with prime veterans (VIR >= 1.25)."
        )
        st.plotly_chart(fig_age, use_container_width=True, theme=None)
        st.markdown("<br>", unsafe_allow_html=True)

        # Full-Width Chart 5B: Playing Style Synergy
        fig_hit = px.scatter(
            df_ts_plot,
            x="xgf_pct",
            y="pfhs_pct",
            color="macro_regime",
            color_discrete_map={"Peak": "#15803D", "Rebuild": "#EA580C", "Inaction": "#334155"},
            size="bubble_sz",
            hover_name="franchise_id",
            hover_data={
                "season_label": True,
                "substate_clean": True,
                "xgf_pct": ":.1f",
                "pfhs_pct": ":.1f",
                "ozone_hit_pct": ":.1f",
                "playoff_games_won": True,
                "playoff_rounds_won": True,
                "bubble_sz": False,
            },
            labels={
                "xgf_pct": "Buts attendus à 5c5 (xGF%)" if is_fr else "5-on-5 Expected Goals (xGF%)",
                "pfhs_pct": "Échec avant zone offensive (%)" if is_fr else "Proactive Forecheck Hit Share (%)",
                "macro_regime": "Macro-régime" if is_fr else "Macro Regime",
                "season_label": "Saison" if is_fr else "Season",
                "substate_clean": "État du régime" if is_fr else "Regime State",
                "ozone_hit_pct": "% Mises en échec zone off." if is_fr else "O-Zone Hit %",
                "playoff_games_won": "Matchs séries gagnés" if is_fr else "Playoff Game Wins",
                "playoff_rounds_won": "Séries remportées" if is_fr else "Playoff Series Won",
            },
            title="Synergie Habileté × Échec avant physique" if is_fr else "Skill × Forecheck Physicality Synergy",
        )
        fig_hit.add_vline(
            x=50.0,
            line_dash="dash",
            line_color="#475569",
            annotation_text="Seuil d'équilibre 50 % xGF" if is_fr else "50% Expected Goals Break-Even",
            annotation_position="top left",
        )
        fig_hit.update_layout(
            height=580,
            margin=dict(l=65, r=40, t=65, b=115),
            title=dict(
                text="Synergie Habileté × Échec avant physique (Taille = Séries gagnées)"
                if is_fr
                else "Skill × Forecheck Physicality Synergy (Circle Size = Series Won)",
                font=dict(size=14, color="#0F172A"),
                y=0.98,
                x=0.0,
                xanchor="left",
            ),
            xaxis=dict(
                title="Part des buts attendus à 5 contre 5 (xGF%, %)"
                if is_fr
                else "5-on-5 Expected Goals Share (xGF%, %)",
            ),
            yaxis=dict(
                title="Part des mises en échec offensives sur échec avant (%)"
                if is_fr
                else "Proactive Offensive-Zone Forecheck Hit Share (%)",
            ),
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        )
        style_plotly_high_contrast(fig_hit)
        st.caption(
            "💡 Le quadrant supérieur droit combine possession de tir et jeu physique offensif."
            if is_fr
            else "💡 Top-right quadrant teams pair superior shot share with aggressive forechecking."
        )
        st.plotly_chart(fig_hit, use_container_width=True, theme=None)
        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown(
            "#### Rapports méthodologiques et tableaux économétriques"
            if is_fr
            else "#### Deep-Dive Reports & Econometric Tables (Click Any Report Below to Read)"
        )
        if is_fr:
            doc_map = {
                "Résumé analytique de l'étude sur 25 ans (EXECUTIVE_SUMMARY.md)": INSIGHTS_DIR / "EXECUTIVE_SUMMARY.md",
                "Autopsie des échecs : Pourquoi les reconstructions calent (AUTOPSY_FAILED_REBUILDS.md)": INSIGHTS_DIR / "AUTOPSY_FAILED_REBUILDS.md",
                "Études de cas : Comment les champions ont été bâtis (CASE_STUDIES_CHAMPIONS.md)": INSIGHTS_DIR / "CASE_STUDIES_CHAMPIONS.md",
                "Évolution des reconstructions selon les ères salariales (REGIME_TRENDS_AND_ERA_SHIFTS.md)": INSIGHTS_DIR / "REGIME_TRENDS_AND_ERA_SHIFTS.md",
                "Tableau 1 : Distribution des régimes et présence des 32 équipes (tab01)": TABLES_DIR / "tab01_regime_distribution_summary.md",
                "Tableau 2 : ADN des prétendants par ère salariale (tab02)": TABLES_DIR / "tab02_contender_dna_by_era.md",
                "Tableau 3 : Modèle de survie et horloge d'attente des partisans (tab03)": TABLES_DIR / "tab03_survival_cox_frailty_aft.md",
                "Tableau 4 : Modèle à changement de régime d'état-espace (tab04)": TABLES_DIR / "tab04_state_space_sem_posteriors.md",
                "Tableau 5 : Audit quasi-expérimental par variable instrumentale (tab05)": TABLES_DIR / "tab05_iv_quasi_experimental_diagnostics.md",
                "Tableau 6 : Risque de congédiement des DG et tri des joueurs autonomes (tab06)": TABLES_DIR / "tab06_gm_hazard_and_market_sorting.md",
                "Tableau 7 : Suite de robustesse et sensibilité en 11 piliers (tab07)": TABLES_DIR / "tab07_eleven_pillar_robustness_suite.md",
                "Tableau 8 : Structure d'âge et synergie d'échec avant (tab08)": TABLES_DIR / "tab08_age_experience_and_hit_style_synergy.md",
            }
        else:
            doc_map = {
                "Executive Summary of the 25-Year Study (EXECUTIVE_SUMMARY.md)": INSIGHTS_DIR / "EXECUTIVE_SUMMARY.md",
                "Why Rebuilds Fail: Forensic Autopsies of Purgatory Teams (AUTOPSY_FAILED_REBUILDS.md)": INSIGHTS_DIR / "AUTOPSY_FAILED_REBUILDS.md",
                "How Cup Champions Were Built: Case Studies (CASE_STUDIES_CHAMPIONS.md)": INSIGHTS_DIR / "CASE_STUDIES_CHAMPIONS.md",
                "How Rebuilding Changed Across Salary-Cap Eras (REGIME_TRENDS_AND_ERA_SHIFTS.md)": INSIGHTS_DIR / "REGIME_TRENDS_AND_ERA_SHIFTS.md",
                "Table 1: Regime Distribution & 32-Team Occupancy (tab01)": TABLES_DIR / "tab01_regime_distribution_summary.md",
                "Table 2: Contender DNA by Salary-Cap Era (tab02)": TABLES_DIR / "tab02_contender_dna_by_era.md",
                "Table 3: Rebuild Survival & Fan Waiting Clock Model (tab03)": TABLES_DIR / "tab03_survival_cox_frailty_aft.md",
                "Table 4: State-Space Regime Transition Model (tab04)": TABLES_DIR / "tab04_state_space_sem_posteriors.md",
                "Table 5: Draft Lottery Instrumental Variable Audit (tab05)": TABLES_DIR / "tab05_iv_quasi_experimental_diagnostics.md",
                "Table 6: GM Firing Hazard & Free-Agent Market Sorting (tab06)": TABLES_DIR / "tab06_gm_hazard_and_market_sorting.md",
                "Table 7: Eleven-Pillar Sensitivity & Robustness Checks (tab07)": TABLES_DIR / "tab07_eleven_pillar_robustness_suite.md",
                "Table 8: Roster Age Structure & Forecheck Hit Synergy (tab08)": TABLES_DIR / "tab08_age_experience_and_hit_style_synergy.md",
            }
        doc_choice = st.selectbox(
            "Sélectionner un rapport ou tableau à lire :" if is_fr else "Select a Report or Table to Read Below:",
            list(doc_map.keys()),
        )
        sel_path = doc_map.get(doc_choice)
        if sel_path and sel_path.exists():
            st.markdown(sel_path.read_text(encoding="utf-8"))

    # =========================================================================
    # TAB 6: Interactive Rebuild Strategy Simulator (What-If Sandbox)
    # =========================================================================
    with t6:
        st.subheader(
            "6. Simulateur stratégique de reconstruction DG"
            if is_fr
            else "6. Interactive GM Rebuild Simulator: Test Your Own Roster Blueprint"
        )
        if is_fr:
            st.markdown(
                """
                Ajustez les six curseurs ci-dessous pour simuler l'impact de la composition d'effectif et du style de jeu d'une équipe en reconstruction sur ses chances historiques d'accéder au **Sommet (« Peak »)** ou de s'enliser en **Inaction / Purgatoire** :
                """
            )
        else:
            st.markdown(
                """
                Adjust the six sliders below to test how changing a rebuilding team's **roster mix** and **playing style** changes its historical odds of breaking out into **`Peak` Playoff Contention** vs. getting trapped in **`Inaction` / Rebuild Purgatory**:
                """
            )
        sim1, sim2, sim3 = st.columns(3)
        with sim1:
            s_cd_spine = st.slider(
                "1. Part des Centres + Défenseurs (% temps clé)"
                if is_fr
                else "1. Top Centers + Defensemen Share of Core Ice Time (%)",
                30, 85, 68, step=2,
                help=(
                    "Part du temps de jeu de l'effectif clé jouée par les centres et défenseurs plutôt que les ailiers."
                    if is_fr
                    else "Share of core ice time played by Centers and Defensemen rather than Wingers."
                ),
            )
            s_trade_core = st.slider(
                "2. Part du noyau acquise par échange / autonomes (%)"
                if is_fr
                else "2. Share of Core Acquired via Prime Trades / Cap Arbitrage (%)",
                0, 80, 45, step=5,
                help=(
                    "Part de l'effectif clé bâtie en échangeant pour des joueurs de 21 à 25 ans plutôt qu'en attendant uniquement le repêchage."
                    if is_fr
                    else "Share of core built by trading for 21–25 year-old players rather than waiting only on draft picks."
                ),
            )
        with sim2:
            s_elc_youth = st.slider(
                "3. Part de temps de jeu recrues 0–3 ans (%)"
                if is_fr
                else "3. Entry-Level Youth (0–3 Yrs Experience) Share of Ice Time (%)",
                10, 50, 24, step=2,
                help=(
                    "Attention : Pousser les recrues au-dessus de 30 % du temps de jeu déclenche la pénalité de surcharge jeunesse !"
                    if is_fr
                    else "Warning: Pushing Entry-Level Youth above 30% of team ice time triggers the Youth Overload penalty!"
                ),
            )
            s_vir = st.slider(
                "4. Ratio d'encadrement vétéran (VIR : Vétérans ÷ Recrues)"
                if is_fr
                else "4. Veteran Insulation Ratio (Prime Vets ÷ Entry-Level Youth)",
                0.25, 2.50, 1.35, step=0.05,
                help=(
                    "Ratio des vétérans de 4 à 9 ans par rapport aux recrues de 0 à 3 ans. Les valeurs >= 1,25 protègent les jeunes patineurs."
                    if is_fr
                    else "Ratio of 4–9 year veterans to 0–3 year rookies. Values >= 1.25 shelter young players from tough matchups."
                ),
            )
        with sim3:
            s_ozone_hit = st.slider(
                "5. Mises en échec zone offensive (% du total)"
                if is_fr
                else "5. Offensive-Zone Forecheck Hit Share (% of All Hits)",
                28, 56, 46, step=2,
                help=(
                    "Pourcentage des mises en échec distribuées en zone offensive sur échec avant pour forcer des revirements."
                    if is_fr
                    else "Percentage of team hits delivered on the offensive forecheck forcing turnovers."
                ),
            )
            s_skill_xg = st.slider(
                "6. Part des buts attendus à 5c5 (xGF% Habileté)"
                if is_fr
                else "6. 5-on-5 Expected Goals Share (xGF% Skill Baseline)",
                44.0, 58.0, 52.5, step=0.5,
                help=(
                    "Part des buts attendus et de la possession à 5c5 (50 % = moyenne LNH)."
                    if is_fr
                    else "5-on-5 puck possession and shot quality share (50% = league average)."
                ),
            )

        yoe_pen = max(0.0, (s_elc_youth / 100.0) - 0.30) ** 2
        style_syn = np.sqrt((s_skill_xg / 55.0) * (s_ozone_hit / 48.0))
        logit_peak = (
            -0.35
            + 1.65 * ((s_cd_spine / 100.0) - 0.55)
            + 1.40 * ((s_trade_core / 100.0) - 0.35)
            + 0.75 * (s_vir - 0.80)
            - 14.5 * yoe_pen
            + 1.55 * (style_syn - 0.92)
        )
        logit_inact = (
            -0.45
            - 1.20 * ((s_cd_spine / 100.0) - 0.55)
            + 18.0 * yoe_pen
            - 0.85 * (s_vir - 0.80)
            - 1.10 * ((s_ozone_hit - 42.0) / 20.0)
        )
        p_peak = float(1.0 / (1.0 + np.exp(-logit_peak)))
        p_inact = float((1.0 - p_peak) * (1.0 / (1.0 + np.exp(-logit_inact))))
        est_years = round(max(2.2, 6.2 - 3.2 * p_peak + 2.8 * p_inact), 1)

        st.markdown("<br>", unsafe_allow_html=True)
        k_top1, k_top2 = st.columns(2)
        if is_fr:
            k_top1.metric(
                "Chances d'accès aux séries en 5 ans (Reconstruction → Peak)",
                f"{100*p_peak:.1f} %",
                f"{100*(p_peak - 0.44):+.1f} points de % vs moyenne LNH",
            )
            k_top2.metric(
                "Risque d'échec / purgatoire (5+ ans)",
                f"{100*p_inact:.1f} %",
                "Surcharge jeunesse (>30 % recrues) !" if s_elc_youth > 30 else "Jeunesse convenablement encadrée",
                delta_color="inverse",
            )
        else:
            k_top1.metric(
                "Predicted 5-Year Playoff Breakout Odds (Rebuild → Peak)",
                f"{100*p_peak:.1f}%",
                f"{100*(p_peak - 0.44):+.1f} percentage points vs. League Average Rebuild",
            )
            k_top2.metric(
                "Predicted Rebuild Purgatory Failure Risk (5+ Years Stuck)",
                f"{100*p_inact:.1f}%",
                "Youth Overload (>30% ELC Youth)!" if s_elc_youth > 30 else "Youth Properly Insulated by Prime Veterans",
                delta_color="inverse",
            )
        k_bot1, k_bot2 = st.columns(2)
        if is_fr:
            k_bot1.metric(
                "Délai prévu pour devenir prétendant aux séries",
                f"{est_years} saisons",
                "Filière hybride rapide" if s_trade_core >= 40 else "Filière repêchage pur plus lente",
            )
            k_bot2.metric(
                "Score combiné Synergie Habileté × Échec avant (SPSI)",
                f"{style_syn:.2f}",
                "Identité physique et talent prête pour les séries" if style_syn >= 0.98 else "Sous le repère historique des prétendants",
            )
        else:
            k_bot1.metric(
                "Expected Time Needed to Reach Playoff Contention",
                f"{est_years} Seasons",
                "Fast Hybrid Rebuild Track" if s_trade_core >= 40 else "Slower Pure-Draft Incubation Track",
            )
            k_bot2.metric(
                "Combined Skill × Forecheck Hit Synergy Score (SPSI)",
                f"{style_syn:.2f}",
                "Playoff-Ready Physical + Skill Identity" if style_syn >= 0.98 else "Below Historical Contender Baseline",
            )
        st.markdown("<br>", unsafe_allow_html=True)

        if s_elc_youth > 30 and s_vir < 0.75:
            st.warning(
                "⚠️ **Avertissement de surcharge jeunesse (`Recrues > 30 %` avec faible encadrement vétéran)** : "
                "Lorsque les recrues jouent plus de 30 % des minutes de l'équipe sans vétérans d'expérience (`4–9 ans`) pour les protéger, "
                "les équipes subissent historiquement de lourdes défaillances défensives et s'enlisent dans le purgatoire `R5`."
                if is_fr
                else "⚠️ **Youth Overload Warning (`Entry-Level Youth > 30%` with Low Veteran Insulation)**: "
                "When rookies play more than 30% of team ice time without prime veterans (`4–9 yrs experience`) to shelter them, "
                "teams historically suffer severe defensive breakdowns and get stuck in `R5: Purgatory`."
            )
        elif p_peak >= 0.58:
            st.success(
                "✅ **Plan de reconstruction hybride à haut taux de succès** : "
                "Bâtir autour de centres et de défenseurs, acquérir des joueurs de 22 à 25 ans par échange, entourer les recrues de vétérans "
                "et déployer un échec avant offensif physique maximise la vitesse d'accès au statut de prétendant (« Peak »)."
                if is_fr
                else "✅ **High-Success Hybrid Rebuild Blueprint**: "
                "Building around Centers and Defensemen, acquiring prime 22–25 year-old players via trade, sheltering rookies "
                "with prime veterans, and playing an aggressive offensive-zone forecheck maximizes breakout speed into `Peak` contention."
            )

        gm_cheat_dir = REPO_ROOT / "dev" / "player_team_success" / "rebuild" / "deepdive" / "gm_cheatcode"
        gm_ladder_csv = gm_cheat_dir / "tables" / "gm_priority_checklist_optimization_ladder.csv"
        gm_sens_csv = gm_cheat_dir / "tables" / "gm_sensitivity_tornado_ranking.csv"
        gm_seq_csv = gm_cheat_dir / "tables" / "gm_720_sequence_permutation_optimization.csv"
        gm_bench_csv = gm_cheat_dir / "tables" / "gm_computational_benchmarks_and_cv.csv"
        gm_fig1 = gm_cheat_dir / "figures" / "fig1_gm_cheatcode_sensitivity_and_checklist.png"
        gm_fig2 = gm_cheat_dir / "figures" / "fig2_720_sequence_landscape_and_diverging_paths.png"
        gm_fig3 = gm_cheat_dir / "figures" / "fig3_non_linear_response_surfaces_and_franchise_audit.png"
        gm_fig4 = gm_cheat_dir / "figures" / "fig4_bootstrap_distributions_and_sobol_shapley_decomposition.png"
        if gm_ladder_csv.exists() and gm_sens_csv.exists():
            st.markdown("---")
            st.markdown(
                "#### 🕹️ Guide d'optimisation des priorités du DG (638 080 simulations, Shapley exact & 720 séquences)"
                if is_fr
                else "#### 🕹️ The 'Konami Code' GM Priority Checklist (638,080-Run Sensitivity, Exact Shapley & 6! = 720 Sequence Optimization)"
            )
            st.caption(
                "Dans quel ordre précis un directeur général de la LNH doit-il activer ces leviers ? Résultats détaillés ci-dessous (`deepdive/gm_cheatcode`)."
                if is_fr
                else "What order should an NHL General Manager pull these levers? Below are the exact sensitivity rankings and sequence permutations (`deepdive/gm_cheatcode`)."
            )
            if gm_fig1.exists():
                st.markdown(
                    "**Figure 1 : Classement de sensibilité globale (512 000 runs Sobol) et échelle de priorité en 6 étapes**"
                    if is_fr
                    else "**Figure 1: Global Sensitivity Ranking (512,000 Sobol Runs) & 6-Step Priority Optimization Ladder**"
                )
                st.image(str(gm_fig1), use_container_width=True)
            if gm_fig2.exists():
                st.markdown(
                    "**Figure 2 : Paysage exhaustif des 720 séquences avec trajectoires divergentes selon l'an 1**"
                    if is_fr
                    else "**Figure 2: Exhaustive 6! = 720 Sequence Landscape Colored by Year-1 Opening Move & Stage-by-Stage Trajectory Divergence**"
                )
                st.image(str(gm_fig2), use_container_width=True)
            if gm_fig3.exists():
                st.markdown(
                    "**Figure 3 : Surface de réponse non-linéaire de surcharge jeunesse et audit réel 2024/25**"
                    if is_fr
                    else "**Figure 3: 10,000-Grid Non-Linear Youth Overload Response Surface & 2024/25 Real-Franchise Simulator Audit**"
                )
                st.image(str(gm_fig3), use_container_width=True)
            if gm_fig4.exists():
                st.markdown(
                    "**Figure 4 : Convergence des trois méthodes de sensibilité (Sobol Si, STi & Shapley 2^6) + validation croisée**"
                    if is_fr
                    else "**Figure 4: Three-Method Sensitivity Convergence (Sobol Si, STi & Exact 2^6 Shapley) + 5-Fold Out-of-Sample Cross-Validation**"
                )
                st.image(str(gm_fig4), use_container_width=True)

            table_renames = BLOG_TABLE_COLUMN_RENAMES_FR if is_fr else BLOG_TABLE_COLUMN_RENAMES

            st.markdown(
                "**1. Échelle de priorité d'émergence en 6 étapes (avec bandes Monte Carlo à 90 %)**"
                if is_fr
                else "**1. The 6-Step GM Priority Escape Ladder (With 10,000-Rollout 90% Monte Carlo Bands)**"
            )
            df_lad_disp = pd.read_csv(gm_ladder_csv).rename(columns=table_renames)
            st.dataframe(df_lad_disp, use_container_width=True, hide_index=True)

            st.markdown(
                "**2. Classement de sensibilité globale (512 000 évaluations Sobol + attribution Shapley 2^6)**"
                if is_fr
                else "**2. Global Sensitivity Ranking (512,000 Sobol Evaluations + Exact 2^6 Coalition Shapley Attribution)**"
            )
            df_sens_disp = pd.read_csv(gm_sens_csv).rename(columns=table_renames)
            st.dataframe(df_sens_disp, use_container_width=True, hide_index=True)

            if gm_seq_csv.exists():
                st.markdown(
                    "**3. Pourquoi l'ordre d'exécution compte : Optimisation des 720 séquences pluriannuelles**"
                    if is_fr
                    else "**3. Why Execution Order Matters: Exhaustive 6! = 720 Multi-Year Sequence Optimization (Best vs. Median vs. Worst Order)**"
                )
                df_seq_disp = pd.read_csv(gm_seq_csv).rename(columns=table_renames)
                st.dataframe(df_seq_disp, use_container_width=True, hide_index=True)

            if gm_bench_csv.exists():
                st.markdown(
                    "**4. Repères informatiques et validation croisée hors échantillon (638 080 évaluations au total)**"
                    if is_fr
                    else "**4. Computational Benchmarks & Out-of-Sample Cross-Validation Summary (638,080 Total Evaluations)**"
                )
                df_bench_disp = pd.read_csv(gm_bench_csv).rename(columns=table_renames)
                st.dataframe(df_bench_disp, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 7: 6-Part Medium Blog Series & MTL vs. CHI Case Study Library
    # =========================================================================
    with t7:
        st.subheader(
            "7. Série de 6 articles : La science des reconstructions (Montréal vs Chicago)"
            if is_fr
            else "7. 6-Part Medium Blog Series: The Math Behind NHL Rebuilds (Montréal vs. Chicago)"
        )
        if is_fr:
            st.markdown(
                """
                Découvrez les **6 articles de vulgarisation analytique (moins de 600 mots chacun)** rédigés dans un style narratif accessible.
                Chaque publication propose des infographies claires, des données approfondies et des tableaux économiques complets.
                """
            )
        else:
            st.markdown(
                """
                Read all **6 self-contained blog posts (under 600 words each)** written in a casual, fan-friendly data-storytelling style.
                Each article features **spacious, full-width 2-panel vertical infographics** (with plain-English labels and zero overlapping text), plus expandable technical reports and plain-English data tables.
                """
            )
        if is_fr:
            blog_catalog = [
                (
                    "Article 1 : Couler au classement, faire du surplace ou décoller ? Reconstructions vs purges générationnelles (`deepdive/basics`)",
                    REPO_ROOT / "dev" / "player_team_success" / "rebuild" / "deepdive" / "basics",
                    "fig1_mtl_vs_chi_regime_trajectories.png",
                ),
                (
                    "Article 2 : Dans la boîte noire de la reconstruction : Cohortes d'expérience, physique et régimes (`deepdive/explain`)",
                    REPO_ROOT / "dev" / "player_team_success" / "rebuild" / "deepdive" / "explain",
                    "fig1_four_pillar_architecture_mtl_chi.png",
                ),
                (
                    "Article 3 : Combien de temps les partisans doivent-ils attendre ? L'horloge actuarielle des reconstructions (`deepdive/timeline`)",
                    REPO_ROOT / "dev" / "player_team_success" / "rebuild" / "deepdive" / "timeline",
                    "fig1_cumulative_incidence_fan_wait_clock.png",
                ),
                (
                    "Article 4 : Croissance vs Régression — Étude économétrique Montréal vs Chicago post-2019 (`deepdive/mtl_vs_chi`)",
                    REPO_ROOT / "dev" / "player_team_success" / "rebuild" / "deepdive" / "mtl_vs_chi",
                    "fig1_mtl_vs_chi_structural_decomposition_2019_2025.png",
                ),
                (
                    "Article 5 : Pourquoi la reconstruction d'aujourd'hui n'est plus celle d'antan — Arbitrage salarial & échanges (`deepdive/modern_shift`)",
                    REPO_ROOT / "dev" / "player_team_success" / "rebuild" / "deepdive" / "modern_shift",
                    "fig1_scorched_earth_vs_hybrid_trade_era_shift.png",
                ),
                (
                    "Article 6 : Le code secret des DG de la LNH — 638 080 évaluations simulées et liste d'évasion en 6 étapes (`deepdive/gm_cheatcode`)",
                    REPO_ROOT / "dev" / "player_team_success" / "rebuild" / "deepdive" / "gm_cheatcode",
                    "fig1_gm_cheatcode_sensitivity_and_checklist.png",
                ),
            ]
        else:
            blog_catalog = [
                (
                    "Post 1: Tanking, Treading Water, or Taking Off? Defining Generational Struggle vs. Rebuilds (`deepdive/basics`)",
                    REPO_ROOT / "dev" / "player_team_success" / "rebuild" / "deepdive" / "basics",
                    "fig1_mtl_vs_chi_regime_trajectories.png",
                ),
                (
                    "Post 2: Inside the Rebuild Black Box — Roster Experience Clusters, Physicality & Regimes (`deepdive/explain`)",
                    REPO_ROOT / "dev" / "player_team_success" / "rebuild" / "deepdive" / "explain",
                    "fig1_four_pillar_architecture_mtl_chi.png",
                ),
                (
                    "Post 3: How Long Should Hockey Fans Wait? The Actuarial Clock on NHL Rebuilds (`deepdive/timeline`)",
                    REPO_ROOT / "dev" / "player_team_success" / "rebuild" / "deepdive" / "timeline",
                    "fig1_cumulative_incidence_fan_wait_clock.png",
                ),
                (
                    "Post 4: Growth vs. Regression — Montréal vs. Chicago Post-2019 Econometric Case Study (`deepdive/mtl_vs_chi`)",
                    REPO_ROOT / "dev" / "player_team_success" / "rebuild" / "deepdive" / "mtl_vs_chi",
                    "fig1_mtl_vs_chi_structural_decomposition_2019_2025.png",
                ),
                (
                    "Post 5: Why Today's Rebuild Is Not Your Father's Rebuild — Cap Arbitrage & Trading vs. Tanking (`deepdive/modern_shift`)",
                    REPO_ROOT / "dev" / "player_team_success" / "rebuild" / "deepdive" / "modern_shift",
                    "fig1_scorched_earth_vs_hybrid_trade_era_shift.png",
                ),
                (
                    "Post 6: The 'Konami Code' for NHL GMs — 638,080 Simulated Evaluations & The 6-Step Escape Checklist (`deepdive/gm_cheatcode`)",
                    REPO_ROOT / "dev" / "player_team_success" / "rebuild" / "deepdive" / "gm_cheatcode",
                    "fig1_gm_cheatcode_sensitivity_and_checklist.png",
                ),
            ]

        blog_titles = [b[0] for b in blog_catalog]
        sel_blog_title = st.selectbox(
            "Choisir l'un des 6 articles à lire :" if is_fr else "Choose One of the 6 Blog Posts to Read:",
            blog_titles,
            index=5,
        )
        sel_entry = next(b for b in blog_catalog if b[0] == sel_blog_title)
        proj_dir = sel_entry[1]
        fig_name = sel_entry[2]

        bp_file = proj_dir / "BLOG_POST.md"
        ta_file = proj_dir / "TECHNICAL_ANALYSIS.md"
        fig_file = proj_dir / "figures" / fig_name

        if bp_file.exists():
            bp_text = bp_file.read_text(encoding="utf-8")
            word_count = len(bp_text.split())
            st.markdown(
                f"**{'Dossier du projet' if is_fr else 'Project Folder'}** : `dev/player_team_success/{proj_dir.name}/`  |  "
                f"**{'Longueur' if is_fr else 'Article Length'}** : `{word_count} {'mots' if is_fr else 'words'}` ({'Cible :' if is_fr else 'Target:'} `<= 600 words`)"
            )
            parts = bp_text.split(f"figures/{fig_name})")
            if len(parts) == 2:
                before_img = parts[0].rsplit("![", 1)[0]
                after_img = parts[1]
                st.markdown(before_img)
                if fig_file.exists():
                    st.image(str(fig_file), use_container_width=True)
                st.markdown(after_img)
            else:
                st.markdown(bp_text)
                if fig_file.exists():
                    st.image(str(fig_file), use_container_width=True)

        extra_figs = sorted(
            [p for p in (proj_dir / "figures").glob("*.png") if p.name != fig_name]
        ) if (proj_dir / "figures").exists() else []
        if extra_figs:
            exp_extra_lbl = (
                f"📈 Voir les figures computationnelles complémentaires pour `{proj_dir.name}` ({len(extra_figs)} figures haute résolution)"
                if is_fr
                else f"📈 View Extended Computational Figure Suite for `{proj_dir.name}` ({len(extra_figs)} Additional High-Resolution Figures)"
            )
            with st.expander(exp_extra_lbl, expanded=True):
                for ef in extra_figs:
                    st.markdown(f"**Figure : `{ef.name}`**")
                    st.image(str(ef), use_container_width=True)

        if ta_file.exists():
            exp_ta_lbl = (
                "🔬 Lire l'analyse technique approfondie et les tableaux de paramètres (`TECHNICAL_ANALYSIS.md`)"
                if is_fr
                else "🔬 Read Companion Technical Deep-Dive & Parameter Tables (`TECHNICAL_ANALYSIS.md`)"
            )
            with st.expander(exp_ta_lbl, expanded=False):
                st.markdown(ta_file.read_text(encoding="utf-8"))

        csv_files = sorted((proj_dir / "tables").glob("*.csv")) if (proj_dir / "tables").exists() else []
        if csv_files:
            exp_csv_lbl = (
                f"📊 Voir les tableaux de données formatés pour `{proj_dir.name}` ({len(csv_files)} tableaux avec en-têtes clairs)"
                if is_fr
                else f"📊 View Formatted Data Tables for `{proj_dir.name}` ({len(csv_files)} Tables with Plain-English Headers)"
            )
            table_renames_blog = BLOG_TABLE_COLUMN_RENAMES_FR if is_fr else BLOG_TABLE_COLUMN_RENAMES
            with st.expander(exp_csv_lbl, expanded=True):
                for cf in csv_files:
                    st.markdown(f"**Tableau : `{cf.name}`**" if is_fr else f"**Table: `{cf.name}`**")
                    raw_df = pd.read_csv(cf)
                    clean_df = raw_df.rename(columns=table_renames_blog)
                    st.dataframe(clean_df, use_container_width=True, hide_index=True)


render_page()
