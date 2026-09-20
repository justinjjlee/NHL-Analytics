'''
Hit on Impact: Sequential EDA & Tactical Ice Analytics
Analyzes event chains (Events +1 to +4), 30-second shot & goal windows,
interactive spatial heatmaps with click-to-inspect subsequent events,
and team/season comparative benchmarking across the NHL.
'''

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import importlib
import i18n
try:
    importlib.reload(i18n)
except Exception:
    pass
from i18n import t
from exe.rink_utils import add_rink_shapes

RINK_INTERACTION_TEXTS = {
    "EN": {
        "hit_rink_click_instruction": "Click any colored hotspot tile on the rink diagram below to enable location-specific analytics: downstream event distributions, puck retention win rates, subsequent shot frequencies, and goal trajectory vectors will be loaded and displayed in the inspection section below.",
        "hit_rink_select_prompt": "Click any colored tile on the ice rink above (or choose a tactical preset) to enable and inspect its detailed metrics and subsequent play distributions here.",
        "hit_rink_text_heading": "Selected Location Details: Coordinates ({x}, {y}) — {zone}",
        "hit_rink_text_body": "Hits Recorded: {hits} ({pct}% of league volume) | Puck Win Rate: {win}% ({win_diff}% vs avg) | Immediate Shots For: {s_for}% | Immediate Shots Conceded: {s_agst}% | 30s Window: {net_shots} net shots ({g_for} goals for vs {g_agst} goals against).",
        "hit_rink_clear_btn": "Clear Location Selection"
    },
    "FR": {
        "hit_rink_click_instruction": "Cliquez sur n'importe quelle cellule colorée de la patinoire ci-dessous pour activer les analyses détaillées par emplacement : la distribution des événements consécutifs, le taux de récupération de la rondelle, la fréquence des tirs et les vecteurs de trajectoire de but s'afficheront dans la section d'inspection ci-dessous.",
        "hit_rink_select_prompt": "Cliquez sur une cellule colorée sur la patinoire ci-dessus (ou sélectionnez un préréglage tactique) pour activer et inspecter ses métriques détaillées et la distribution des jeux suivants.",
        "hit_rink_text_heading": "Détails de l'emplacement sélectionné : Coordonnées ({x}, {y}) — {zone}",
        "hit_rink_text_body": "Mises en échec enregistrées : {hits} ({pct} % de la ligue) | Récupération de rondelle : {win} % ({win_diff} % vs moy.) | Tirs immédiats Pour : {s_for} % | Tirs immédiats Contre : {s_agst} % | Fenêtre 30s : {net_shots} tirs nets ({g_for} buts pour vs {g_agst} buts contre).",
        "hit_rink_clear_btn": "Désélectionner l'emplacement"
    }
}

def get_rink_text(key):
    cur_lang = st.session_state.get("lang", "EN")
    val = t(key)
    if val == key or val is None:
        return RINK_INTERACTION_TEXTS.get(cur_lang, RINK_INTERACTION_TEXTS["EN"]).get(key, key)
    return val

def get_data_path(filename):
    """Resolve absolute path to tabulated impact_of_hits datasets."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "../.."))
    return os.path.join(repo_root, "dev/decision_science/impact_of_hits/data", filename)

@st.cache_data
def load_all_data():
    """Load tabulated sequence, KPI, grid, and team-season datasets."""
    kpi_file = get_data_path("hit_zone_summary_kpi.csv")
    seq_file = get_data_path("tabulated_hit_sequences.csv")
    goals_file = get_data_path("hit_goal_window_30s.csv")
    grid_file = get_data_path("rink_grid_hit_sequences.csv")
    team_kpi_file = get_data_path("team_season_hit_kpi.csv")
    team_spatial_file = get_data_path("team_season_spatial_hits.csv")

    df_kpi = pd.read_csv(kpi_file) if os.path.exists(kpi_file) else None
    df_seq = pd.read_csv(seq_file) if os.path.exists(seq_file) else None
    df_goals = pd.read_csv(goals_file) if os.path.exists(goals_file) else None
    df_grid = pd.read_csv(grid_file) if os.path.exists(grid_file) else None
    df_team_kpi = pd.read_csv(team_kpi_file) if os.path.exists(team_kpi_file) else None
    df_team_spatial = pd.read_csv(team_spatial_file) if os.path.exists(team_spatial_file) else None

    # Load teamlist for full team names
    current_dir = os.path.dirname(os.path.abspath(__file__))
    teamlist_path = os.path.abspath(os.path.join(current_dir, "../data/teamlist.csv"))
    if os.path.exists(teamlist_path):
        tl = pd.read_csv(teamlist_path)
        team_name_map = dict(zip(tl["tricode"], tl["tricode"] + " - " + tl["team"]))
    else:
        team_name_map = {}

    return df_kpi, df_seq, df_goals, df_grid, df_team_kpi, df_team_spatial, team_name_map

def normalize_zone(zone_val):
    """Safely normalizes any zone input (short code, localized string, or legacy label) to 'All', 'O', 'D', or 'N'."""
    if not zone_val:
        return "All"
    z = str(zone_val).strip()
    zl = z.lower()
    if z in ("All", "ALL") or "all" in zl or "toutes" in zl or "complet" in zl or "compreh" in zl:
        return "All"
    if z == "O" or "offens" in zl:
        return "O"
    if z == "D" or "defens" in zl or "défens" in zl:
        return "D"
    if z == "N" or "neutr" in zl:
        return "N"
    return "All"

def create_sankey_sequence(df_seq, selected_zone="All"):
    """Constructs an interactive multi-layer Sankey diagram."""
    norm_zone = normalize_zone(selected_zone)
    filtered = df_seq.copy()
    if norm_zone != "All":
        filtered = filtered[filtered["hit_zone"] == norm_zone]

    step1 = filtered[filtered["event_step"] == 1].groupby(["hit_zone", "event_category"])["count"].sum().reset_index()
    step2 = filtered[filtered["event_step"] == 2].groupby(["event_category", "event_owner"])["count"].sum().reset_index()

    zones = list(filtered["hit_zone"].unique())
    categories = list(filtered["event_category"].unique())
    owners = list(filtered["event_owner"].unique())

    zone_map = {
        "O": t("hit_tree_zone_o"),
        "D": t("hit_tree_zone_d"),
        "N": t("hit_tree_zone_n")
    }
    cat_map = {
        "Shot Attempt": t("hit_cat_shot"),
        "Goal": t("hit_cat_goal"),
        "Possession Change": t("hit_cat_poss"),
        "Physical Battle": t("hit_cat_battle"),
        "Stoppage": t("hit_cat_stop"),
        "Other": t("hit_cat_other")
    }
    owner_map = {
        "Hitting Team": t("hit_owner_for"),
        "Opponent": t("hit_owner_against"),
        "Neutral / Stoppage": t("hit_owner_neutral")
    }

    zone_nodes = [f"{t('hit_th_origin_zone')}: {zone_map.get(z, z)}" for z in zones]
    step1_nodes = [f"{t('hit_step_prefix')} +1: {cat_map.get(c, c)}" for c in categories]
    step2_nodes = [f"{t('hit_step_prefix')} +2: {owner_map.get(o, o)}" for o in owners]

    node_labels = zone_nodes + step1_nodes + step2_nodes
    node_idx = {name: i for i, name in enumerate(node_labels)}

    sources, targets, values, colors = [], [], [], []

    category_colors = {
        "Shot Attempt": "#3b82f6",
        "Goal": "#10b981",
        "Possession Change": "#f59e0b",
        "Physical Battle": "#ef4444",
        "Stoppage": "#8b5cf6",
        "Other": "#64748b"
    }

    for _, row in step1.iterrows():
        z_node = f"{t('hit_th_origin_zone')}: {zone_map.get(row['hit_zone'], row['hit_zone'])}"
        c_node = f"{t('hit_step_prefix')} +1: {cat_map.get(row['event_category'], row['event_category'])}"
        sources.append(node_idx[z_node])
        targets.append(node_idx[c_node])
        values.append(row["count"])
        colors.append(category_colors.get(row["event_category"], "rgba(100, 116, 139, 0.4)"))

    for _, row in step2.iterrows():
        c_node = f"{t('hit_step_prefix')} +1: {cat_map.get(row['event_category'], row['event_category'])}"
        o_node = f"{t('hit_step_prefix')} +2: {owner_map.get(row['event_owner'], row['event_owner'])}"
        if c_node in node_idx and o_node in node_idx:
            sources.append(node_idx[c_node])
            targets.append(node_idx[o_node])
            values.append(row["count"])
            colors.append("rgba(148, 163, 184, 0.3)")

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=18,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=["#ce0e2d" if t('hit_th_origin_zone') in n else "#3b82f6" if "+1" in n else "#f59e0b" for n in node_labels]
        ),
        link=dict(source=sources, target=targets, value=values, color=colors)
    )])

    fig.update_layout(
        title_text=t("hit_sankey_heading"),
        title_font_size=16,
        font=dict(size=11, color="white"),
        height=500,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=20)
    )
    return fig

def render_interactive_rink_heatmap(df_grid, df_goals, selected_loc=None, half_rink=False, min_density=0.0):
    """
    Renders 2D density heatmap of hit occurrences with clickable hotspot tiles
    and downstream trajectory vectors. Displays all rink tiles for full interactivity.
    """
    grid = df_grid.copy()
    if min_density > 0:
        grid = grid[grid["density_pct"] >= min_density]
    if half_rink:
        grid = grid[grid["x_bin"] >= 0]

    fig = go.Figure()
    fig = add_rink_shapes(fig, half_rink=half_rink, theme="dark")

    # Heatmap scatter tiles (each 10ft x 10ft cell)
    customdata = np.stack((
        grid["total_hits"],
        grid["density_pct"],
        grid["puck_win_pct"],
        grid["shot_for_pct"],
        grid["shot_against_pct"],
        grid["spatial_zone"],
        grid["x_bin"],
        grid["y_bin"]
    ), axis=-1)

    # Bright, luminous multi-hue scale optimized for dark ice rink (#0f172a)
    bright_interactive_colorscale = [
        [0.0, "#00f0ff"],   # Electric Radiant Cyan (pops at low hit density)
        [0.2, "#22c55e"],   # Bright Emerald Green
        [0.4, "#a3e635"],   # Bright Lime
        [0.6, "#facc15"],   # Vibrant Golden Yellow
        [0.8, "#fb923c"],   # Radiant Coral Orange
        [1.0, "#ff1e56"]    # Intense Neon Crimson Red
    ]

    fig.add_trace(go.Scatter(
        x=grid["x_bin"],
        y=grid["y_bin"],
        mode="markers",
        marker=dict(
            symbol="square",
            size=18 if not half_rink else 22,
            color=grid["total_hits"],
            colorscale=bright_interactive_colorscale,
            showscale=True,
            colorbar=dict(
                title=dict(text=t("hit_th_total_hits"), font=dict(color="white", size=11)),
                tickfont=dict(color="white"),
                thickness=12,
                len=0.75,
                y=0.5
            ),
            line=dict(color="rgba(255,255,255,0.22)", width=0.5)
        ),
        customdata=customdata,
        hoverinfo="none",
        name=t("hit_sub_rink_heading")
    ))

    # If a location is selected, plot trajectories and highlight marker
    if selected_loc is not None:
        sel_x, sel_y = selected_loc
        # Highlight marker
        fig.add_trace(go.Scatter(
            x=[sel_x],
            y=[sel_y],
            mode="markers",
            marker=dict(
                symbol="circle-open",
                size=32,
                line=dict(color="#facc15", width=3.5)
            ),
            name=f"{t('hit_val_selected_focus')} ({sel_x}, {sel_y})",
            hoverinfo="skip"
        ))

        # Filter goals and shots originating near this grid cell (+/- 7 ft)
        loc_events = df_goals[
            (df_goals["x_bin"] == sel_x) & (df_goals["y_bin"] == sel_y)
        ].copy()

        if not loc_events.empty:
            sample_events = loc_events.head(75)  # Cap for speed
            x_lines_for, y_lines_for = [], []
            x_lines_against, y_lines_against = [], []

            for _, row in sample_events.iterrows():
                if row["outcome_side"] == "Hitting Team (For)":
                    x_lines_for.extend([row["hit_norm_x"], row["event_norm_x"], None])
                    y_lines_for.extend([row["hit_norm_y"], row["event_norm_y"], None])
                else:
                    x_lines_against.extend([row["hit_norm_x"], row["event_norm_x"], None])
                    y_lines_against.extend([row["hit_norm_y"], row["event_norm_y"], None])

            if x_lines_for:
                fig.add_trace(go.Scatter(
                    x=x_lines_for, y=y_lines_for, mode="lines",
                    line=dict(color="rgba(16, 185, 129, 0.45)", width=1.4, dash="dot"),
                    name=t("hit_th_for_shots"),
                    hoverinfo="skip"
                ))
            if x_lines_against:
                fig.add_trace(go.Scatter(
                    x=x_lines_against, y=y_lines_against, mode="lines",
                    line=dict(color="rgba(239, 68, 68, 0.45)", width=1.4, dash="dot"),
                    name=t("hit_th_agst_shots"),
                    hoverinfo="skip"
                ))

            # Add markers for goals
            goals_from_loc = loc_events[loc_events["is_goal"] == 1]
            if not goals_from_loc.empty:
                fig.add_trace(go.Scatter(
                    x=goals_from_loc["event_norm_x"],
                    y=goals_from_loc["event_norm_y"],
                    mode="markers",
                    marker=dict(
                        symbol="star",
                        size=14,
                        color=["#22c55e" if o == "Hitting Team (For)" else "#f43f5e" for o in goals_from_loc["outcome_side"]],
                        line=dict(color="white", width=1.5)
                    ),
                    name=t("hit_panel_goals_scored"),
                    text=[f"{t('hit_tree_event_goal')}! {r['outcome_side']} (+{r['delta_sec']}s)" for _, r in goals_from_loc.iterrows()],
                    hoverinfo="skip"
                ))

    fig.update_layout(
        title=t("hit_sub_rink_heading"),
        title_font_size=15,
        height=540,
        legend=dict(
            orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5,
            font=dict(color="white")
        )
    )
    return fig

def render_team_rink_heatmap(df_spatial, team_code, season_code, half_rink=False, display_name=None, is_benchmark=False, season_label=None, min_density=1.0):
    """Renders a team-specific spatial hit density heatmap on the rink. Drops tiles lower than min_density%."""
    sub = df_spatial[(df_spatial["team"] == team_code) & (df_spatial["season"] == season_code)].copy()

    # Fallback for LEAGUE_AVG if specific season combination not pre-tabulated
    if sub.empty and team_code == "LEAGUE_AVG":
        base = df_spatial[(df_spatial["team"] != "LEAGUE_AVG") & (df_spatial["season"] == season_code)]
        if not base.empty:
            sub = base.groupby(["x_bin", "y_bin"], as_index=False).agg(
                hit_count=("hit_count", "mean"),
                density_pct=("density_pct", "mean")
            )
            sub["team"] = "LEAGUE_AVG"
            sub["season"] = season_code

    if min_density > 0:
        sub = sub[sub["density_pct"] >= min_density]

    if half_rink:
        sub = sub[sub["x_bin"] >= 0]

    fig = go.Figure()
    fig = add_rink_shapes(fig, half_rink=half_rink, theme="dark")

    label = display_name or (t("hit_team_bench_league") if team_code == "LEAGUE_AVG" else team_code)
    season_text = season_label or season_code

    if not sub.empty:
        hover_texts = []
        for _, r in sub.iterrows():
            hit_str = f"{r['hit_count']:.1f}" if team_code == "LEAGUE_AVG" else f"{int(r['hit_count']):,}"
            hover_texts.append(
                f"<b>{label}</b> ({r['x_bin']:+.0f}, {r['y_bin']:+.0f})<br>"
                f"{t('hit_th_total_hits')}: {hit_str}<br>"
                f"{t('hit_panel_rate_col')}: {r['density_pct']:.2f}%"
            )

        # High-luminance colorscales tailored for dark ice background (#0f172a)
        # Primary: Radiant Solar Warm (Yellow -> Orange -> Red -> Crimson)
        bright_primary_scale = [
            [0.0, "#fde047"],
            [0.35, "#fb923c"],
            [0.7, "#ef4444"],
            [1.0, "#be123c"]
        ]
        # Benchmark: Radiant Electric Cool (Bright Aqua -> Sky Blue -> Violet -> Neon Rose)
        bright_bench_scale = [
            [0.0, "#67e8f9"],
            [0.35, "#38bdf8"],
            [0.7, "#a855f7"],
            [1.0, "#f43f5e"]
        ]
        colorscale = bright_bench_scale if is_benchmark else bright_primary_scale

        fig.add_trace(go.Scatter(
            x=sub["x_bin"],
            y=sub["y_bin"],
            mode="markers",
            marker=dict(
                symbol="square",
                size=16 if not half_rink else 20,
                color=sub["density_pct"],
                colorscale=colorscale,
                showscale=True,
                colorbar=dict(
                    title=dict(text=t("hit_panel_rate_col"), font=dict(color="white", size=10)),
                    tickfont=dict(color="white"),
                    thickness=10,
                    len=0.7,
                    y=0.5
                ),
                line=dict(color="rgba(255,255,255,0.22)", width=0.5)
            ),
            text=hover_texts,
            hoverinfo="text",
            name=f"{label} {t('hit_th_total_hits')}"
        ))

    prefix = f"{t('hit_val_benchmark')}: " if is_benchmark else f"{t('hit_val_selected_focus')}: "
    fig.update_layout(
        title=f"{prefix}{label} ({season_text})",
        title_font_size=13,
        height=420,
        margin=dict(l=10, r=10, t=35, b=10)
    )
    return fig

def create_shot_outcome_decision_tree(df_goals, selected_zone="All", highlight=None):
    """
    Renders an expansive, structured Decision Tree diagram for 30-second post-hit shot outcomes:
    Origin Hit Zone -> Possession Branch (For/Against) -> Terminal Shot Outcome (Goal, SOG, Miss, Block).
    Supports interactive category highlighting and dimming:
    When a box/category is clicked, same-category boxes are highlighted (full opacity, thick vibrant border)
    and others become slightly transparent/dimmed.
    """
    data = df_goals.copy()
    norm_zone = normalize_zone(selected_zone)
    if norm_zone != "All":
        data = data[data["hit_zone"] == norm_zone]

    zones = ["O", "D", "N"] if norm_zone == "All" else [norm_zone]
    zone_names = {
        "O": t("hit_tree_zone_o"),
        "D": t("hit_tree_zone_d"),
        "N": t("hit_tree_zone_n"),
        "All": t("hit_tree_opt_all")
    }
    zone_border_colors = {
        "O": "#38bdf8",  # Sky blue
        "D": "#f59e0b",  # Amber
        "N": "#a855f7"   # Purple
    }

    sides = ["Hitting Team (For)", "Opponent (Against)"]
    side_names = {
        "Hitting Team (For)": t("hit_tree_side_for"),
        "Opponent (Against)": t("hit_tree_side_agst")
    }
    side_border_colors = {
        "Hitting Team (For)": "#10b981",  # Emerald
        "Opponent (Against)": "#ef4444"   # Rose
    }

    event_order = ["goal", "shot-on-goal", "missed-shot", "blocked-shot"]
    event_names = {
        "goal": t("hit_tree_event_goal"),
        "shot-on-goal": t("hit_tree_event_sog"),
        "missed-shot": t("hit_tree_event_miss"),
        "blocked-shot": t("hit_tree_event_block")
    }
    event_border_colors = {
        "goal": "#facc15",        # Bright Gold
        "shot-on-goal": "#38bdf8",# Cyan
        "missed-shot": "#94a3b8", # Slate
        "blocked-shot": "#fb923c" # Coral Orange
    }

    total_shots = len(data)

    # Determine active highlight filters
    hl_layer = highlight.get("layer") if highlight else None
    hl_cat = highlight.get("category") if highlight else None

    # Build hierarchical tree data
    leaves = []
    for z in zones:
        z_df = data[data["hit_zone"] == z]
        z_count = len(z_df)
        z_pct = (z_count / total_shots * 100) if total_shots > 0 else 0

        for s in sides:
            s_df = z_df[z_df["outcome_side"] == s]
            s_count = len(s_df)
            s_pct_z = (s_count / z_count * 100) if z_count > 0 else 0
            s_avg_time = s_df["delta_sec"].mean() if s_count > 0 else 0
            goals_count = len(s_df[s_df["is_goal"] == 1])
            goal_conv = (goals_count / s_count * 100) if s_count > 0 else 0

            for e in event_order:
                e_df = s_df[s_df["event_type"] == e]
                e_count = len(e_df)
                e_pct_s = (e_count / s_count * 100) if s_count > 0 else 0
                e_avg_time = e_df["delta_sec"].mean() if e_count > 0 else 0

                leaves.append({
                    "zone": z,
                    "zone_name": zone_names.get(z, t("hit_tree_zone_o") if "O" in str(z) else (t("hit_tree_zone_d") if "D" in str(z) else t("hit_tree_zone_n"))),
                    "side": s,
                    "event_type": e,
                    "event_name": event_names[e],
                    "count": e_count,
                    "pct_s": e_pct_s,
                    "avg_time": e_avg_time,
                    "z_count": z_count,
                    "z_pct": z_pct,
                    "s_count": s_count,
                    "s_pct_z": s_pct_z,
                    "s_avg_time": s_avg_time,
                    "goal_conv": goal_conv
                })

    n_leaves = len(leaves)
    y_gap = 48
    for i, leaf in enumerate(leaves):
        leaf["y"] = (n_leaves - 1 - i) * y_gap

    # Calculate side node positions
    side_nodes = {}
    for leaf in leaves:
        key = (leaf["zone"], leaf["side"])
        if key not in side_nodes:
            side_nodes[key] = {
                "zone": leaf["zone"],
                "side": leaf["side"],
                "count": leaf["s_count"],
                "pct_z": leaf["s_pct_z"],
                "avg_time": leaf["s_avg_time"],
                "goal_conv": leaf["goal_conv"],
                "y_vals": []
            }
        side_nodes[key]["y_vals"].append(leaf["y"])

    for key, s_node in side_nodes.items():
        s_node["y"] = float(np.mean(s_node["y_vals"]))

    # Calculate zone node positions
    zone_nodes = {}
    for key, s_node in side_nodes.items():
        z = s_node["zone"]
        if z not in zone_nodes:
            zone_nodes[z] = {
                "zone": z,
                "name": zone_names.get(z, t("hit_tree_zone_o") if "O" in str(z) else (t("hit_tree_zone_d") if "D" in str(z) else t("hit_tree_zone_n"))),
                "count": [l["z_count"] for l in leaves if l["zone"] == z][0],
                "pct": [l["z_pct"] for l in leaves if l["zone"] == z][0],
                "y_vals": []
            }
        zone_nodes[z]["y_vals"].append(s_node["y"])

    for z, z_node in zone_nodes.items():
        z_node["y"] = float(np.mean(z_node["y_vals"]))

    # Stage X coordinates
    x_zone = 0.0
    x_side = 1.0
    x_event = 2.1

    fig = go.Figure()

    # Connecting branch lines: Zone -> Side
    edge_x1, edge_y1 = [], []
    edge_x1_hl, edge_y1_hl = [], []
    for key, s_node in side_nodes.items():
        z_node = zone_nodes[s_node["zone"]]
        x0, y0 = x_zone + 0.35, z_node["y"]
        x1, y1 = x_side - 0.35, s_node["y"]
        xm = (x0 + x1) / 2

        is_hl = False
        if hl_layer == 1 and hl_cat == s_node["zone"]:
            is_hl = True
        elif hl_layer == 2 and hl_cat == s_node["side"]:
            is_hl = True
        elif hl_layer == 3:
            is_hl = True  # Upper levels remain active when comparing terminal outcome

        if is_hl:
            edge_x1_hl.extend([x0, xm, xm, x1, None])
            edge_y1_hl.extend([y0, y0, y1, y1, None])
        else:
            edge_x1.extend([x0, xm, xm, x1, None])
            edge_y1.extend([y0, y0, y1, y1, None])

    # Base/dimmed Zone->Side lines
    fig.add_trace(go.Scatter(
        x=edge_x1, y=edge_y1,
        mode="lines",
        line=dict(color="rgba(148, 163, 184, 0.15)" if hl_layer else "rgba(148, 163, 184, 0.4)", width=1.5 if hl_layer else 2.5),
        hoverinfo="skip",
        showlegend=False
    ))
    # Highlighted Zone->Side lines
    if edge_x1_hl:
        fig.add_trace(go.Scatter(
            x=edge_x1_hl, y=edge_y1_hl,
            mode="lines",
            line=dict(color="rgba(56, 189, 248, 0.7)", width=2.8),
            hoverinfo="skip",
            showlegend=False
        ))

    # Connecting branch lines: Side -> Event
    edge_x2, edge_y2 = [], []
    edge_x2_hl, edge_y2_hl = [], []
    for leaf in leaves:
        s_node = side_nodes[(leaf["zone"], leaf["side"])]
        x0, y0 = x_side + 0.35, s_node["y"]
        x1, y1 = x_event - 0.05, leaf["y"]
        xm = (x0 + x1) / 2

        is_hl = False
        if hl_layer == 3 and hl_cat == leaf["event_type"]:
            is_hl = True
        elif hl_layer == 2 and hl_cat == leaf["side"]:
            is_hl = True
        elif hl_layer == 1 and hl_cat == leaf["zone"]:
            is_hl = True

        if is_hl:
            edge_x2_hl.extend([x0, xm, xm, x1, None])
            edge_y2_hl.extend([y0, y0, y1, y1, None])
        else:
            edge_x2.extend([x0, xm, xm, x1, None])
            edge_y2.extend([y0, y0, y1, y1, None])

    # Base/dimmed Side->Event lines
    fig.add_trace(go.Scatter(
        x=edge_x2, y=edge_y2,
        mode="lines",
        line=dict(color="rgba(148, 163, 184, 0.1)" if hl_layer else "rgba(148, 163, 184, 0.3)", width=1 if hl_layer else 1.8),
        hoverinfo="skip",
        showlegend=False
    ))
    # Highlighted Side->Event lines
    if edge_x2_hl:
        fig.add_trace(go.Scatter(
            x=edge_x2_hl, y=edge_y2_hl,
            mode="lines",
            line=dict(color="rgba(250, 204, 21, 0.85)" if (hl_layer == 3 and hl_cat == "goal") else "rgba(56, 189, 248, 0.75)", width=2.8),
            hoverinfo="skip",
            showlegend=False
        ))

    # Annotations for Decision Tree Nodes
    annotations = []

    # 1. Zone Node Cards
    for z in zone_nodes.values():
        bcolor = zone_border_colors[z["zone"]]
        is_active = (hl_layer is None) or (hl_layer == 1 and hl_cat == z["zone"]) or (hl_layer in (2, 3))

        bgcolor = "#1e293b" if is_active else "rgba(15, 23, 42, 0.4)"
        border_c = bcolor if is_active else "rgba(148, 163, 184, 0.2)"
        border_w = 3 if (hl_layer == 1 and hl_cat == z["zone"]) else (2 if is_active else 1)
        font_c = "white" if is_active else "rgba(148, 163, 184, 0.3)"
        sub_c = "#cbd5e1" if is_active else "rgba(148, 163, 184, 0.2)"

        annotations.append(dict(
            x=x_zone, y=z["y"],
            text=f"<b>{z['name']}</b><br><span style='font-size:10px;color:{sub_c}'>{t('hit_th_total_30s')}: <b>{z['count']:,}</b> ({z['pct']:.1f}%)</span>",
            showarrow=False,
            xanchor="center", yanchor="middle",
            bgcolor=bgcolor,
            bordercolor=border_c,
            borderwidth=border_w,
            borderpad=6,
            font=dict(color=font_c, size=11),
            align="center"
        ))

    # 2. Side Node Cards (Possession Branch)
    for s in side_nodes.values():
        bcolor = side_border_colors[s["side"]]
        is_same_cat = (hl_layer == 2 and hl_cat == s["side"])
        is_active = (hl_layer is None) or is_same_cat or (hl_layer == 1 and hl_cat == s["zone"]) or (hl_layer == 3)

        bgcolor = ("#064e3b" if "For" in s["side"] else "#4c0519") if is_same_cat else ("#1e293b" if is_active else "rgba(15, 23, 42, 0.4)")
        border_c = bcolor if is_active else "rgba(148, 163, 184, 0.2)"
        border_w = 3 if is_same_cat else (2 if is_active else 1)
        font_c = "white" if is_active else "rgba(148, 163, 184, 0.3)"
        sub_c = "#cbd5e1" if is_active else "rgba(148, 163, 184, 0.2)"

        disp_side = side_names.get(s["side"], s["side"])
        annotations.append(dict(
            x=x_side, y=s["y"],
            text=f"<b>{disp_side}</b><br><span style='font-size:10px;color:{sub_c}'>{s['count']:,} {t('hit_th_30s_shots').lower()} ({s['pct_z']:.1f}%) | Avg: <b>{s['avg_time']:.1f}s</b></span>",
            showarrow=False,
            xanchor="center", yanchor="middle",
            bgcolor=bgcolor,
            bordercolor=border_c,
            borderwidth=border_w,
            borderpad=6,
            font=dict(color=font_c, size=11),
            align="center"
        ))

    # 3. Leaf Node Cards (Terminal Shot Outcome)
    for l in leaves:
        bcolor = event_border_colors[l["event_type"]]
        is_goal = (l["event_type"] == "goal")
        is_same_cat = (hl_layer == 3 and hl_cat == l["event_type"])
        is_parent_active = (hl_layer == 2 and hl_cat == l["side"]) or (hl_layer == 1 and hl_cat == l["zone"])
        is_active = (hl_layer is None) or is_same_cat or is_parent_active

        if is_same_cat:
            bgcolor = "#451a03" if is_goal else "#082f49"
            border_c = bcolor
            border_w = 3.5
            font_c = "#fde047" if is_goal else "#38bdf8"
            sub_c = "#e2e8f0"
        elif is_active:
            bgcolor = "#2e1065" if is_goal else "#0f172a"
            border_c = bcolor
            border_w = 2 if is_goal else 1.5
            font_c = "#fde047" if is_goal else "white"
            sub_c = "#94a3b8"
        else:  # Dimmed
            bgcolor = "rgba(15, 23, 42, 0.3)"
            border_c = "rgba(148, 163, 184, 0.15)"
            border_w = 1
            font_c = "rgba(148, 163, 184, 0.25)"
            sub_c = "rgba(148, 163, 184, 0.18)"

        goal_extra = t("hit_tree_high_danger") if is_goal else ""
        annotations.append(dict(
            x=x_event, y=l["y"],
            text=f"<b>{l['event_name']}{goal_extra}</b>: <b>{l['count']:,}</b> ({l['pct_s']:.1f}%) <span style='font-size:10px;color:{sub_c}'>| {l['avg_time']:.1f}s</span>",
            showarrow=False,
            xanchor="left", yanchor="middle",
            bgcolor=bgcolor,
            bordercolor=border_c,
            borderwidth=border_w,
            borderpad=5,
            font=dict(color=font_c, size=11),
            align="left"
        ))

    # Clickable invisible/semi-transparent scatter markers covering each node for interactive selection
    # Zone markers (Layer 1)
    fig.add_trace(go.Scatter(
        x=[x_zone] * len(zone_nodes),
        y=[z["y"] for z in zone_nodes.values()],
        mode="markers",
        marker=dict(symbol="square", size=36, color="rgba(255,255,255,0.01)", line=dict(width=0)),
        customdata=[[1, z["zone"], z["name"]] for z in zone_nodes.values()],
        hoverinfo="text",
        hovertext=[f"{t('hit_tree_stage1')}: {z['name']}" for z in zone_nodes.values()],
        showlegend=False
    ))

    # Side markers (Layer 2)
    fig.add_trace(go.Scatter(
        x=[x_side] * len(side_nodes),
        y=[s["y"] for s in side_nodes.values()],
        mode="markers",
        marker=dict(symbol="square", size=36, color="rgba(255,255,255,0.01)", line=dict(width=0)),
        customdata=[[2, s["side"], side_names.get(s["side"], s["side"])] for s in side_nodes.values()],
        hoverinfo="text",
        hovertext=[f"{t('hit_tree_stage2')}: {side_names.get(s['side'], s['side'])}" for s in side_nodes.values()],
        showlegend=False
    ))

    # Leaf markers (Layer 3)
    fig.add_trace(go.Scatter(
        x=[x_event + 0.3] * len(leaves),
        y=[l["y"] for l in leaves],
        mode="markers",
        marker=dict(symbol="square", size=36, color="rgba(255,255,255,0.01)", line=dict(width=0)),
        customdata=[[3, l["event_type"], l["event_name"]] for l in leaves],
        hoverinfo="text",
        hovertext=[f"{t('hit_tree_stage3')}: {l['event_name']}" for l in leaves],
        showlegend=False
    ))

    chart_height = max(460, n_leaves * y_gap + 80)

    base_title = t("hit_tree_title_base")
    hl_subtitle = ""
    if highlight:
        hl_subtitle = t("hit_tree_title_filter").format(label=highlight.get('label', ''))

    fig.update_layout(
        title=dict(
            text=f"{base_title}{hl_subtitle}",
            font=dict(color="white", size=15)
        ),
        annotations=annotations,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=chart_height,
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=True,
            tickmode="array", tickvals=[x_zone, x_side, x_event + 0.3],
            ticktext=[t("hit_tree_stage1"), t("hit_tree_stage2"), t("hit_tree_stage3")],
            tickfont=dict(color="white", size=12),
            range=[-0.5, 3.2]
        ),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-20, n_leaves * y_gap + 20]),
        margin=dict(l=40, r=40, t=50, b=40)
    )

    return fig

def get_decision_matrix(df_goals, selected_zone="All"):
    """Generates tabular summary of the decision tree branches."""
    data = df_goals.copy()
    norm_zone = normalize_zone(selected_zone)
    if norm_zone != "All":
        data = data[data["hit_zone"] == norm_zone]

    zones = ["O", "D", "N"] if norm_zone == "All" else [norm_zone]
    zone_names = {
        "O": t("hit_tree_zone_o"),
        "D": t("hit_tree_zone_d"),
        "N": t("hit_tree_zone_n"),
        "All": t("hit_tree_opt_all")
    }
    side_map = {
        "Hitting Team (For)": t("hit_tree_side_for"),
        "Opponent (Against)": t("hit_tree_side_agst")
    }

    rows = []
    for z in zones:
        z_df = data[data["hit_zone"] == z]
        z_total = len(z_df)
        if z_total == 0:
            continue
        for s in ["Hitting Team (For)", "Opponent (Against)"]:
            s_df = z_df[z_df["outcome_side"] == s]
            s_count = len(s_df)
            if s_count == 0:
                continue
            pct_z = s_count / z_total * 100
            avg_t = s_df["delta_sec"].mean()
            goals = len(s_df[s_df["event_type"] == "goal"])
            sog = len(s_df[s_df["event_type"] == "shot-on-goal"])
            miss = len(s_df[s_df["event_type"] == "missed-shot"])
            block = len(s_df[s_df["event_type"] == "blocked-shot"])

            rows.append({
                t("hit_th_origin_zone"): zone_names.get(z, t("hit_tree_zone_o") if "O" in str(z) else (t("hit_tree_zone_d") if "D" in str(z) else t("hit_tree_zone_n"))),
                t("hit_th_possession_branch"): side_map.get(s, s),
                t("hit_th_30s_shots"): s_count,
                t("hit_th_branch_share"): pct_z,
                t("hit_th_avg_elapsed"): avg_t,
                t("hit_th_goals"): goals,
                t("hit_th_goal_conv"): goals / s_count * 100,
                t("hit_th_sog_pct"): sog / s_count * 100,
                t("hit_th_miss_pct"): miss / s_count * 100,
                t("hit_th_blocked_pct"): block / s_count * 100
            })
    return pd.DataFrame(rows)

def get_layer_3_comparison(df_goals, event_type, selected_zone="All"):
    """
    Comparative tabulation across TWO higher levels for a Terminal Shot Outcome:
    Level 2: Possession Branch (Hitting Team vs Opponent)
    Level 1: Hit Origin Zone (Forecheck, D-Zone, Neutral Zone)
    """
    data = df_goals.copy()
    norm_zone = normalize_zone(selected_zone)
    if norm_zone != "All":
        data = data[data["hit_zone"] == norm_zone]

    zones = ["O", "D", "N"] if norm_zone == "All" else [norm_zone]
    zone_map = {
        "O": t("hit_tree_zone_o"),
        "D": t("hit_tree_zone_d"),
        "N": t("hit_tree_zone_n"),
        "All": t("hit_tree_opt_all")
    }
    side_map = {
        "Hitting Team (For)": t("hit_tree_side_for"),
        "Opponent (Against)": t("hit_tree_side_agst")
    }

    rows = []
    for z in zones:
        z_df = data[data["hit_zone"] == z]
        z_total = len(z_df)
        if z_total == 0:
            continue
        for s in ["Hitting Team (For)", "Opponent (Against)"]:
            s_df = z_df[z_df["outcome_side"] == s]
            s_total = len(s_df)
            e_df = s_df[s_df["event_type"] == event_type]
            e_count = len(e_df)
            pct_branch = (e_count / s_total * 100) if s_total > 0 else 0
            pct_zone = (e_count / z_total * 100) if z_total > 0 else 0
            avg_time = e_df["delta_sec"].mean() if e_count > 0 else 0
            rows.append({
                t("hit_th_origin_zone"): zone_map.get(z, t("hit_tree_zone_o") if "O" in str(z) else (t("hit_tree_zone_d") if "D" in str(z) else t("hit_tree_zone_n"))),
                t("hit_th_possession_branch"): side_map.get(s, s),
                t("hit_th_event_count"): e_count,
                t("hit_th_branch_share"): pct_branch,
                t("hit_th_zone_share"): pct_zone,
                t("hit_th_avg_elapsed"): avg_time
            })
    return pd.DataFrame(rows)

def get_layer_2_comparison(df_goals, outcome_side, selected_zone="All"):
    """
    Comparative tabulation across ONE higher level for a Possession Branch:
    Level 1: Hit Origin Zone (Forecheck, D-Zone, Neutral Zone)
    """
    data = df_goals.copy()
    norm_zone = normalize_zone(selected_zone)
    if norm_zone != "All":
        data = data[data["hit_zone"] == norm_zone]

    zones = ["O", "D", "N"] if norm_zone == "All" else [norm_zone]
    zone_map = {
        "O": t("hit_tree_zone_o"),
        "D": t("hit_tree_zone_d"),
        "N": t("hit_tree_zone_n"),
        "All": t("hit_tree_opt_all")
    }

    rows = []
    for z in zones:
        z_df = data[data["hit_zone"] == z]
        z_total = len(z_df)
        if z_total == 0:
            continue
        s_df = z_df[z_df["outcome_side"] == outcome_side]
        s_count = len(s_df)
        pct_z = (s_count / z_total * 100) if z_total > 0 else 0
        avg_t = s_df["delta_sec"].mean() if s_count > 0 else 0
        goals = len(s_df[s_df["event_type"] == "goal"])
        sog = len(s_df[s_df["event_type"] == "shot-on-goal"])
        miss = len(s_df[s_df["event_type"] == "missed-shot"])
        block = len(s_df[s_df["event_type"] == "blocked-shot"])
        rows.append({
            t("hit_th_origin_zone"): zone_map.get(z, t("hit_tree_zone_o") if "O" in str(z) else (t("hit_tree_zone_d") if "D" in str(z) else t("hit_tree_zone_n"))),
            t("hit_th_30s_shots"): s_count,
            t("hit_th_zone_poss_share"): pct_z,
            t("hit_th_avg_elapsed"): avg_t,
            t("hit_th_goals"): goals,
            t("hit_th_goal_conv"): (goals / s_count * 100) if s_count > 0 else 0,
            t("hit_th_sog_pct"): (sog / s_count * 100) if s_count > 0 else 0,
            t("hit_th_miss_pct"): (miss / s_count * 100) if s_count > 0 else 0,
            t("hit_th_blocked_pct"): (block / s_count * 100) if s_count > 0 else 0
        })
    return pd.DataFrame(rows)

def get_layer_1_comparison(df_goals, zone_code):
    """
    Comparative tabulation for an Origin Zone compared across all zones.
    """
    norm_target = normalize_zone(zone_code)
    target_code = norm_target if norm_target in ["O", "D", "N"] else "O"
    zone_map = {
        "O": t("hit_tree_zone_o"),
        "D": t("hit_tree_zone_d"),
        "N": t("hit_tree_zone_n")
    }
    rows = []
    for z in ["O", "D", "N"]:
        z_df = df_goals[df_goals["hit_zone"] == z]
        z_total = len(z_df)
        for_df = z_df[z_df["outcome_side"] == "Hitting Team (For)"]
        agst_df = z_df[z_df["outcome_side"] == "Opponent (Against)"]
        for_goals = len(for_df[for_df["is_goal"] == 1])
        agst_goals = len(agst_df[agst_df["is_goal"] == 1])

        rows.append({
            t("hit_th_origin_zone"): zone_map.get(z, z),
            t("hit_th_total_30s"): z_total,
            t("hit_th_for_shots"): len(for_df),
            t("hit_th_agst_shots"): len(agst_df),
            t("hit_th_puck_win"): (len(for_df) / z_total * 100) if z_total > 0 else 0,
            t("hit_th_goals_for"): for_goals,
            t("hit_th_goals_agst"): agst_goals,
            t("hit_th_net_goal_diff"): for_goals - agst_goals,
            t("hit_th_status"): t("hit_val_selected_focus") if z == target_code else t("hit_val_benchmark")
        })
    return pd.DataFrame(rows)

def show():
    st.title(t("hit_title"))
    st.markdown(t("hit_subtitle"))

    df_kpi, df_seq, df_goals, df_grid, df_team_kpi, df_team_spatial, team_name_map = load_all_data()

    if any(x is None for x in [df_kpi, df_seq, df_goals, df_grid, df_team_kpi, df_team_spatial]):
        st.error(t("hit_data_not_found"))
        return

    # Top Executive Metric Ribbon
    c1, c2, c3, c4 = st.columns(4)
    total_hits = df_kpi["total_hits"].sum()
    ozone_hits = df_kpi[df_kpi["hit_zone"] == "O"]["total_hits"].values[0]
    dzone_hits = df_kpi[df_kpi["hit_zone"] == "D"]["total_hits"].values[0]
    total_goals_30s = df_kpi["goals_for"].sum() + df_kpi["goals_against"].sum()

    c1.metric(t("hit_metric_hits"), f"{total_hits:,}")
    c2.metric(t("hit_metric_ozone"), f"{ozone_hits:,}", t("hit_metric_ozone_delta"))
    c3.metric(t("hit_metric_dzone"), f"{dzone_hits:,}", t("hit_metric_dzone_delta"), delta_color="inverse")
    c4.metric(t("hit_metric_goals"), f"{int(total_goals_30s):,}", t("hit_metric_goals_delta"))

    st.markdown("---")

    # Two-Layer Main Analysis Navigation
    main_tabs = st.tabs([
        t("hit_layer_sequence"),
        t("hit_layer_rink"),
        t("hit_layer_tactics")
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 1 - TAB 1: SEQUENCE
    # ══════════════════════════════════════════════════════════════════════════
    with main_tabs[0]:
        with st.expander(t("hit_exp_seq_title"), expanded=False):
            st.markdown(f"""
            {t("hit_exp_seq_intro")}

            **{t("hit_exp_seq_sec1_title")}**

            {t("hit_exp_seq_sec1_p1")}
            """)

            st.latex(r"\operatorname{eventId}(E_k) = \operatorname{eventId}(E_0) + k \quad \text{where} \quad \operatorname{period}(E_k) = P, \quad \operatorname{gameId}(E_k) = G")

            st.markdown(f"""
            {t("hit_exp_seq_sec1_p2")}
            """)

            st.latex(r"""
            O_k = \begin{cases} 
            \text{Hitting Team}, & \text{if } \operatorname{team}(E_k) = T_{\text{hit}} \\ 
            \text{Opposing Team}, & \text{if } \operatorname{team}(E_k) = T_{\text{recv}} \\ 
            \text{Neutral / Stoppage}, & \text{otherwise} 
            \end{cases}
            """)

            st.markdown(f"""
            {t("hit_exp_seq_sec1_p3")}
            """)

            st.latex(r"P(C_k = c, O_k = o \mid Z_0) = \frac{1}{N_{Z_0}} \sum_{i=1}^{N_{Z_0}} \mathbb{I}(C_{k, i} = c \ \land \ O_{k, i} = o)")

            st.markdown(f"""
            {t("hit_exp_seq_sec1_p4")}

            **{t("hit_exp_seq_sec2_title")}**

            {t("hit_exp_seq_sec2_p1")}
            """)

            st.latex(r"\Omega(E_0) = \left\{ E_j \ \middle|\ \operatorname{eventId}(E_j) > \operatorname{eventId}(E_0), \ \operatorname{period}(E_j) = P, \ 0 \le t_j - t_0 \le 30 \right\}")

            st.markdown(f"""
            {t("hit_exp_seq_sec2_p2")}
            """)

            st.latex(r"\text{Shots For / 100 Hits} = \frac{100}{N} \sum_{i=1}^{N} S_{\text{for}}(E_{0, i}), \quad \text{Shots Against / 100 Hits} = \frac{100}{N} \sum_{i=1}^{N} S_{\text{against}}(E_{0, i})")

            st.latex(r"\text{Net Shot Differential} = \frac{100}{N} \sum_{i=1}^{N} \Big( S_{\text{for}}(E_{0, i}) - S_{\text{against}}(E_{0, i}) \Big)")

            st.latex(r"\text{Net Goal Rate / 100 Hits} = \frac{100}{N} \sum_{i=1}^{N} \Big( G_{\text{for}}(E_{0, i}) - G_{\text{against}}(E_{0, i}) \Big)")

            st.markdown(f"""
            **{t("hit_exp_seq_sec3_title")}**

            {t("hit_exp_seq_sec3_p1")}
            """)

        subtab_seq, subtab_goal = st.tabs([
            t("hit_tab_seq"),
            t("hit_tab_goal")
        ])

        # ──────────────────────────────────────────────────────────────────────
        # SUBTAB 1.1: Sequence Statistics (+1 to +4)
        # ──────────────────────────────────────────────────────────────────────
        with subtab_seq:
            st.markdown(t("hit_sub_seq_scope"))
            st.subheader(t("hit_sub_seq_heading"))
            st.markdown(t("hit_sub_seq_intro"))

            zone_col, step_col = st.columns([1, 1])
            with zone_col:
                cur_l = st.session_state.get("lang", "EN")
                prev_zone = st.session_state.get("hit_zone_scope_val", "All")
                zone_opts = ["All", "O", "D", "N"]
                z_idx = zone_opts.index(prev_zone) if prev_zone in zone_opts else 0
                selected_zone = st.selectbox(
                    t("hit_zone_scope_label"),
                    zone_opts,
                    format_func=lambda x, l=cur_l: t("hit_tree_opt_all", lang=l) if x == "All" else (t("hit_tree_opt_o", lang=l) if x == "O" else (t("hit_tree_opt_d", lang=l) if x == "D" else t("hit_tree_opt_n", lang=l))),
                    index=z_idx,
                    key=f"hit_zone_scope_{cur_l}"
                )
                st.session_state.hit_zone_scope_val = selected_zone
            with step_col:
                selected_step = st.slider(t("hit_event_step_label"), 1, 4, 1)

            # Flow Diagram
            st.plotly_chart(create_sankey_sequence(df_seq, selected_zone), width="stretch")

            step_df = df_seq[df_seq["event_step"] == selected_step].copy()
            if selected_zone != "All":
                step_df = step_df[step_df["hit_zone"] == selected_zone]

            st.markdown(f"#### {t('hit_step_dist_heading').format(step=selected_step)}")
            cat_display_map = {
                "Shot Attempt": t("hit_cat_shot"),
                "Goal": t("hit_cat_goal"),
                "Possession Change": t("hit_cat_poss"),
                "Physical Battle": t("hit_cat_battle"),
                "Stoppage": t("hit_cat_stop"),
                "Other": t("hit_cat_other")
            }
            owner_display_map = {
                "Hitting Team": t("hit_owner_for"),
                "Opponent": t("hit_owner_against"),
                "Neutral / Stoppage": t("hit_owner_neutral")
            }
            step_df_plot = step_df.copy()
            step_df_plot["category_disp"] = step_df_plot["event_category"].map(cat_display_map).fillna(step_df_plot["event_category"])
            step_df_plot["owner_disp"] = step_df_plot["event_owner"].map(owner_display_map).fillna(step_df_plot["event_owner"])

            bar_data = step_df_plot.groupby(["category_disp", "owner_disp"])["count"].sum().reset_index()
            fig_bar = px.bar(
                bar_data,
                x="category_disp",
                y="count",
                color="owner_disp",
                barmode="group",
                color_discrete_map={t("hit_owner_for"): "#10b981", t("hit_owner_against"): "#ef4444", t("hit_owner_neutral"): "#94a3b8"},
                labels={"category_disp": t("hit_chart_sub_cat"), "count": t("hit_chart_freq"), "owner_disp": t("hit_chart_initiated")}
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                height=380,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_bar, width="stretch")

        # ──────────────────────────────────────────────────────────────────────
        # SUBTAB 1.2: 30s Goal & Shot Dynamics
        # ──────────────────────────────────────────────────────────────────────
        with subtab_goal:
            st.markdown(t("hit_sub_goal_scope"))
            st.subheader(t("hit_sub_goal_heading"))
            st.markdown(t("hit_sub_goal_intro"))

            st.markdown(f"#### {t('hit_zone_conversion_heading')}")
            kpi_display = df_kpi[[
                "zone_name", "total_hits", "immediate_possession_pct",
                "shots_for_per_100_hits", "shots_against_per_100_hits", "net_shot_diff_per_100",
                "goals_for", "goals_against"
            ]].copy()
            col_ctx = t("hit_th_zone_context")
            col_hits = t("hit_th_total_hits")
            col_pwin = t("hit_th_possession_win_pct")
            col_s100_f = t("hit_th_shots_for_100")
            col_s100_a = t("hit_th_shots_agst_100")
            col_net_s = t("hit_th_net_shot_diff")
            col_gf = t("hit_th_goals_for")
            col_ga = t("hit_th_goals_agst")

            kpi_display.columns = [
                col_ctx, col_hits, col_pwin,
                col_s100_f, col_s100_a, col_net_s,
                col_gf, col_ga
            ]
            st.dataframe(kpi_display.style.format({
                col_hits: "{:,}",
                col_pwin: "{:.1f}%",
                col_s100_f: "{:.2f}",
                col_s100_a: "{:.2f}",
                col_net_s: "{:+.2f}",
                col_gf: "{:.0f}",
                col_ga: "{:.0f}"
            }), width="stretch")

            st.markdown(f"#### {t('hit_time_elapsed_heading')}")
            fig_hist = px.histogram(
                df_goals,
                x="delta_sec",
                color="outcome_side",
                nbins=30,
                barmode="overlay",
                color_discrete_map={"Hitting Team (For)": "#10b981", "Opponent (Against)": "#ef4444"},
                labels={"delta_sec": t("hit_hist_x_label"), "outcome_side": t("hit_hist_color_label")}
            )
            fig_hist.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="white"),
                height=320,
                margin=dict(l=10, r=10, t=20, b=10)
            )
            st.plotly_chart(fig_hist, width="stretch")

            st.markdown("---")
            st.markdown(f"#### {t('hit_decision_tree_heading')}")
            st.markdown(t("hit_decision_tree_desc"))

            tree_col1, tree_col2 = st.columns([2, 1])
            with tree_col1:
                cur_l = st.session_state.get("lang", "EN")
                prev_tree_zone = st.session_state.get("select_tree_zone_scope_val", "All")
                tree_zone_opts = ["All", "O", "D", "N"]
                tz_idx = tree_zone_opts.index(prev_tree_zone) if prev_tree_zone in tree_zone_opts else 0
                selected_tree_zone = st.selectbox(
                    t("hit_tree_select_label"),
                    tree_zone_opts,
                    format_func=lambda x, l=cur_l: t("hit_tree_opt_all", lang=l) if x == "All" else (t("hit_tree_opt_o", lang=l) if x == "O" else (t("hit_tree_opt_d", lang=l) if x == "D" else t("hit_tree_opt_n", lang=l))),
                    index=tz_idx,
                    key=f"select_tree_zone_scope_{cur_l}"
                )
                if selected_tree_zone != prev_tree_zone:
                    # Scope changed: if a layer-1 zone was highlighted that doesn't match the new scope, reset it
                    if st.session_state.get("tree_highlight") and st.session_state["tree_highlight"].get("layer") == 1:
                        if selected_tree_zone != "All" and st.session_state["tree_highlight"].get("category") != selected_tree_zone:
                            st.session_state.tree_highlight = None
                            st.session_state.tree_highlight_source = None
                            st.session_state.last_clicked_tree_pt = None
                    st.session_state.select_tree_zone_scope_val = selected_tree_zone

            # Initialize session state for interactive tree highlighting
            if "tree_highlight" not in st.session_state:
                st.session_state.tree_highlight = None
            if "last_clicked_tree_pt" not in st.session_state:
                st.session_state.last_clicked_tree_pt = None
            if "tree_highlight_source" not in st.session_state:
                st.session_state.tree_highlight_source = None

            # Category filter buttons in order of tree hierarchy
            st.success(t("hit_btn_success_desc"))

            st.markdown("""
            <style>
            /* Equal size and typography for decision tree category buttons */
            div[class*="st-key-btn_tree_"] button {
                width: 100% !important;
                min-height: 42px !important;
                font-weight: 700 !important;
                font-size: 13px !important;
                border-radius: 6px !important;
                text-align: center !important;
                padding: 6px 4px !important;
                transition: transform 0.15s ease, filter 0.15s ease !important;
            }
            div[class*="st-key-btn_tree_"] button:hover {
                transform: translateY(-1px) !important;
                filter: brightness(1.1) !important;
            }

            /* 1. Reset View: Slate neutral */
            div.st-key-btn_tree_reset button {
                background-color: #334155 !important;
                color: #f8fafc !important;
                border: 1.5px solid #64748b !important;
            }

            /* 2. Stage 2 - Hitting Team (For): Bright Emerald (#10b981) */
            div.st-key-btn_tree_for button {
                background-color: #10b981 !important;
                color: #ffffff !important;
                border: 1.5px solid #34d399 !important;
            }

            /* 3. Stage 2 - Opponent (Against): Bright Rose / Red (#ef4444) */
            div.st-key-btn_tree_agst button {
                background-color: #ef4444 !important;
                color: #ffffff !important;
                border: 1.5px solid #f87171 !important;
            }

            /* 4. Stage 3 - Goals: Bright Gold (#facc15) */
            div.st-key-btn_tree_goal button {
                background-color: #facc15 !important;
                color: #0f172a !important;
                border: 1.5px solid #fde047 !important;
            }

            /* 5. Stage 3 - Shots on Goal: Bright Cyan (#38bdf8) */
            div.st-key-btn_tree_sog button {
                background-color: #38bdf8 !important;
                color: #0f172a !important;
                border: 1.5px solid #7dd3fc !important;
            }

            /* 6. Stage 3 - Missed Shots: Bright Slate (#94a3b8) */
            div.st-key-btn_tree_miss button {
                background-color: #94a3b8 !important;
                color: #0f172a !important;
                border: 1.5px solid #cbd5e1 !important;
            }

            /* 7. Stage 3 - Blocked Shots: Bright Coral Orange (#fb923c) */
            div.st-key-btn_tree_block button {
                background-color: #fb923c !important;
                color: #0f172a !important;
                border: 1.5px solid #fdba74 !important;
            }
            </style>
            """, unsafe_allow_html=True)

            chart_key = f"tree_chart_{selected_tree_zone}"

            # Callback for category button clicks to guarantee state updates before rerun
            def on_tree_ribbon_click(layer, cat, lbl, ck):
                if layer == "clear":
                    st.session_state.tree_highlight = None
                    st.session_state.tree_highlight_source = None
                    st.session_state.last_clicked_tree_pt = None
                else:
                    st.session_state.tree_highlight = {"layer": layer, "category": cat, "label": lbl}
                    st.session_state.tree_highlight_source = "button"
                    st.session_state.last_clicked_tree_pt = f"{layer}_{cat}"
                if ck in st.session_state and isinstance(st.session_state[ck], dict):
                    st.session_state[ck] = {"selection": {"points": []}}

            # Button ribbon ordered by Tree Hierarchy:
            # Stage 2: Hitting Team (For) -> Opponent (Against)
            # Stage 3: Goals -> Shots on Goal -> Missed Shots -> Blocked Shots
            btn_cols = st.columns(7)
            with btn_cols[0]:
                st.button(t("hit_btn_reset"), key="btn_tree_reset", width="stretch",
                          on_click=on_tree_ribbon_click, args=("clear", None, None, chart_key))
            with btn_cols[1]:
                st.button(t("hit_btn_for"), key="btn_tree_for", width="stretch",
                          on_click=on_tree_ribbon_click, args=(2, "Hitting Team (For)", t("hit_btn_for"), chart_key))
            with btn_cols[2]:
                st.button(t("hit_btn_agst"), key="btn_tree_agst", width="stretch",
                          on_click=on_tree_ribbon_click, args=(2, "Opponent (Against)", t("hit_btn_agst"), chart_key))
            with btn_cols[3]:
                st.button(t("hit_btn_goal"), key="btn_tree_goal", width="stretch",
                          on_click=on_tree_ribbon_click, args=(3, "goal", t("hit_btn_goal"), chart_key))
            with btn_cols[4]:
                st.button(t("hit_btn_sog"), key="btn_tree_sog", width="stretch",
                          on_click=on_tree_ribbon_click, args=(3, "shot-on-goal", t("hit_btn_sog"), chart_key))
            with btn_cols[5]:
                st.button(t("hit_btn_miss"), key="btn_tree_miss", width="stretch",
                          on_click=on_tree_ribbon_click, args=(3, "missed-shot", t("hit_btn_miss"), chart_key))
            with btn_cols[6]:
                st.button(t("hit_btn_block"), key="btn_tree_block", width="stretch",
                          on_click=on_tree_ribbon_click, args=(3, "blocked-shot", t("hit_btn_block"), chart_key))

            # Check for chart point click in session state (populated by Streamlit before script execution)
            chart_state = st.session_state.get(chart_key)
            if chart_state and isinstance(chart_state, dict):
                sel = chart_state.get("selection", {})
                pts = sel.get("points", [])
                if pts:
                    cd = pts[0].get("customdata")
                    if cd and len(cd) >= 3 and cd[0] is not None and cd[1] is not None:
                        click_id = f"{cd[0]}_{cd[1]}"
                        st.session_state.last_clicked_tree_pt = click_id
                        st.session_state.tree_highlight_source = "chart"
                        st.session_state.tree_highlight = {
                            "layer": int(cd[0]),
                            "category": str(cd[1]),
                            "label": str(cd[2])
                        }
                else:
                    # Only deselect if the current highlight was initiated by a chart click
                    if st.session_state.get("tree_highlight_source") == "chart":
                        st.session_state.last_clicked_tree_pt = None
                        st.session_state.tree_highlight = None
                        st.session_state.tree_highlight_source = None

            # Render Decision Tree with Plotly and on_select="rerun"
            st.plotly_chart(
                create_shot_outcome_decision_tree(df_goals, selected_tree_zone, highlight=st.session_state.tree_highlight),
                width="stretch",
                on_select="rerun",
                selection_mode=["points"],
                key=chart_key
            )

            # Common column labels for comparative tabulations
            col_zone = t("hit_th_origin_zone")
            col_branch = t("hit_th_possession_branch")
            col_count = t("hit_th_event_count")
            col_b_share = t("hit_th_branch_share")
            col_z_share = t("hit_th_zone_share")
            col_time = t("hit_th_avg_elapsed")

            # Render Comparative Tabulation or Standard Decision Matrix
            hl = st.session_state.tree_highlight
            if hl and hl.get("layer") == 3:
                # Terminal shot outcome highlighted -> Compare across TWO higher levels: Possession Branch and Hit Origin Zone
                cat_type = hl["category"]
                cat_label_map = {
                    "goal": t("hit_btn_goal"),
                    "shot-on-goal": t("hit_btn_sog"),
                    "missed-shot": t("hit_btn_miss"),
                    "blocked-shot": t("hit_btn_block")
                }
                cat_label = cat_label_map.get(cat_type, hl.get("label", cat_type))
                st.markdown(f"##### {t('hit_comp_l3_heading').format(label=cat_label)}")
                st.info(t("hit_comp_l3_info").format(label=cat_label))

                df_comp3 = get_layer_3_comparison(df_goals, cat_type, selected_tree_zone)
                for_cnt = int(df_comp3[df_comp3[col_branch] == t("hit_tree_side_for")][col_count].sum()) if not df_comp3.empty and col_branch in df_comp3 else 0
                agst_cnt = int(df_comp3[df_comp3[col_branch] == t("hit_tree_side_agst")][col_count].sum()) if not df_comp3.empty and col_branch in df_comp3 else 0
                net_diff = for_cnt - agst_cnt

                active_events = df_comp3[df_comp3[col_count] > 0] if not df_comp3.empty and col_count in df_comp3 else pd.DataFrame()
                if not active_events.empty:
                    fastest_idx = active_events[col_time].idxmin()
                    fastest_row = active_events.loc[fastest_idx]
                    fastest_desc = f"{fastest_row[col_zone]} ({fastest_row[col_time]:.1f}s)"
                else:
                    fastest_desc = "N/A"

                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric(t("hit_comp_l3_m1"), f"{for_cnt:,}")
                m_col2.metric(t("hit_comp_l3_m2"), f"{agst_cnt:,}")
                m_col3.metric(t("hit_comp_l3_m3"), f"{net_diff:+,}")
                m_col4.metric(t("hit_comp_l3_m4"), fastest_desc)

                st.dataframe(df_comp3.style.format({
                    col_count: "{:,}",
                    col_b_share: "{:.1f}%",
                    col_z_share: "{:.1f}%",
                    col_time: "{:.1f}s"
                }), width="stretch")

            elif hl and hl.get("layer") == 2:
                # Possession branch highlighted -> Compare across ONE higher level: Hit Origin Zone
                cat_side = hl["category"]
                side_label_map = {
                    "Hitting Team (For)": t("hit_btn_for"),
                    "Opponent (Against)": t("hit_btn_agst")
                }
                cat_label = side_label_map.get(cat_side, hl.get("label", cat_side))
                st.markdown(f"##### {t('hit_comp_l2_heading').format(label=cat_label)}")
                st.info(t("hit_comp_l2_info").format(label=cat_label))

                df_comp2 = get_layer_2_comparison(df_goals, cat_side, selected_tree_zone)
                col_shots = t("hit_th_30s_shots")
                col_goals = t("hit_th_goals")
                tot_shots = int(df_comp2[col_shots].sum()) if not df_comp2.empty and col_shots in df_comp2 else 0
                tot_goals = int(df_comp2[col_goals].sum()) if not df_comp2.empty and col_goals in df_comp2 else 0
                w_time = (df_comp2[col_time] * df_comp2[col_shots]).sum() / tot_shots if tot_shots > 0 else 0
                top_zone = df_comp2.sort_values(col_shots, ascending=False).iloc[0][col_zone] if not df_comp2.empty and col_shots in df_comp2 else "N/A"

                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric(t("hit_comp_l2_m1"), f"{tot_shots:,}")
                m_col2.metric(t("hit_comp_l2_m2"), f"{tot_goals:,}")
                m_col3.metric(t("hit_comp_l2_m3"), f"{w_time:.1f}s")
                m_col4.metric(t("hit_comp_l2_m4"), top_zone)

                st.dataframe(df_comp2.style.format({
                    col_shots: "{:,}",
                    t("hit_th_zone_poss_share"): "{:.1f}%",
                    col_time: "{:.1f}s",
                    col_goals: "{:,}",
                    t("hit_th_goal_conv"): "{:.1f}%",
                    t("hit_th_sog_pct"): "{:.1f}%",
                    t("hit_th_miss_pct"): "{:.1f}%",
                    t("hit_th_blocked_pct"): "{:.1f}%"
                }), width="stretch")

            elif hl and hl.get("layer") == 1:
                # Hit origin zone highlighted -> Compare against all zones
                cat_zone = normalize_zone(hl["category"])
                zone_label_map = {
                    "O": t("hit_tree_zone_o"),
                    "D": t("hit_tree_zone_d"),
                    "N": t("hit_tree_zone_n")
                }
                cat_label = zone_label_map.get(cat_zone, hl.get("label", cat_zone))
                st.markdown(f"##### {t('hit_comp_l1_heading').format(label=cat_label)}")
                st.info(t("hit_comp_l1_info").format(label=cat_label))

                col_tot_30s = t("hit_th_total_30s")
                col_status = t("hit_th_status")
                col_for_s = t("hit_th_for_shots")
                col_agst_s = t("hit_th_agst_shots")
                col_puck = t("hit_th_puck_win")
                col_net_g = t("hit_th_net_goal_diff")

                df_comp1 = get_layer_1_comparison(df_goals, cat_zone)
                sel_mask = df_comp1[col_status] == t("hit_val_selected_focus")
                sel_row = df_comp1[sel_mask].iloc[0] if not df_comp1[sel_mask].empty else df_comp1.iloc[0]

                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric(t("hit_comp_l1_m1"), f"{int(sel_row[col_tot_30s]):,}")
                m_col2.metric(t("hit_comp_l1_m2"), f"{sel_row[col_puck]:.1f}%")
                m_col3.metric(t("hit_comp_l1_m3"), f"{int(sel_row[col_for_s] - sel_row[col_agst_s]):+,}")
                m_col4.metric(t("hit_comp_l1_m4"), f"{int(sel_row[col_net_g]):+,}")

                st.dataframe(df_comp1.style.format({
                    col_tot_30s: "{:,}",
                    col_for_s: "{:,}",
                    col_agst_s: "{:,}",
                    col_puck: "{:.1f}%",
                    t("hit_th_goals_for"): "{:,}",
                    t("hit_th_goals_agst"): "{:,}",
                    col_net_g: "{:+,}"
                }), width="stretch")

            else:
                # Default: No highlight active -> Display comprehensive Decision Matrix
                st.markdown(f"##### {t('hit_comp_default_heading')}")
                st.caption(t("hit_comp_default_tip"))
                matrix_df = get_decision_matrix(df_goals, selected_tree_zone)
                st.dataframe(matrix_df.style.format({
                    t("hit_th_30s_shots"): "{:,}",
                    t("hit_th_branch_share"): "{:.1f}%",
                    t("hit_th_avg_elapsed"): "{:.1f}s",
                    t("hit_th_goals"): "{:,}",
                    t("hit_th_goal_conv"): "{:.1f}%",
                    t("hit_th_sog_pct"): "{:.1f}%",
                    t("hit_th_miss_pct"): "{:.1f}%",
                    t("hit_th_blocked_pct"): "{:.1f}%"
                }), width="stretch")

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 1 - TAB 2: RINK LOCATION
    # ══════════════════════════════════════════════════════════════════════════
    with main_tabs[1]:
        with st.expander(t("hit_exp_spatial_title"), expanded=False):
            st.markdown(f"""
            {t("hit_exp_spatial_intro")}

            **{t("hit_exp_spatial_sec1_title")}**

            {t("hit_exp_spatial_sec1_p1")}
            """)

            st.latex(r"\tilde{x} = x_{\text{raw}} \cdot \delta(G, P, T), \quad \tilde{y} = y_{\text{raw}} \cdot \delta(G, P, T)")

            st.markdown(f"""
            {t("hit_exp_spatial_sec1_p2")}

            **{t("hit_exp_spatial_sec2_title")}**

            {t("hit_exp_spatial_sec2_p1")}
            """)

            st.latex(r"x_{\text{bin}} = 10 \cdot \left\lfloor \frac{\tilde{x}}{10} + 0.5 \right\rfloor, \quad y_{\text{bin}} = 10 \cdot \left\lfloor \frac{\tilde{y}}{10} + 0.5 \right\rfloor")

            st.markdown(f"""
            {t("hit_exp_spatial_sec2_p2")}
            """)

            st.latex(r"\mathcal{C} = \left\{ c_j = (x_{\text{bin}}, y_{\text{bin}}) \ \middle|\ |x_{\text{bin}}| \le 100, \ |y_{\text{bin}}| \le 42.5, \ \text{dist}\big(c_j, \partial\Omega_{\text{rink}}\big) \ge 0 \right\}")

            st.markdown(f"""
            **{t("hit_exp_spatial_sec3_title")}**

            {t("hit_exp_spatial_sec3_p1")}
            """)

            st.latex(r"P(c_j \mid \mathcal{H}) = \frac{1}{N(\mathcal{H})} \sum_{i=1}^{N(\mathcal{H})} \mathbb{I}\left( (\tilde{x}_i, \tilde{y}_i) \in c_j \right) \times 100\% \quad \text{such that} \quad \sum_{c_j \in \mathcal{C}} P(c_j \mid \mathcal{H}) = 100\%")

            st.markdown(f"""
            **{t("hit_exp_spatial_sec4_title")}**

            {t("hit_exp_spatial_sec4_p1")}
            """)

            st.latex(r"\text{Puck Win \%}(c_j) = \frac{1}{n(c_j)} \sum_{i=1}^{n(c_j)} \mathbb{I}\left( O_{1, i} = \text{Hitting Team} \right) \times 100\%")

            st.markdown(f"""
            {t("hit_exp_spatial_sec4_p2")}
            """)

            st.latex(r"\Delta S_{30}(c_j) = \sum_{i=1}^{n(c_j)} \Big( S_{\text{for}, i} - S_{\text{against}, i} \Big)")

            st.markdown(f"""
            **{t("hit_exp_spatial_sec5_title")}**

            {t("hit_exp_spatial_sec5_p1")}
            """)

            st.latex(r"\Delta P(c_j; T, B) = P(c_j \mid T, S) - P(c_j \mid B, S)")

            st.markdown(f"""
            {t("hit_exp_spatial_sec5_p2")}
            """)

            st.latex(r"\mathcal{C}_{\text{profile}}(T) = \left\{ c_j \in \mathcal{C} \ \middle|\ P(c_j \mid T, S) \ge 1.0\% \right\}")

            st.markdown(f"""
            {t("hit_exp_spatial_sec5_p3")}
            """)

        subtab_rink, subtab_team = st.tabs([
            t("hit_tab_rink"),
            t("hit_tab_team")
        ])

        # ──────────────────────────────────────────────────────────────────────
        # SUBTAB 2.1: Interactive Rink Heatmap & Inspection
        # ──────────────────────────────────────────────────────────────────────
        with subtab_rink:
            st.markdown(t("hit_sub_rink_scope"))
            st.subheader(t("hit_sub_rink_heading"))
            st.info(get_rink_text("hit_rink_click_instruction"))

            # Initialize session state for selected location
            if "selected_hit_loc" not in st.session_state:
                st.session_state.selected_hit_loc = None
            if "last_rink_clicked_pt" not in st.session_state:
                st.session_state.last_rink_clicked_pt = None

            rink_ctrl_1, rink_ctrl_2 = st.columns([1, 1])
            with rink_ctrl_1:
                cur_l = st.session_state.get("lang", "EN")
                prev_rink_view = st.session_state.get("heat_rink_view_val", "full")
                view_opts = ["full", "half"]
                v_idx = view_opts.index(prev_rink_view) if prev_rink_view in view_opts else 0

                view_mode = st.radio(
                    t("hit_rink_view_mode_label"),
                    view_opts,
                    format_func=lambda x, l=cur_l: t("hit_rink_full_label", lang=l) if x == "full" else t("hit_rink_half_label", lang=l),
                    index=v_idx,
                    horizontal=True,
                    key=f"heat_rink_view_{cur_l}"
                )
                st.session_state.heat_rink_view_val = view_mode

                # If in half rink mode and selected location is in defensive half (x < 0), clear selection
                if view_mode == "half" and st.session_state.selected_hit_loc is not None:
                    if st.session_state.selected_hit_loc[0] < 0:
                        st.session_state.selected_hit_loc = None
                        st.session_state.last_rink_clicked_pt = None
                        if "rink_heat_chart" in st.session_state and isinstance(st.session_state["rink_heat_chart"], dict):
                            st.session_state["rink_heat_chart"] = {"selection": {"points": []}}

            with rink_ctrl_2:
                hotspot_preset_coords = {
                    "o_corner": (90.0, -30.0),
                    "o_corner_l": (90.0, 30.0),
                    "behind_net": (100.0, 0.0),
                    "slot": (60.0, 0.0),
                    "center": (0.0, 0.0),
                    "d_corner": (-90.0, -30.0),
                    "d_corner_l": (-90.0, 30.0),
                    "d_net": (-80.0, 0.0)
                }
                preset_labels = {
                    "o_corner": t("hit_rink_preset_o_corner", lang=cur_l),
                    "o_corner_l": t("hit_rink_preset_o_corner_l", lang=cur_l),
                    "behind_net": t("hit_rink_preset_behind_net", lang=cur_l),
                    "slot": t("hit_rink_preset_slot", lang=cur_l),
                    "center": t("hit_rink_preset_center", lang=cur_l),
                    "d_corner": t("hit_rink_preset_d_corner", lang=cur_l),
                    "d_corner_l": t("hit_rink_preset_d_corner_l", lang=cur_l),
                    "d_net": t("hit_rink_preset_d_net", lang=cur_l)
                }
                preset_opts = [k for k, coords in hotspot_preset_coords.items() if (coords[0] >= 0 if view_mode == "half" else True)]
                prev_preset_key = st.session_state.get("select_rink_preset_key_val", "o_corner")
                pr_idx = preset_opts.index(prev_preset_key) if prev_preset_key in preset_opts else 0

                preset_key = st.selectbox(
                    t("hit_rink_quick_select"),
                    preset_opts,
                    format_func=lambda k, p=preset_labels: p.get(k, k),
                    index=pr_idx,
                    key=f"select_rink_preset_key_{cur_l}"
                )
                st.session_state.select_rink_preset_key_val = preset_key

                def on_apply_rink_preset(coords):
                    st.session_state.selected_hit_loc = coords
                    st.session_state.last_rink_clicked_pt = None
                    if "rink_heat_chart" in st.session_state and isinstance(st.session_state["rink_heat_chart"], dict):
                        st.session_state["rink_heat_chart"] = {"selection": {"points": []}}

                st.button(
                    t("hit_rink_preset_btn"),
                    key="btn_apply_rink_preset",
                    on_click=on_apply_rink_preset,
                    args=(hotspot_preset_coords[preset_key],)
                )

            # Process Reactive Chart Click from Session State (populated by Streamlit before script run)
            chart_state = st.session_state.get("rink_heat_chart")
            if chart_state and isinstance(chart_state, dict):
                sel = chart_state.get("selection", {})
                pts = sel.get("points", [])
                if pts:
                    pt = pts[0]
                    # Only accept clicks on curve 0 (the heatmap tiles trace)
                    if pt.get("curve_number", 0) == 0:
                        px_val = pt.get("x")
                        py_val = pt.get("y")
                        if px_val is not None and py_val is not None:
                            try:
                                clicked_pt = (float(px_val), float(py_val))
                                if ((df_grid["x_bin"] == clicked_pt[0]) & (df_grid["y_bin"] == clicked_pt[1])).any():
                                    st.session_state.selected_hit_loc = clicked_pt
                                    st.session_state.last_rink_clicked_pt = clicked_pt
                            except (ValueError, TypeError):
                                pass
                else:
                    # User clicked chart background to deselect (only if prior selection was from chart)
                    if st.session_state.get("last_rink_clicked_pt") is not None:
                        st.session_state.selected_hit_loc = None
                        st.session_state.last_rink_clicked_pt = None

            # Render Heatmap figure (all tiles visible and clickable, min_density=0.0)
            fig_rink = render_interactive_rink_heatmap(
                df_grid,
                df_goals,
                selected_loc=st.session_state.selected_hit_loc,
                half_rink=(view_mode == "half"),
                min_density=0.0
            )

            # Reactive click selection via Streamlit plotly_chart on_select
            st.plotly_chart(
                fig_rink,
                width="stretch",
                on_select="rerun",
                selection_mode=["points"],
                key="rink_heat_chart"
            )

            # Detailed Inspection Panel for Selected Location
            if st.session_state.selected_hit_loc is not None:
                sel_x, sel_y = st.session_state.selected_hit_loc
                matched_cell = df_grid[(df_grid["x_bin"] == sel_x) & (df_grid["y_bin"] == sel_y)]

                if not matched_cell.empty:
                    cell = matched_cell.iloc[0]
                    zone_desc = t("hit_tree_zone_o") if cell["spatial_zone"] == "O" else (t("hit_tree_zone_d") if cell["spatial_zone"] == "D" else t("hit_tree_zone_n"))

                    def on_clear_rink_selection():
                        st.session_state.selected_hit_loc = None
                        st.session_state.last_rink_clicked_pt = None
                        if "rink_heat_chart" in st.session_state and isinstance(st.session_state["rink_heat_chart"], dict):
                            st.session_state["rink_heat_chart"] = {"selection": {"points": []}}

                    col_title, col_clear = st.columns([4, 1])
                    with col_title:
                        st.markdown(f"### {t('hit_rink_inspected_loc').format(x=f'{sel_x:+.0f}', y=f'{sel_y:+.0f}', zone=zone_desc)}")
                    with col_clear:
                        st.button(
                            get_rink_text("hit_rink_clear_btn"),
                            key="btn_clear_rink_sel",
                            on_click=on_clear_rink_selection
                        )

                    st.markdown(
                        f"""
                        <div style="background-color: rgba(15, 23, 42, 0.75); border: 1px solid rgba(56, 189, 248, 0.4); border-left: 4px solid #38bdf8; border-radius: 8px; padding: 14px 18px; margin-bottom: 16px;">
                            <div style="font-size: 1.02rem; font-weight: 600; color: #f8fafc; margin-bottom: 6px;">
                                {get_rink_text('hit_rink_text_heading').format(x=f'{sel_x:+.0f}', y=f'{sel_y:+.0f}', zone=zone_desc)}
                            </div>
                            <div style="color: #cbd5e1; font-size: 0.92rem; line-height: 1.6;">
                                {get_rink_text('hit_rink_text_body').format(
                                    x=f'{sel_x:+.0f}',
                                    y=f'{sel_y:+.0f}',
                                    zone=zone_desc,
                                    hits=f"{int(cell['total_hits']):,}",
                                    pct=f"{cell['density_pct']:.2f}",
                                    win=f"{cell['puck_win_pct']:.1f}",
                                    win_diff=f"{cell['puck_win_pct'] - 46.2:+.1f}",
                                    s_for=f"{cell['shot_for_pct']:.1f}",
                                    s_agst=f"{cell['shot_against_pct']:.1f}",
                                    net_shots=f"{int(cell['net_shots_30s']):+d}",
                                    g_for=int(cell['goals_for_30s']),
                                    g_agst=int(cell['goals_against_30s'])
                                )}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric(t("hit_panel_m1_label"), f"{int(cell['total_hits']):,}", f"{cell['density_pct']}%")
                    m2.metric(t("hit_panel_m2_label"), f"{cell['puck_win_pct']:.1f}%", f"{cell['puck_win_pct'] - 46.2:+.1f}%")
                    m3.metric(t("hit_panel_m3_label"), f"{cell['shot_for_pct']:.1f}%")
                    m4.metric(t("hit_panel_m4_label"), f"{cell['shot_against_pct']:.1f}%", delta_color="inverse")
                    m5.metric(t("hit_panel_m5_label"), f"{int(cell['net_shots_30s']):+d}", t("hit_panel_m5_delta").format(g_for=int(cell['goals_for_30s']), g_agst=int(cell['goals_against_30s'])))

                    # Subsequent event categorical breakdown bar chart
                    st.markdown(f"#### {t('hit_rink_sub_dist_heading').format(x=f'{sel_x:+.0f}', y=f'{sel_y:+.0f}')}")
                    cat_col = t("hit_panel_category_col")
                    rate_col = t("hit_panel_rate_col")
                    side_col = t("hit_panel_side_col")
                    cat_data = pd.DataFrame([
                        {cat_col: t("hit_panel_shots_for"), rate_col: cell["shot_for_pct"], side_col: t("hit_owner_for")},
                        {cat_col: t("hit_panel_takeaways_won"), rate_col: cell["takeaway_for_pct"], side_col: t("hit_owner_for")},
                        {cat_col: t("hit_panel_goals_scored"), rate_col: cell["goal_for_pct"], side_col: t("hit_owner_for")},
                        {cat_col: t("hit_panel_shots_conceded"), rate_col: cell["shot_against_pct"], side_col: t("hit_owner_against")},
                        {cat_col: t("hit_panel_giveaways_committed"), rate_col: cell["giveaway_against_pct"], side_col: t("hit_owner_against")},
                        {cat_col: t("hit_panel_goals_conceded"), rate_col: cell["goal_against_pct"], side_col: t("hit_owner_against")},
                        {cat_col: t("hit_panel_phys_battle"), rate_col: cell["physical_battle_pct"], side_col: t("hit_panel_neutral_battle")},
                        {cat_col: t("hit_panel_stoppage"), rate_col: cell["stoppage_pct"], side_col: t("hit_panel_neutral_battle")}
                    ])

                    fig_loc_bar = px.bar(
                        cat_data,
                        x=cat_col,
                        y=rate_col,
                        color=side_col,
                        color_discrete_map={t("hit_owner_for"): "#10b981", t("hit_owner_against"): "#ef4444", t("hit_panel_neutral_battle"): "#94a3b8"},
                        text_auto=".1f"
                    )
                    fig_loc_bar.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white"),
                        height=340,
                        margin=dict(l=10, r=10, t=20, b=10)
                    )
                    st.plotly_chart(fig_loc_bar, width="stretch")
                else:
                    st.info(t("hit_rink_no_data").format(x=sel_x, y=sel_y))
            else:
                st.info(get_rink_text("hit_rink_select_prompt"))

        # ──────────────────────────────────────────────────────────────────────
        # SUBTAB 2.2: Team & Season Profiles (Drops anywhere < 1.0%)
        # ──────────────────────────────────────────────────────────────────────
        with subtab_team:
            st.markdown(t("hit_sub_team_scope"))

            st.subheader(t("hit_sub_team_heading"))
            st.markdown(t("hit_sub_team_desc"))

            team_list = sorted([tm for tm in df_team_kpi["team"].unique() if tm != "LEAGUE_AVG"])
            team_options = [team_name_map.get(tm, tm) for tm in team_list]
            tri_to_opt = {tm: team_name_map.get(tm, tm) for tm in team_list}
            opt_to_tri = {v: k for k, v in tri_to_opt.items()}

            col_s, col_t1, col_t2 = st.columns(3)
            with col_s:
                cur_l = st.session_state.get("lang", "EN")
                season_labels = {
                    "ALL": t("hit_season_all", lang=cur_l),
                    "20252026": t("hit_season_20252026", lang=cur_l),
                    "20242025": t("hit_season_20242025", lang=cur_l)
                }
                prev_season_choice = st.session_state.get("select_team_season_choice_val", "ALL")
                season_choice_opts = list(season_labels.keys())
                s_idx = season_choice_opts.index(prev_season_choice) if prev_season_choice in season_choice_opts else 0

                selected_season = st.selectbox(
                    t("hit_team_sel_season_label"),
                    season_choice_opts,
                    format_func=lambda k, s=season_labels: s.get(k, k),
                    index=s_idx,
                    key=f"select_team_season_choice_{cur_l}"
                )
                st.session_state.select_team_season_choice_val = selected_season
                selected_season_label = season_labels[selected_season]

            with col_t1:
                prev_primary = st.session_state.get("select_primary_team_tri_val", "FLA" if "FLA" in team_list else (team_list[0] if team_list else "FLA"))
                p_idx = team_list.index(prev_primary) if prev_primary in team_list else (team_list.index("FLA") if "FLA" in team_list else 0)

                primary_tri = st.selectbox(
                    t("hit_team_sel_primary"),
                    team_list,
                    format_func=lambda tm, m=team_name_map: m.get(tm, tm),
                    index=p_idx,
                    key=f"select_primary_team_tri_{cur_l}"
                )
                st.session_state.select_primary_team_tri_val = primary_tri
                primary_choice = team_name_map.get(primary_tri, primary_tri)

            with col_t2:
                bench_options = ["LEAGUE_AVG"] + [tm for tm in team_list if tm != primary_tri]
                prev_bench = st.session_state.get("select_bench_team_tri_val", "LEAGUE_AVG")
                b_idx = bench_options.index(prev_bench) if prev_bench in bench_options else 0

                benchmark_tri = st.selectbox(
                    t("hit_team_sel_bench"),
                    bench_options,
                    format_func=lambda tm, l=cur_l, m=team_name_map: t("hit_team_bench_league", lang=l) if tm == "LEAGUE_AVG" else m.get(tm, tm),
                    index=b_idx,
                    key=f"select_bench_team_tri_{cur_l}"
                )
                st.session_state.select_bench_team_tri_val = benchmark_tri
                benchmark_label = t("hit_team_bench_league", lang=cur_l) if benchmark_tri == "LEAGUE_AVG" else team_name_map.get(benchmark_tri, benchmark_tri)

            # Retrieve Team Metrics
            t1_data = df_team_kpi[(df_team_kpi["team"] == primary_tri) & (df_team_kpi["season"] == selected_season)]
            t2_data = df_team_kpi[(df_team_kpi["team"] == benchmark_tri) & (df_team_kpi["season"] == selected_season)]

            if not t1_data.empty and not t2_data.empty:
                r1 = t1_data.iloc[0]
                r2 = t2_data.iloc[0]

                st.markdown(f"#### {t('hit_team_h2h_title')}: **{primary_choice}** vs. **{benchmark_label}**")

                k1, k2, k3, k4, k5 = st.columns(5)
                d_hpg = r1["hits_per_game"] - r2["hits_per_game"]
                d_o = r1["ozone_pct"] - r2["ozone_pct"]
                d_puck = r1["puck_win_pct"] - r2["puck_win_pct"]
                d_net_shots = r1["net_shots_per_100"] - r2["net_shots_per_100"]
                d_net_goals = r1["net_goals_per_100"] - r2["net_goals_per_100"]

                vs_bench_str = t("hit_vs_bench")
                k1.metric(t("hit_team_card_hpg_label"), f"{r1['hits_per_game']:.1f}", f"{d_hpg:+.1f} {vs_bench_str}")
                k2.metric(t("hit_team_card_ozone_label"), f"{r1['ozone_pct']:.1f}%", f"{d_o:+.1f}% {vs_bench_str}")
                k3.metric(t("hit_team_card_win_label"), f"{r1['puck_win_pct']:.1f}%", f"{d_puck:+.1f}% {vs_bench_str}")
                k4.metric(t("hit_team_card_shots_label"), f"{r1['net_shots_per_100']:+.2f}", f"{d_net_shots:+.2f} {vs_bench_str}")
                k5.metric(t("hit_team_card_goals_label"), f"{r1['net_goals_per_100']:+.3f}", f"{d_net_goals:+.3f} {vs_bench_str}")

                st.markdown("---")

                # Side-by-side Spatial Heatmaps (drops < 1.0%)
                st.markdown(f"#### {t('hit_team_spatial_comp_heading')}")
                h_col1, h_col2 = st.columns(2)
                with h_col1:
                    fig_t1 = render_team_rink_heatmap(
                        df_team_spatial, primary_tri, selected_season,
                        display_name=primary_choice,
                        is_benchmark=False,
                        season_label=selected_season_label,
                        min_density=1.0
                    )
                    st.plotly_chart(fig_t1, width="stretch")
                with h_col2:
                    fig_t2 = render_team_rink_heatmap(
                        df_team_spatial, benchmark_tri, selected_season,
                        display_name=benchmark_label,
                        is_benchmark=True,
                        season_label=selected_season_label,
                        min_density=1.0
                    )
                    st.plotly_chart(fig_t2, width="stretch")

                # League-wide 32-Team Scatter Plot
                st.markdown(f"#### {t('hit_team_landscape_heading')}")
                league_teams = df_team_kpi[
                    (df_team_kpi["season"] == selected_season) & (df_team_kpi["team"] != "LEAGUE_AVG")
                ].copy()

                if not league_teams.empty:
                    league_teams["team_label"] = league_teams["team"].map(team_name_map).fillna(league_teams["team"])
                    league_teams["is_selected"] = league_teams["team"] == primary_tri

                    fig_scatter = px.scatter(
                        league_teams,
                        x="ozone_pct",
                        y="puck_win_pct",
                        size="total_hits",
                        color="net_shots_per_100",
                        color_continuous_scale="RdYlGn",
                        hover_name="team_label",
                        hover_data={
                            "total_hits": ":,",
                            "hits_per_game": ":.1f",
                            "ozone_pct": ":.1f%",
                            "puck_win_pct": ":.1f%",
                            "net_shots_per_100": ":+.2f",
                            "is_selected": False
                        },
                        labels={
                            "ozone_pct": t("hit_scatter_x_label"),
                            "puck_win_pct": t("hit_scatter_y_label"),
                            "net_shots_per_100": t("hit_scatter_color_label"),
                            "total_hits": t("hit_scatter_total_hits"),
                            "hits_per_game": t("hit_scatter_hits_per_game")
                        },
                        title=t("hit_team_scatter_title_full").format(season=selected_season_label)
                    )

                    # Highlight the primary selected team
                    t1_row = league_teams[league_teams["team"] == primary_tri]
                    if not t1_row.empty:
                        fig_scatter.add_trace(go.Scatter(
                            x=[t1_row["ozone_pct"].values[0]],
                            y=[t1_row["puck_win_pct"].values[0]],
                            mode="markers+text",
                            marker=dict(symbol="star", size=22, color="#facc15", line=dict(color="black", width=1.5)),
                            text=[f"  [{primary_tri}]"],
                            textposition="top right",
                            textfont=dict(color="#facc15", size=13),
                            name=f"{t('hit_scatter_selected_prefix')} {primary_tri}"
                        ))

                    fig_scatter.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white"),
                        height=480,
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    st.plotly_chart(fig_scatter, width="stretch")
            else:
                st.warning(f"No metric data available for {primary_choice} or {benchmark_label} for season {selected_season_label}.")

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 1 - TAB 3: TACTICAL INSIGHTS
    # ══════════════════════════════════════════════════════════════════════════
    with main_tabs[2]:
        st.markdown(t("hit_sub_tactics_scope"))

        st.subheader(t("hit_sub_tactics_heading"))
        st.markdown(f"""
        ### {t('hit_tactics_sec1_title')}
        {t('hit_tactics_sec1_p1')}
        {t('hit_tactics_sec1_p2')}
        {t('hit_tactics_sec1_p3')}

        ---

        ### {t('hit_tactics_sec2_title')}
        {t('hit_tactics_sec2_p1')}
        {t('hit_tactics_sec2_p2')}
        {t('hit_tactics_sec2_p3')}

        ---

        ### {t('hit_tactics_sec3_title')}
        {t('hit_tactics_sec3_p1')}
        {t('hit_tactics_sec3_p2')}
        """)

show()