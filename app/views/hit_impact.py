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
    """Resolve absolute path to tabulated hit datasets prioritizing latest/action/hits."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(current_dir, "../.."))
    candidates = [
        os.path.join(repo_root, "latest/action/hits", filename),
        os.path.join(repo_root, "dev/decision_science/impact_of_hits/data", filename),
        os.path.join(current_dir, "../data/hit_eda", filename),
        os.path.join(repo_root, "dev/decision_science/hit_eda/data", filename)
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]

try:
    from exe.data_loader import load_hit_analytics_datasets as load_hf_hit_analytics
except ImportError:
    try:
        from data_loader import load_hit_analytics_datasets as load_hf_hit_analytics
    except ImportError:
        load_hf_hit_analytics = None

@st.cache_data(max_entries=1, ttl=3600)
def load_all_data():
    """
    Load tabulated sequence, KPI, grid, and team-season datasets.
    Primary: Hugging Face datasets (measurement/action/hits/*.parquet).
    Backup: local CSV files in latest/action/hits/ or app/data/hit_eda/.
    """
    if load_hf_hit_analytics is not None:
        try:
            res = load_hf_hit_analytics()
            if res[0] is not None and not res[0].empty:
                return res
        except Exception:
            pass

    # Backup: Local CSV files
    kpi_file = get_data_path("hit_zone_summary_kpi.csv")
    seq_file = get_data_path("tabulated_hit_sequences.csv")
    goals_file = get_data_path("hit_goal_window_30s.csv")
    grid_file = get_data_path("rink_grid_hit_sequences.csv")
    team_kpi_file = get_data_path("team_season_hit_kpi.csv")
    team_spatial_file = get_data_path("team_season_spatial_hits.csv")
    eda_summary_file = get_data_path("hit_zone_summary.csv")
    eda_events_file = get_data_path("hit_next_events.csv")
    trans_file = get_data_path("hit_sequence_transitions.csv")

    df_kpi = pd.read_csv(kpi_file) if os.path.exists(kpi_file) else None
    df_seq = pd.read_csv(seq_file) if os.path.exists(seq_file) else None
    df_goals = pd.read_csv(goals_file) if os.path.exists(goals_file) else None
    df_grid = pd.read_csv(grid_file) if os.path.exists(grid_file) else None
    df_team_kpi = pd.read_csv(team_kpi_file) if os.path.exists(team_kpi_file) else None
    df_team_spatial = pd.read_csv(team_spatial_file) if os.path.exists(team_spatial_file) else None
    df_eda_summary = pd.read_csv(eda_summary_file) if os.path.exists(eda_summary_file) else None
    df_eda_events = pd.read_csv(eda_events_file) if os.path.exists(eda_events_file) else None
    df_trans = pd.read_csv(trans_file) if os.path.exists(trans_file) else None

    # Load teamlist for full team names
    current_dir = os.path.dirname(os.path.abspath(__file__))
    teamlist_path = os.path.abspath(os.path.join(current_dir, "../data/teamlist.csv"))
    if os.path.exists(teamlist_path):
        tl = pd.read_csv(teamlist_path)
        team_name_map = dict(zip(tl["tricode"], tl["tricode"] + " - " + tl["team"]))
    else:
        team_name_map = {}
    team_name_map["ARI"] = "ARI - Arizona Coyotes"

    return df_kpi, df_seq, df_goals, df_grid, df_team_kpi, df_team_spatial, team_name_map, df_eda_summary, df_eda_events, df_trans

def create_eda_first_event_chart(df_events, selected_zone="All"):
    """
    Constructs a horizontal grouped bar chart showing the immediate first event categories
    (Step +1) by initiation ownership (Hitting Team vs. Opponent).
    """
    if df_events is None or df_events.empty:
        fig = go.Figure()
        fig.add_annotation(text="No event frequency data available", showarrow=False, font=dict(color="#94a3b8"))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280)
        return fig

    norm_zone = normalize_zone(selected_zone)
    if norm_zone != "All":
        filtered = df_events[df_events["zoneCode"] == norm_zone].copy()
    else:
        filtered = df_events.copy()

    # Consolidate by next_category and next_owner_rel across multi-season records
    if not filtered.empty:
        grouped = filtered.groupby(["next_category", "next_owner_rel"], as_index=False).apply(
            lambda g: pd.Series({
                "event_count": int(g["event_count"].sum()),
                "avg_delta_sec": round(float(np.average(g["avg_delta_sec"], weights=g["event_count"])), 1) if g["event_count"].sum() > 0 else 10.0,
                "median_delta_sec": round(float(np.average(g["median_delta_sec"], weights=g["event_count"])), 1) if g["event_count"].sum() > 0 else 10.0,
            }),
            include_groups=False
        )
        tot_zone_hits = grouped["event_count"].sum()
        grouped["pct_of_zone"] = (grouped["event_count"] / tot_zone_hits * 100.0).round(2) if tot_zone_hits > 0 else 0.0
        filtered = grouped

    cat_map = {
        "Shot Attempt": t("hit_cat_shot"),
        "Physical Battle": t("hit_cat_battle"),
        "Possession Change": t("hit_cat_poss"),
        "Stoppage": t("hit_cat_stop"),
        "Goal": t("hit_cat_goal"),
        "Penalty": t("hit_cat_penalty", default="Penalty")
    }
    owner_map = {
        "Hitting Team": t("hit_owner_for"),
        "Opponent": t("hit_owner_against")
    }

    filtered["cat_disp"] = filtered["next_category"].map(cat_map).fillna(filtered["next_category"])
    filtered["owner_disp"] = filtered["next_owner_rel"].map(owner_map).fillna(filtered["next_owner_rel"])

    order_df = filtered.groupby("cat_disp")["event_count"].sum().sort_values(ascending=True).reset_index()
    cat_order = list(order_df["cat_disp"])

    fig = px.bar(
        filtered,
        y="cat_disp",
        x="event_count",
        color="owner_disp",
        barmode="group",
        category_orders={"cat_disp": cat_order},
        color_discrete_map={
            t("hit_owner_for"): "#10b981",
            t("hit_owner_against"): "#ef4444"
        },
        labels={
            "cat_disp": t("hit_chart_sub_cat", default="Event Category"),
            "event_count": t("hit_chart_freq", default="Event Count"),
            "owner_disp": t("hit_chart_initiated", default="Initiating Team")
        },
        custom_data=["pct_of_zone", "median_delta_sec", "avg_delta_sec"]
    )

    cur_lang = st.session_state.get("lang", "EN")
    if cur_lang == "FR":
        hover_tmpl = (
            "<b>%{y}</b> (%{data.name})<br>"
            "Nombre d'événements : %{x:,}<br>"
            "Part de la zone : %{customdata[0]:.1f} %<br>"
            "Délai médian : %{customdata[1]:.1f} s<br>"
            "Délai moyen : %{customdata[2]:.1f} s<extra></extra>"
        )
    else:
        hover_tmpl = (
            "<b>%{y}</b> (%{data.name})<br>"
            "Event Count: %{x:,}<br>"
            "Share of Zone: %{customdata[0]:.1f}%<br>"
            "Median Latency: %{customdata[1]:.1f}s<br>"
            "Mean Latency: %{customdata[2]:.1f}s<extra></extra>"
        )

    fig.update_traces(hovertemplate=hover_tmpl)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_eda_latency_chart(df_events, selected_zone="All"):
    """
    Constructs a horizontal bar chart showing reaction duration (delta_t median seconds)
    from hit to next event for Hitting Team vs. Opponent.
    """
    if df_events is None or df_events.empty:
        fig = go.Figure()
        fig.add_annotation(text="No latency duration data available", showarrow=False, font=dict(color="#94a3b8"))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=280)
        return fig

    norm_zone = normalize_zone(selected_zone)
    if norm_zone != "All":
        filtered = df_events[df_events["zoneCode"] == norm_zone].copy()
    else:
        filtered = df_events.copy()

    # Consolidate by next_category and next_owner_rel across multi-season records
    if not filtered.empty:
        grouped = filtered.groupby(["next_category", "next_owner_rel"], as_index=False).apply(
            lambda g: pd.Series({
                "event_count": int(g["event_count"].sum()),
                "avg_delta_sec": round(float(np.average(g["avg_delta_sec"], weights=g["event_count"])), 1) if g["event_count"].sum() > 0 else 10.0,
                "median_delta_sec": round(float(np.average(g["median_delta_sec"], weights=g["event_count"])), 1) if g["event_count"].sum() > 0 else 10.0,
            }),
            include_groups=False
        )
        tot_zone_hits = grouped["event_count"].sum()
        grouped["pct_of_zone"] = (grouped["event_count"] / tot_zone_hits * 100.0).round(2) if tot_zone_hits > 0 else 0.0
        filtered = grouped

    cat_map = {
        "Shot Attempt": t("hit_cat_shot"),
        "Physical Battle": t("hit_cat_battle"),
        "Possession Change": t("hit_cat_poss"),
        "Stoppage": t("hit_cat_stop"),
        "Goal": t("hit_cat_goal"),
        "Penalty": t("hit_cat_penalty", default="Penalty")
    }
    owner_map = {
        "Hitting Team": t("hit_owner_for"),
        "Opponent": t("hit_owner_against")
    }

    filtered["cat_disp"] = filtered["next_category"].map(cat_map).fillna(filtered["next_category"])
    filtered["owner_disp"] = filtered["next_owner_rel"].map(owner_map).fillna(filtered["next_owner_rel"])

    order_df = filtered.groupby("cat_disp")["median_delta_sec"].mean().sort_values(ascending=True).reset_index()
    cat_order = list(order_df["cat_disp"])

    cur_lang = st.session_state.get("lang", "EN")
    dur_label = "Délai médian d'action (secondes)" if cur_lang == "FR" else "Median Duration (Seconds)"

    fig = px.bar(
        filtered,
        y="cat_disp",
        x="median_delta_sec",
        color="owner_disp",
        barmode="group",
        category_orders={"cat_disp": cat_order},
        color_discrete_map={
            t("hit_owner_for"): "#10b981",
            t("hit_owner_against"): "#ef4444"
        },
        labels={
            "cat_disp": t("hit_chart_sub_cat", default="Event Category"),
            "median_delta_sec": dur_label,
            "owner_disp": t("hit_chart_initiated", default="Initiating Team")
        },
        custom_data=["avg_delta_sec", "event_count"]
    )

    if cur_lang == "FR":
        hover_lat = (
            "<b>%{y}</b> (%{data.name})<br>"
            "Délai médian : %{x:.1f} s<br>"
            "Délai moyen : %{customdata[0]:.1f} s<br>"
            "Total actions : %{customdata[1]:,}<extra></extra>"
        )
    else:
        hover_lat = (
            "<b>%{y}</b> (%{data.name})<br>"
            "Median Latency: %{x:.1f}s<br>"
            "Average Latency: %{customdata[0]:.1f}s<br>"
            "Total Plays: %{customdata[1]:,}<extra></extra>"
        )

    fig.update_traces(hovertemplate=hover_lat)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=380,
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_eda_zone_split_charts(df_summary):
    """
    Constructs a dual-bar comparison chart illustrating:
    1) Puck Recovery % by Zone (vs 50% parity)
    2) Net Shots on Goal per 100 Hits (Diverging bar chart)
    """
    if df_summary is None or df_summary.empty:
        return go.Figure(), go.Figure()

    # Ensure aggregation across multi-season rows so exactly 1 bar and 1 label render per zone
    if len(df_summary["zoneCode"].drop_duplicates()) < len(df_summary):
        df_summary = aggregate_eda_summary(df_summary)

    cur_lang = st.session_state.get("lang", "EN")
    if cur_lang == "FR":
        zone_map = {"D": "Défensive (D)", "N": "Neutre (N)", "O": "Offensive (O)"}
        parity_text = "Parité 50 %"
        x_title = "Zone sur la glace"
        y_rec_title = "Récupération de rondelle (%)"
        y_sog_title = "Tirs nets / 100 mises en échec"
    else:
        zone_map = {"D": "Defensive (D)", "N": "Neutral (N)", "O": "Offensive (O)"}
        parity_text = "50% Parity"
        x_title = "Rink Zone"
        y_rec_title = "Puck Recovery %"
        y_sog_title = "Net SOG / 100 Hits"

    df_plot = df_summary.copy()
    df_plot["zone_name"] = df_plot["zoneCode"].map(zone_map)
    zone_order = [zone_map["D"], zone_map["N"], zone_map["O"]]

    # Chart A: Puck Recovery Rate
    fig_rec = px.bar(
        df_plot,
        x="zone_name",
        y="recovery_pct",
        category_orders={"zone_name": zone_order},
        color="recovery_pct",
        color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
        range_color=[35, 55],
        labels={"zone_name": x_title, "recovery_pct": y_rec_title},
        text=df_plot["recovery_pct"].apply(lambda x: f"{x:.1f}%")
    )
    fig_rec.add_hline(y=50, line_dash="dash", line_color="#94a3b8", annotation_text=parity_text, annotation_position="top left")
    fig_rec.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        coloraxis_showscale=False
    )
    if cur_lang == "FR":
        hover_rec = "<b>%{x}</b><br>Taux de récupération : %{y:.1f} %<extra></extra>"
    else:
        hover_rec = "<b>%{x}</b><br>Recovery Rate: %{y:.1f}%<extra></extra>"
    fig_rec.update_traces(textposition="outside", hovertemplate=hover_rec)

    # Chart B: Net Shot Differential per 100 Hits
    colors = ["#ef4444" if v < 0 else "#10b981" for v in df_plot["net_sog_per_100"]]
    if cur_lang == "FR":
        hover_sog = "<b>%{x}</b><br>Diff. net tirs / 100 : %{y:+.2f}<extra></extra>"
    else:
        hover_sog = "<b>%{x}</b><br>Net SOG / 100: %{y:+.2f}<extra></extra>"
    fig_sog = go.Figure(go.Bar(
        x=df_plot["zone_name"],
        y=df_plot["net_sog_per_100"],
        marker_color=colors,
        text=df_plot["net_sog_per_100"].apply(lambda x: f"{x:+.2f}"),
        textposition="outside",
        hovertemplate=hover_sog
    ))
    fig_sog.add_hline(y=0, line_dash="solid", line_color="#94a3b8")
    fig_sog.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=320,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title=y_sog_title,
        xaxis_title=x_title
    )

    return fig_rec, fig_sog


def create_eda_funnel_chart(df_summary, selected_zone="All"):
    """
    Constructs a bilateral conversion funnel comparing:
    Hits (100) -> Shots on Goal Generated -> Goals Scored for Hitting Team vs. Opponent.
    """
    if df_summary is None or df_summary.empty:
        return go.Figure()

    if len(df_summary["zoneCode"].drop_duplicates()) < len(df_summary):
        df_summary = aggregate_eda_summary(df_summary)

    norm_zone = normalize_zone(selected_zone)
    if norm_zone != "All":
        matching = df_summary[df_summary["zoneCode"] == norm_zone]
        if not matching.empty:
            row = matching.iloc[0]
            sog_for = row["sog_for_per_100"]
            sog_agst = row["sog_against_per_100"]
            goals_for = row["goals_for_per_100"]
            goals_agst = row["goals_against_per_100"]
        else:
            sog_for, sog_agst, goals_for, goals_agst = 0, 0, 0, 0
    else:
        tot_hits = df_summary["total_hits"].sum()
        sog_for = (df_summary["sog_for"].sum() / tot_hits) * 100 if tot_hits > 0 else 0
        sog_agst = (df_summary["sog_against"].sum() / tot_hits) * 100 if tot_hits > 0 else 0
        goals_for = (df_summary["goals_for"].sum() / tot_hits) * 100 if tot_hits > 0 else 0
        goals_agst = (df_summary["goals_against"].sum() / tot_hits) * 100 if tot_hits > 0 else 0

    cur_lang = st.session_state.get("lang", "EN")
    if cur_lang == "FR":
        stages = ["Mises en échec (Base 100)", "Tirs au but générés", "Buts marqués"]
        y_label = "Taux pour 100 mises en échec"
    else:
        stages = ["Hits Delivered (100)", "Shots on Goal Generated", "Goals Scored"]
        y_label = "Rate per 100 Hits"

    for_vals = [100.0, sog_for, goals_for]
    agst_vals = [100.0, sog_agst, goals_agst]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=t("hit_owner_for", default="Hitting Team"),
        x=stages,
        y=for_vals,
        marker_color="#10b981",
        text=[f"{v:.1f}" if i < 2 else f"{v:.2f}" for i, v in enumerate(for_vals)],
        textposition="outside"
    ))
    fig.add_trace(go.Bar(
        name=t("hit_owner_against", default="Opponent"),
        x=stages,
        y=agst_vals,
        marker_color="#ef4444",
        text=[f"{v:.1f}" if i < 2 else f"{v:.2f}" for i, v in enumerate(agst_vals)],
        textposition="outside"
    ))

    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=340,
        margin=dict(l=10, r=10, t=30, b=10),
        yaxis_title=y_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_eda_zone_exit_donut(df_summary):
    """
    Constructs an interactive donut chart showing Defensive Zone Exit vs. In-Zone Trap rates.
    """
    if df_summary is None or df_summary.empty:
        return go.Figure()

    if len(df_summary["zoneCode"].drop_duplicates()) < len(df_summary):
        df_summary = aggregate_eda_summary(df_summary)

    row_d = df_summary[df_summary["zoneCode"] == "D"]
    if row_d.empty:
        return go.Figure()
    r = row_d.iloc[0]

    exit_count = int(r["zone_exits"]) if pd.notna(r.get("zone_exits")) else 0
    tot_d = int(r["total_hits"])
    pinned_count = tot_d - exit_count
    exit_pct = (exit_count / tot_d) * 100
    pinned_pct = (pinned_count / tot_d) * 100

    cur_lang = st.session_state.get("lang", "EN")
    if cur_lang == "FR":
        labels = ["Sortie de zone propre (D → N/O)", "Enfermé en zone déf."]
        center_text = f"<b>{exit_pct:.1f} %</b><br>Taux de sortie"
    else:
        labels = ["Clean Zone Exit (D → N/O)", "Pinned In-Zone (Trapped)"]
        center_text = f"<b>{exit_pct:.1f}%</b><br>Exit Rate"

    values = [exit_count, pinned_count]
    colors = ["#10b981", "#ef4444"]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker_colors=colors,
        textinfo="percent+label",
        hoverinfo="label+value+percent"
    )])

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        annotations=[dict(text=center_text, x=0.5, y=0.5, font_size=18, showarrow=False)]
    )
    return fig


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

def create_sankey_sequence(df_trans=None, selected_zone="All", selected_step=1, df_seq_fallback=None):
    """
    Constructs an interactive multi-layer Sankey sequence transition diagram.
    Dynamically expands from Step +1 through Step +4 based on the user's selected step depth.
    """
    norm_zone = normalize_zone(selected_zone)
    cur_lang = st.session_state.get("lang", "EN")

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

    category_rgba_map = {
        "Shot Attempt": "rgba(59, 130, 246, 0.4)",
        "Goal": "rgba(16, 185, 129, 0.45)",
        "Possession Change": "rgba(245, 158, 11, 0.4)",
        "Physical Battle": "rgba(239, 68, 68, 0.4)",
        "Stoppage": "rgba(139, 92, 246, 0.4)",
        "Other": "rgba(100, 116, 139, 0.35)"
    }

    step_colors = ["#3b82f6", "#f59e0b", "#8b5cf6", "#10b981"]

    max_step = max(1, min(int(selected_step), 4))
    node_labels = []
    node_idx = {}
    node_colors = []
    sources, targets, values, colors = [], [], [], []

    def get_or_create_node(label, stage_type="cat", step_num=1):
        if label not in node_idx:
            node_idx[label] = len(node_labels)
            node_labels.append(label)
            if stage_type == "zone":
                node_colors.append("#ce0e2d")
            elif stage_type == "owner":
                if t("hit_owner_for") in label:
                    node_colors.append("#10b981")
                elif t("hit_owner_against") in label:
                    node_colors.append("#ef4444")
                else:
                    node_colors.append("#94a3b8")
            else:
                c = step_colors[(step_num - 1) % len(step_colors)]
                node_colors.append(c)
        return node_idx[label]

    has_trans = df_trans is not None and isinstance(df_trans, pd.DataFrame) and not df_trans.empty

    if has_trans:
        df_use = df_trans.copy()
        if norm_zone != "All" and "hit_zone" in df_use.columns:
            df_use = df_use[df_use["hit_zone"] == norm_zone]

        if max_step == 1:
            # 1-Step flow: Zone -> Step +1 Category -> Step +1 Owner
            s0 = df_use[(df_use["from_step"] == 0) & (df_use["to_step"] == 1)].groupby(["source_node", "target_node"])["count"].sum().reset_index()
            for _, r in s0.iterrows():
                z_name = zone_map.get(r["source_node"], r["source_node"])
                c_name = cat_map.get(r["target_node"], r["target_node"])
                u = get_or_create_node(f"{t('hit_th_origin_zone')}: {z_name}", "zone", 0)
                v = get_or_create_node(f"{t('hit_step_prefix')} +1: {c_name}", "cat", 1)
                sources.append(u)
                targets.append(v)
                values.append(int(r["count"]))
                colors.append("rgba(59, 130, 246, 0.4)")

            s1 = df_use[(df_use["from_step"] == 1) & (df_use["to_step"] == 11)].groupby(["source_node", "target_node"])["count"].sum().reset_index()
            for _, r in s1.iterrows():
                c_name = cat_map.get(r["source_node"], r["source_node"])
                o_name = owner_map.get(r["target_node"], r["target_node"])
                u_label = f"{t('hit_step_prefix')} +1: {c_name}"
                if u_label in node_idx:
                    u = node_idx[u_label]
                    v = get_or_create_node(f"{t('hit_step_prefix')} +1: {o_name}", "owner", 1)
                    sources.append(u)
                    targets.append(v)
                    values.append(int(r["count"]))
                    c = "rgba(16, 185, 129, 0.4)" if r["target_node"] == "Hitting Team" else ("rgba(239, 68, 68, 0.4)" if r["target_node"] == "Opponent" else "rgba(148, 163, 184, 0.4)")
                    colors.append(c)
        else:
            # Multi-step flow: Zone -> Step +1 -> ... -> Step +max_step
            for s in range(max_step):
                sub = df_use[(df_use["from_step"] == s) & (df_use["to_step"] == s + 1)].groupby(["source_node", "target_node"])["count"].sum().reset_index()
                for _, r in sub.iterrows():
                    if s == 0:
                        z_name = zone_map.get(r["source_node"], r["source_node"])
                        src_label = f"{t('hit_th_origin_zone')}: {z_name}"
                        src_type = "zone"
                    else:
                        c_src = cat_map.get(r["source_node"], r["source_node"])
                        src_label = f"{t('hit_step_prefix')} +{s}: {c_src}"
                        src_type = "cat"

                    c_tgt = cat_map.get(r["target_node"], r["target_node"])
                    tgt_label = f"{t('hit_step_prefix')} +{s+1}: {c_tgt}"

                    u = get_or_create_node(src_label, src_type, s)
                    v = get_or_create_node(tgt_label, "cat", s + 1)
                    sources.append(u)
                    targets.append(v)
                    values.append(int(r["count"]))
                    colors.append(category_rgba_map.get(r["target_node"], "rgba(59, 130, 246, 0.4)"))
    else:
        # Graceful fallback using df_seq_fallback (tabulated sequences)
        fb = df_seq_fallback.copy() if df_seq_fallback is not None else pd.DataFrame()
        if not fb.empty:
            if norm_zone != "All" and "hit_zone" in fb.columns:
                fb = fb[fb["hit_zone"] == norm_zone]
            s1_fb = fb[fb["event_step"] == 1].groupby(["hit_zone", "event_category"])["count"].sum().reset_index()
            for _, r in s1_fb.iterrows():
                z_name = zone_map.get(r["hit_zone"], r["hit_zone"])
                c_name = cat_map.get(r["event_category"], r["event_category"])
                u = get_or_create_node(f"{t('hit_th_origin_zone')}: {z_name}", "zone", 0)
                v = get_or_create_node(f"{t('hit_step_prefix')} +1: {c_name}", "cat", 1)
                sources.append(u)
                targets.append(v)
                values.append(int(r["count"]))
                colors.append("rgba(59, 130, 246, 0.4)")

            for s in range(1, max_step):
                cur_step_df = fb[fb["event_step"] == s].groupby("event_category")["count"].sum()
                next_step_df = fb[fb["event_step"] == s + 1].groupby("event_category")["count"].sum()
                tot_next = next_step_df.sum()
                for c1, cnt1 in cur_step_df.items():
                    u_label = f"{t('hit_step_prefix')} +{s}: {cat_map.get(c1, c1)}"
                    if u_label in node_idx and tot_next > 0:
                        u = node_idx[u_label]
                        for c2, cnt2 in next_step_df.items():
                            v = get_or_create_node(f"{t('hit_step_prefix')} +{s+1}: {cat_map.get(c2, c2)}", "cat", s + 1)
                            flow_val = int(round(cnt1 * (cnt2 / tot_next)))
                            if flow_val > 0:
                                sources.append(u)
                                targets.append(v)
                                values.append(flow_val)
                                colors.append("rgba(148, 163, 184, 0.3)")

    if not sources:
        fig = go.Figure()
        fig.update_layout(
            title_text=t("hit_sankey_heading"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )
        return fig

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=18,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=node_labels,
            color=node_colors
        ),
        link=dict(source=sources, target=targets, value=values, color=colors)
    )])

    title_range = f"(+1)" if max_step == 1 else (f"(+1 à +{max_step})" if cur_lang == "FR" else f"(+1 to +{max_step})")
    full_title = f"{t('hit_sankey_heading')} {title_range}"

    fig.update_layout(
        title_text=full_title,
        title_font_size=16,
        font=dict(size=11, color="white"),
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=20)
    )
    return fig

def compute_density_contour_and_outliers(df_events, half_rink=False, sigma=7.0, min_threshold_pct=0.25, min_cluster_density=1.35):
    """
    Computes a narrow 2D Gaussian density kernel for downstream events (goals or shots).
    Classifies events into:
      - Inside contour range: covered by the glowing contour fill, suppressed from discrete markers
      - Outside contour range: outliers/isolated events, explicitly displayed with discrete markers & vectors

    Returns:
      (xg, yg, Z_masked, df_outliers, has_contour)
    """
    if df_events is None or df_events.empty:
        return None, None, None, pd.DataFrame(), False

    xs = df_events["event_norm_x"].dropna().to_numpy()
    ys = df_events["event_norm_y"].dropna().to_numpy()
    n = len(xs)
    if n == 0:
        return None, None, None, pd.DataFrame(), False
    if n == 1:
        return None, None, None, df_events.copy(), False

    dist_sq_matrix = (xs[:, None] - xs[None, :]) ** 2 + (ys[:, None] - ys[None, :]) ** 2
    density_at_pts = np.exp(-dist_sq_matrix / (2.0 * (sigma ** 2))).sum(axis=1)
    max_dens = density_at_pts.max()

    if max_dens < min_cluster_density:
        return None, None, None, df_events.copy(), False

    contour_min_val = max(min_cluster_density, min_threshold_pct * max_dens)
    is_inside = density_at_pts >= contour_min_val
    df_outliers = df_events[~is_inside].copy()

    xg = np.linspace(0, 100, 61) if half_rink else np.linspace(-100, 100, 101)
    yg = np.linspace(-42.5, 42.5, 51)
    X, Y = np.meshgrid(xg, yg)

    dist_sq_grid = (X[:, :, None] - xs[None, None, :]) ** 2 + (Y[:, :, None] - ys[None, None, :]) ** 2
    Z = np.exp(-dist_sq_grid / (2.0 * (sigma ** 2))).sum(axis=-1)

    Z_norm = Z / max_dens
    cutoff_norm = contour_min_val / max_dens
    Z_masked = np.where(Z_norm >= cutoff_norm, Z_norm, np.nan)

    return xg, yg, Z_masked, df_outliers, True


def aggregate_rink_grid(df_grid):
    """
    Consolidates multi-season rink grid tiles into a single unique (x_bin, y_bin) spatial surface
    so that exactly 1 tile is rendered per rink cell, enabling true interactive dimming and consistent coloring.
    """
    if df_grid is None or df_grid.empty:
        return pd.DataFrame()
    if len(df_grid[["x_bin", "y_bin"]].drop_duplicates()) == len(df_grid):
        return df_grid.copy()

    def _agg_tile(g):
        tot = float(g["total_hits"].sum())
        puck_win = float(np.average(g["puck_win_pct"], weights=g["total_hits"])) if tot > 0 and "puck_win_pct" in g else 50.0
        shot_for = float(np.average(g["shot_for_pct"], weights=g["total_hits"])) if tot > 0 and "shot_for_pct" in g else 0.0
        shot_against = float(np.average(g["shot_against_pct"], weights=g["total_hits"])) if tot > 0 and "shot_against_pct" in g else 0.0
        avg_delta = float(np.average(g["avg_delta_sec"], weights=g["total_hits"])) if tot > 0 and "avg_delta_sec" in g else 10.0
        shots_f = float(g["shots_for_30s"].sum()) if "shots_for_30s" in g else 0.0
        shots_a = float(g["shots_against_30s"].sum()) if "shots_against_30s" in g else 0.0
        goals_f = float(g["goals_for_30s"].sum()) if "goals_for_30s" in g else 0.0
        goals_a = float(g["goals_against_30s"].sum()) if "goals_against_30s" in g else 0.0
        takeaway_f = float(np.average(g["takeaway_for_pct"], weights=g["total_hits"])) if tot > 0 and "takeaway_for_pct" in g else 0.0
        giveaway_a = float(np.average(g["giveaway_against_pct"], weights=g["total_hits"])) if tot > 0 and "giveaway_against_pct" in g else 0.0
        phys_battle = float(np.average(g["physical_battle_pct"], weights=g["total_hits"])) if tot > 0 and "physical_battle_pct" in g else 0.0
        stoppage = float(np.average(g["stoppage_pct"], weights=g["total_hits"])) if tot > 0 and "stoppage_pct" in g else 0.0
        goal_f = float(np.average(g["goal_for_pct"], weights=g["total_hits"])) if tot > 0 and "goal_for_pct" in g else 0.0
        goal_a = float(np.average(g["goal_against_pct"], weights=g["total_hits"])) if tot > 0 and "goal_against_pct" in g else 0.0
        return pd.Series({
            "spatial_zone": g["spatial_zone"].iloc[0] if "spatial_zone" in g else ("D" if g["x_bin"].iloc[0] < -25 else ("O" if g["x_bin"].iloc[0] > 25 else "N")),
            "total_hits": int(tot),
            "puck_win_pct": round(puck_win, 1),
            "shot_for_pct": round(shot_for, 1),
            "shot_against_pct": round(shot_against, 1),
            "takeaway_for_pct": round(takeaway_f, 1),
            "giveaway_against_pct": round(giveaway_a, 1),
            "physical_battle_pct": round(phys_battle, 1),
            "stoppage_pct": round(stoppage, 1),
            "goal_for_pct": round(goal_f, 1),
            "goal_against_pct": round(goal_a, 1),
            "avg_delta_sec": round(avg_delta, 1),
            "shots_for_30s": shots_f,
            "shots_against_30s": shots_a,
            "goals_for_30s": goals_f,
            "goals_against_30s": goals_a,
        })

    agg = df_grid.groupby(["x_bin", "y_bin"], as_index=False).apply(_agg_tile, include_groups=False)
    tot_hits = agg["total_hits"].sum()
    agg["density_pct"] = (agg["total_hits"] / tot_hits * 100.0) if tot_hits > 0 else 0.0
    agg["sog_for_per_100"] = (agg["shots_for_30s"] / agg["total_hits"] * 100.0).fillna(0.0)
    agg["sog_against_per_100"] = (agg["shots_against_30s"] / agg["total_hits"] * 100.0).fillna(0.0)
    agg["net_sog_per_100"] = agg["sog_for_per_100"] - agg["sog_against_per_100"]
    agg["goals_for_per_100"] = (agg["goals_for_30s"] / agg["total_hits"] * 100.0).fillna(0.0)
    agg["goals_against_per_100"] = (agg["goals_against_30s"] / agg["total_hits"] * 100.0).fillna(0.0)
    agg["net_goals_per_100"] = agg["goals_for_per_100"] - agg["goals_against_per_100"]
    agg["season"] = "ALL"
    return agg


def aggregate_zone_kpi(df_kpi):
    """
    Consolidates multi-season zone KPI into exactly 3 rows (O, D, N) with weighted statistics.
    """
    if df_kpi is None or df_kpi.empty or len(df_kpi) <= 3:
        return df_kpi.copy() if df_kpi is not None else pd.DataFrame()

    rows = []
    for zone in ["O", "D", "N"]:
        sub = df_kpi[df_kpi["hit_zone"] == zone]
        if sub.empty:
            continue
        tot_hits = int(sub["total_hits"].sum())
        gf = int(sub["goals_for"].sum()) if "goals_for" in sub else 0
        ga = int(sub["goals_against"].sum()) if "goals_against" in sub else 0
        pwin = float(np.average(sub["immediate_possession_pct"], weights=sub["total_hits"])) if tot_hits > 0 else 50.0
        sf100 = float(np.average(sub["shots_for_per_100_hits"], weights=sub["total_hits"])) if tot_hits > 0 else 0.0
        sa100 = float(np.average(sub["shots_against_per_100_hits"], weights=sub["total_hits"])) if tot_hits > 0 else 0.0
        ns100 = sf100 - sa100
        z_name = sub["zone_name"].iloc[0] if "zone_name" in sub else zone
        rows.append({
            "season": "ALL",
            "hit_zone": zone,
            "zone_name": z_name,
            "total_hits": tot_hits,
            "immediate_possession_pct": round(pwin, 1),
            "shots_for_per_100_hits": round(sf100, 2),
            "shots_against_per_100_hits": round(sa100, 2),
            "net_shot_diff_per_100": round(ns100, 2),
            "goals_for": gf,
            "goals_against": ga,
        })
    return pd.DataFrame(rows)


def aggregate_eda_summary(df_summary):
    """
    Consolidates multi-season EDA zone summary into exactly 3 rows (D, N, O)
    with total sums and weighted rates to prevent bar/label stacking.
    """
    if df_summary is None or df_summary.empty:
        return pd.DataFrame()
    if len(df_summary["zoneCode"].drop_duplicates()) == len(df_summary):
        return df_summary.copy()

    rows = []
    tot_all = float(df_summary["total_hits"].sum())
    for z in ["D", "N", "O"]:
        sub = df_summary[df_summary["zoneCode"] == z]
        if sub.empty:
            continue
        tot_hits = float(sub["total_hits"].sum())
        rec_pct = float(np.average(sub["recovery_pct"], weights=sub["total_hits"])) if tot_hits > 0 else 50.0
        opp_pct = float(np.average(sub["opponent_possession_pct"], weights=sub["total_hits"])) if tot_hits > 0 and "opponent_possession_pct" in sub else (100.0 - rec_pct)
        avg_d = float(np.average(sub["avg_delta_sec"], weights=sub["total_hits"])) if tot_hits > 0 and "avg_delta_sec" in sub else 10.0
        med_d = float(np.average(sub["median_delta_sec"], weights=sub["total_hits"])) if tot_hits > 0 and "median_delta_sec" in sub else 10.0
        sog_f = float(sub["sog_for"].sum()) if "sog_for" in sub else 0.0
        sog_a = float(sub["sog_against"].sum()) if "sog_against" in sub else 0.0
        goals_f = float(sub["goals_for"].sum()) if "goals_for" in sub else 0.0
        goals_a = float(sub["goals_against"].sum()) if "goals_against" in sub else 0.0

        exits = float(sub["zone_exits"].dropna().sum()) if "zone_exits" in sub and not sub["zone_exits"].dropna().empty else np.nan
        exit_pct = (exits / tot_hits * 100.0) if (pd.notna(exits) and tot_hits > 0) else np.nan

        sog_f_100 = (sog_f / tot_hits * 100.0) if tot_hits > 0 else 0.0
        sog_a_100 = (sog_a / tot_hits * 100.0) if tot_hits > 0 else 0.0
        net_sog_100 = sog_f_100 - sog_a_100
        goals_f_100 = (goals_f / tot_hits * 100.0) if tot_hits > 0 else 0.0
        goals_a_100 = (goals_a / tot_hits * 100.0) if tot_hits > 0 else 0.0
        net_goals_100 = goals_f_100 - goals_a_100
        share_pct = (tot_hits / tot_all * 100.0) if tot_all > 0 else 0.0

        rows.append({
            "season": "ALL",
            "zoneCode": z,
            "total_hits": int(tot_hits),
            "recovery_pct": round(rec_pct, 1),
            "opponent_possession_pct": round(opp_pct, 1),
            "avg_delta_sec": round(avg_d, 1),
            "median_delta_sec": round(med_d, 1),
            "sog_for": sog_f,
            "sog_against": sog_a,
            "goals_for": goals_f,
            "goals_against": goals_a,
            "zone_exits": exits,
            "share_of_all_hits_pct": round(share_pct, 1),
            "sog_for_per_100": round(sog_f_100, 2),
            "sog_against_per_100": round(sog_a_100, 2),
            "net_sog_per_100": round(net_sog_100, 2),
            "goals_for_per_100": round(goals_f_100, 3),
            "goals_against_per_100": round(goals_a_100, 3),
            "net_goals_per_100": round(net_goals_100, 3),
            "zone_exit_pct": round(exit_pct, 1) if pd.notna(exit_pct) else np.nan,
        })
    return pd.DataFrame(rows)


def aggregate_eda_events(df_events):
    """
    Consolidates multi-season EDA first event telemetry by (zoneCode, next_category, next_owner_rel)
    with total sums and weighted latency averages.
    """
    if df_events is None or df_events.empty:
        return pd.DataFrame()
    if len(df_events[["zoneCode", "next_category", "next_owner_rel"]].drop_duplicates()) == len(df_events):
        return df_events.copy()

    def _agg_ev(g):
        tot = float(g["event_count"].sum())
        avg_d = float(np.average(g["avg_delta_sec"], weights=g["event_count"])) if tot > 0 and "avg_delta_sec" in g else 10.0
        med_d = float(np.average(g["median_delta_sec"], weights=g["event_count"])) if tot > 0 and "median_delta_sec" in g else 10.0
        return pd.Series({
            "event_count": int(tot),
            "avg_delta_sec": round(avg_d, 1),
            "median_delta_sec": round(med_d, 1),
        })

    agg = df_events.groupby(["zoneCode", "next_category", "next_owner_rel"], as_index=False).apply(_agg_ev, include_groups=False)
    zone_totals = agg.groupby("zoneCode")["event_count"].transform("sum")
    agg["pct_of_zone"] = (agg["event_count"] / zone_totals * 100.0).round(2)
    agg["season"] = "ALL"
    return agg


def aggregate_team_kpi(df_team_kpi):
    """
    Consolidates multi-season team KPI into exactly 1 row per team (including LEAGUE_AVG)
    with total sums and weighted averages across seasons.
    """
    if df_team_kpi is None or df_team_kpi.empty:
        return pd.DataFrame()
    if len(df_team_kpi["team"].drop_duplicates()) == len(df_team_kpi):
        return df_team_kpi.copy()

    rows = []
    for team, grp in df_team_kpi.groupby("team"):
        tot_hits = float(grp["total_hits"].sum())
        gp = float(grp["games_played"].sum())
        w = grp["total_hits"].values
        if tot_hits > 0:
            ozone = float(np.average(grp["ozone_pct"], weights=w))
            dzone = float(np.average(grp["dzone_pct"], weights=w))
            nzone = float(np.average(grp["nzone_pct"], weights=w))
            puck_win = float(np.average(grp["puck_win_pct"], weights=w))
            s1_f = float(np.average(grp["step1_shot_for_pct"], weights=w))
            s1_a = float(np.average(grp["step1_shot_against_pct"], weights=w))
        else:
            ozone = float(grp["ozone_pct"].mean())
            dzone = float(grp["dzone_pct"].mean())
            nzone = float(grp["nzone_pct"].mean())
            puck_win = float(grp["puck_win_pct"].mean())
            s1_f = float(grp["step1_shot_for_pct"].mean())
            s1_a = float(grp["step1_shot_against_pct"].mean())

        s_for = float(grp["shots_for_30s"].sum())
        s_agst = float(grp["shots_against_30s"].sum())
        g_for = float(grp["goals_for_30s"].sum())
        g_agst = float(grp["goals_against_30s"].sum())

        hpg = tot_hits / gp if gp > 0 else 0.0
        net_s100 = (s_for - s_agst) / tot_hits * 100.0 if tot_hits > 0 else 0.0
        net_g100 = (g_for - g_agst) / tot_hits * 100.0 if tot_hits > 0 else 0.0

        rows.append({
            "season": "ALL",
            "team": team,
            "games_played": gp,
            "total_hits": int(tot_hits),
            "hits_per_game": round(hpg, 1),
            "ozone_pct": round(ozone, 1),
            "dzone_pct": round(dzone, 1),
            "nzone_pct": round(nzone, 1),
            "puck_win_pct": round(puck_win, 1),
            "step1_shot_for_pct": round(s1_f, 1),
            "step1_shot_against_pct": round(s1_a, 1),
            "shots_for_30s": s_for,
            "shots_against_30s": s_agst,
            "shots_for_per_100": round(s_for / tot_hits * 100.0, 2) if tot_hits > 0 else 0.0,
            "shots_against_per_100": round(s_agst / tot_hits * 100.0, 2) if tot_hits > 0 else 0.0,
            "net_shots_per_100": round(net_s100, 2),
            "goals_for_30s": g_for,
            "goals_against_30s": g_agst,
            "net_goals_per_100": round(net_g100, 3),
        })
    return pd.DataFrame(rows)



def render_interactive_rink_heatmap(df_grid, df_goals, selected_loc=None, half_rink=False, min_density=0.0, event_type="goals", side_filter="both"):
    """
    Renders 2D density heatmap of hit occurrences with clickable hotspot tiles
    and clean downstream trajectory vectors.
    When a tile is selected, all other tiles are shaded out (dimmed) and the selected
    tile + its clear destination vectors and endpoints are highlighted without visual overlap.
    """
    grid = df_grid.copy() if df_grid is not None else pd.DataFrame()
    if not grid.empty and len(grid[["x_bin", "y_bin"]].drop_duplicates()) < len(grid):
        grid = aggregate_rink_grid(grid)

    if min_density > 0 and not grid.empty and "density_pct" in grid.columns:
        grid = grid[grid["density_pct"] >= min_density]
    if half_rink and not grid.empty and "x_bin" in grid.columns:
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

    # Dynamic shading: when selected_loc is set, shade out (dim) all unselected tiles
    if selected_loc is not None:
        sel_x, sel_y = float(selected_loc[0]), float(selected_loc[1])
        is_sel_list = [(abs(float(x) - sel_x) < 1e-3 and abs(float(y) - sel_y) < 1e-3) for x, y in zip(grid["x_bin"], grid["y_bin"])]
        opacities = [1.0 if s else 0.16 for s in is_sel_list]
        border_colors = ["#facc15" if s else "rgba(255,255,255,0.06)" for s in is_sel_list]
        border_widths = [3.0 if s else 0.5 for s in is_sel_list]
        marker_sizes = [24 if s else (18 if not half_rink else 22) for s in is_sel_list]
        sel_indices = [i for i, s in enumerate(is_sel_list) if s]
    else:
        opacities = 0.90
        border_colors = "rgba(255,255,255,0.22)"
        border_widths = 0.5
        marker_sizes = 18 if not half_rink else 22
        sel_indices = None

    scatter_kwargs = dict(
        x=grid["x_bin"],
        y=grid["y_bin"],
        mode="markers",
        marker=dict(
            symbol="square",
            size=marker_sizes,
            opacity=opacities,
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
            line=dict(color=border_colors, width=border_widths)
        ),
        customdata=customdata,
        hoverinfo="none",
        name=t("hit_sub_rink_heading"),
        showlegend=False
    )
    if sel_indices:
        scatter_kwargs["selectedpoints"] = sel_indices
        scatter_kwargs["selected"] = dict(marker=dict(size=24, opacity=1.0))
        scatter_kwargs["unselected"] = dict(marker=dict(opacity=0.16))

    fig.add_trace(go.Scatter(**scatter_kwargs))

    # If a location is selected, plot clean bounded trajectories and distinct endpoints
    if selected_loc is not None:
        sel_x, sel_y = selected_loc

        # Highlight reticle halo around origin hit tile
        fig.add_trace(go.Scatter(
            x=[sel_x],
            y=[sel_y],
            mode="markers",
            marker=dict(
                symbol="circle-open",
                size=34,
                line=dict(color="#facc15", width=3.5)
            ),
            name=f"{t('hit_val_selected_focus')} ({sel_x:+.0f}, {sel_y:+.0f})",
            hoverinfo="skip"
        ))

        # Filter downstream events originating near this grid cell
        if df_goals is not None and not df_goals.empty:
            loc_events = df_goals[
                (abs(df_goals["x_bin"] - sel_x) < 1e-3) & (abs(df_goals["y_bin"] - sel_y) < 1e-3)
            ].copy()

            if not loc_events.empty:
                # 1. Filter by event_type ("goals" vs "shots")
                if event_type == "shots":
                    filtered_events = loc_events[loc_events["is_goal"] == 0].copy()
                    type_label = t("hit_rink_event_shots")
                else:
                    filtered_events = loc_events[loc_events["is_goal"] == 1].copy()
                    type_label = t("hit_rink_event_goals")

                ev_for = filtered_events[filtered_events["outcome_side"] == "Hitting Team (For)"].copy()
                ev_agst = filtered_events[filtered_events["outcome_side"] == "Opponent (Against)"].copy()

                # 2. Offense / Hitting Team (For)
                if side_filter in ("both", "for") and not ev_for.empty:
                    xg_f, yg_f, Z_f, outliers_f, has_contour_f = compute_density_contour_and_outliers(
                        ev_for, half_rink=half_rink, sigma=7.0
                    )
                    if has_contour_f and Z_f is not None:
                        emerald_contour_scale = [
                            [0.0, "rgba(16, 185, 129, 0.20)"],
                            [0.35, "rgba(52, 211, 153, 0.50)"],
                            [0.70, "rgba(250, 204, 21, 0.75)"],
                            [1.0, "rgba(251, 191, 36, 0.95)"]
                        ]
                        n_clustered_f = len(ev_for) - len(outliers_f)
                        count_lbl_f = f"{n_clustered_f}" if outliers_f.empty else f"{n_clustered_f} clustered"
                        c_title_f = (
                            f"🟢 {t('hit_contour_goals_for')} ({count_lbl_f})"
                            if event_type == "goals"
                            else f"🟢 {t('hit_contour_shots_for')} ({count_lbl_f})"
                        )
                        fig.add_trace(go.Contour(
                            x=xg_f, y=yg_f, z=Z_f,
                            contours=dict(coloring="fill", showlines=True, start=0.2, end=1.0, size=0.15),
                            line=dict(width=2.0, color="#ffffff"),
                            colorscale=emerald_contour_scale,
                            opacity=0.88,
                            showscale=False,
                            showlegend=True,
                            hoverinfo="skip",
                            name=c_title_f
                        ))

                    # Discrete markers and vectors ONLY for outliers (or for all events if no contour formed)
                    # This leaves the contour completely unobstructed so the user can see the cluster!
                    if not outliers_f.empty:
                        x_lines_f, y_lines_f = [], []
                        for _, row in outliers_f.iterrows():
                            x_lines_f.extend([sel_x, row["event_norm_x"], None])
                            y_lines_f.extend([sel_y, row["event_norm_y"], None])
                        fig.add_trace(go.Scatter(
                            x=x_lines_f, y=y_lines_f, mode="lines",
                            line=dict(color="rgba(52, 211, 153, 0.55)", width=1.4, dash="dot"),
                            showlegend=False,
                            hoverinfo="skip"
                        ))
                        sym_f = "star" if event_type == "goals" else "circle"
                        sz_f = 13 if event_type == "goals" else 7
                        color_f = "#fbbf24" if event_type == "goals" else "rgba(16, 185, 129, 0.85)"
                        prefix_f = "⭐" if event_type == "goals" else "🎯"
                        sub_lbl_f = f", Outliers: {len(outliers_f)}" if has_contour_f else f": {len(outliers_f)}"
                        title_f = f"{prefix_f} {type_label} (For{sub_lbl_f})"
                        hover_f = [
                            f"<b>{prefix_f} {type_label} (For)</b><br>"
                            f"({r['event_norm_x']:+.1f}, {r['event_norm_y']:+.1f})<br>"
                            f"⏱️ +{r.get('delta_sec', 0):.0f}s"
                            for _, r in outliers_f.iterrows()
                        ]
                        fig.add_trace(go.Scatter(
                            x=outliers_f["event_norm_x"], y=outliers_f["event_norm_y"], mode="markers",
                            marker=dict(symbol=sym_f, size=sz_f, color=color_f, line=dict(color="#10b981", width=1.5)),
                            name=title_f,
                            text=hover_f,
                            hoverinfo="text"
                        ))

                # 3. Defense / Opponent Conceded (Against)
                if side_filter in ("both", "against") and not ev_agst.empty:
                    xg_a, yg_a, Z_a, outliers_a, has_contour_a = compute_density_contour_and_outliers(
                        ev_agst, half_rink=half_rink, sigma=7.0
                    )
                    if has_contour_a and Z_a is not None:
                        crimson_contour_scale = [
                            [0.0, "rgba(225, 29, 72, 0.35)"],
                            [0.35, "rgba(244, 63, 94, 0.65)"],
                            [0.70, "rgba(251, 113, 133, 0.85)"],
                            [1.0, "rgba(255, 255, 255, 0.98)"]
                        ]
                        n_clustered_a = len(ev_agst) - len(outliers_a)
                        count_lbl_a = f"{n_clustered_a}" if outliers_a.empty else f"{n_clustered_a} clustered"
                        c_title_a = (
                            f"🔴 {t('hit_contour_goals_agst')} ({count_lbl_a})"
                            if event_type == "goals"
                            else f"🔴 {t('hit_contour_shots_agst')} ({count_lbl_a})"
                        )
                        fig.add_trace(go.Contour(
                            x=xg_a, y=yg_a, z=Z_a,
                            contours=dict(coloring="fill", showlines=True, start=0.2, end=1.0, size=0.15),
                            line=dict(width=2.2, color="#ffffff"),
                            colorscale=crimson_contour_scale,
                            opacity=0.90,
                            showscale=False,
                            showlegend=True,
                            hoverinfo="skip",
                            name=c_title_a
                        ))

                    if not outliers_a.empty:
                        x_lines_a, y_lines_a = [], []
                        for _, row in outliers_a.iterrows():
                            x_lines_a.extend([sel_x, row["event_norm_x"], None])
                            y_lines_a.extend([sel_y, row["event_norm_y"], None])
                        fig.add_trace(go.Scatter(
                            x=x_lines_a, y=y_lines_a, mode="lines",
                            line=dict(color="rgba(248, 113, 113, 0.55)", width=1.4, dash="dot"),
                            showlegend=False,
                            hoverinfo="skip"
                        ))
                        sym_a = "star" if event_type == "goals" else "circle"
                        sz_a = 13 if event_type == "goals" else 7
                        color_a = "#ef4444" if event_type == "goals" else "rgba(239, 68, 68, 0.85)"
                        prefix_a = "🚨" if event_type == "goals" else "🛑"
                        sub_lbl_a = f", Outliers: {len(outliers_a)}" if has_contour_a else f": {len(outliers_a)}"
                        title_a = f"{prefix_a} {type_label} (Agst{sub_lbl_a})"
                        hover_a = [
                            f"<b>{prefix_a} {type_label} (Agst)</b><br>"
                            f"({r['event_norm_x']:+.1f}, {r['event_norm_y']:+.1f})<br>"
                            f"⏱️ +{r.get('delta_sec', 0):.0f}s"
                            for _, r in outliers_a.iterrows()
                        ]
                        fig.add_trace(go.Scatter(
                            x=outliers_a["event_norm_x"], y=outliers_a["event_norm_y"], mode="markers",
                            marker=dict(symbol=sym_a, size=sz_a, color=color_a, line=dict(color="#ffffff", width=1.5)),
                            name=title_a,
                            text=hover_a,
                            hoverinfo="text"
                        ))

    fig.update_layout(
        xaxis=dict(range=[-5, 105] if half_rink else [-105, 105]),
        yaxis=dict(range=[-45, 45]),
        margin=dict(l=5, r=5, t=0, b=30),
        height=420 if half_rink else 350,
        legend=dict(
            orientation="h", yanchor="top", y=-0.04, xanchor="center", x=0.5,
            font=dict(color="white", size=11)
        )
    )
    return fig

def render_team_rink_heatmap(df_spatial, team_code, season_code, half_rink=False, display_name=None, is_benchmark=False, season_label=None, min_density=1.0, show_title=False):
    """Renders a team-specific spatial hit density heatmap on the rink. Drops tiles lower than min_density%."""
    is_all = str(season_code).upper() == "ALL"
    if is_all:
        if team_code == "LEAGUE_AVG":
            base = df_spatial[df_spatial["team"] != "LEAGUE_AVG"]
            if not base.empty:
                team_totals = base.groupby(["team", "x_bin", "y_bin"], as_index=False)["hit_count"].sum()
                team_sums = team_totals.groupby("team")["hit_count"].transform("sum")
                team_totals["density_pct"] = (team_totals["hit_count"] / team_sums) * 100.0
                sub = team_totals.groupby(["x_bin", "y_bin"], as_index=False).agg(
                    hit_count=("hit_count", "mean"),
                    density_pct=("density_pct", "mean")
                )
                sub["team"] = "LEAGUE_AVG"
                sub["season"] = "ALL"
            else:
                sub = pd.DataFrame()
        else:
            base = df_spatial[df_spatial["team"] == team_code]
            if not base.empty:
                sub = base.groupby(["x_bin", "y_bin"], as_index=False)["hit_count"].sum()
                tot_h = sub["hit_count"].sum()
                sub["density_pct"] = (sub["hit_count"] / tot_h * 100.0) if tot_h > 0 else 0.0
                sub["team"] = team_code
                sub["season"] = "ALL"
            else:
                sub = pd.DataFrame()
    else:
        sub = df_spatial[(df_spatial["team"] == team_code) & (df_spatial["season"].astype(str) == str(season_code))].copy()
        # Fallback for LEAGUE_AVG if specific season combination not pre-tabulated
        if sub.empty and team_code == "LEAGUE_AVG":
            base = df_spatial[(df_spatial["team"] != "LEAGUE_AVG") & (df_spatial["season"].astype(str) == str(season_code))]
            if not base.empty:
                sub = base.groupby(["x_bin", "y_bin"], as_index=False).agg(
                    hit_count=("hit_count", "mean"),
                    density_pct=("density_pct", "mean")
                )
                sub["team"] = "LEAGUE_AVG"
                sub["season"] = season_code

    if min_density > 0 and not sub.empty:
        sub = sub[sub["density_pct"] >= min_density]

    if half_rink and not sub.empty:
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

    if show_title:
        prefix = f"{t('hit_val_benchmark')}: " if is_benchmark else f"{t('hit_val_selected_focus')}: "
        title_config = dict(
            text=f"<b>{prefix}{label}</b> <span style='font-size:11px;color:#94a3b8;'>({season_text})</span>",
            font=dict(size=12, color="white"),
            y=0.96,
            x=0.04,
            xanchor="left",
            yanchor="top"
        )
        t_margin = 28
    else:
        title_config = None
        t_margin = 2

    fig.update_layout(
        title=title_config,
        height=240 if half_rink else 195,
        margin=dict(l=4, r=4, t=t_margin, b=2)
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

    st.markdown("""
        <style>
        /* Widen selectboxes and dropdown menus so full option text is clearly visible */
        div[data-baseweb="select"] {
            min-width: 100% !important;
        }
        div[data-baseweb="select"] > div {
            white-space: normal !important;
            text-overflow: clip !important;
            overflow: visible !important;
            min-height: 42px !important;
        }
        div[data-baseweb="select"] span {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
        }
        div[data-baseweb="popover"] {
            min-width: max-content !important;
            max-width: 95vw !important;
        }
        div[data-baseweb="popover"] ul[role="listbox"] {
            min-width: max-content !important;
            max-width: 95vw !important;
        }
        div[data-baseweb="popover"] li[role="option"] {
            white-space: normal !important;
            word-break: break-word !important;
            line-height: 1.4 !important;
            padding: 8px 12px !important;
        }
        </style>
    """, unsafe_allow_html=True)

    df_kpi, df_seq, df_goals, df_grid, df_team_kpi, df_team_spatial, team_name_map, df_eda_summary, df_eda_events, df_trans = load_all_data()

    if any(x is None for x in [df_kpi, df_seq, df_goals, df_grid, df_team_kpi, df_team_spatial]):
        st.error(t("hit_data_not_found"))
        return

    # --------------------------------------------------------------------------
    # TOP-LEVEL SEASON FILTER & UNIFIED DATASET SLICING
    # --------------------------------------------------------------------------
    cur_l = st.session_state.get("lang", "EN")
    avail_seasons = [str(s) for s in df_kpi["season"].unique() if str(s) != "ALL"] if "season" in df_kpi.columns else []
    avail_seasons = sorted(avail_seasons, reverse=True)
    season_opts = ["ALL"] + avail_seasons

    season_labels = {"ALL": t("hit_season_all", lang=cur_l)}
    for s in avail_seasons:
        season_labels[s] = f"{s[:4]}-{s[4:]}" if len(s) == 8 else s

    prev_season = st.session_state.get("hit_global_season_val", "ALL")
    s_idx = season_opts.index(prev_season) if prev_season in season_opts else 0

    col_season_sel, col_season_info = st.columns([1.6, 2])
    with col_season_sel:
        selected_season = st.selectbox(
            t("hit_global_season_label"),
            season_opts,
            format_func=lambda s, sl=season_labels: sl.get(s, s),
            index=s_idx,
            key=f"hit_global_season_select_{cur_l}"
        )
        st.session_state.hit_global_season_val = selected_season

    selected_season_label = season_labels.get(selected_season, selected_season)

    # Slice all master DataFrames uniformly based on selected season
    if selected_season == "ALL":
        active_kpi = aggregate_zone_kpi(df_kpi)
        active_seq = df_seq.copy()
        active_trans = df_trans.copy() if df_trans is not None else None
        active_goals = df_goals.copy()
        active_grid = aggregate_rink_grid(df_grid)
        active_team_kpi = aggregate_team_kpi(df_team_kpi)
        active_summary = aggregate_eda_summary(df_eda_summary)
        active_events = aggregate_eda_events(df_eda_events)
    else:
        active_kpi = df_kpi[df_kpi["season"].astype(str) == str(selected_season)].copy() if "season" in df_kpi.columns else df_kpi.copy()
        if active_kpi.empty and not df_kpi.empty:
            active_kpi = df_kpi.copy()

        active_seq = df_seq[df_seq["season"].astype(str) == str(selected_season)].copy() if "season" in df_seq.columns else df_seq.copy()
        if active_seq.empty and not df_seq.empty:
            active_seq = df_seq.copy()

        active_trans = df_trans[df_trans["season"].astype(str) == str(selected_season)].copy() if (df_trans is not None and "season" in df_trans.columns) else (df_trans.copy() if df_trans is not None else None)
        if active_trans is not None and active_trans.empty and not df_trans.empty:
            active_trans = df_trans.copy()

        if "idx_season" in df_goals.columns:
            try:
                s_int = int(selected_season)
                active_goals = df_goals[
                    (df_goals["idx_season"] == s_int) |
                    (df_goals["idx_season"] // 10000 == s_int) |
                    (df_goals["idx_season"] == s_int // 10000)
                ].copy()
            except (ValueError, TypeError):
                active_goals = df_goals[df_goals["idx_season"].astype(str).str.startswith(str(selected_season)[:4])].copy()
        elif "season" in df_goals.columns:
            active_goals = df_goals[df_goals["season"].astype(str).str.startswith(str(selected_season)[:4])].copy()
        else:
            active_goals = df_goals.copy()
        if active_goals.empty and not df_goals.empty:
            active_goals = df_goals.copy()

        active_grid = df_grid[df_grid["season"].astype(str) == str(selected_season)].copy() if "season" in df_grid.columns else df_grid.copy()
        if active_grid.empty and not df_grid.empty:
            active_grid = df_grid.copy()

        active_team_kpi = df_team_kpi[df_team_kpi["season"].astype(str) == str(selected_season)].copy() if "season" in df_team_kpi.columns else df_team_kpi.copy()
        if active_team_kpi.empty and not df_team_kpi.empty:
            active_team_kpi = df_team_kpi.copy()

        active_summary = df_eda_summary[df_eda_summary["season"].astype(str) == str(selected_season)].copy() if "season" in df_eda_summary.columns else df_eda_summary.copy()
        if active_summary.empty and not df_eda_summary.empty:
            active_summary = df_eda_summary.copy()

        active_events = df_eda_events[df_eda_events["season"].astype(str) == str(selected_season)].copy() if "season" in df_eda_events.columns else df_eda_events.copy()
        if active_events.empty and not df_eda_events.empty:
            active_events = df_eda_events.copy()

    # Top Executive Metric Ribbon
    c1, c2, c3, c4 = st.columns(4)
    total_hits = int(active_kpi["total_hits"].sum()) if not active_kpi.empty else 0
    ozone_hits = int(active_kpi[active_kpi["hit_zone"] == "O"]["total_hits"].sum()) if not active_kpi[active_kpi["hit_zone"] == "O"].empty else 0
    dzone_hits = int(active_kpi[active_kpi["hit_zone"] == "D"]["total_hits"].sum()) if not active_kpi[active_kpi["hit_zone"] == "D"].empty else 0
    total_goals_30s = int(active_kpi["goals_for"].sum() + active_kpi["goals_against"].sum()) if not active_kpi.empty else 0

    o_row = active_kpi[active_kpi["hit_zone"] == "O"]
    o_puck_win = float(np.average(o_row["immediate_possession_pct"], weights=o_row["total_hits"])) if not o_row.empty and o_row["total_hits"].sum() > 0 else 52.9

    d_row = active_kpi[active_kpi["hit_zone"] == "D"]
    d_sog_diff = float(np.average(d_row["net_shot_diff_per_100"], weights=d_row["total_hits"])) if not d_row.empty and d_row["total_hits"].sum() > 0 else -79.1

    gf_tot = int(active_kpi["goals_for"].sum()) if not active_kpi.empty else 0
    ga_tot = int(active_kpi["goals_against"].sum()) if not active_kpi.empty else 0

    c1.metric(t("hit_metric_hits"), f"{total_hits:,}", help=f"{t('hit_team_sel_season_label')} {selected_season_label}")
    c2.metric(t("hit_metric_ozone"), f"{ozone_hits:,}", f"{o_puck_win:+.1f}% {t('hit_th_puck_win')}")
    c3.metric(t("hit_metric_dzone"), f"{dzone_hits:,}", f"{d_sog_diff:+.1f} {t('hit_th_net_shot_diff')}", delta_color="inverse")
    c4.metric(t("hit_metric_goals"), f"{total_goals_30s:,}", f"{gf_tot - ga_tot:+,} {t('hit_th_net_goal_diff')}", help=f"{gf_tot:,} {t('hit_owner_for')} / {ga_tot:,} {t('hit_owner_against')}")

    st.markdown("---")

    # Four-Layer Main Analysis Navigation
    main_tabs = st.tabs([
        t("hit_layer_eda"),
        t("hit_layer_sequence"),
        t("hit_layer_rink"),
        t("hit_layer_tactics")
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 1 - TAB 1: EXPLORATORY (EDA) FOUNDATIONS
    # ══════════════════════════════════════════════════════════════════════════
    with main_tabs[0]:
        st.markdown(t("hit_eda_scope"))
        st.subheader(t("hit_eda_heading"))
        st.markdown(t("hit_eda_intro"))

        # Dropdown write up for methodology (short and simple)
        with st.expander(t("hit_exp_eda_title"), expanded=False):
            st.markdown(f"""
            {t("hit_exp_eda_intro")}

            **{t("hit_exp_eda_sec1_title")}**
            {t("hit_exp_eda_sec1_p1")}
            """)
            st.latex(r"\operatorname{eventId}(E_{k+1}) = \operatorname{eventId}(E_k) + 1 \quad \text{and} \quad \Delta t = t_{k+1} - t_k \ge 0")

            st.markdown(f"""
            **{t("hit_exp_eda_sec2_title")}**
            {t("hit_exp_eda_sec2_p1")}
            """)
            st.latex(r"\text{Puck Recovery Rate } \Pi_{\text{recovery}}(Z) = \frac{1}{N_Z} \sum_{i=1}^{N_Z} \mathbb{I}(\operatorname{Owner}(E_{k+1, i}) = T_{\text{hit}})")
            st.latex(r"\text{Zone Exit }(Z = \text{D}) \iff \operatorname{Zone}(E_{k+1}) \in \{\text{Neutral (N)}, \text{Offensive (O)}\}")

            st.markdown(f"""
            **{t("hit_exp_eda_sec3_title")}**
            {t("hit_exp_eda_sec3_p1")}
            """)
            st.latex(r"\text{SOG Rate / 100 Hits} = \frac{100}{N} \sum_{i=1}^{N} \mathbb{I}(E_{k+1, i} = \text{Shot on Goal})")
            st.latex(r"\text{Goal Rate / 100 Hits} = \frac{100}{N} \sum_{i=1}^{N} \mathbb{I}(E_{k+1, i} = \text{Goal})")

        # Top EDA Baseline KPI Cards (Clean metrics without misleading trend arrows)
        tot_all_hits = active_summary["total_hits"].sum() if not active_summary.empty else 0
        overall_rec = (active_summary["recovery_pct"] * active_summary["total_hits"]).sum() / tot_all_hits if tot_all_hits > 0 else 46.3
        d_row_kpi = active_summary[active_summary["zoneCode"] == "D"].iloc[0] if not active_summary[active_summary["zoneCode"] == "D"].empty else None
        o_row_kpi = active_summary[active_summary["zoneCode"] == "O"].iloc[0] if not active_summary[active_summary["zoneCode"] == "O"].empty else None

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(
            t("hit_eda_kpi_sample"),
            f"{tot_all_hits:,}",
            help=f"{t('hit_team_sel_season_label')} {selected_season_label}"
        )
        k2.metric(
            t("hit_eda_kpi_overall_rec"),
            f"{overall_rec:.1f}%",
            f"{overall_rec - 50.0:+.1f}% {t('hit_eda_kpi_parity')}",
            delta_color="inverse",
            help=t("hit_eda_kpi_rec_help")
        )
        k3.metric(
            t("hit_eda_kpi_forecheck_rec"),
            f"{o_row_kpi['recovery_pct']:.1f}%" if o_row_kpi is not None else "53.1%",
            f"{o_row_kpi['recovery_pct'] - 50.0:+.1f}% {t('hit_eda_kpi_parity')}" if o_row_kpi is not None else f"+3.1% {t('hit_eda_kpi_parity')}",
            help=t("hit_eda_kpi_forecheck_help")
        )
        k4.metric(
            t("hit_eda_kpi_exit_rate"),
            f"{d_row_kpi['zone_exit_pct']:.1f}%" if d_row_kpi is not None and pd.notna(d_row_kpi['zone_exit_pct']) else "69.7%",
            help=t("hit_eda_kpi_exit_help")
        )

        st.markdown("---")

        # ----------------------------------------------------------------------
        # SECTION 1: FIRST EVENT BREAKDOWN & LATENCY
        # ----------------------------------------------------------------------
        st.subheader(t("hit_eda_sec1_heading"))
        st.markdown(t("hit_eda_sec1_intro"))

        # Origin Zone Filter specifically for Section 1 First Event Telemetry
        eda_zone_opts = ["All", "D", "N", "O"]
        prev_eda_zone = st.session_state.get("eda_sec1_zone_val", "All")
        z1_idx = eda_zone_opts.index(prev_eda_zone) if prev_eda_zone in eda_zone_opts else 0

        col_z1_sel, _ = st.columns([2.5, 1])
        with col_z1_sel:
            selected_eda_zone = st.selectbox(
                t("hit_eda_zone_label"),
                eda_zone_opts,
                format_func=lambda x, l=cur_l: t("hit_tree_opt_all", lang=l) if x == "All" else (
                    t("hit_tree_opt_d", lang=l) if x == "D" else (
                        t("hit_tree_opt_n", lang=l) if x == "N" else t("hit_tree_opt_o", lang=l)
                    )
                ),
                index=z1_idx,
                key=f"eda_sec1_zone_select_{cur_l}"
            )
            st.session_state.eda_sec1_zone_val = selected_eda_zone

        c_ev1, c_ev2 = st.columns(2)
        with c_ev1:
            st.markdown(f"##### {t('hit_eda_sec1_sub1')}")
            st.plotly_chart(create_eda_first_event_chart(active_events, selected_eda_zone), width="stretch", key="eda_first_event_chart")
        with c_ev2:
            st.markdown(f"##### {t('hit_eda_sec1_sub2')}")
            st.plotly_chart(create_eda_latency_chart(active_events, selected_eda_zone), width="stretch", key="eda_latency_chart")

        st.info(t("hit_eda_sec1_takeaways"))

        st.markdown("---")

        # ----------------------------------------------------------------------
        # SECTION 2: DEFENSIVE VS. OFFENSIVE ZONE SPLIT
        # ----------------------------------------------------------------------
        st.subheader(t("hit_eda_sec2_heading"))
        st.markdown(t("hit_eda_sec2_intro"))

        c_z1, c_z2 = st.columns(2)
        fig_rec, fig_sog = create_eda_zone_split_charts(active_summary)
        with c_z1:
            st.markdown(f"##### {t('hit_eda_sec2_sub1')}")
            st.plotly_chart(fig_rec, width="stretch", key="eda_zone_split_recovery")
        with c_z2:
            st.markdown(f"##### {t('hit_eda_sec2_sub2')}")
            st.plotly_chart(fig_sog, width="stretch", key="eda_zone_split_sog")

        st.info(t("hit_eda_sec2_takeaways"))

        st.markdown("---")

        # ----------------------------------------------------------------------
        # SECTION 3: SHOT-TO-GOAL CONVERSION FUNNEL
        # ----------------------------------------------------------------------
        st.subheader(t("hit_eda_sec3_heading"))
        st.markdown(t("hit_eda_sec3_intro"))

        col_f_sel, _ = st.columns([2.5, 1])
        with col_f_sel:
            prev_funnel_zone = st.session_state.get("eda_sec3_funnel_zone_val", "All")
            zf_idx = eda_zone_opts.index(prev_funnel_zone) if prev_funnel_zone in eda_zone_opts else 0
            selected_funnel_zone = st.selectbox(
                t("hit_eda_funnel_zone_label"),
                eda_zone_opts,
                format_func=lambda x, l=cur_l: t("hit_tree_opt_all", lang=l) if x == "All" else (
                    t("hit_tree_opt_d", lang=l) if x == "D" else (
                        t("hit_tree_opt_n", lang=l) if x == "N" else t("hit_tree_opt_o", lang=l)
                    )
                ),
                index=zf_idx,
                key=f"eda_sec3_funnel_zone_select_{cur_l}"
            )
            st.session_state.eda_sec3_funnel_zone_val = selected_funnel_zone

        st.plotly_chart(create_eda_funnel_chart(active_summary, selected_funnel_zone), width="stretch", key="eda_funnel_chart")

        st.info(t("hit_eda_sec3_takeaways"))

        st.markdown("---")

        # ----------------------------------------------------------------------
        # SECTION 4: DEFENSIVE ZONE CLEARANCE & BREAKOUT
        # ----------------------------------------------------------------------
        st.subheader(t("hit_eda_sec4_heading"))
        st.markdown(t("hit_eda_sec4_intro"))

        c_ex1, c_ex2 = st.columns([1, 1])
        with c_ex1:
            st.markdown(f"##### {t('hit_eda_sec4_sub1')}")
            st.plotly_chart(create_eda_zone_exit_donut(active_summary), width="stretch", key="eda_zone_exit_donut")
        with c_ex2:
            st.markdown(f"##### {t('hit_eda_sec4_sub2')}")
            d_row_trap = active_summary[active_summary["zoneCode"] == "D"].iloc[0] if not active_summary[active_summary["zoneCode"] == "D"].empty else None
            d_hits = int(d_row_trap["total_hits"]) if d_row_trap is not None else 0
            d_exits = int(d_row_trap["zone_exits"]) if d_row_trap is not None and pd.notna(d_row_trap.get("zone_exits")) else 0
            d_pinned = d_hits - d_exits
            d_exit_pct = (d_exits / d_hits * 100) if d_hits > 0 else 0
            d_pinned_pct = (d_pinned / d_hits * 100) if d_hits > 0 else 0
            d_sog_against = int(d_row_trap["sog_against"]) if d_row_trap is not None else 0
            d_goals_against = int(d_row_trap["goals_against"]) if d_row_trap is not None else 0

            breakdown_text = f"""
            {t("hit_eda_sec4_breakdown_intro").format(hits=f"{d_hits:,}", season=selected_season_label)}
            {t("hit_eda_sec4_clean_exit").format(exits=f"{d_exits:,}", pct=f"{d_exit_pct:.1f}")}
            {t("hit_eda_sec4_pinned_trap").format(pinned=f"{d_pinned:,}", pct=f"{d_pinned_pct:.1f}")}

            {t("hit_eda_sec4_consequences_title")}
            {t("hit_eda_sec4_consequences_sog").format(sog=f"{d_sog_against:,}")}
            {t("hit_eda_sec4_consequences_goals").format(goals=f"{d_goals_against:,}")}
            """
            st.markdown(breakdown_text)

        st.info(t("hit_eda_sec4_takeaways"))

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 1 - TAB 2: SEQUENCE
    # ══════════════════════════════════════════════════════════════════════════
    with main_tabs[1]:
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

            zone_col, step_col = st.columns([1.6, 1.2])
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
            st.plotly_chart(create_sankey_sequence(active_trans, selected_zone, selected_step, active_seq), width="stretch", key="sankey_sequence_chart")

            step_df = active_seq[active_seq["event_step"] == selected_step].copy()
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
            st.plotly_chart(fig_bar, width="stretch", key="event_progression_bar_chart")

        # ──────────────────────────────────────────────────────────────────────
        # SUBTAB 1.2: 30s Goal & Shot Dynamics
        # ──────────────────────────────────────────────────────────────────────
        with subtab_goal:
            st.markdown(t("hit_sub_goal_scope"))
            st.subheader(t("hit_sub_goal_heading"))
            st.markdown(t("hit_sub_goal_intro"))

            st.markdown(f"#### {t('hit_zone_conversion_heading')}")
            kpi_display = active_kpi[[
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
                col_gf: "{:,.0f}",
                col_ga: "{:,.0f}"
            }), width="stretch")

            st.markdown(f"#### {t('hit_time_elapsed_heading')}")
            fig_hist = px.histogram(
                active_goals,
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
            st.plotly_chart(fig_hist, width="stretch", key="possession_duration_hist_chart")

            st.markdown("---")
            st.markdown(f"#### {t('hit_decision_tree_heading')}")
            st.markdown(t("hit_decision_tree_desc"))

            tree_col1, tree_col2 = st.columns([3, 1])
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
                    if "tree_chart_version" in st.session_state:
                        st.session_state.tree_chart_version += 1
                    st.session_state.select_tree_zone_scope_val = selected_tree_zone

            # Initialize session state for interactive tree highlighting
            if "tree_highlight" not in st.session_state:
                st.session_state.tree_highlight = None
            if "last_clicked_tree_pt" not in st.session_state:
                st.session_state.last_clicked_tree_pt = None
            if "tree_highlight_source" not in st.session_state:
                st.session_state.tree_highlight_source = None
            if "tree_chart_version" not in st.session_state:
                st.session_state.tree_chart_version = 0

            # Category filter buttons in order of tree hierarchy
            st.success(t("hit_btn_success_desc"))

            st.markdown("""
            <style>
            /* Equal size and multiline typography for decision tree category buttons */
            div[class*="st-key-btn_tree_"] button {
                width: 100% !important;
                min-height: 48px !important;
                height: auto !important;
                font-weight: 700 !important;
                font-size: 13px !important;
                border-radius: 6px !important;
                text-align: center !important;
                padding: 6px 4px !important;
                white-space: pre-wrap !important;
                word-break: break-word !important;
                line-height: 1.25 !important;
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

            /* Responsive metric card typography */
            div[data-testid="stMetricValue"] {
                font-size: 1.35rem !important;
                white-space: normal !important;
                word-break: break-word !important;
                line-height: 1.25 !important;
            }
            div[data-testid="stMetricLabel"] {
                white-space: normal !important;
                word-break: break-word !important;
                line-height: 1.2 !important;
            }
            </style>
            """, unsafe_allow_html=True)

            chart_key = f"tree_chart_{selected_tree_zone}_{st.session_state.tree_chart_version}"

            # Callback for category button clicks to guarantee state updates before rerun
            def on_tree_ribbon_click(layer, cat, lbl):
                if layer == "clear":
                    st.session_state.tree_highlight = None
                    st.session_state.tree_highlight_source = None
                    st.session_state.last_clicked_tree_pt = None
                else:
                    st.session_state.tree_highlight = {"layer": layer, "category": cat, "label": (lbl or "").replace("\n", " ")}
                    st.session_state.tree_highlight_source = "button"
                    st.session_state.last_clicked_tree_pt = f"{layer}_{cat}"
                st.session_state.tree_chart_version += 1

            # Button ribbon ordered by Tree Hierarchy:
            # Stage 2: Hitting Team (For) -> Opponent (Against)
            # Stage 3: Goals -> Shots on Goal -> Missed Shots -> Blocked Shots
            btn_cols = st.columns(7)
            with btn_cols[0]:
                st.button(t("hit_btn_reset"), key="btn_tree_reset", width="stretch",
                          on_click=on_tree_ribbon_click, args=("clear", None, None))
            with btn_cols[1]:
                st.button(t("hit_btn_for"), key="btn_tree_for", width="stretch",
                          on_click=on_tree_ribbon_click, args=(2, "Hitting Team (For)", t("hit_btn_for").replace("\n", " ")))
            with btn_cols[2]:
                st.button(t("hit_btn_agst"), key="btn_tree_agst", width="stretch",
                          on_click=on_tree_ribbon_click, args=(2, "Opponent (Against)", t("hit_btn_agst").replace("\n", " ")))
            with btn_cols[3]:
                st.button(t("hit_btn_goal"), key="btn_tree_goal", width="stretch",
                          on_click=on_tree_ribbon_click, args=(3, "goal", t("hit_btn_goal").replace("\n", " ")))
            with btn_cols[4]:
                st.button(t("hit_btn_sog"), key="btn_tree_sog", width="stretch",
                          on_click=on_tree_ribbon_click, args=(3, "shot-on-goal", t("hit_btn_sog").replace("\n", " ")))
            with btn_cols[5]:
                st.button(t("hit_btn_miss"), key="btn_tree_miss", width="stretch",
                          on_click=on_tree_ribbon_click, args=(3, "missed-shot", t("hit_btn_miss").replace("\n", " ")))
            with btn_cols[6]:
                st.button(t("hit_btn_block"), key="btn_tree_block", width="stretch",
                          on_click=on_tree_ribbon_click, args=(3, "blocked-shot", t("hit_btn_block").replace("\n", " ")))

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
                            "label": str(cd[2]).replace("\n", " ")
                        }
                else:
                    # Only deselect if the current highlight was initiated by a chart click
                    if st.session_state.get("tree_highlight_source") == "chart":
                        st.session_state.last_clicked_tree_pt = None
                        st.session_state.tree_highlight = None
                        st.session_state.tree_highlight_source = None

            # Render Decision Tree with Plotly and on_select="rerun"
            st.plotly_chart(
                create_shot_outcome_decision_tree(active_goals, selected_tree_zone, highlight=st.session_state.tree_highlight),
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
                    "goal": t("hit_btn_goal").replace("\n", " "),
                    "shot-on-goal": t("hit_btn_sog").replace("\n", " "),
                    "missed-shot": t("hit_btn_miss").replace("\n", " "),
                    "blocked-shot": t("hit_btn_block").replace("\n", " ")
                }
                cat_label = cat_label_map.get(cat_type, hl.get("label", cat_type)).replace("\n", " ")
                st.markdown(f"##### {t('hit_comp_l3_heading').format(label=cat_label)}")
                st.info(t("hit_comp_l3_info").format(label=cat_label))

                df_comp3 = get_layer_3_comparison(active_goals, cat_type, selected_tree_zone)
                for_cnt = int(df_comp3[df_comp3[col_branch] == t("hit_tree_side_for")][col_count].sum()) if not df_comp3.empty and col_branch in df_comp3 else 0
                agst_cnt = int(df_comp3[df_comp3[col_branch] == t("hit_tree_side_agst")][col_count].sum()) if not df_comp3.empty and col_branch in df_comp3 else 0
                tot_cnt = for_cnt + agst_cnt
                net_diff = for_cnt - agst_cnt

                for_share = (for_cnt / tot_cnt * 100.0) if tot_cnt > 0 else 0.0
                agst_share = (agst_cnt / tot_cnt * 100.0) if tot_cnt > 0 else 0.0
                net_margin = for_share - agst_share

                w_b_share = (df_comp3[col_b_share] * df_comp3[col_count]).sum() / tot_cnt if tot_cnt > 0 else 0.0
                w_z_share = (df_comp3[col_z_share] * df_comp3[col_count]).sum() / tot_cnt if tot_cnt > 0 else 0.0
                w_time = (df_comp3[col_time] * df_comp3[col_count]).sum() / tot_cnt if tot_cnt > 0 else 0.0

                active_events = df_comp3[df_comp3[col_count] > 0] if not df_comp3.empty and col_count in df_comp3 else pd.DataFrame()
                if not active_events.empty:
                    fastest_idx = active_events[col_time].idxmin()
                    fastest_row = active_events.loc[fastest_idx]
                    fastest_desc = f"{fastest_row[col_zone]} ({fastest_row[col_time]:.1f}s)"
                else:
                    fastest_desc = "N/A"

                # 2 Row Metrics with 3 Metric Columns (Comparable % Shares & Differentials)
                r1_c1, r1_c2, r1_c3 = st.columns(3)
                r1_c1.metric(t("hit_l3_m_for_share"), f"{for_share:.1f}%", f"{for_cnt:,} events ({for_share - 50.0:+.1f}% vs 50%)")
                r1_c2.metric(t("hit_l3_m_agst_share"), f"{agst_share:.1f}%", f"{agst_cnt:,} events", delta_color="inverse")
                r1_c3.metric(t("hit_l3_m_net_margin"), f"{net_margin:+.1f}%", f"{net_diff:+,} net differential")

                r2_c1, r2_c2, r2_c3 = st.columns(3)
                r2_c1.metric(t("hit_th_branch_share"), f"{w_b_share:.1f}%", t("hit_l3_m_branch_help"))
                r2_c2.metric(t("hit_th_zone_share"), f"{w_z_share:.1f}%", t("hit_l3_m_zone_help"))
                r2_c3.metric(t("hit_th_avg_elapsed"), f"{w_time:.1f}s", f"Fastest: {fastest_desc}")

                st.markdown("---")
                # Measurements in 3 columns across rows: Branch Share %, Zone Share %, and Average Elapsed Time
                st.markdown(f"**{cat_label} — Measurements by Origin Zone & Possession Branch**")
                for _, r in df_comp3.iterrows():
                    z_name = r[col_zone]
                    b_name = r[col_branch]
                    c_cnt = int(r[col_count])
                    b_pct = r[col_b_share]
                    z_pct = r[col_z_share]
                    t_sec = r[col_time]

                    st.markdown(f"**{z_name} — {b_name}** (`{c_cnt:,}` events)")
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric(t("hit_th_branch_share"), f"{b_pct:.1f}%")
                    mc2.metric(t("hit_th_zone_share"), f"{z_pct:.1f}%")
                    mc3.metric(t("hit_th_avg_elapsed"), f"{t_sec:.1f}s")

            elif hl and hl.get("layer") == 2:
                # Possession branch highlighted -> Compare across ONE higher level: Hit Origin Zone
                cat_side = hl["category"]
                side_label_map = {
                    "Hitting Team (For)": t("hit_btn_for").replace("\n", " "),
                    "Opponent (Against)": t("hit_btn_agst").replace("\n", " ")
                }
                cat_label = side_label_map.get(cat_side, hl.get("label", cat_side)).replace("\n", " ")
                st.markdown(f"##### {t('hit_comp_l2_heading').format(label=cat_label)}")
                st.info(t("hit_comp_l2_info").format(label=cat_label))

                df_comp2 = get_layer_2_comparison(active_goals, cat_side, selected_tree_zone)
                col_shots = t("hit_th_30s_shots")
                col_goals = t("hit_th_goals")
                tot_shots = int(df_comp2[col_shots].sum()) if not df_comp2.empty and col_shots in df_comp2 else 0
                tot_goals = int(df_comp2[col_goals].sum()) if not df_comp2.empty and col_goals in df_comp2 else 0
                conv_rate = (tot_goals / tot_shots * 100.0) if tot_shots > 0 else 0.0
                w_time = (df_comp2[col_time] * df_comp2[col_shots]).sum() / tot_shots if tot_shots > 0 else 0
                w_sog = (df_comp2[t("hit_th_sog_pct")] * df_comp2[col_shots]).sum() / tot_shots if tot_shots > 0 else 0
                raw_top_zone = df_comp2.sort_values(col_shots, ascending=False).iloc[0][col_zone] if not df_comp2.empty and col_shots in df_comp2 else "N/A"
                clean_top_zone = (
                    raw_top_zone.replace(" Zone", "")
                    .replace(" zone", "")
                    .replace("Zone ", "")
                    .replace("zone ", "")
                    .replace("D-Zone", "D")
                    .replace("Zone D", "D")
                    .replace("zone D", "D")
                    .strip()
                )
                if clean_top_zone:
                    clean_top_zone = clean_top_zone[0].upper() + clean_top_zone[1:]

                # 2 Row Metrics with 3 Metric Columns
                r1_c1, r1_c2, r1_c3 = st.columns(3)
                r1_c1.metric(t("hit_l2_m_tot_shots"), f"{tot_shots:,}")
                r1_c2.metric(t("hit_l2_m_tot_goals"), f"{tot_goals:,}")
                r1_c3.metric(t("hit_l2_m_goal_conv"), f"{conv_rate:.1f}%", f"{tot_goals} goals / {tot_shots:,} shots")

                r2_c1, r2_c2, r2_c3 = st.columns(3)
                r2_c1.metric(t("hit_l2_m_sog_rate"), f"{w_sog:.1f}%")
                r2_c2.metric(t("hit_l2_m_top_zone"), clean_top_zone)
                r2_c3.metric(t("hit_l2_m_avg_time"), f"{w_time:.1f}s")

                st.markdown("---")
                st.markdown(f"**{cat_label} — Measurements by Origin Zone**")
                for _, r in df_comp2.iterrows():
                    z_name = r[col_zone]
                    s_cnt = int(r[col_shots])
                    z_share = r[t("hit_th_zone_poss_share")]
                    g_conv = r[t("hit_th_goal_conv")]
                    t_sec = r[col_time]

                    z_lower = str(z_name).lower()
                    z_icon = "⚔️" if any(k in z_lower for k in ["offens", "forecheck", "échec"]) else ("🛡️" if any(k in z_lower for k in ["défens", "defens", "zone d"]) else "🧊")
                    st.markdown(f"**{z_icon} {z_name}** (`{s_cnt:,}` shots, `{int(r[col_goals]):,}` goals)")
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric(t("hit_th_zone_poss_share"), f"{z_share:.1f}%")
                    mc2.metric(t("hit_th_goal_conv"), f"{g_conv:.1f}%")
                    mc3.metric(t("hit_th_avg_elapsed"), f"{t_sec:.1f}s")

            elif hl and hl.get("layer") == 1:
                # Hit origin zone highlighted -> Compare against all zones
                cat_zone = normalize_zone(hl["category"])
                zone_label_map = {
                    "O": t("hit_tree_zone_o"),
                    "D": t("hit_tree_zone_d"),
                    "N": t("hit_tree_zone_n")
                }
                cat_label = zone_label_map.get(cat_zone, hl.get("label", cat_zone)).replace("\n", " ")
                st.markdown(f"##### {t('hit_comp_l1_heading').format(label=cat_label)}")
                st.info(t("hit_comp_l1_info").format(label=cat_label))

                col_tot_30s = t("hit_th_total_30s")
                col_status = t("hit_th_status")
                col_for_s = t("hit_th_for_shots")
                col_agst_s = t("hit_th_agst_shots")
                col_puck = t("hit_th_puck_win")
                col_net_g = t("hit_th_net_goal_diff")

                df_comp1 = get_layer_1_comparison(active_goals, cat_zone)
                sel_mask = df_comp1[col_status] == t("hit_val_selected_focus")
                sel_row = df_comp1[sel_mask].iloc[0] if not df_comp1[sel_mask].empty else df_comp1.iloc[0]

                puck_win_val = sel_row[col_puck]
                tot_shots_val = int(sel_row[col_tot_30s])
                net_shots_val = int(sel_row[col_for_s] - sel_row[col_agst_s])
                gf_val = int(sel_row[t("hit_th_goals_for")])
                ga_val = int(sel_row[t("hit_th_goals_agst")])
                net_goals_val = int(sel_row[col_net_g])

                # 2 Row Metrics with 3 Metric Columns
                r1_c1, r1_c2, r1_c3 = st.columns(3)
                r1_c1.metric(t("hit_l1_m_puck_win"), f"{puck_win_val:.1f}%", f"{puck_win_val - 50.0:+.1f}% vs 50%")
                r1_c2.metric(t("hit_l1_m_shots_vol"), f"{tot_shots_val:,}")
                r1_c3.metric(t("hit_l1_m_net_shots"), f"{net_shots_val:+,}")

                r2_c1, r2_c2, r2_c3 = st.columns(3)
                r2_c1.metric(t("hit_l1_m_goals_for"), f"{gf_val:,}")
                r2_c2.metric(t("hit_l1_m_goals_agst"), f"{ga_val:,}", delta_color="inverse")
                r2_c3.metric(t("hit_l1_m_net_goals"), f"{net_goals_val:+,}")

                st.markdown("---")
                st.markdown(f"**{cat_label} — Measurements across Origin Zones**")
                for _, r in df_comp1.iterrows():
                    z_name = r[col_zone]
                    p_win = r[col_puck]
                    n_s = int(r[col_for_s] - r[col_agst_s])
                    n_g = int(r[col_net_g])
                    is_focus = r[col_status] == t("hit_val_selected_focus")
                    tag = f" 🌟 *({t('hit_val_selected_focus')})*" if is_focus else ""

                    z_lower = str(z_name).lower()
                    z_icon = "⚔️" if any(k in z_lower for k in ["offens", "forecheck", "échec"]) else ("🛡️" if any(k in z_lower for k in ["défens", "defens", "zone d"]) else "🧊")
                    st.markdown(f"**{z_icon} {z_name}**{tag} (`{int(r[col_tot_30s]):,}` total 30s shots)")
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric(t("hit_th_puck_win"), f"{p_win:.1f}%")
                    mc2.metric(t("hit_th_net_shot_diff"), f"{n_s:+,}")
                    mc3.metric(t("hit_th_net_goal_diff"), f"{n_g:+,}")

            else:
                # Default: No highlight active -> Display comprehensive Decision Matrix
                st.markdown(f"##### {t('hit_comp_default_heading')}")
                st.caption(t("hit_comp_default_tip"))
                matrix_df = get_decision_matrix(active_goals, selected_tree_zone)

                st.markdown(f"**{t('hit_comp_default_heading')} — Measurements by Trajectory Branch**")
                zone_desc_map = {
                    t("hit_tree_zone_o"): t("hit_comp_zone_desc_o"),
                    t("hit_tree_zone_d"): t("hit_comp_zone_desc_d"),
                    t("hit_tree_zone_n"): t("hit_comp_zone_desc_n")
                }

                col_zone_hdr = t("hit_th_origin_zone")
                col_branch_hdr = t("hit_th_possession_branch")
                col_shots_hdr = t("hit_th_30s_shots")
                col_goals_hdr = t("hit_th_goals")
                col_bshare_hdr = t("hit_th_branch_share")
                col_gconv_hdr = t("hit_th_goal_conv")
                col_time_hdr = t("hit_th_avg_elapsed")

                zone_icon_map = {
                    t("hit_tree_zone_o"): "⚔️",
                    t("hit_tree_zone_d"): "🛡️",
                    t("hit_tree_zone_n"): "🧊"
                }

                for z_name, z_group in matrix_df.groupby(col_zone_hdr, sort=False):
                    st.markdown("---")
                    z_icon = zone_icon_map.get(z_name)
                    if not z_icon:
                        z_lower = str(z_name).lower()
                        if any(k in z_lower for k in ["offens", "forecheck", "échec"]):
                            z_icon = "⚔️"
                        elif any(k in z_lower for k in ["défens", "defens", "zone d"]):
                            z_icon = "🛡️"
                        elif any(k in z_lower for k in ["neutr", "rupture"]):
                            z_icon = "🧊"
                        else:
                            z_icon = "🏒"
                    st.markdown(f"##### {z_icon} {z_name}")
                    z_desc = zone_desc_map.get(z_name, "")
                    if z_desc:
                        st.caption(z_desc)

                    for _, r in z_group.iterrows():
                        b_name = r[col_branch_hdr]
                        s_cnt = int(r[col_shots_hdr])
                        b_share = r[col_bshare_hdr]
                        g_conv = r[col_gconv_hdr]
                        t_sec = r[col_time_hdr]

                        st.markdown(f"**{b_name}** (`{s_cnt:,}` shots, `{int(r[col_goals_hdr]):,}` goals)")
                        mc1, mc2, mc3 = st.columns(3)
                        mc1.metric(col_bshare_hdr, f"{b_share:.1f}%")
                        mc2.metric(col_gconv_hdr, f"{g_conv:.1f}%")
                        mc3.metric(col_time_hdr, f"{t_sec:.1f}s")

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 1 - TAB 3: RINK LOCATION
    # ══════════════════════════════════════════════════════════════════════════
    with main_tabs[2]:
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
            st.subheader(t("hit_sub_rink_heading"))
            st.caption(f"{t('hit_sub_rink_scope')} — 💡 {get_rink_text('hit_rink_click_instruction')}")

            # Initialize session state for selected location
            if "selected_hit_loc" not in st.session_state:
                st.session_state.selected_hit_loc = None
            if "last_rink_clicked_pt" not in st.session_state:
                st.session_state.last_rink_clicked_pt = None
            if "rink_chart_version" not in st.session_state:
                st.session_state.rink_chart_version = 0

            cur_l = st.session_state.get("lang", "EN")
            prev_rink_view = st.session_state.get("heat_rink_view_val", "full")
            view_opts = ["full", "half"]
            v_idx = view_opts.index(prev_rink_view) if prev_rink_view in view_opts else 0

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

            # Row 1: Selectors & View Filters
            rink_r1_c1, rink_r1_c2, rink_r1_c3, rink_r1_c4 = st.columns([1.1, 1.4, 1.1, 1.3])
            with rink_r1_c1:
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
                        st.session_state.rink_chart_version += 1

            with rink_r1_c2:
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

            with rink_r1_c3:
                ev_type_opts = ["goals", "shots"]
                prev_ev_type = st.session_state.get("rink_ev_type_val", "goals")
                ev_type_idx = ev_type_opts.index(prev_ev_type) if prev_ev_type in ev_type_opts else 0
                ev_type_labels = {
                    "goals": t("hit_rink_event_goals", lang=cur_l),
                    "shots": t("hit_rink_event_shots", lang=cur_l)
                }
                event_type_choice = st.radio(
                    t("hit_rink_event_type_label"),
                    ev_type_opts,
                    format_func=lambda x, p=ev_type_labels: p.get(x, x),
                    index=ev_type_idx,
                    horizontal=True,
                    key=f"rink_ev_type_{cur_l}"
                )
                st.session_state.rink_ev_type_val = event_type_choice

            with rink_r1_c4:
                side_opts = ["both", "for", "against"]
                prev_side = st.session_state.get("rink_side_filter_val", "both")
                side_idx = side_opts.index(prev_side) if prev_side in side_opts else 0
                side_labels = {
                    "both": t("hit_rink_side_both", lang=cur_l),
                    "for": t("hit_rink_side_for", lang=cur_l),
                    "against": t("hit_rink_side_agst", lang=cur_l)
                }
                side_choice = st.selectbox(
                    t("hit_rink_side_label"),
                    side_opts,
                    format_func=lambda x, p=side_labels: p.get(x, x),
                    index=side_idx,
                    key=f"rink_side_filter_{cur_l}"
                )
                st.session_state.rink_side_filter_val = side_choice

            def on_apply_rink_preset(coords):
                st.session_state.selected_hit_loc = coords
                st.session_state.last_rink_clicked_pt = None
                st.session_state.rink_chart_version += 1

            def on_clear_rink_selection():
                st.session_state.selected_hit_loc = None
                st.session_state.last_rink_clicked_pt = None
                st.session_state.rink_chart_version += 1

            # Row 2: Action Buttons placed on second line under selector (wide, comfortable layout)
            col_btn_set, col_btn_clear = st.columns([1, 1])
            with col_btn_set:
                st.button(
                    t("hit_rink_preset_btn"),
                    key="btn_apply_rink_preset",
                    on_click=on_apply_rink_preset,
                    args=(hotspot_preset_coords[preset_key],),
                    use_container_width=True
                )
            with col_btn_clear:
                st.button(
                    get_rink_text("hit_rink_clear_btn"),
                    key="btn_clear_rink_sel",
                    on_click=on_clear_rink_selection,
                    disabled=(st.session_state.selected_hit_loc is None),
                    use_container_width=True
                )

            rink_chart_key = f"rink_heat_chart_{st.session_state.rink_chart_version}"

            # 1. Process Reactive Chart Click from Session State before building figure
            chart_state = st.session_state.get(rink_chart_key)
            if chart_state and isinstance(chart_state, dict):
                sel = chart_state.get("selection", {})
                pts = sel.get("points", [])
                if pts:
                    pt = pts[0]
                    px_val = pt.get("x")
                    py_val = pt.get("y")
                    if px_val is not None and py_val is not None:
                        try:
                            # Snap click to nearest 10-ft rink grid tile
                            clicked_x = round(float(px_val) / 10.0) * 10.0
                            clicked_y = round(float(py_val) / 10.0) * 10.0
                            if view_mode == "half" and clicked_x < 0:
                                clicked_x = 0.0
                            clicked_pt = (clicked_x, clicked_y)
                            if ((abs(active_grid["x_bin"] - clicked_x) < 1e-3) & (abs(active_grid["y_bin"] - clicked_y) < 1e-3)).any():
                                st.session_state.selected_hit_loc = clicked_pt
                        except (ValueError, TypeError):
                            pass

            # Render Heatmap figure (with active cell highlighted, unselected cells shaded out)
            fig_rink = render_interactive_rink_heatmap(
                active_grid,
                active_goals,
                selected_loc=st.session_state.selected_hit_loc,
                half_rink=(view_mode == "half"),
                min_density=0.0,
                event_type=event_type_choice,
                side_filter=side_choice
            )

            # 2. Reactive click selection via Streamlit plotly_chart on_select
            chart_event = st.plotly_chart(
                fig_rink,
                width="stretch",
                on_select="rerun",
                selection_mode=["points"],
                key=rink_chart_key
            )

            # 3. Immediate reactive check: if click occurred on this rerun and wasn't set yet, update & rerun
            if chart_event and isinstance(chart_event, dict):
                sel = chart_event.get("selection", {})
                pts = sel.get("points", [])
                if pts:
                    pt = pts[0]
                    px_val = pt.get("x")
                    py_val = pt.get("y")
                    if px_val is not None and py_val is not None:
                        try:
                            clicked_x = round(float(px_val) / 10.0) * 10.0
                            clicked_y = round(float(py_val) / 10.0) * 10.0
                            if view_mode == "half" and clicked_x < 0:
                                clicked_x = 0.0
                            clicked_pt = (clicked_x, clicked_y)
                            if ((abs(active_grid["x_bin"] - clicked_x) < 1e-3) & (abs(active_grid["y_bin"] - clicked_y) < 1e-3)).any():
                                if st.session_state.get("selected_hit_loc") != clicked_pt:
                                    st.session_state.selected_hit_loc = clicked_pt
                                    st.rerun()
                        except (ValueError, TypeError):
                            pass

            # Detailed Inspection Panel for Selected Location
            if st.session_state.selected_hit_loc is not None:
                sel_x, sel_y = st.session_state.selected_hit_loc
                matched_cell = active_grid[(abs(active_grid["x_bin"] - sel_x) < 1e-3) & (abs(active_grid["y_bin"] - sel_y) < 1e-3)]

                if not matched_cell.empty:
                    cell = matched_cell.iloc[0]
                    zone_desc = t("hit_tree_zone_o") if cell["spatial_zone"] == "O" else (t("hit_tree_zone_d") if cell["spatial_zone"] == "D" else t("hit_tree_zone_n"))

                    # Safe metric extractions with robust zero fallbacks
                    s_for_30s = int(cell.get("shots_for_30s", 0)) if "shots_for_30s" in cell else 0
                    s_agst_30s = int(cell.get("shots_against_30s", 0)) if "shots_against_30s" in cell else 0
                    net_shots_val = int(cell.get("net_shots_30s", s_for_30s - s_agst_30s))
                    g_for_30s = int(cell.get("goals_for_30s", 0)) if "goals_for_30s" in cell else 0
                    g_agst_30s = int(cell.get("goals_against_30s", 0)) if "goals_against_30s" in cell else 0
                    tot_hits = int(cell.get("total_hits", 0))
                    density_val = float(cell.get("density_pct", 0.0))
                    puck_win_val = float(cell.get("puck_win_pct", 0.0))
                    s_for_pct = float(cell.get("shot_for_pct", 0.0))
                    s_agst_pct = float(cell.get("shot_against_pct", 0.0))

                    st.markdown(f"### {t('hit_rink_inspected_loc').format(x=f'{sel_x:+.0f}', y=f'{sel_y:+.0f}', zone=zone_desc)}")

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
                                    hits=f"{tot_hits:,}",
                                    pct=f"{density_val:.2f}",
                                    win=f"{puck_win_val:.1f}",
                                    win_diff=f"{puck_win_val - 46.2:+.1f}",
                                    s_for=f"{s_for_pct:.1f}",
                                    s_agst=f"{s_agst_pct:.1f}",
                                    net_shots=f"{net_shots_val:+d}",
                                    g_for=g_for_30s,
                                    g_agst=g_agst_30s
                                )}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric(t("hit_panel_m1_label"), f"{tot_hits:,}", f"{density_val:.2f}%")
                    m2.metric(t("hit_panel_m2_label"), f"{puck_win_val:.1f}%", f"{puck_win_val - 46.2:+.1f}%")
                    m3.metric(t("hit_panel_m3_label"), f"{s_for_pct:.1f}%")
                    m4.metric(t("hit_panel_m4_label"), f"{s_agst_pct:.1f}%", delta_color="inverse")
                    m5.metric(t("hit_panel_m5_label"), f"{net_shots_val:+d}", t("hit_panel_m5_delta").format(g_for=g_for_30s, g_agst=g_agst_30s))

                    # Subsequent event categorical breakdown bar chart
                    st.markdown(f"#### {t('hit_rink_sub_dist_heading').format(x=f'{sel_x:+.0f}', y=f'{sel_y:+.0f}')}")
                    cat_col = t("hit_panel_category_col")
                    rate_col = t("hit_panel_rate_col")
                    side_col = t("hit_panel_side_col")
                    cat_data = pd.DataFrame([
                        {cat_col: t("hit_panel_shots_for"), rate_col: float(cell.get("shot_for_pct", 0)), side_col: t("hit_owner_for")},
                        {cat_col: t("hit_panel_takeaways_won"), rate_col: float(cell.get("takeaway_for_pct", 0)), side_col: t("hit_owner_for")},
                        {cat_col: t("hit_panel_goals_scored"), rate_col: float(cell.get("goal_for_pct", 0)), side_col: t("hit_owner_for")},
                        {cat_col: t("hit_panel_shots_conceded"), rate_col: float(cell.get("shot_against_pct", 0)), side_col: t("hit_owner_against")},
                        {cat_col: t("hit_panel_giveaways_committed"), rate_col: float(cell.get("giveaway_against_pct", 0)), side_col: t("hit_owner_against")},
                        {cat_col: t("hit_panel_goals_conceded"), rate_col: float(cell.get("goal_against_pct", 0)), side_col: t("hit_owner_against")},
                        {cat_col: t("hit_panel_phys_battle"), rate_col: float(cell.get("physical_battle_pct", 0)), side_col: t("hit_panel_neutral_battle")},
                        {cat_col: t("hit_panel_stoppage"), rate_col: float(cell.get("stoppage_pct", 0)), side_col: t("hit_panel_neutral_battle")}
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
                    st.plotly_chart(fig_loc_bar, width="stretch", key="spatial_zone_distribution_bar")
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

            if len(active_team_kpi["team"].drop_duplicates()) < len(active_team_kpi):
                active_team_kpi = aggregate_team_kpi(active_team_kpi)

            team_list = sorted([tm for tm in active_team_kpi["team"].unique() if tm != "LEAGUE_AVG"])
            if not team_list:
                team_list = sorted([tm for tm in df_team_kpi["team"].unique() if tm != "LEAGUE_AVG"])
            team_options = [team_name_map.get(tm, tm) for tm in team_list]
            tri_to_opt = {tm: team_name_map.get(tm, tm) for tm in team_list}
            opt_to_tri = {v: k for k, v in tri_to_opt.items()}

            st.caption(f"📅 **{t('hit_team_sel_season_label')}** {selected_season_label}")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                cur_l = st.session_state.get("lang", "EN")
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
            t1_data = active_team_kpi[active_team_kpi["team"] == primary_tri]
            t2_data = active_team_kpi[active_team_kpi["team"] == benchmark_tri]

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
                    focus_prefix = t("hit_val_selected_focus")
                    st.markdown(
                        f"<div style='font-size: 13.5px; font-weight: 700; color: #f8fafc; margin-bottom: 2px; padding-left: 2px;'>"
                        f"{focus_prefix}: {primary_choice} "
                        f"<span style='font-size: 11.5px; font-weight: 400; color: #94a3b8;'>({selected_season_label})</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    fig_t1 = render_team_rink_heatmap(
                        df_team_spatial, primary_tri, selected_season,
                        display_name=primary_choice,
                        is_benchmark=False,
                        season_label=selected_season_label,
                        min_density=1.0,
                        show_title=False
                    )
                    st.plotly_chart(fig_t1, width="stretch", key="team_spatial_density_t1")
                with h_col2:
                    bench_prefix = t("hit_val_benchmark")
                    st.markdown(
                        f"<div style='font-size: 13.5px; font-weight: 700; color: #f8fafc; margin-bottom: 2px; padding-left: 2px;'>"
                        f"{bench_prefix}: {benchmark_label} "
                        f"<span style='font-size: 11.5px; font-weight: 400; color: #94a3b8;'>({selected_season_label})</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                    fig_t2 = render_team_rink_heatmap(
                        df_team_spatial, benchmark_tri, selected_season,
                        display_name=benchmark_label,
                        is_benchmark=True,
                        season_label=selected_season_label,
                        min_density=1.0,
                        show_title=False
                    )
                    st.plotly_chart(fig_t2, width="stretch", key="team_spatial_density_t2")

                # League-wide 32-Team Scatter Plot
                st.markdown(f"#### {t('hit_team_landscape_heading')}")
                st.markdown(t("hit_team_landscape_purpose"))
                st.info(t("hit_team_landscape_takeaways"))
                league_teams = active_team_kpi[active_team_kpi["team"] != "LEAGUE_AVG"].copy()

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

                    # Benchmark dashed quadrant lines at league mean
                    x_mid = float(league_teams["ozone_pct"].mean())
                    y_mid = float(league_teams["puck_win_pct"].mean())
                    fig_scatter.add_vline(x=x_mid, line_dash="dash", line_color="rgba(255,255,255,0.35)", line_width=1.5)
                    fig_scatter.add_hline(y=y_mid, line_dash="dash", line_color="rgba(255,255,255,0.35)", line_width=1.5)

                    # Quadrant Corner Annotations matching analytical takeaways
                    fig_scatter.add_annotation(
                        x=float(league_teams["ozone_pct"].max()), y=float(league_teams["puck_win_pct"].max()),
                        text=f"<b>{t('hit_scatter_quad_top_right')}</b>", showarrow=False,
                        xanchor="right", yanchor="top", font=dict(color="#86efac", size=10)
                    )
                    fig_scatter.add_annotation(
                        x=float(league_teams["ozone_pct"].max()), y=float(league_teams["puck_win_pct"].min()),
                        text=f"<b>{t('hit_scatter_quad_bot_right')}</b>", showarrow=False,
                        xanchor="right", yanchor="bottom", font=dict(color="#fca5a5", size=10)
                    )
                    fig_scatter.add_annotation(
                        x=float(league_teams["ozone_pct"].min()), y=float(league_teams["puck_win_pct"].max()),
                        text=f"<b>{t('hit_scatter_quad_top_left')}</b>", showarrow=False,
                        xanchor="left", yanchor="top", font=dict(color="#93c5fd", size=10)
                    )
                    fig_scatter.add_annotation(
                        x=float(league_teams["ozone_pct"].min()), y=float(league_teams["puck_win_pct"].min()),
                        text=f"<b>{t('hit_scatter_quad_bot_left')}</b>", showarrow=False,
                        xanchor="left", yanchor="bottom", font=dict(color="#cbd5e1", size=10)
                    )

                    fig_scatter.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="white"),
                        height=520,
                        margin=dict(l=10, r=10, t=40, b=10)
                    )
                    st.plotly_chart(fig_scatter, width="stretch", key="team_efficiency_scatter_chart")
            else:
                st.warning(f"No metric data available for {primary_choice} or {benchmark_label} for season {selected_season_label}.")

    # ══════════════════════════════════════════════════════════════════════════
    # LAYER 1 - TAB 4: TACTICAL INSIGHTS
    # ══════════════════════════════════════════════════════════════════════════
    with main_tabs[3]:
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