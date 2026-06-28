"""
goalie_stats.py — NHL Goalie Statistics
Tab hierarchy:
  Top-level Tab 1: Goalie Profile + Game Log (goalie-specific, weekly game log)
  Top-level Tab 2: PPI Leaderboard (season-wide, no goalie selection needed)
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys, os
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'exe'))

from i18n import t
import importlib
import player_stats
importlib.reload(player_stats)

from player_stats import (
    load_player_season_data, load_player_game_data,
    compute_goalie_ppi, get_available_seasons, get_player_list,
    resolve_game_player_name, aggregate_weekly_goalie,
)


# ── Page Setup ────────────────────────────────────────────────────────────────
st.title(t("gl_title"))
st.markdown(t("gl_desc"))
st.divider()

# ── Load Data ─────────────────────────────────────────────────────────────────
is_fr = st.session_state.get('lang', 'FR') == 'FR'
loading_msg = "Chargement..." if is_fr else "Loading..."
with st.spinner(loading_msg):
    _, goalies = load_player_season_data()
    _, goalie_games = load_player_game_data()

if goalies is None or goalies.empty:
    st.error(t("gl_no_data"))
    st.stop()

seasons = get_available_seasons(goalies)
season_labels = [s[1] for s in seasons]
season_years  = [s[0] for s in seasons]
all_teams     = sorted(goalies['team_tri'].dropna().unique().tolist())
all_names     = get_player_list(goalies)
DEFAULT       = "Jakub Dobeš"

# ── Top-level Tabs ────────────────────────────────────────────────────────────
TAB_PLAYER     = "👤 " + ("Gardien" if is_fr else "Goalie")
TAB_LEADERBOARD = t("gl_tab_leaderboard")
top_tab1, top_tab2 = st.tabs([TAB_PLAYER, TAB_LEADERBOARD])

# ══════════════════════════════════════════════════════════════════════════════
# TOP TAB 1 — Goalie Profile + Game Log
# ══════════════════════════════════════════════════════════════════════════════
with top_tab1:
    # Row 1: Team, Season
    fr1_col1, fr1_col2 = st.columns(2)
    with fr1_col1:
        selected_team = st.selectbox(t("gl_select_team"), [t("gl_all_teams")] + all_teams, key="gl_team_filter")
    with fr1_col2:
        sel_season_lbl = st.selectbox(t("gl_select_season"), season_labels[::-1], key="gl_season_filter")
        sel_season_yr  = season_years[season_labels.index(sel_season_lbl)]

    # Filter goalies to restrict goalie name list
    filtered_goalies = goalies.copy()
    if selected_team != t("gl_all_teams"):
        filtered_goalies = filtered_goalies[filtered_goalies['team_tri'] == selected_team]

    filtered_names = get_player_list(filtered_goalies)

    if not filtered_names:
        st.warning("Aucun gardien ne correspond à ces critères. / No goalies match these criteria.")
        st.stop()

    # Row 2: Goalie selection
    default_idx = filtered_names.index(DEFAULT) if DEFAULT in filtered_names else 0
    selected_goalie = st.selectbox(t("gl_search"), filtered_names, index=default_idx, key="gl_goalie_select")


    # ── Inner Tabs ──
    inner1, inner2 = st.tabs([t("gl_tab_profile"), t("gl_tab_gamelog")])

    # ── INNER TAB 1: Season Profile ───────────────────────────────────────────
    with inner1:
        goalie_df = goalies[goalies['fullName'] == selected_goalie].copy()
        if selected_team != t("gl_all_teams"):
            goalie_df = goalie_df[goalie_df['team_tri'] == selected_team]

        if goalie_df.empty:
            st.warning(t("gl_no_data"))
        else:
            latest = goalie_df.sort_values('season_year', ascending=False).iloc[0]
            st.markdown(f"### {selected_goalie} · {latest.get('team_tri','')} · {t('gl_tab_profile')}")

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("PJ" if is_fr else "GP", int(latest.get('gamesPlayed', 0) or 0))
            m2.metric("V"  if is_fr else "W",  int(latest.get('wins', 0) or 0))
            m3.metric("D"  if is_fr else "L",  int(latest.get('losses', 0) or 0))
            m4.metric("DPR" if is_fr else "OTL", int(latest.get('overtimeLosses', 0) or 0))
            gaa = latest.get('goalsAgainstAverage')
            svp = latest.get('savePercentage')
            m5.metric("MAB" if is_fr else "GAA", f"{gaa:.2f}" if pd.notna(gaa) else "—")
            m6.metric("% Arr." if is_fr else "SV%", f"{svp:.3f}" if pd.notna(svp) else "—")

            st.divider()
            st.subheader(t("gl_career_trend"))
            trend = goalie_df.sort_values('season_year')

            fig = go.Figure()
            fig.add_trace(go.Bar(x=trend['season_label'], y=trend['savePercentage'],
                                 name=t("chart_save_pct"), marker_color='#1a85ff', opacity=0.75))
            fig.add_trace(go.Scatter(x=trend['season_label'], y=trend['goalsAgainstAverage'],
                                     name=t("chart_gaa"), mode='lines+markers', yaxis='y2',
                                     line=dict(color='#CE0E2D', width=3),
                                     marker=dict(size=8, color='#CE0E2D')))
            fig.add_trace(go.Scatter(x=trend['season_label'], y=trend['wins'],
                                     name=t("chart_wins"), mode='lines+markers', yaxis='y3',
                                     line=dict(color='#f4a418', width=2, dash='dot'),
                                     marker=dict(size=6)))
            fig.update_layout(
                yaxis=dict(title=t("chart_y_svpct"), range=[0.88, 0.95], gridcolor='rgba(255,255,255,0.1)'),
                yaxis2=dict(title=t("chart_gaa"), overlaying='y', side='right', showgrid=False, range=[1.5, 4.0]),
                yaxis3=dict(title=t("chart_wins"), overlaying='y', side='right', anchor='free', position=1.0, showgrid=False),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation='h', y=1.14), margin=dict(t=60, b=40, r=100),
                hovermode='x unified',
            )
            st.plotly_chart(fig, width="stretch")

            st.divider()
            st.subheader(t("gl_season_stats"))
            col_map = {
                'season_label': 'Saison' if is_fr else 'Season',
                'team_tri':     'Équipe' if is_fr else 'Team',
                'gamesPlayed':  'PJ' if is_fr else 'GP',
                'gamesStarted': 'PD' if is_fr else 'GS',
                'wins':         'V' if is_fr else 'W',
                'losses':       'D' if is_fr else 'L',
                'overtimeLosses': 'DPR' if is_fr else 'OTL',
                'win_pct':      'V%' if is_fr else 'W%',
                'goalsAgainstAverage': 'MAB' if is_fr else 'GAA',
                'savePercentage': '% Arr.' if is_fr else 'SV%',
                'shutouts':     'BL' if is_fr else 'SO',
                'saves':        'Arr.' if is_fr else 'Saves',
                'shotsAgainst': 'TC' if is_fr else 'SA',
                'saves_per_game': 'Arr./PJ' if is_fr else 'Saves/GP',
            }
            avail = {k: v for k, v in col_map.items() if k in goalie_df.columns}
            tbl = goalie_df[list(avail.keys())].sort_values('season_label', ascending=False).copy()
            tbl.rename(columns=avail, inplace=True)
            wkey = 'V%' if is_fr else 'W%'
            if wkey in tbl.columns:
                tbl[wkey] = tbl[wkey].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
            for col in (['MAB','% Arr.','Arr./PJ'] if is_fr else ['GAA','SV%','Saves/GP']):
                if col in tbl.columns:
                    tbl[col] = tbl[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
            int_keys = ['PJ','GP','PD','GS','V','W','D','L','DPR','OTL','BL','SO','Arr.','Saves','TC','SA']
            for col in int_keys:
                if col in tbl.columns:
                    tbl[col] = tbl[col].apply(lambda x: str(int(x)) if pd.notna(x) else "—")
            st.dataframe(tbl.reset_index(drop=True), width="stretch", hide_index=True)

    # ── INNER TAB 2: Weekly Game Log ──────────────────────────────────────────
    with inner2:
        if goalie_games is None or goalie_games.empty:
            st.info(t("gl_no_game_data"))
        else:
            # Lookup player_id from goalies dataframe
            sel_goalie_row = goalies[goalies['fullName'] == selected_goalie]
            sel_goalie_id = sel_goalie_row.iloc[0]['playerId'] if not sel_goalie_row.empty else None

            game_df = resolve_game_player_name(goalie_games, sel_goalie_id, lang=st.session_state.get('lang', 'FR'))
            game_df = game_df[game_df['season_year'] == sel_season_yr]
            if selected_team != t("gl_all_teams"):
                game_df = game_df[game_df['own_team'] == selected_team]


            if game_df.empty:
                st.warning(t("gl_no_game_data"))
            else:
                st.subheader(t("gl_game_log"))
                weekly = aggregate_weekly_goalie(game_df)

                # Trend chart
                weekly_asc = weekly.sort_values('week_start')
                fig_gl = go.Figure()
                fig_gl.add_trace(go.Bar(
                    x=weekly_asc['week_start'].dt.strftime('%Y-%m-%d'),
                    y=weekly_asc['sa_week'],
                    name=t("chart_shots_against"),
                    marker_color='rgba(206, 14, 45, 0.4)',
                ))
                fig_gl.add_trace(go.Bar(
                    x=weekly_asc['week_start'].dt.strftime('%Y-%m-%d'),
                    y=weekly_asc['ga_week'],
                    name=t("chart_goals_against"),
                    marker_color='rgba(206, 14, 45, 0.85)',
                ))
                fig_gl.add_trace(go.Scatter(
                    x=weekly_asc['week_start'].dt.strftime('%Y-%m-%d'),
                    y=weekly_asc['sv_pct_week'],
                    name=t("chart_save_pct"),
                    mode='lines+markers', yaxis='y2',
                    line=dict(color='#1a85ff', width=2), marker=dict(size=6),
                ))
                fig_gl.update_layout(
                    barmode='overlay',
                    yaxis=dict(title=t("chart_y_shotGoals")),
                    yaxis2=dict(title=t("chart_y_svpct"), overlaying='y', side='right', showgrid=False, range=[0.80, 1.0]),
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    legend=dict(orientation='h', y=1.12),
                    hovermode='x unified', margin=dict(t=20, b=40),
                )
                st.plotly_chart(fig_gl, width="stretch")

                # Weekly table
                col_map_w = {
                    'week_start':  'Semaine' if is_fr else 'Week',
                    'team':        'Équipe' if is_fr else 'Team',
                    'opponents':   'Adversaires' if is_fr else 'Opponents',
                    'gp_week':     'PJ' if is_fr else 'GP',
                    'saves_week':  'Arr.' if is_fr else 'Saves',
                    'sa_week':     'TC' if is_fr else 'SA',
                    'ga_week':     'BC' if is_fr else 'GA',
                    'wins_week':   'V' if is_fr else 'W',
                    'losses_week': 'D' if is_fr else 'L',
                    'sv_pct_week': '% Arr.' if is_fr else 'SV%',
                    'gaa_week':    'MAB' if is_fr else 'GAA',
                }
                avail_w = {k: v for k, v in col_map_w.items() if k in weekly.columns}
                tbl_w = weekly[list(avail_w.keys())].copy()
                tbl_w['week_start'] = tbl_w['week_start'].dt.strftime('%Y-%m-%d')
                tbl_w.rename(columns=avail_w, inplace=True)
                for col in (['MAB','% Arr.'] if is_fr else ['GAA','SV%']):
                    if col in tbl_w.columns:
                        tbl_w[col] = tbl_w[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
                int_keys_w = ['PJ','GP','Arr.','Saves','TC','SA','BC','GA','V','W','D','L']
                for col in int_keys_w:
                    if col in tbl_w.columns:
                        tbl_w[col] = tbl_w[col].apply(lambda x: str(int(x)) if pd.notna(x) else "—")
                st.dataframe(tbl_w.reset_index(drop=True), width="stretch", hide_index=True)
                total_gp = int(game_df['gameDate'].notna().sum())
                st.caption(t("chart_games_count").format(n=total_gp, season=sel_season_lbl))

# ══════════════════════════════════════════════════════════════════════════════
# TOP TAB 2 — PPI Leaderboard
# ══════════════════════════════════════════════════════════════════════════════
with top_tab2:
    st.subheader(t("gl_ppi_title"))
    st.info(t("gl_ppi_desc"))

    lb_c1, lb_c2 = st.columns([2, 2])
    with lb_c1:
        lb_lbl = st.selectbox(t("gl_select_season"), season_labels[::-1], key="gl_lb_season")
        lb_yr  = season_years[season_labels.index(lb_lbl)]
    with lb_c2:
        min_gp = st.slider(t("gl_min_games"), 5, 50, 15, key="gl_lb_mingames")

    ppi_df = compute_goalie_ppi(goalies, season_year=lb_yr, min_games=min_gp)

    if ppi_df.empty:
        st.warning(t("chart_not_enough"))
    else:
        top15 = ppi_df.head(15).copy()
        top15.insert(0, 'Rang', range(1, len(top15)+1))

        colors = ['#CE0E2D' if i==0 else '#f4a418' if i<3 else '#1a85ff' for i in range(len(top15))]
        fig_ppi = go.Figure()
        fig_ppi.add_trace(go.Bar(
            y=top15['fullName'], x=top15['PPI'], orientation='h',
            marker=dict(color=colors),
            text=[f"{v:+.3f}" for v in top15['PPI']], textposition='outside',
            hovertemplate="<b>%{y}</b><br>PPI: %{x:.3f}<extra></extra>",
        ))
        fig_ppi.add_vline(x=0, line_dash='dash', line_color='white', opacity=0.4)
        fig_ppi.update_layout(
            title=f"{t('gl_ppi_chart')} — {lb_lbl}",
            yaxis=dict(autorange='reversed', tickfont=dict(size=12)),
            xaxis=dict(title=t("chart_y_ppi"), gridcolor='rgba(255,255,255,0.1)'),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            height=520, margin=dict(l=20, r=60, t=50, b=40),
        )
        st.plotly_chart(fig_ppi, width="stretch")

        lb_col_map = {
            'Rang': 'Rang', 'fullName': 'Gardien' if is_fr else 'Goalie',
            'team_tri': 'Équipe' if is_fr else 'Team',
            'gamesPlayed': 'PJ' if is_fr else 'GP',
            'gamesStarted': 'PD' if is_fr else 'GS',
            'wins': 'V' if is_fr else 'W', 'losses': 'D' if is_fr else 'L',
            'overtimeLosses': 'DPR' if is_fr else 'OTL',
            'win_pct': 'V%' if is_fr else 'W%',
            'goalsAgainstAverage': 'MAB' if is_fr else 'GAA',
            'savePercentage': '% Arr.' if is_fr else 'SV%',
            'shutouts': 'BL' if is_fr else 'SO',
            'saves': 'Arr.' if is_fr else 'Saves',
            'PPI': 'PPI',
        }
        avail_lb = {k: v for k, v in lb_col_map.items() if k in top15.columns}
        lb_tbl = top15[list(avail_lb.keys())].copy()
        lb_tbl.rename(columns=avail_lb, inplace=True)
        wkey = 'V%' if is_fr else 'W%'
        if wkey in lb_tbl.columns:
            lb_tbl[wkey] = lb_tbl[wkey].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
        for col in (['MAB','% Arr.'] if is_fr else ['GAA','SV%']):
            if col in lb_tbl.columns:
                lb_tbl[col] = lb_tbl[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
        st.dataframe(lb_tbl.reset_index(drop=True), width="stretch", hide_index=True)
