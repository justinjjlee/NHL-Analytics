"""
skater_stats.py — NHL Skater Statistics
Tab hierarchy:
  Top-level Tab 1: Player Profile + Game Log (player-specific, weekly game log)
  Top-level Tab 2: PPI Leaderboard (season-wide, no player selection needed)
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
    compute_skater_ppi, get_available_seasons, get_player_list,
    resolve_game_player_name, aggregate_weekly_skater, _sec_to_mmss,
)


# ── Page ──────────────────────────────────────────────────────────────────────
st.title(t("sk_title"))
st.markdown(t("sk_desc"))
st.divider()

# ── Load ──────────────────────────────────────────────────────────────────────
is_fr = st.session_state.get('lang', 'FR') == 'FR'
loading_msg = "Chargement..." if is_fr else "Loading..."
with st.spinner(loading_msg):
    skaters, _ = load_player_season_data()
    skater_games, _ = load_player_game_data()

if skaters is None or skaters.empty:
    st.error(t("sk_no_data"))
    st.stop()

seasons = get_available_seasons(skaters)
season_labels = [s[1] for s in seasons]
season_years  = [s[0] for s in seasons]
all_teams     = sorted(skaters['team_tri'].dropna().unique().tolist())
all_names     = get_player_list(skaters)
DEFAULT       = "Connor Bedard"

# ── Top-level Tabs ─────────────────────────────────────────────────────────
TAB_PLAYER     = "👤 " + ("Joueur" if is_fr else "Player")
TAB_LEADERBOARD = t("sk_tab_leaderboard")
top_tab1, top_tab2 = st.tabs([TAB_PLAYER, TAB_LEADERBOARD])

# ══════════════════════════════════════════════════════════════════════════════
# TOP TAB 1 — Player Profile + Game Log
# ══════════════════════════════════════════════════════════════════════════════
with top_tab1:
    # ── Filters ──
    st.markdown("#### Filtres / Filters")
    pos_map = {
        t("sk_pos_all"): None,
        t("sk_pos_C"): ["C"],
        t("sk_pos_LR"): ["L", "R"],
        t("sk_pos_D"): ["D"],
    }
    
    # Row 1: Position, Team, Season
    fr1_col1, fr1_col2, fr1_col3 = st.columns(3)
    with fr1_col1:
        pos_choice = st.selectbox(t("sk_select_position"), list(pos_map.keys()), key="sk_pos_filter")
    with fr1_col2:
        selected_team = st.selectbox(t("sk_select_team"), [t("sk_all_teams")] + all_teams, key="sk_team_filter")
    with fr1_col3:
        sel_season_lbl = st.selectbox(t("sk_select_season"), season_labels[::-1], key="sk_season_filter")
        sel_season_yr  = season_years[season_labels.index(sel_season_lbl)]

    # Filter skaters to restrict player name list
    filtered_skaters = skaters.copy()
    if selected_team != t("sk_all_teams"):
        filtered_skaters = filtered_skaters[filtered_skaters['team_tri'] == selected_team]
    if pos_map[pos_choice]:
        filtered_skaters = filtered_skaters[filtered_skaters['positionCode'].isin(pos_map[pos_choice])]

    filtered_names = get_player_list(filtered_skaters)

    if not filtered_names:
        st.warning("Aucun joueur ne correspond à ces critères. / No players match these criteria.")
        st.stop()

    # Row 2: Player selection
    default_idx = filtered_names.index(DEFAULT) if DEFAULT in filtered_names else 0
    selected_player = st.selectbox(t("sk_search"), filtered_names, index=default_idx, key="sk_player_select")


    # ── Inner Tabs ──
    inner1, inner2 = st.tabs([t("sk_tab_profile"), t("sk_tab_gamelog")])

    # ── INNER TAB 1: Season Profile ───────────────────────────────────────────
    with inner1:
        player_df = skaters[skaters['fullName'] == selected_player].copy()
        if selected_team != t("sk_all_teams"):
            player_df = player_df[player_df['team_tri'] == selected_team]
        if pos_map[pos_choice]:
            player_df = player_df[player_df['positionCode'].isin(pos_map[pos_choice])]

        if player_df.empty:
            st.warning(t("sk_no_data"))
        else:
            latest = player_df.sort_values('season_year', ascending=False).iloc[0]
            pos_labels = {
                "C": "Centre",
                "L": "Ailier G" if is_fr else "Left Wing",
                "R": "Ailier D" if is_fr else "Right Wing",
                "D": "Défenseur" if is_fr else "Defenseman",
            }
            pos_label = pos_labels.get(latest.get('positionCode',''), latest.get('positionCode',''))
            st.markdown(f"### {selected_player} · {latest.get('team_tri','')} · {pos_label}")

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            ppg = latest.get('points_per_game')
            m1.metric("PJ" if is_fr else "GP",  int(latest.get('gamesPlayed',0) or 0))
            m2.metric("B"  if is_fr else "G",   int(latest.get('goals',0) or 0))
            m3.metric("A",                       int(latest.get('assists',0) or 0))
            m4.metric("PTS",                     int(latest.get('points',0) or 0))
            m5.metric("+/-",                     int(latest.get('plusMinus',0) or 0))
            m6.metric("PTS/PJ" if is_fr else "PTS/GP", f"{ppg:.2f}" if pd.notna(ppg) else "—")

            st.divider()
            st.subheader(t("sk_career_trend"))
            trend = player_df.sort_values('season_year')

            fig = go.Figure()
            fig.add_trace(go.Bar(x=trend['season_label'], y=trend['points'],
                                 name=t("chart_points"), marker_color='#CE0E2D', opacity=0.7))
            fig.add_trace(go.Scatter(x=trend['season_label'], y=trend['points_per_game'],
                                     name=t("chart_pts_per_gp"), mode='lines+markers',
                                     yaxis='y2', line=dict(color='#f4a418', width=3),
                                     marker=dict(size=8, color='#f4a418')))
            fig.add_trace(go.Scatter(x=trend['season_label'], y=trend['goals'],
                                     name=t("chart_goals"), mode='lines+markers',
                                     line=dict(color='#1a85ff', width=2, dash='dot'),
                                     marker=dict(size=6)))
            fig.update_layout(
                yaxis=dict(title=t("chart_y_pts"), gridcolor='rgba(255,255,255,0.1)'),
                yaxis2=dict(title=t("chart_y_ptsGP"), overlaying='y', side='right', showgrid=False),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation='h', y=1.12), margin=dict(t=30, b=40),
                hovermode='x unified',
            )
            st.plotly_chart(fig, width="stretch")

            st.divider()
            st.subheader(t("sk_season_stats"))
            col_map = {
                'season_label': 'Saison' if is_fr else 'Season',
                'team_tri':     'Équipe' if is_fr else 'Team',
                'positionCode': 'Pos',
                'gamesPlayed':  'PJ' if is_fr else 'GP',
                'goals':        'B' if is_fr else 'G',
                'assists': 'A', 'points': 'PTS',
                'points_per_game': 'PTS/PJ' if is_fr else 'PTS/GP',
                'plusMinus': '+/-',
                'shots': 'COC' if is_fr else 'SOG',
                'shootingPctg': 'Tir%' if is_fr else 'Sh%',
                'powerPlayGoals': 'BP' if is_fr else 'PPG',
                'shorthandedGoals': 'BIN' if is_fr else 'SHG',
                'gameWinningGoals': 'BBV' if is_fr else 'GWG',
                'overtimeGoals': 'BPR' if is_fr else 'OTG',
                'toi_per_game_min': 'TMG/PJ' if is_fr else 'TOI/GP',
                'faceoffWinPctg': 'MF%' if is_fr else 'FO%',
                'penaltyMinutes': 'MPP' if is_fr else 'PIM',
            }
            avail = {k: v for k, v in col_map.items() if k in player_df.columns}
            tbl = player_df[list(avail.keys())].sort_values('season_label', ascending=False).copy()
            tbl.rename(columns=avail, inplace=True)
            for col in ['PTS/PJ','PTS/GP','Tir%','Sh%','MF%','FO%']:
                if col in tbl.columns:
                    tbl[col] = tbl[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
            int_keys = ['PJ','GP','B','G','A','PTS','+/-','COC','SOG','BP','PPG','BIN','SHG','BBV','GWG','BPR','OTG','MPP','PIM']
            for col in int_keys:
                if col in tbl.columns:
                    tbl[col] = tbl[col].apply(lambda x: str(int(x)) if pd.notna(x) else "—")
            st.dataframe(tbl.reset_index(drop=True), width="stretch", hide_index=True)

    # ── INNER TAB 2: Weekly Game Log ──────────────────────────────────────────
    with inner2:
        if skater_games is None or skater_games.empty:
            st.info(t("sk_no_game_data"))
        else:
            # Lookup player_id from skaters dataframe
            sel_player_row = skaters[skaters['fullName'] == selected_player]
            sel_player_id = sel_player_row.iloc[0]['playerId'] if not sel_player_row.empty else None

            game_df = resolve_game_player_name(skater_games, sel_player_id, lang=st.session_state.get('lang', 'FR'))
            game_df = game_df[game_df['season_year'] == sel_season_yr]
            if selected_team != t("sk_all_teams"):
                game_df = game_df[game_df['own_team'] == selected_team]


            if game_df.empty:
                st.warning(t("sk_no_game_data"))
            else:
                st.subheader(t("sk_game_log"))
                weekly = aggregate_weekly_skater(game_df)

                # Trend chart (weekly)
                weekly_asc = weekly.sort_values('week_start')
                weekly_asc['cum_pts'] = weekly_asc['points'].cumsum()

                fig_gl = go.Figure()
                fig_gl.add_trace(go.Bar(
                    x=weekly_asc['week_start'].dt.strftime('%Y-%m-%d'),
                    y=weekly_asc['points'],
                    name=t("chart_points"),
                    marker_color='#CE0E2D', opacity=0.65,
                ))
                fig_gl.add_trace(go.Scatter(
                    x=weekly_asc['week_start'].dt.strftime('%Y-%m-%d'),
                    y=weekly_asc['cum_pts'],
                    name=t("chart_cum_points"),
                    mode='lines', line=dict(color='#f4a418', width=2), yaxis='y2',
                ))
                fig_gl.update_layout(
                    yaxis=dict(title=t("chart_y_ptsPerGame")),
                    yaxis2=dict(title=t("chart_y_cumPts"), overlaying='y', side='right', showgrid=False),
                    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                    legend=dict(orientation='h', y=1.12),
                    hovermode='x unified', margin=dict(t=20, b=40),
                )
                st.plotly_chart(fig_gl, width="stretch")

                # Weekly table
                col_map_w = {
                    'week_start': 'Semaine' if is_fr else 'Week',
                    'team':       'Équipe' if is_fr else 'Team',
                    'opponents':  'Adversaires' if is_fr else 'Opponents',
                    'gp_week':    'PJ' if is_fr else 'GP',
                    'goals': 'B' if is_fr else 'G',
                    'assists': 'A', 'points': 'PTS', 'plusMinus': '+/-',
                    'sog':        'COC' if is_fr else 'SOG',
                    'hits':       'Mises' if is_fr else 'Hits',
                    'blockedShots': 'Bloqués' if is_fr else 'Blocks',
                    'powerPlayGoals': 'BP' if is_fr else 'PPG',
                    'toi_week':   'TMG total' if is_fr else 'TOI total',
                    'fo_pct_week': 'MF%' if is_fr else 'FO%',
                    'giveaways':  'Pertes' if is_fr else 'GA',
                    'takeaways':  'Gains' if is_fr else 'TA',
                }
                avail_w = {k: v for k, v in col_map_w.items() if k in weekly.columns}
                tbl_w = weekly[list(avail_w.keys())].copy()
                tbl_w['week_start'] = tbl_w['week_start'].dt.strftime('%Y-%m-%d')
                tbl_w.rename(columns=avail_w, inplace=True)
                fo_key = 'MF%' if is_fr else 'FO%'
                if fo_key in tbl_w.columns:
                    tbl_w[fo_key] = tbl_w[fo_key].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "—")
                int_keys_w = ['PJ','GP','B','G','A','PTS','+/-','COC','SOG','Mises','Hits','Bloqués','Blocks','BP','PPG','Pertes','GA','Gains','TA']
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
    st.subheader(t("sk_ppi_title"))
    st.info(t("sk_ppi_desc"))

    lb_c1, lb_c2, lb_c3 = st.columns([1.5, 1.5, 2])
    with lb_c1:
        lb_lbl = st.selectbox(t("sk_select_season"), season_labels[::-1], key="sk_lb_season")
        lb_yr  = season_years[season_labels.index(lb_lbl)]
    with lb_c2:
        min_gp = st.slider(t("sk_min_games"), 5, 60, 20, key="sk_lb_mingames")
    with lb_c3:
        pos_opts_lb = {
            t("sk_pos_all"): None, t("sk_pos_C"): ["C"],
            t("sk_pos_LR"): ["L","R"], t("sk_pos_D"): ["D"],
        }
        pos_lb = st.selectbox(t("sk_select_position"), list(pos_opts_lb.keys()), key="sk_lb_pos")

    ppi_df = compute_skater_ppi(skaters, season_year=lb_yr, min_games=min_gp)
    if pos_opts_lb[pos_lb]:
        ppi_df = ppi_df[ppi_df['positionCode'].isin(pos_opts_lb[pos_lb])]

    if ppi_df.empty:
        st.warning(t("chart_not_enough"))
    else:
        top15 = ppi_df.head(15).copy()
        top15.insert(0, 'Rang', range(1, len(top15)+1))

        colors = ['#CE0E2D' if i==0 else '#f4a418' if i<3 else '#4a90d9' for i in range(len(top15))]
        fig_ppi = go.Figure()
        fig_ppi.add_trace(go.Bar(
            y=top15['fullName'], x=top15['PPI'], orientation='h',
            marker=dict(color=colors),
            text=[f"{v:+.3f}" for v in top15['PPI']], textposition='outside',
            hovertemplate="<b>%{y}</b><br>PPI: %{x:.3f}<extra></extra>",
        ))
        fig_ppi.add_vline(x=0, line_dash='dash', line_color='white', opacity=0.4)
        fig_ppi.update_layout(
            title=f"{t('sk_ppi_chart')} — {lb_lbl}",
            yaxis=dict(autorange='reversed', tickfont=dict(size=12)),
            xaxis=dict(title=t("chart_y_ppi"), gridcolor='rgba(255,255,255,0.1)'),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            height=520, margin=dict(l=20, r=60, t=50, b=40),
        )
        st.plotly_chart(fig_ppi, width="stretch")

        lb_col_map = {
            'Rang': 'Rang', 'fullName': 'Joueur' if is_fr else 'Player',
            'team_tri': 'Équipe' if is_fr else 'Team', 'positionCode': 'Pos',
            'gamesPlayed': 'PJ' if is_fr else 'GP',
            'goals': 'B' if is_fr else 'G', 'assists': 'A', 'points': 'PTS',
            'points_per_game': 'PTS/PJ' if is_fr else 'PTS/GP',
            'plusMinus': '+/-',
            'toi_per_game_min': 'TMG/PJ' if is_fr else 'TOI/GP',
            'shootingPctg': 'Tir%' if is_fr else 'Sh%',
            'powerPlayGoals': 'BP' if is_fr else 'PPG',
            'gameWinningGoals': 'BBV' if is_fr else 'GWG',
            'PPI': 'PPI',
        }
        avail_lb = {k: v for k, v in lb_col_map.items() if k in top15.columns}
        lb_tbl = top15[list(avail_lb.keys())].copy()
        lb_tbl.rename(columns=avail_lb, inplace=True)
        for col in ['PTS/PJ','PTS/GP','Tir%','Sh%']:
            if col in lb_tbl.columns:
                lb_tbl[col] = lb_tbl[col].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
        st.dataframe(lb_tbl.reset_index(drop=True), width="stretch", hide_index=True)
