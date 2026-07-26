import streamlit as st
import pandas as pd
import glob
import os
import base64
import plotly.express as px
from i18n import t

def get_data_path(rel_path):
    """Helper function to get correct data paths regardless of run location"""
    # Start with the current file location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to the project root (from /app/views to /)
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    # Construct the absolute path to the data
    return os.path.join(project_root, rel_path)

def get_logo_url(team_tri):
    """Helper to get SVG URL for st.dataframe ImageColumn"""
    return f"https://assets.nhle.com/logos/nhl/svg/{team_tri}_light.svg"

@st.cache_data
def load_season_data():
    """Load and aggregate box scores, merge with team_season stats across all available years."""
    # Find all team season files
    season_dir = get_data_path("latest/team/season")
    season_files = glob.glob(f"{season_dir}/*_team_season.csv")
    
    all_data = []
    
    for fpath in season_files:
        filename = os.path.basename(fpath)
        year = filename[:4]
        
        # Load advanced stats
        try:
            df_season = pd.read_csv(fpath)
        except Exception as e:
            continue
            
        # Load box scores to compute W, L, OTL, PTS
        box_path = get_data_path(f"latest/box/{year}_box.csv")
        if os.path.exists(box_path):
            try:
                df_box = pd.read_csv(box_path)
                
                # Melt the box score to have a row for each team-game
                df_for = df_box[['tricode_for', 'tricode_winteam', 'period_ending']].rename(columns={'tricode_for': 'team'})
                df_against = df_box[['tricode_against', 'tricode_winteam', 'period_ending']].rename(columns={'tricode_against': 'team'})
                df_teams = pd.concat([df_for, df_against])
                
                # W, L, OTL logic
                df_teams['W'] = (df_teams['team'] == df_teams['tricode_winteam']).astype(int)
                df_teams['L'] = ((df_teams['team'] != df_teams['tricode_winteam']) & (df_teams['period_ending'] == 'REG')).astype(int)
                df_teams['OTL'] = ((df_teams['team'] != df_teams['tricode_winteam']) & (df_teams['period_ending'].isin(['OT', 'SO']))).astype(int)
                
                # Group by team
                df_agg = df_teams.groupby('team')[['W', 'L', 'OTL']].sum().reset_index()
                df_agg['PTS'] = df_agg['W'] * 2 + df_agg['OTL']
                df_agg['GP'] = df_agg['W'] + df_agg['L'] + df_agg['OTL']
                
                # Merge with advanced stats
                df_merged = pd.merge(df_agg, df_season, left_on='team', right_on='team_tri_for', how='inner')
                df_merged['Season'] = year
                
                all_data.append(df_merged)
            except Exception as e:
                continue
            
    if all_data:
        # Combine all years into one DataFrame
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

# Main Page UI
st.title(t("tss_title"))

with st.spinner(t("tss_loading")):
    df = load_season_data()

if df.empty:
    st.warning(t("tss_no_data"))
    st.stop()

# Ensure integer types for standard stats
for col in ['GP', 'W', 'L', 'OTL', 'PTS', 'pairwise_win']:
    if col in df.columns:
        df[col] = df[col].astype(int)

# Round specific metrics to 3 decimal places
for col in ['wp', 'rpe', 'kpi_pe', 'kpi_pairwise', 'kpi_wp']:
    if col in df.columns:
        df[col] = df[col].round(3)

# Get available seasons
seasons = sorted(df['Season'].unique().tolist(), reverse=True)

# Define clean labels for metrics
METRIC_NAMES = {
    'team': 'Team',
    'GP': 'GP',
    'W': 'W',
    'L': 'L',
    'OTL': 'OTL',
    'PTS': 'PTS',
    'wp': 'Win %',
    'kpi_corsi': 'Corsi (KPI)',
    'kpi_fenwick': 'Fenwick (KPI)',
    'rpi': 'RPI',
    'kpi_rpi': 'RPI (KPI)',
    'rpe': 'Pyth. Exp. (PE)',
    'kpi_pe': 'Pyth. Exp.',
    'pairwise_win': 'Pairwise Rank',
    'kpi_pairwise': 'Pairwise KPI',
    'kpi_wp': 'Win %',
    'rwin_fittedPE': 'Fitted PE Wins',
    'rwin': 'RWin',
    'rgame': 'RGame',
    'wp_own': 'WP Own',
    'wp_ow': 'WP Opponents',
    'wp_oow': 'WP Opponents of Opponents',
    'Final': 'Final Ranking'
}

def format_metric(m):
    return METRIC_NAMES.get(m, m)

# Create layout tabs
tab1, tab2 = st.tabs([t("tss_tab1"), t("tss_tab2")])

with tab1:
    # Top split row for Season Overview: metric guide on left, season selector on right
    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.info(
            f"**{t('tss_metric_guide_title')}**\n\n"
            f"- {t('tss_wp_desc')}\n"
            f"- {t('tss_pairwise_desc')}\n"
            f"- {t('tss_pe_desc')}"
        )

    with col_right:
        selected_season = st.selectbox(t("tss_select_season"), seasons, key="season_overview_selector")

    # Filter data for selected season
    df_season = df[df['Season'] == selected_season].copy()
    
    # Sort by Points (PTS) by default
    df_season = df_season.sort_values(by='PTS', ascending=False).reset_index(drop=True)
    
    # Define columns to show in the primary data table (Pairwise Rank next to PE)
    default_cols = ['team', 'GP', 'W', 'L', 'OTL', 'PTS', 'wp', 'pairwise_win', 'rpe']
    cols_to_show = [c for c in default_cols if c in df_season.columns]
    
    st.subheader(f"{t('tss_standings')} ({selected_season})")
    
    # Prepare dataframe for display with SVG logos
    df_display = df_season[cols_to_show].copy()
    df_display.insert(0, " ", df_display['team'].apply(get_logo_url))
    
    # Rename columns to beautified names
    df_display = df_display.rename(columns=METRIC_NAMES)
    
    # Render table with st.dataframe for sortability and 3-decimal alignment
    st.dataframe(
        df_display,
        column_config={
            " ": st.column_config.ImageColumn(" ", help="Team Logo"),
            METRIC_NAMES['wp']: st.column_config.NumberColumn(METRIC_NAMES['wp'], format="%.3f", help="Winning percentage (W / GP)"),
            METRIC_NAMES['rpe']: st.column_config.NumberColumn(METRIC_NAMES['rpe'], format="%.3f", help="Pythagorean Expectation (GF^2 / (GF^2 + GA^2))"),
            METRIC_NAMES['pairwise_win']: st.column_config.NumberColumn(METRIC_NAMES['pairwise_win'], format="%d", help="Pairwise Rank (Head-to-head win strength)"),
            METRIC_NAMES['PTS']: st.column_config.NumberColumn(METRIC_NAMES['PTS'], format="%d"),
            METRIC_NAMES['W']: st.column_config.NumberColumn(METRIC_NAMES['W'], format="%d"),
            METRIC_NAMES['L']: st.column_config.NumberColumn(METRIC_NAMES['L'], format="%d"),
            METRIC_NAMES['OTL']: st.column_config.NumberColumn(METRIC_NAMES['OTL'], format="%d"),
            METRIC_NAMES['GP']: st.column_config.NumberColumn(METRIC_NAMES['GP'], format="%d"),
        },
        hide_index=True,
        width="stretch"
    )
    
    st.markdown("---")
    st.subheader(t("tss_scatter_title"))
    
    # Let user pick metrics to plot
    numeric_cols = df_season.select_dtypes(include=['float64', 'int64']).columns.tolist()
    available_metrics = [c for c in numeric_cols if c not in ['Season']]
    
    # Define trend metrics for scatter plot and trend chart (including Corsi, Fenwick, RPI)
    trend_metrics = [c for c in ['wp', 'pairwise_win', 'rpe', 'kpi_corsi', 'kpi_fenwick', 'rpi', 'kpi_pe', 'kpi_pairwise', 'kpi_rpi', 'kpi_wp'] if c in available_metrics]
    if not trend_metrics:
        trend_metrics = available_metrics
    
    col_x, col_y = st.columns(2)
    with col_x:
        default_x = 'pairwise_win' if 'pairwise_win' in trend_metrics else trend_metrics[0]
        x_metric = st.selectbox(t("tss_x_axis"), trend_metrics, index=trend_metrics.index(default_x), format_func=format_metric)
    with col_y:
        default_y = 'wp' if 'wp' in trend_metrics else (trend_metrics[1] if len(trend_metrics) > 1 else trend_metrics[0])
        y_metric = st.selectbox(t("tss_y_axis"), trend_metrics, index=trend_metrics.index(default_y), format_func=format_metric)
        
    # Scatter plot using Plotly Express for better interactivity and tooltips
    fig_scatter = px.scatter(
        df_season, 
        x=x_metric, 
        y=y_metric, 
        text="team",
        hover_data=['W', 'L', 'OTL'],
        title=f"{format_metric(y_metric)} vs {format_metric(x_metric)} ({selected_season})",
        template="plotly_dark"
    )
    fig_scatter.update_layout(xaxis_title=format_metric(x_metric), yaxis_title=format_metric(y_metric))
    fig_scatter.update_traces(textposition='top center', marker=dict(size=10, opacity=0.8, color='#CE0E2D'))
    st.plotly_chart(fig_scatter, width="stretch")

with tab2:
    st.header(t("tss_trend_title"))
    
    all_teams = sorted(df['team'].unique().tolist())
    
    # Select teams and metric
    col_teams, col_metric = st.columns(2)
    with col_teams:
        default_teams = [t for t in ['NYR', 'BOS', 'PIT', 'MTL', 'TOR', 'CHI'] if t in all_teams]
        if not default_teams:
            default_teams = all_teams[:3]
        selected_teams = st.multiselect(t("tss_select_teams"), all_teams, default=default_teams)
    with col_metric:
        trend_metric = st.selectbox(t("tss_select_trend_metric"), trend_metrics, index=0, format_func=format_metric)
    
    if selected_teams:
        df_trend = df[df['team'].isin(selected_teams)].copy()
        
        # Sort by season so the x-axis is chronological
        df_trend = df_trend.sort_values('Season')
        
        # Line chart using Plotly Express
        fig_line = px.line(
            df_trend,
            x="Season",
            y=trend_metric,
            color="team",
            markers=True,
            title=f"{format_metric(trend_metric)} - {t('tss_trend_chart_title')}",
            template="plotly_dark"
        )
        
        # Force X-axis to show discrete categorical years (so 2024 doesn't render as 2,024.5)
        fig_line.update_xaxes(type='category')
        fig_line.update_layout(yaxis_title=format_metric(trend_metric))
        
        st.plotly_chart(fig_line, width="stretch")
    else:
        st.info(t("tss_no_teams"))
