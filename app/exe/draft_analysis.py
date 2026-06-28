import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from i18n import t

def get_data_path(rel_path):
    """Helper function to get correct data paths regardless of run location"""
    # Start with the current file location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to the project root (from /exe to /)
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    # Construct the absolute path to the data
    return os.path.join(project_root, rel_path)

def load_draft_data():
    """Load all necessary data for draft analysis"""
    try:
        # Define relative paths to data files
        relative_paths = {
            'exceptional_players': 'dev/player/player_draft/draftee_lifecycle/results/multi_exceptional_players.csv',
            'exceptional_seasons': 'dev/player/player_draft/draftee_lifecycle/results/exceptional_seasons_details.csv',
            'merged_stats': 'dev/player/player_draft/data/merged_draft_player_stats.csv',
            'player_stats': 'dev/player/player_draft/data/all_drafted_players_chronicle.csv'
        }
        
        # Convert to absolute paths
        data_paths = {key: get_data_path(path) for key, path in relative_paths.items()}
        
        # Check if files exist
        missing_files = []
        for name, path in data_paths.items():
            if not os.path.exists(path):
                missing_files.append(path)
        
        if missing_files:
            return False, f"Files not found: {', '.join(missing_files)}"
        
        return True, data_paths
    
    except Exception as e:
        return False, f"Error loading draft data: {str(e)}"

def render_exceptional_players_analysis():
    """Render the exceptional players analysis tab"""
    # Load exceptional players data from results directory
    files_exist, data_paths = load_draft_data()
    
    # Try to load the files
    try:
        if files_exist:
            # Load data files
            multi_exceptional_players = pd.read_csv(data_paths['exceptional_players'])
            exceptional_seasons_details = pd.read_csv(data_paths['exceptional_seasons'])
            
            # Format season column to YYYY-YYYY format
            if 'season' in exceptional_seasons_details.columns:
                exceptional_seasons_details['season_marking'] = exceptional_seasons_details['season'].astype(str).apply(
                    lambda x: f"{x[:4]}-{x[4:]}" if len(x) >= 8 else x
                )
            # Create full name for easier filter
            exceptional_seasons_details['fullName'] = (
                exceptional_seasons_details['firstName'] + " " + exceptional_seasons_details['lastName']
            )
            multi_exceptional_players['fullName'] = (
                multi_exceptional_players['firstName'] + " " + multi_exceptional_players['lastName']
            )
            
            # Add career lines first if career data is available
            career_data = pd.read_csv(data_paths['merged_stats'])
            # Create full name for easier filter
            career_data['fullName'] = (
                career_data['firstName'] + " " + career_data['lastName']
            )
            # join with player stats
            player_stats = pd.read_csv(data_paths['player_stats'])
            # Some cleaning needed: Player season level stats (disregarding team movements)
            # by player id, season - aggregate up statistics 
            # Before all that, just account for NHL career stats + Regular season only
            player_stats = player_stats[
                    (player_stats['leagueAbbrev'] == 'NHL') & 
                    (player_stats['gameTypeId'] == 2) 
                ]\
                .groupby(['id', 'season']).agg(
                    {
                        'points': 'sum',
                        'goals': 'sum',
                        'assists': 'sum',
                        'gamesPlayed': 'sum',
                        'faceoffWinningPctg': 'mean',
                        'gameWinningGoals':'sum',
                        'plusMinus':'sum',
                        'powerPlayGoals':'sum',
                        'shorthandedGoals':'sum',
                        'avgToi': 'last',
                        'otGoals':'sum',
                        'powerPlayPoints':'sum',
                        'shootingPctg': 'mean',
                        'shorthandedPoints':'sum',
                        'shots':'sum'
                    }).reset_index()
            # create simplifed version for player career at the time of the analysis
            player_stats_simple = player_stats.groupby(['id'])\
                .agg({'season': 'count'})
            # Left join nto multi exceptional players
            multi_exceptional_players = multi_exceptional_players.merge(
                player_stats_simple,
                on=['id'],
                how='left'
            )
            # Convert data formats to int: excluding pct, all should be int
            # run through for loop: points, goals, assists, gamesPlayed, plusminus, Goals, points etc...
            for col in ['points', 'goals', 'assists', 'gamesPlayed', 'plusMinus', 'gameWinningGoals',
                        'powerPlayGoals', 'shorthandedGoals', 'otGoals', 'powerPlayPoints',
                        'shorthandedPoints', 'shots']:
                player_stats[col] = player_stats[col].astype(int)
            # Calculate points per game
            player_stats['pointspergame'] = player_stats['points'] / player_stats['gamesPlayed']
            # Rename some columns for consistency
            player_stats.rename(columns={
                'points': 'points_tempseasoy',
            }, inplace=True)
            # Merge with career data
            career_data = career_data.merge(
                player_stats,
                on=['id', 'season'],
                how='inner'
            )

            # Format season column to YYYY-YYYY format
            if 'season' in career_data.columns:
                career_data['season_marking'] = career_data['season'].astype(str).apply(
                    lambda x: f"{x[:4]}-{x[4:]}" if len(x) >= 8 else x
                )
            # Simple filter for first round picks
            if 'round' in career_data.columns:
                career_data = career_data[career_data['round'] == 1]
        
            # Calculate year_inNHL if not present
            if 'year_inNHL' not in career_data.columns:
                career_data['year_inNHL'] = career_data.groupby('id')['season'].rank(method='dense').astype(int)

            # Add a new section for exceptional players
            st.header(t("da_first_round"))
    
            # Update the title and description with more emphasis on the 95th percentile calculation
            st.markdown(t("da_draft_years").format(
                min_year=career_data.year.min(), 
                max_year=career_data.year.max(), 
                max_season=exceptional_seasons_details['season_marking'].max()
            ))
            st.subheader(t("da_exceptional"))
            st.markdown(t("da_exceptional_desc"))
            
            # Display summary metrics
            cols = st.columns([1,2,3])
            with cols[0]:
                st.metric(t("da_skaters"), len(multi_exceptional_players))
            with cols[1]:
                st.metric(t("da_most_seasons"), multi_exceptional_players['exceptional_seasons'].max())
            with cols[2]:
                # find player with most exceptional seasons
                top_player = multi_exceptional_players.loc[multi_exceptional_players['exceptional_seasons'].idxmax()]
                st.metric(t("da_most_skater"), f"{top_player['firstName']} {top_player['lastName']}")
            
            fig = px.scatter(
                multi_exceptional_players,
                x="season",  # Using season count as x-axis
                y="exceptional_seasons",
                color="exceptional_seasons",
                size="exceptional_seasons",  # Size points by number of exceptional seasons
                hover_name="fullName",
                hover_data=["firstName", "lastName", "overallPick", "exceptional_seasons"],
                title=t("da_chart1_title"),
                labels={
                    "season": t("da_chart1_x"), 
                    "exceptional_seasons": t("da_chart1_y"),
                    "fullName": t("da_chart1_player")
                },
                color_continuous_scale="viridis",
                range_color=[2, multi_exceptional_players['exceptional_seasons'].max()],
            )

            # Customize layout
            fig.update_layout(
                xaxis_title=t("da_chart1_x"),
                yaxis_title=t("da_chart1_y"),
                height=500,
                coloraxis_showscale=False,
                hovermode="closest"
            )

            # Add player names as text labels
            fig.update_traces(
                textposition="top center",
                textfont=dict(size=10),
                text=[f"{row['lastName']} (#{row['overallPick']})" for _, row in multi_exceptional_players.iterrows()]
            )

            # Show only integer ticks on y-axis
            fig.update_yaxes(
                dtick=1,
                tick0=0
            )

            # Custom hover template
            fig.update_traces(
                hovertemplate="<b>%{hovertext}</b><br>" +
                            t("da_chart1_draft") + ": #%{customdata[2]}<br>" +
                            t("da_chart1_hover") + "<br>" +
                            "<extra></extra>"
            )

            st.plotly_chart(fig, width="stretch")
            
            st.subheader(t("da_tracking"))
            # Create detailed view of exceptional seasons
            st.markdown("""
                Select skater(s) to view point productions over their career and exceptional seasons in detail.
            """)
            # Allow user to filter by player
            default_players = []
            player_list = sorted(career_data["fullName"].unique())

            # Check if our desired default players are in the dataset
            for player in ["Patrick Kane", "Duncan Keith", "Jonathan Toews", "Connor Bedard"]:
                if player in player_list:
                    default_players.append(player)

            # Create the multiselect with our default players
            selected_players = st.multiselect(
                t("da_select_skaters"),
                options=player_list,
                default=default_players
            )
            
            # Create scatter plot of exceptional seasons with complete career lines
            if selected_players:
                # Filter exceptional seasons data for selected players
                filtered_exceptional = exceptional_seasons_details[
                    exceptional_seasons_details["fullName"].isin(selected_players)
                ]
                
                # Create figure for our visualization
                fig = go.Figure()
                
                # For each selected player, add lines from career data
                for i, player in enumerate(selected_players):
                    # Find player in career data - first try by name
                    player_career = career_data[(career_data['fullName'] == player)]
                    
                    if len(player_career) > 0:
                        # Sort and plot
                        player_career = player_career.sort_values('year_inNHL')
                        fig.add_trace(go.Scatter(
                            x=player_career['year_inNHL'],
                            y=player_career['points'],
                            mode='lines',
                            name=f"{player}",
                            line=dict(color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)])
                        , hovertemplate=
                            "<b>%{text}</b><br>" +
                            t("da_chart2_hover_pts") + ": %{y}<br>" +
                            t("da_chart2_hover_gp") + ": " + player_career['gamesPlayed'].astype(int).astype(str) + "<br>" +
                            t("da_chart2_hover_yr") + ": %{x}<br>" +
                            t("da_chart2_hover_tm") + ": " + player_career['teamAbbrev_stats'] + "<br>" +
                            t("da_chart2_hover_draft") + " " + \
                                player_career['year'].astype(str) + " " + t("da_chart2_hover_by") + " " +\
                                player_career['teamAbbrev_draft'] + " " + t("da_chart2_hover_overall") + " " +\
                                player_career['overallPick'].astype(str) + "<br>",
                        text=[f"{row['firstName']} {row['lastName']} ({row['season_marking']})" 
                            for _, row in player_career.iterrows()],
                        ))
                
                # Add scatter points for exceptional seasons
                for i, player in enumerate(selected_players):
                    player_exceptional = filtered_exceptional[
                        filtered_exceptional['fullName'] == player
                    ]
                    # Merge with career data to get other information
                    player_exceptional_seasonstats = career_data[
                        (career_data['fullName'] == player)
                    ][['fullName', 'season', 'year', 'teamAbbrev_stats', 'teamAbbrev_draft']]
                    # left join with same columns
                    player_exceptional = player_exceptional.merge(
                        player_exceptional_seasonstats,
                        on=['fullName', 'season'],
                        how='left'
                    )
                    # Plot exceptional seasons as markers
                    fig.add_trace(go.Scatter(
                        x=player_exceptional['year_inNHL'],
                        y=player_exceptional['points'],
                        mode='markers',
                        name=f"{player} (95%+)",
                        marker=dict(
                            color=px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)],
                            size=12,
                            line=dict(width=2, color='white')
                        ),
                        hovertemplate=
                            "<b>%{text}</b><br>" +
                            t("da_chart2_hover_pts") + ": %{y}<br>" +
                            t("da_chart2_hover_gp") + ": " + player_exceptional['gamesPlayed'].astype(int).astype(str) + "<br>" +
                            t("da_chart2_hover_yr") + ": %{x}<br>" +
                            t("da_chart2_hover_tm") + ": " + player_exceptional['teamAbbrev_stats'] + "<br>" +
                            t("da_chart2_hover_draft") + " " + \
                                player_exceptional['year'].astype(str) + " " + t("da_chart2_hover_by") + " " +\
                                player_exceptional['teamAbbrev_draft'] + " " + t("da_chart2_hover_overall") + " " +\
                                player_exceptional['overallPick'].astype(str) + "<br>",
                        text=[f"{row['firstName']} {row['lastName']} ({row['season_marking']})" 
                            for _, row in player_exceptional.iterrows()],
                    ))
                
                # Update layout
                fig.update_layout(
                    title=t("da_chart2_title"),
                    xaxis_title=t("da_chart2_x"),
                    yaxis_title=t("da_chart2_y"),
                    legend_title=t("da_chart2_legend"),
                    height=500
                )
                
                st.plotly_chart(fig, width="stretch")
                
                # Show table with exceptional seasons details
                with st.expander(t("da_view_details")):
                    st.markdown(t("da_table_desc"))
                    
                    display_df = filtered_exceptional[
                        ["firstName", "lastName", "season_marking", "year_inNHL", 
                        "points", "goals", "assists", "gamesPlayed", "pointspergame"]
                    ].copy()
                    
                    display_df = display_df.sort_values(["lastName", "year_inNHL"])
                    display_df["pointspergame"] = display_df["pointspergame"].round(2)
                    
                    display_df.columns = [
                        t("da_col_fname"), t("da_col_lname"), t("da_col_season"),
                        t("da_col_year"), t("da_col_pts"), t("da_col_g"),
                        t("da_col_a"), t("da_col_gp"), t("da_col_ppg")
                    ]
                    
                    st.dataframe(display_df, width="stretch", hide_index=True)
            else:
                st.info(t("da_no_skater"))
                
        else:
            st.warning(t("da_no_data"))
            
            # If files don't exist, provide info about what they would contain
            st.info(t("da_requires_data"))
            st.info("""
            The exceptional skaters analysis shows players who have had multiple seasons performing above
            the 95th percentile in points production compared to their peers. This identifies the most 
            consistently elite performers among NHL draft picks.
            """)
            
    except Exception as e:
        st.error(t("da_error") + f": {e}")
        st.info(t("da_requires_data"))


def render_not_so_magnificent_analysis():
    st.subheader(t("nsm_title"))
    st.markdown(t("nsm_desc"))

    # Load data
    files_exist, data_paths = load_draft_data()
    if not files_exist:
        st.warning(t("da_no_data"))
        return

    try:
        draft_df = pd.read_csv(get_data_path('dev/player/player_draft/data/all_drafted_players.csv'))
        merged_df = pd.read_csv(data_paths['merged_stats'])
        chronicle_df = pd.read_csv(data_paths['player_stats'])
    except Exception as e:
        st.error(f"Error loading draft data: {e}")
        return

    is_fr = st.session_state.get('lang', 'FR') == 'FR'

    # Dropdowns for Round and Cohorts
    col1, col2 = st.columns(2)
    with col1:
        rounds = sorted(draft_df['round'].dropna().unique().tolist())
        selected_round = st.selectbox(
            t("nsm_select_round"),
            options=rounds,
            index=0,
            key="nsm_round_select"
        )
    with col2:
        years = sorted(draft_df['year'].dropna().unique().tolist())
        import datetime
        current_year = datetime.datetime.now().year
        default_end = current_year - 5
        default_start = default_end - 6
        
        min_yr_bound = int(min(years))
        max_yr_bound = int(max(years))
        
        default_start = max(min_yr_bound, min(default_start, max_yr_bound))
        default_end = max(min_yr_bound, min(default_end, max_yr_bound))
        if default_start > default_end:
            default_start = default_end

        # Slider range for draft years
        selected_years = st.slider(
            t("nsm_select_cohort_range"),
            min_value=min_yr_bound,
            max_value=max_yr_bound,
            value=(default_start, default_end),
            key="nsm_years_slider"
        )
        min_yr, max_yr = selected_years

    # Filter drafted players for the selected round and cohort range
    cohort_drafted = draft_df[
        (draft_df['round'] == selected_round) &
        (draft_df['year'] >= min_yr) &
        (draft_df['year'] <= max_yr)
    ].copy()

    if cohort_drafted.empty:
        st.warning("No drafted players found for the selected criteria. / Aucun joueur trouvé.")
        return

    # Link drafted players to player stats ID
    player_id_map = merged_df[
        (merged_df['round'] == selected_round)
    ][['firstName', 'lastName', 'year', 'id']].drop_duplicates(subset=['firstName', 'lastName', 'year'])

    cohort_players = cohort_drafted.merge(
        player_id_map,
        on=['firstName', 'lastName', 'year'],
        how='left'
    )

    # Total drafted players in each year
    drafted_counts = cohort_players.groupby('year').size().to_dict()
    total_drafted = len(cohort_players)

    # Filter NHL regular season games in chronicle
    nhl_chron = chronicle_df[
        (chronicle_df['leagueAbbrev'] == 'NHL')
    ].copy()
    
    # Group by id and season to handle trades / mid-season moves
    nhl_chron = nhl_chron.groupby(['id', 'season']).agg({
        'gamesPlayed': 'sum'
    }).reset_index()
    
    # Calculate years since draft safely
    nhl_chron = nhl_chron.dropna(subset=['season'])
    nhl_chron['season'] = nhl_chron['season'].astype(float).astype(int).astype(str)
    nhl_chron['season_year'] = nhl_chron['season'].str[:4].astype(int)

    # Join chronicle with cohort_players on id
    player_games = nhl_chron.merge(
        cohort_players[['id', 'year', 'firstName', 'lastName']],
        on='id',
        how='inner'
    )
    player_games['years_since_draft'] = player_games['season_year'] - player_games['year']

    # Filter to only years since draft between 0 and 10
    player_games = player_games[
        (player_games['years_since_draft'] >= 0) &
        (player_games['years_since_draft'] <= 10)
    ]

    # Chart 1: Survival curves (Percentage Active playing >= 40 Games Played)
    active_games = player_games[player_games['gamesPlayed'] >= 40]

    # Calculate proportion active by years since draft for each cohort year
    years_list = list(range(11))
    cohort_years = sorted(cohort_players['year'].unique().tolist())

    plot_data = []
    
    # Aggregate across all selected cohort years (Average curve)
    avg_active_counts = active_games.groupby('years_since_draft')['id'].nunique().to_dict()
    for ysd in years_list:
        active_cnt = avg_active_counts.get(ysd, 0)
        pct = (active_cnt / total_drafted) * 100 if total_drafted > 0 else 0
        plot_data.append({
            t("nsm_chart_col_ysd"): ysd,
            t("nsm_chart_col_pct"): pct,
            t("nsm_chart_col_cohort"): t("nsm_chart_val_avg"),
            'Count': active_cnt,
            'Total': total_drafted
        })

    # Aggregate across all selected cohort years (Cumulative curve)
    # Find the very first year each player played >= 40 games
    first_active_year = active_games.groupby('id')['years_since_draft'].min().reset_index()
    avg_first_active_counts = first_active_year.groupby('years_since_draft')['id'].nunique().to_dict()
    
    cumulative_active = 0
    for ysd in years_list:
        cumulative_active += avg_first_active_counts.get(ysd, 0)
        pct = (cumulative_active / total_drafted) * 100 if total_drafted > 0 else 0
        plot_data.append({
            t("nsm_chart_col_ysd"): ysd,
            t("nsm_chart_col_pct"): pct,
            t("nsm_chart_col_cohort"): t("nsm_chart_val_cum_avg"),
            'Count': cumulative_active,
            'Total': total_drafted
        })

    # Individual cohort years (optional checkbox)
    show_individual = st.checkbox(t("nsm_show_individual"), value=False)
    if show_individual:
        for cy in cohort_years:
            cy_drafted = drafted_counts.get(cy, 0)
            if cy_drafted == 0:
                continue
            cy_active = active_games[active_games['year'] == cy]
            cy_active_counts = cy_active.groupby('years_since_draft')['id'].nunique().to_dict()
            for ysd in years_list:
                active_cnt = cy_active_counts.get(ysd, 0)
                pct = (active_cnt / cy_drafted) * 100
                plot_data.append({
                    t("nsm_chart_col_ysd"): ysd,
                    t("nsm_chart_col_pct"): pct,
                    t("nsm_chart_col_cohort"): f"{t('nsm_cohort_label')} {cy}",
                    'Count': active_cnt,
                    'Total': cy_drafted
                })

    plot_df = pd.DataFrame(plot_data)

    fig1 = px.line(
        plot_df,
        x=t("nsm_chart_col_ysd"),
        y=t("nsm_chart_col_pct"),
        color=t("nsm_chart_col_cohort"),
        title=t("nsm_chart1_title").format(r=selected_round),
        labels={
            t("nsm_chart_col_ysd"): t("nsm_chart_axis_ysd"),
            t("nsm_chart_col_pct"): t("nsm_chart_axis_pct")
        },
        markers=True
    )
    
    fig1.update_layout(
        xaxis=dict(tickmode='linear', tick0=0, dtick=1, range=[0, 10]),
        yaxis=dict(ticksuffix='%', range=[0, 105]),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend_title="Cohort"
    )
    fig1.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    fig1.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    
    st.info(t("nsm_chart1_explain"))
    st.plotly_chart(fig1, width="stretch")

    # ── Chart 2: Career Games Played Distribution ──
    st.markdown("---")
    st.subheader(t("nsm_chart2_header"))
    st.info(t("nsm_chart2_explain"))
    
    # Selected cohort year
    selected_cohort_year = st.selectbox(
        t("nsm_select_cohort_year"),
        options=cohort_years,
        index=0,
        key="nsm_cohort_year"
    )


    # Filter to selected cohort year
    cohort_year_players = cohort_players[cohort_players['year'] == selected_cohort_year].copy()
    cy_drafted_count = len(cohort_year_players)

    # Calculate career games played for each player
    career_gp = nhl_chron.groupby('id')['gamesPlayed'].sum().reset_index()
    career_gp.rename(columns={'gamesPlayed': 'careerGamesPlayed'}, inplace=True)

    cohort_year_players = cohort_year_players.merge(
        career_gp,
        on='id',
        how='left'
    )
    cohort_year_players['careerGamesPlayed'] = cohort_year_players['careerGamesPlayed'].fillna(0).astype(int)

    # Create games played bins
    bins = [-1, 0, 99, 399, 799, 9999]
    bin_labels = [
        t("nsm_bin_0"),
        t("nsm_bin_1_99"),
        t("nsm_bin_100_399"),
        t("nsm_bin_400_799"),
        t("nsm_bin_800")
    ]
    cohort_year_players['GP_Group'] = pd.cut(
        cohort_year_players['careerGamesPlayed'],
        bins=bins,
        labels=bin_labels
    )

    gp_dist = cohort_year_players['GP_Group'].value_counts().reindex(bin_labels).reset_index()
    gp_dist.columns = [t("nsm_table_gp_cat"), t("nsm_table_player_cnt")]
    gp_dist['Percentage'] = (gp_dist[t("nsm_table_player_cnt")] / cy_drafted_count) * 100 if cy_drafted_count > 0 else 0

    # Build the bar chart
    fig2 = px.bar(
        gp_dist,
        x=t("nsm_table_gp_cat"),
        y=t("nsm_table_player_cnt"),
        title=t("nsm_chart2_title").format(r=selected_round, y=selected_cohort_year),
        labels={
            t("nsm_table_gp_cat"): t("nsm_chart2_axis_x"),
            t("nsm_table_player_cnt"): t("nsm_chart2_axis_y")
        },
        text=gp_dist[t("nsm_table_player_cnt")].astype(str) + " (" + gp_dist['Percentage'].round(1).astype(str) + "%)"
    )
    fig2.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    fig2.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(255,255,255,0.1)')
    st.plotly_chart(fig2, width="stretch")

    # Expandable player details
    with st.expander(t("nsm_view_player_details").format(y=selected_cohort_year)):
        display_tbl = cohort_year_players[['pickInRound', 'overallPick', 'teamAbbrev', 'firstName', 'lastName', 'position', 'careerGamesPlayed']].copy()
        display_tbl.columns = [
            t("nsm_tbl_pick"), t("nsm_tbl_overall"), t("nsm_tbl_team"),
            t("nsm_tbl_fname"), t("nsm_tbl_lname"), t("nsm_tbl_pos"), t("nsm_tbl_gp")
        ]
        st.dataframe(display_tbl.sort_values(t("nsm_tbl_overall")), width="stretch", hide_index=True)