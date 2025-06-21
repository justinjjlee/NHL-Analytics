import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

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
            'exceptional_players': 'dev/player_draft/draftee_lifecycle/results/multi_exceptional_players.csv',
            'exceptional_seasons': 'dev/player_draft/draftee_lifecycle/results/exceptional_seasons_details.csv',
            'merged_stats': 'dev/player_draft/data/merged_draft_player_stats.csv',
            'player_stats': 'dev/player_draft/data/all_drafted_players_chronicle.csv'
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
            st.header("1st Round Drafted Skaters")
    
            # Update the title and description with more emphasis on the 95th percentile calculation
            st.markdown(f"""
                Draft Years: {career_data.year.min()} - {career_data.year.max()} (Last updated for {exceptional_seasons_details['season_marking'].max()} season)
            """)
            st.subheader("Seasons of Exceptional Point Production")
            st.markdown("""
                An **exceptional season** is defined as a season where a skater's point production **exceeds the 95%** 
                of all other active 1st round drafted skaters in the same regular season. **Most exceptional skaters** 
                had long careers with greater number of the exceptional seasons.
            """)
            
            # Display summary metrics
            cols = st.columns([1,2,3])
            with cols[0]:
                st.metric("Skaters", len(multi_exceptional_players))
            with cols[1]:
                st.metric("Most Exceptional Seasons by a Skater", multi_exceptional_players['exceptional_seasons'].max())
            with cols[2]:
                top_player = multi_exceptional_players.iloc[0]
                st.metric("Most Exceptional Skater", f"{top_player['firstName']} {top_player['lastName']}")
            
            fig = px.scatter(
                multi_exceptional_players,
                x="season",  # Using season count as x-axis
                y="exceptional_seasons",
                color="exceptional_seasons",
                size="exceptional_seasons",  # Size points by number of exceptional seasons
                hover_name="fullName",
                hover_data=["firstName", "lastName", "overallPick", "exceptional_seasons"],
                title="Skaters with Multiple Exceptional Seasons",
                labels={
                    "season": "NHL Seasons Played", 
                    "exceptional_seasons": "Number of Exceptional Seasons",
                    "fullName": "Player"
                },
                color_continuous_scale="viridis",
                range_color=[2, multi_exceptional_players['exceptional_seasons'].max()],
            )

            # Customize layout
            fig.update_layout(
                xaxis_title="Total NHL Seasons Played",
                yaxis_title="Number of Exceptional Seasons",
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
                            "Draft Position: #%{customdata[2]}<br>" +
                            "Exceptional Seasons: %{y} / %{x} total seasons<br>" +
                            "<extra></extra>"
            )

            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Tracking Career Point Productions of Exceptional Skaters")
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
                "Select Skaters to View Details",
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
                            "Points: %{y}<br>" +
                            "Games Played: " + player_career['gamesPlayed'].astype(int).astype(str) + "<br>" +
                            "Year: %{x}<br>" +
                            "Team: " + player_career['teamAbbrev_stats'] + "<br>"
                            "Draft: On " + \
                                player_career['year'].astype(str) + " by " +\
                                player_career['teamAbbrev_draft'] + " in 1st round drafted on " +\
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
                            "Points: %{y}<br>" +
                            "Games Played: " + player_exceptional['gamesPlayed'].astype(int).astype(str) + "<br>" +
                            "Year: %{x}<br>" +
                            "Team: " + player_exceptional['teamAbbrev_stats'] + "<br>"
                            "Draft: On " + \
                                player_exceptional['year'].astype(str) + " by " +\
                                player_exceptional['teamAbbrev_draft'] + " in 1st round draft of " +\
                                player_exceptional['overallPick'].astype(str) + "<br>",
                        text=[f"{row['firstName']} {row['lastName']} ({row['season_marking']})" 
                            for _, row in player_exceptional.iterrows()],
                    ))
                
                # Update layout
                fig.update_layout(
                    title="Point Productions by 1st Round Drafted Skaters",
                    xaxis_title="n-th Year in NHL (Regular Season)",
                    yaxis_title="Points",
                    legend_title="Skaters",
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show table with exceptional seasons details
                with st.expander("View Exceptional Seasons Details"):
                    st.dataframe(filtered_exceptional[
                        ["firstName", "lastName", "season_marking", "year_inNHL", 
                        "points", "goals", "assists", "gamesPlayed", "pointspergame"]
                    ].sort_values(["lastName", "year_inNHL"]))
            else:
                st.info("Please select at least one skater to view details.")
                
        else:
            st.warning("Exceptional skaters data files not found. Please ensure the files exist in the results directory.")
            
            # If files don't exist, provide info about what they would contain
            st.info("""
            The exceptional skaters analysis shows players who have had multiple seasons performing above
            the 95th percentile in points production compared to their peers. This identifies the most 
            consistently elite performers among NHL draft picks.
            """)
            
    except Exception as e:
        st.error(f"An error occurred loading exceptional players data: {e}")
        st.info("The exceptional skaters analysis requires data files generated from the draftee_lifecycle analysis.")