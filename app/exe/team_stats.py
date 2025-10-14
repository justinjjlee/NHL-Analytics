import streamlit as st
import pandas as pd
import os
import glob
from datetime import datetime, timedelta
import plotly.graph_objects as go

def get_data_path(rel_path):
    """Helper function to get correct data paths regardless of run location"""
    # Start with the current file location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to the project root (from /app/exe to /)
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    # Construct the absolute path to the data
    return os.path.join(project_root, rel_path)

def load_box_scores_and_odds():
    """
    Load all box score data and betting odds data from the latest/box directory
    Returns dataframes with the scores, odds, and a list of available dates
    """
    try:
        # Use absolute paths for box score files
        box_dir = get_data_path('latest/box')
        box_files = glob.glob(f"{box_dir}/*box.csv")
        
        if not box_files:
            st.error(f"No box score files found. Please check data directory: {box_dir}")
            return None, None, []
        
        # Sort files for consistent processing
        box_files.sort(reverse=True)
        
        # Create empty list to store dataframes
        box_dfs = []
        
        # Load data from all files
        for file in box_files:
            try:
                temp_df = pd.read_csv(file)
                # Add to list of dataframes
                box_dfs.append(temp_df)
            except Exception as e:
                st.warning(f"Error loading file {file}: {e}")
                continue
        
        if not box_dfs:
            st.error("No valid data found in any box score files.")
            return None, None, []
        
        # Concatenate all box score dataframes
        box_df = pd.concat(box_dfs, ignore_index=True)
        
        # Find all odds files
        odds_files = glob.glob(f"{box_dir}/*odds.csv")
        # 2025, added US and Canada, use US odds for 
        odds_files_us = glob.glob(f"{box_dir}/*odds_US.csv")
        # Combine the two
        odds_files = odds_files + odds_files_us
        odds_df = None
        
        if odds_files:
            # Sort files for consistent processing
            odds_files.sort(reverse=True)
            
            # Create empty list to store dataframes
            odds_dfs = []
            
            # Load data from all files
            for file in odds_files:
                try:
                    temp_df = pd.read_csv(file)
                    # If gameId column exist, change the name to gameid
                    if 'gameId' in temp_df.columns:
                        temp_df = temp_df.rename(columns={'gameId': 'gameid'})
                    # Filter for MONEY_LINE_2_WAY odds only
                    temp_df = temp_df[temp_df['odds_description'] == 'MONEY_LINE_2_WAY']
                    # Add to list of dataframes
                    odds_dfs.append(temp_df)
                except Exception as e:
                    st.warning(f"Error loading odds file {file}: {e}")
                    continue
            
            if odds_dfs:
                # Concatenate all odds dataframes
                odds_df = pd.concat(odds_dfs, ignore_index=True)
                # Drop duplicates to ensure we have one odds entry per game
                odds_df = odds_df.drop_duplicates(subset=['gameid'])
        
        # Convert date column to datetime
        box_df['date'] = pd.to_datetime(box_df['date'])
        
        # Remove duplicates if any
        box_df = box_df.drop_duplicates()
        
        # Sort by date descending
        box_df = box_df.sort_values('date', ascending=False)
        
        # Get list of available dates
        available_dates = box_df['date'].dt.date.unique()
        
        return box_df, odds_df, available_dates
    
    except Exception as e:
        st.error(f"Error processing data: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, []

def get_team_logo_url(tricode):
    """
    Generate URL for team logo based on tricode
    """
    return f"https://assets.nhle.com/logos/nhl/svg/{tricode}_light.svg"

def create_scoreboard_card(iter_home_team, iter_away_team, home_score, away_score, game_date, game_status, home_odds=None, away_odds=None):
    """
    Create a smaller styled card for a game scoreboard using Streamlit components
    Includes odds data when available
    """
    home_logo = get_team_logo_url(iter_home_team)
    away_logo = get_team_logo_url(iter_away_team)
    
    # Format date
    game_date_str = game_date.strftime("%a, %b %d")  # Shorter date format
    
    # Create a container with built-in styling
    with st.container():
        # Add a lighter border
        st.markdown("<hr style='margin: 5px 0; border-width: 0.5px'>", unsafe_allow_html=True)
        
        # Game date and status - smaller font
        st.markdown(f"<p style='font-size:11px; margin:2px 0; text-align:center;'>{game_date_str} • {game_status}</p>", unsafe_allow_html=True)
        
        # Away team row - more compact
        col1, col2, col_odds, col_score = st.columns([0.8, 1, 0.8, 0.6])
        with col1:
            # Center-align the image with fixed height and proper vertical centering
            st.markdown(f"""
                <div style="display: flex; justify-content: center; align-items: center; height: 40px; margin: 0;">
                    <img src="{away_logo}" width="35" style="vertical-align: middle; margin: auto 0;">
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"<p style='text-align:center; font-weight:bold; margin:5px 0;'>{iter_away_team}</p>", unsafe_allow_html=True)
        with col_odds:
            if away_odds is not None:
                # Money line 2 - always negative and positive - smaller font
                if home_odds > away_odds:
                    odds_str = f"+{int(away_odds)}" if away_odds > 0 else f"{int(away_odds)}"
                    st.markdown(f"<div style='text-align:center;'><span style='color:#28a745;font-size:12px;'>{odds_str}</span></div>", unsafe_allow_html=True)
                else:
                    odds_str = f"+{int(away_odds)}" if away_odds > 0 else f"{int(away_odds)}"
                    st.markdown(f"<div style='text-align:center;'><span style='font-size:12px;'>{odds_str}</span></div>", unsafe_allow_html=True)
        with col_score:
            if away_score > home_score:
                st.markdown(f"<p style='color:#28a745;text-align:center;font-weight:bold;font-size:16px;margin:5px 0;'>{away_score}</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='text-align:center;font-weight:bold;font-size:16px;margin:5px 0;'>{away_score}</p>", unsafe_allow_html=True)
        
        # Home team row - more compact
        col1, col2, col_odds, col_score = st.columns([0.8, 1, 0.8, 0.6])
        with col1:
            # Center-align the image with fixed height and proper vertical centering
            st.markdown(f"""
                <div style="display: flex; justify-content: center; align-items: center; height: 40px; margin: 0;">
                    <img src="{home_logo}" width="35" style="vertical-align: middle; margin: auto 0;">
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"<p style='text-align:center; font-weight:bold; margin:5px 0;'>{iter_home_team}</p>", unsafe_allow_html=True)
        with col_odds:
            if home_odds is not None:
                # Money line 2 - always negative and positive - smaller font
                if home_odds < away_odds:
                    odds_str = f"{int(home_odds)}" if home_odds < 0 else f"+{int(home_odds)}"
                    st.markdown(f"<div style='text-align:center;'><span style='color:#28a745;font-size:12px;'>{odds_str}</span></div>", unsafe_allow_html=True)
                else:
                    odds_str = f"{int(home_odds)}" if home_odds < 0 else f"+{int(home_odds)}"
                    st.markdown(f"<div style='text-align:center;'><span style='font-size:12px;'>{odds_str}</span></div>", unsafe_allow_html=True)
        with col_score:
            if home_score > away_score:
                st.markdown(f"<p style='color:#28a745;text-align:center;font-weight:bold;font-size:16px;margin:5px 0;'>{home_score}</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='text-align:center;font-weight:bold;font-size:16px;margin:5px 0;'>{home_score}</p>", unsafe_allow_html=True)
        
        
def render_scoreboard():
    """
    Main function to render the scoreboard interface
    """
    #st.header("Scoreboard")
    
    # Load box score and odds data
    box_data, odds_data, available_dates = load_box_scores_and_odds()
    
    if box_data is None or len(available_dates) == 0:
        st.warning("No scoreboard data available. Please check data source.")
        return
    
    # Initialize session state for date if not already set
    if 'current_date_idx' not in st.session_state:
        st.session_state.current_date_idx = 0
    
    # Use a simple centered layout for the date selector
    col_date,_, _ = st.columns([2, 1, 1])
    
    # Date selector with calendar view
    with col_date:
        # Get current selected date
        current_date = available_dates[st.session_state.current_date_idx]
        
        # Create calendar date input
        calendar_date = st.date_input(
            "Select Date",
            value=current_date,
            min_value=min(available_dates),
            max_value=max(available_dates),
            key="calendar_date_picker"
        )
        
        # Update index when date is selected from calendar
        if calendar_date != current_date:
            # Find closest available date (since calendar might select dates without games)
            closest_idx = find_closest_date_idx(available_dates, calendar_date)
            if closest_idx is not None:
                st.session_state.current_date_idx = closest_idx
    
    # Get the selected date based on current index
    selected_date = available_dates[st.session_state.current_date_idx]
    
    # Filter data for selected date
    daily_games = box_data[box_data['date'].dt.date == selected_date]
    
    if len(daily_games) == 0:
        st.info(f"No games found for {selected_date}")
        return
    
    # Create columns to display multiple games per row - now 3 instead of 2
    cols_per_row = 4
    
    # Process each game and create scoreboard cards
    for i in range(0, len(daily_games), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j in range(cols_per_row):
            if i + j < len(daily_games):
                game = daily_games.iloc[i + j]
                game_id = game.get('gameid', None)
                
                # Get odds data if available
                home_odds = None
                away_odds = None
                
                if odds_data is not None and game_id is not None:
                    # Find the matching odds row
                    odds_row = odds_data[odds_data['gameid'] == game_id]
                    
                    if not odds_row.empty:
                        # Get home and away odds if columns exist
                        if 'home_odds_value' in odds_row.columns:
                            home_odds = odds_row['home_odds_value'].values[0]
                        if 'away_odds_value' in odds_row.columns:
                            away_odds = odds_row['away_odds_value'].values[0]
                
                with cols[j]:
                    create_scoreboard_card(
                        iter_home_team=game['tricode_for'],
                        iter_away_team=game['tricode_against'],
                        home_score=int(game['metric_score_for']),
                        away_score=int(game['metric_score_against']),
                        game_date=game['date'],
                        game_status=game["period_ending"],
                        home_odds=home_odds,
                        away_odds=away_odds
                    )
            else:
                # Empty column placeholder to maintain layout
                with cols[j]:
                    st.write("")

def find_closest_date_idx(available_dates, target_date):
    """
    Find the index of the date closest to the target date
    """
    if not isinstance(available_dates, list):
        available_dates = available_dates.tolist()
    
    # Calculate the absolute difference between each date and the target
    date_diffs = [(abs((date - target_date).days), i) for i, date in enumerate(available_dates)]
    
    # Sort by difference and get the index of the closest date
    date_diffs.sort()
    if date_diffs:
        return date_diffs[0][1]
    return None

if __name__ == "__main__":
    # For testing the module directly
    render_scoreboard()