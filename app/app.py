'''
    Create NHL Team Geospatial Location through Streamlit

    To Start Locally: Turn on the server in terminal and run this code
        ~ % cd Documents 
        Documents % cd Github/NHL-Analytics/app 
        app % streamlit run team_mapper_app.py
        
        % cd Github/NHL-Analytics/app  
        % streamlit run team_mapper_app.py
'''

import streamlit as st
import folium
from folium import DivIcon
from streamlit_folium import st_folium
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from exe.map_utils import load_map_data, render_team_map
from exe.draft_analysis import render_exceptional_players_analysis
from exe.team_stats import render_scoreboard  # Import the new module

# Sidebar content
# Add logo image
st.sidebar.image("../app/assets/img_main.jpeg", use_container_width=True)

st.sidebar.markdown(
    """
    <hr style="border: none; height: 2px; background-color: #ffb703; margin-top: 5px; margin-bottom: 5px;">
    """,
    unsafe_allow_html=True
)

st.sidebar.title("NHL Analytics")
st.sidebar.markdown(
    """
    <div style="font-size: 14px;">
        I'm JJ - data scientist by day, and hockey enthusiast all day. This is my Streamlit application for my project on 
        analyzing and visualizin NHL team and player performance to unpack my curiosity. See my work on
        <a href="https://github.com/justinjjlee/NHL-Analytics" target="_blank" style="text-decoration: none; color: #ffb703;">
            GitHub
        </a>
        and
        <a href="https://medium.com/@thinkingjustin" target="_blank" style="text-decoration: none; color: #ffb703;">
            Medium articles.
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar subsection
# Add spacing before the subheader
st.sidebar.markdown(
    """
    <div style="margin-top: 15px;"></div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <hr style="border: none; height: 2px; background-color: #ffb703; margin-top: 5px; margin-bottom: 5px;">
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.subheader("Navigation")

# Create a radio button for page selection
page = st.sidebar.radio(
    "Select Analysis",
    ["Teams", "Les magnifiques patineuses"],
    index=0
)

# Load the map data once
map_data = load_map_data()

if page == "Teams":
    st.title("Teams")
    
    # Create tabs within this page for different map views
    map_tabs = st.tabs(["Team Locations & Distance", "Scoreboard"])
    
    with map_tabs[0]:
        # .............................................................................................................
        # (*) The standard map & distance view
        # .............................................................................................................
        #st.header("NHL Team Locations & Distance")

        # Create a row with two columns for the refresh button and the distance string
        col_button, col_text = st.columns([1, 4])

        # Button to refresh the map (reset zoom)
        with col_button:
            refresh_button = st.button("Refresh Map")

        # Handle refresh button click before rendering the widgets
        if refresh_button:
            st.session_state.home_team = ""
            st.session_state.away_team = ""

        # Create two columns for side-by-side dropdowns
        col1, col2 = st.columns(2)

        # Home team selection on the left column
        with col1:
            home_team = st.selectbox("Select Home Team", [""] + map_data["home_team"].unique().tolist(), key="home_team")

        # Away team selection on the right column
        with col2:
            away_team = st.selectbox("Select Away Team", [""] + map_data["away_team"].unique().tolist(), key="away_team")

        # Render the map using the utility function
        m = render_team_map(map_data, home_team, away_team, refresh_button, travel_mode=False)
        
        # Render the map in Streamlit
        st_folium(m, width=700, height=500)

    with map_tabs[1]:
        # Placeholder for future map analysis
        #st.info("Future home for git-compiled data.")
        render_scoreboard()

elif page == "Les magnifiques patineuses":
    # Call the function from the imported module
    st.title("The Magnificant Skaters")
    
    # Create tabs within this page for different draft analyses
    draft_tabs = st.tabs(["Choix du premier tour", "College Alumni"])
    
    with draft_tabs[0]:
        #render_exceptional_players_analysis()
        st.info("à venir bientôt!")
    
    with draft_tabs[1]:
        st.info("à venir bientôt!")
    