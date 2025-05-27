'''
    NHL Analytics Application
    
    To Start Locally: Turn on the server in terminal and run this code
        ~ % cd Documents 
        Documents % cd Github/NHL-Analytics/app 
        app % streamlit run app.py
'''

import streamlit as st
import os

# Set page configuration
st.set_page_config(
    page_title="Rinklytics",
    page_icon="🏒",
    layout="centered"
)

# Add custom CSS for navigation color
st.markdown("""
<style>
    /* Main navigation bar background */
    .st-emotion-cache-z5fcl4 {
        background-color: #CE0E2D !important;
    }
    
    /* Active nav item background */
    .st-emotion-cache-1cypcdb {
        background-color: rgba(255, 255, 255, 0.15) !important; 
    }
    
    /* Nav text color */
    .st-emotion-cache-1cypcdb, .st-emotion-cache-1avcm0n, .st-emotion-cache-j7qwjs {
        color: white !important;
    }
    
    /* Dropdown background */
    .st-emotion-cache-18ni7ap {
        background-color: #CE0E2D !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function for asset paths
def get_asset_path(rel_path):
    """Helper function to get correct asset paths regardless of run location"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "assets", rel_path)

# Define pages for navigation
HOME = st.Page(page="views/home.py", title="Home", default=True)
TEAM_MAP = st.Page(page="views/teams_map.py", title="Team Locations & Travel Distance")
SCOREBOARD = st.Page(page="views/scoreboard.py", title="Scoreboard")
SKATERS = st.Page(page="views/magnifiques_patineuses.py", title="Les magnifiques patineuses")
BLOCKING_SHOTS = st.Page(page="views/blocking_shots.py", title="The Valor of Blocking Shots")
ABOUT = st.Page(page="views/about.py", title="About")

# Add logo at the top
#st.image(get_asset_path("img_main.jpeg"), width=120)

# Create navigation
pages = {
    "Bienvenue": [HOME, ABOUT],
    "Team": [SCOREBOARD, TEAM_MAP],
    "Skaters": [SKATERS, BLOCKING_SHOTS],
}
pg = st.navigation(pages)
pg.run()

# Footer styling 
st.markdown("""
<hr style="border: none; height: 2px; background-color: #cf8a00; margin-top: 25px; margin-bottom: 10px;">
<div style="text-align: center; color: #666; font-size: 12px;">
    Created by JJ - data scientist by day, and hockey enthusiast all day.
</div>
""", unsafe_allow_html=True)
