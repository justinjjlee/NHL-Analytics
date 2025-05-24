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
ABOUT = st.Page(page="views/about.py", title="About")

# Add logo at the top
#st.image(get_asset_path("img_main.jpeg"), width=120)

# Create navigation
pages = {
    "Bienvenue": [HOME,ABOUT],
    "Team": [SCOREBOARD, TEAM_MAP],
    "Skaters": [SKATERS],
}
pg = st.navigation(pages)
pg.run()

# Footer with version
#st.caption("NHL Analytics • Version 1.0.0")

# Footer styling 
st.markdown("""
<hr style="border: none; height: 2px; background-color: #cf8a00; margin-top: 25px; margin-bottom: 10px;">
<div style="text-align: center; color: #666; font-size: 12px;">
    Created by JJ - data scientist by day, and hockey enthusiast all day.
</div>
""", unsafe_allow_html=True)
