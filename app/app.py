'''
    NHL Analytics Application (Rinklytics Dashboard)
    
    To Start Locally:
        ~ % cd Documents 
        Documents % cd Github/NHL-Analytics/app 
        app % streamlit run app.py
'''

import streamlit as st
import os
import importlib
import i18n
try:
    importlib.reload(i18n)
except Exception:
    pass
from i18n import t

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

# Initialize language state
if 'lang' not in st.session_state:
    st.session_state.lang = 'EN'

# Sidebar language toggle
st.sidebar.radio(
    "Language / Langue", 
    ["EN", "FR"], 
    key="lang"
)

# Define pages for navigation with constant url_path to preserve selection across languages
HOME = st.Page(page="views/home.py", title=t("nav_home"), url_path="home", default=True)
TEAM_MAP = st.Page(page="views/teams_map.py", title=t("nav_team_map"), url_path="teams-map")
SCOREBOARD = st.Page(page="views/scoreboard.py", title=t("nav_scoreboard"), url_path="scoreboard")
TEAM_SEASON_STATS = st.Page(page="views/team_season_stats.py", title=t("nav_team_season"), url_path="team-season-stats")
SKATERS = st.Page(page="views/magnifiques_patineuses.py", title=t("nav_mag_patineuses"), url_path="magnifiques-patineuses")
BLOCKING_SHOTS = st.Page(page="views/blocking_shots.py", title=t("nav_blocking_shots"), url_path="blocking-shots")
HIT_IMPACT = st.Page(page="views/hit_impact.py", title=t("nav_hit_impact"), url_path="hit-impact")
SKATER_STATS = st.Page(page="views/skater_stats.py", title=t("nav_skater_stats"), url_path="skater-stats")
GOALIE_STATS = st.Page(page="views/goalie_stats.py", title=t("nav_goalie_stats"), url_path="goalie-stats")
ABOUT = st.Page(page="views/about.py", title=t("nav_about"), url_path="about")

# Add logo at the top
#st.image(get_asset_path("img_main.jpeg"), width=120)

# Create navigation
pages = {
    t("nav_bienvenue"): [HOME, ABOUT],
    t("nav_team"): [SCOREBOARD, TEAM_MAP, TEAM_SEASON_STATS],
    t("nav_skaters"): [SKATER_STATS, GOALIE_STATS, SKATERS],
    t("nav_action"): [BLOCKING_SHOTS, HIT_IMPACT],
}
pg = st.navigation(pages)
pg.run()

# Footer styling 
st.markdown(f"""
<hr style="border: none; height: 2px; background-color: #cf8a00; margin-top: 25px; margin-bottom: 10px;">
<div style="text-align: center; color: #666; font-size: 12px;">
    {t("footer_created_by")}
</div>
""", unsafe_allow_html=True)
