import streamlit as st
import os
from exe.map_utils import load_map_data, render_team_map
from streamlit_folium import st_folium

st.title("NHL Team Locations & Distance")

# Initialize session state for reset functionality
if 'reset_map' not in st.session_state:
    st.session_state.reset_map = False

# Load map data
map_data = load_map_data()

# Create a row with two columns for the refresh button and the distance string
col_button, col_team1, col_team2 = st.columns([1, 2, 2])

# Button to refresh the map (reset zoom)
with col_button:
    # Add some vertical space to push the button down
    st.markdown("<style>.push-button-down {margin-top: auto;}</style>", unsafe_allow_html=True)
    
    # Create a container for the button with bottom alignment
    with st.container():
        # Add class to this container
        st.markdown('<div class="push-button-down">', unsafe_allow_html=True)
        
        # Place the button
        if st.button("Refresh Map"):
            st.session_state.reset_map = True
            st.rerun()
            
        st.markdown('</div>', unsafe_allow_html=True)

# Check if we need to reset values (from button press)
if st.session_state.reset_map:
    # Clear the flag
    st.session_state.reset_map = False
    # Use empty strings as initial values
    initial_home = ""
    initial_away = ""
else:
    # Use existing values if available
    initial_home = st.session_state.get('home_team', "")
    initial_away = st.session_state.get('away_team', "")

# Create two columns for side-by-side dropdowns
#col1, col2 = st.columns(2)

# Home team selection on the left column
with col_team1:
    team_options = [""] + sorted(map_data["home_team"].unique().tolist())
    home_team = st.selectbox(
        "Select Home Team", 
        options=team_options,
        index=0 if initial_home == "" else team_options.index(initial_home)
    )

# Away team selection on the right column
with col_team2:
    away_team = st.selectbox(
        "Select Away Team", 
        options=team_options,
        index=0 if initial_away == "" else team_options.index(initial_away)
    )

# Store selections
st.session_state.home_team = home_team
st.session_state.away_team = away_team

# Render the map using the utility function
refresh_button = st.session_state.reset_map
m = render_team_map(map_data, home_team, away_team, refresh_button, travel_mode=False)

# Render the map in Streamlit
st_folium(m, width=700, height=500)