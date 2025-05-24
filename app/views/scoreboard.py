import streamlit as st
from exe.team_stats import render_scoreboard

st.title("NHL Scoreboard")
st.markdown("View the latest NHL game scores and betting odds.")

# Render the scoreboard component
render_scoreboard()