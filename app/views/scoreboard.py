import streamlit as st
from exe.team_stats import render_scoreboard
from i18n import t

st.title(t("sb_title"))
st.markdown(t("sb_desc"))

# Render the scoreboard component
render_scoreboard()