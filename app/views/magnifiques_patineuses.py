import streamlit as st
from exe.draft_analysis import render_exceptional_players_analysis

st.title("Les Magnifiques Patineuses")
st.markdown("Player analysis and exceptional performance data.")

# Create tabs for different analyses
draft_tabs = st.tabs(["Choix du premier tour: offensif", "College Alumni"])

with draft_tabs[0]:
    render_exceptional_players_analysis()

with draft_tabs[1]:
    st.info("À venir bientôt!")