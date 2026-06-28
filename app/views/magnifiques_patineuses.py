import streamlit as st
from i18n import t
import importlib
import exe.draft_analysis as draft_analysis
importlib.reload(draft_analysis)

from exe.draft_analysis import render_exceptional_players_analysis, render_not_so_magnificent_analysis



st.title(t("mp_title"))
st.markdown(t("mp_desc"))

# Create tabs for different analyses
draft_tabs = st.tabs([t("first_round_offensive"), t("not_so_magnificent")])

with draft_tabs[0]:
    render_exceptional_players_analysis()

with draft_tabs[1]:
    render_not_so_magnificent_analysis()