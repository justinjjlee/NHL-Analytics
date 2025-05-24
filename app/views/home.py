import streamlit as st
import os

def get_asset_path(rel_path):
    """Helper function to get correct asset paths"""
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(current_dir, "assets", rel_path)

st.title("Rinklytics: NHL Analytics App")

st.markdown("""
Welcome to Rinklytics - a hub for insightful hockey statistics and visualizations.
""")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Featured Visualizations")
    st.markdown("""
    * **Team Map** - Explore NHL team locations and distances
    * **Live Scoreboard** - Check current game scores and odds
    * **Les magnifiques patineuses** - Player analysis and performance insights
    """)

with col2:
    st.image(get_asset_path("img_main.jpeg"), width=300)

st.markdown("""
<div style="margin-top: 20px;">
    <a href="https://github.com/justinjjlee/NHL-Analytics" target="_blank" style="text-decoration: none; margin-right: 15px;">
        <img src="https://img.shields.io/badge/GitHub-black?style=for-the-badge&logo=github" width="80">
    </a>
    <a href="https://medium.com/@thinkingjustin" target="_blank" style="text-decoration: none;">
        <img src="https://img.shields.io/badge/Medium-black?style=for-the-badge&logo=medium" width="80">
    </a>
</div>
""", unsafe_allow_html=True)