import streamlit as st
import os
from i18n import t

def get_asset_path(rel_path):
    """Helper function to get correct asset paths."""
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(current_dir, "assets", rel_path)

# Header Section
col_head1, col_head2 = st.columns([3, 2])
with col_head1:
    st.title(t("home_title"))
    st.markdown(t("home_welcome"))
with col_head2:
    st.image(get_asset_path("img_main.jpeg"), width=260)

# Section 1: Platform Capabilities (Grouped at High Level)
st.markdown("---")
st.subheader(t("home_caps_heading"))

c1, c2 = st.columns(2)

with c1:
    st.markdown(f"#### 🏒 {t('home_cap_group1_title')}")
    st.markdown(t("home_cap_group1_desc"))

    st.markdown(f"#### 🎯 {t('home_cap_group2_title')}")
    st.markdown(t("home_cap_group2_desc"))

with c2:
    st.markdown(f"#### 💥 {t('home_cap_group3_title')}")
    st.markdown(t("home_cap_group3_desc"))

    st.markdown(f"#### ⚡ {t('home_cap_group4_title')}")
    st.markdown(t("home_cap_group4_desc"))

# Section 2: About the Creator
st.markdown("---")
st.subheader(t("home_creator_heading"))
st.markdown(t("home_creator_bio"))

st.markdown("""
<div style="display: flex; gap: 14px; align-items: center; margin-top: 15px; margin-bottom: 10px; flex-wrap: wrap;">
    <a href="https://github.com/justinjjlee/NHL-Analytics" target="_blank" style="text-decoration: none;">
        <img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" height="28" alt="GitHub">
    </a>
    <a href="https://medium.com/@thinkingjustin" target="_blank" style="text-decoration: none;">
        <img src="https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white" height="28" alt="Medium">
    </a>
    <a href="https://huggingface.co/jjerlee" target="_blank" style="text-decoration: none;">
        <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000" height="28" alt="Hugging Face">
    </a>
</div>
""", unsafe_allow_html=True)