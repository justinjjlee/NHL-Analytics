import streamlit as st
from i18n import t

st.title(t("about_title"))

st.markdown(f"""
{t("about_desc_1")}

{t("about_desc_2")}

{t("about_desc_3")}
""")

# Contact information
st.subheader(t("about_contact"))
st.markdown(t("about_contact_desc"))