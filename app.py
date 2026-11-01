import streamlit as st

try:
    app_code = st.secrets["APP_CODE"]
except Exception:
    st.error("⚠️ APP_CODE secret not found. Please check Streamlit Secrets.")
    st.stop()

exec(app_code)
