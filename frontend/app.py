"""Main Application Entry Point"""
import streamlit as st
from styles import apply_custom_css
from utils  import init_session_state, ToastNotification
from sidebar import render_sidebar
from chat import render_chat
from api_client import APIClient
from config import API_BASE_URL

st.set_page_config(
    page_title="Chat With Documents using AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

init_session_state()
apply_custom_css()
api_client = APIClient(API_BASE_URL)

is_healthy, health_data = api_client.health_check()
if not is_healthy:
    st.error("❌ Backend unavailable. Start FastAPI server.")
    st.stop()

ToastNotification.render_pending()
st.markdown('<div class="main-header">📚 Chat With Documents using AI</div>', unsafe_allow_html=True)

render_sidebar(api_client)
render_chat(api_client, health_data)
