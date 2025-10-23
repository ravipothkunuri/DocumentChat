# app.py (or streamlit_app.py)
"""
Main RAG Application Entry Point
"""
import streamlit as st
from styles import apply_custom_css
from session_state import init_session_state
from sidebar import render_sidebar
from chat import render_chat  # or from main_app import render_chat
from api_client import RAGAPIClient
from toast import ToastNotification
from config import API_BASE_URL, DEFAULT_LLM_MODEL

# Page config
st.set_page_config(
    page_title="RAG Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize
init_session_state()
apply_custom_css()

# API client
api_client = RAGAPIClient(API_BASE_URL)

# Check backend health
is_healthy, health_data = api_client.health_check()

if not is_healthy:
    st.error("❌ Backend is not available. Please start the FastAPI server.")
    st.stop()

# Render pending toasts
ToastNotification.render_pending()

# Header
st.markdown('<div class="main-header">📚 RAG Chat</div>', unsafe_allow_html=True)

# Sidebar
render_sidebar(api_client)

# Main chat
model = st.session_state.get('current_model', DEFAULT_LLM_MODEL)
render_chat(api_client, health_data, model)
