"""
Enhanced Main RAG Application - Cleaned up version
"""
import streamlit as st
from styles import apply_custom_css
from session_state import init_session_state
from sidebar import render_sidebar
from chat import render_chat
from onboarding import render_onboarding, render_quick_start_card
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
    st.code("python rag_backend.py", language="bash")
    st.stop()

# Render pending toasts
ToastNotification.render_pending()

# Show onboarding for first-time users
if st.session_state.get('show_onboarding', True):
    render_onboarding()

# Header
st.markdown('<div class="main-header">📚 RAG Chat Assistant</div>', unsafe_allow_html=True)

# Show quick start if no documents and onboarding dismissed
if health_data and health_data.get('document_count', 0) == 0 and not st.session_state.get('show_onboarding', True):
    render_quick_start_card()

# Sidebar
render_sidebar(api_client)

# Main chat interface
model = st.session_state.get('current_model', DEFAULT_LLM_MODEL)
render_chat(api_client, health_data, model)
