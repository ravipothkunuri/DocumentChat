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

# Header
st.markdown('<div class="main-header">📚 RAG Chat Assistant</div>', unsafe_allow_html=True)

# Sidebar
render_sidebar(api_client)

# Show onboarding for first-time users or when no document is selected
document_count = health_data.get('document_count', 0) if health_data else 0
has_selected_document = st.session_state.selected_document is not None

# Show onboarding if: first time OR (has documents but none selected)
show_onboarding_now = st.session_state.get('show_onboarding', True)
if show_onboarding_now or (document_count > 0 and not has_selected_document):
    render_onboarding()
elif document_count == 0:
    # Show quick start if no documents uploaded yet
    render_quick_start_card()

# Main chat interface
model = st.session_state.get('current_model', DEFAULT_LLM_MODEL)
if health_data:
    render_chat(api_client, health_data, model)
else:
    st.warning("⚠️ Unable to connect to backend")
