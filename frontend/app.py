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

# Check document state
document_count = health_data.get('document_count', 0) if health_data else 0
has_selected_document = st.session_state.selected_document is not None

# Dismiss onboarding when a document is selected
if has_selected_document:
    st.session_state.show_onboarding = False

# Determine what to show
show_onboarding_now = st.session_state.get('show_onboarding', True)

if not has_selected_document and show_onboarding_now:
    # First time user, no document selected
    render_onboarding()
elif not has_selected_document and document_count == 0:
    # No documents uploaded yet
    render_quick_start_card()
else:
    # Document is selected or user dismissed onboarding - show chat
    model = st.session_state.get('current_model', DEFAULT_LLM_MODEL)
    if health_data:
        render_chat(api_client, health_data, model)
    else:
        st.warning("⚠️ Unable to connect to backend")
