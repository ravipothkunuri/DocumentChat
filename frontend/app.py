"""
Enhanced Main RAG Application with all new features
"""
import streamlit as st
from styles import apply_custom_css
from session_state import init_session_state
from sidebar import render_sidebar
from chat import render_chat
from onboarding import render_onboarding, render_quick_start_card, render_feature_tour
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

# Header with status indicator
col1, col2, col3 = st.columns([6, 2, 2])

with col1:
    st.markdown('<div class="main-header">📚 RAG Chat Assistant</div>', unsafe_allow_html=True)

with col2:
    if health_data:
        doc_count = health_data.get('document_count', 0)
        total_queries = health_data.get('total_queries', 0)
        st.metric("Documents", doc_count)

with col3:
    if health_data:
        st.metric("Total Queries", total_queries)

# Show quick start if no documents and onboarding dismissed
if health_data and health_data.get('document_count', 0) == 0 and not st.session_state.get('show_onboarding', True):
    render_quick_start_card()

# Contextual feature tour
render_feature_tour(health_data)

# Sidebar
render_sidebar(api_client)

# Main chat interface
model = st.session_state.get('current_model', DEFAULT_LLM_MODEL)
render_chat(api_client, health_data, model)

# Footer with quick stats
if health_data and health_data.get('document_count', 0) > 0:
    st.markdown("---")
    footer_col1, footer_col2, footer_col3, footer_col4 = st.columns(4)
    
    with footer_col1:
        st.caption(f"📊 **{health_data.get('total_chunks', 0)}** total chunks indexed")
    
    with footer_col2:
        st.caption(f"🤖 **{health_data['configuration']['model']}** active model")
    
    with footer_col3:
        if st.session_state.selected_document:
            chat_len = len(st.session_state.document_chats.get(st.session_state.selected_document, []))
            st.caption(f"💬 **{chat_len}** messages in current chat")
        else:
            st.caption("💬 No chat selected")
    
    with footer_col4:
        history_count = len(st.session_state.get('conversation_history', []))
        st.caption(f"🗂️ **{history_count}** saved conversations")
