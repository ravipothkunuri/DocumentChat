"""
Main application entry point for RAG Assistant
"""
import streamlit as st
from config import API_BASE_URL
from api_client import RAGAPIClient
from session_state import init_session_state
from toast import ToastNotification
from styles import apply_custom_css
from sidebar import render_sidebar
from chat import render_chat


def main():
    """Main application"""
    st.set_page_config(
        page_title="RAG Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="auto"
    )
    
    apply_custom_css()
    init_session_state()
    
    # Render pending toasts at the start of each render
    ToastNotification.render_pending()
    
    api_client = RAGAPIClient(API_BASE_URL)
    
    health_ok, health_data = api_client.health_check()
    
    if not health_ok:
        st.error("🔴 **Backend Offline**")
        st.markdown("Start the backend:\n```\npython rag_backend.py\n```")
        return
    
    render_sidebar(api_client)
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        title = st.session_state.selected_document or "Chat with Documents"
        st.markdown(f'<h2 style="margin-bottom: 0;">💬 {title}</h2>', unsafe_allow_html=True)
    
    with col2:
        models_data = api_client.get_models()
        llm_models = models_data.get('ollama', {}).get('llm_models', ['phi3'])
        current = health_data.get('configuration', {}).get('model', 'phi3') if health_data else 'phi3'
        
        model = st.selectbox(
            "🤖 Model",
            options=llm_models,
            index=llm_models.index(current) if current in llm_models else 0,
            label_visibility="collapsed",
            disabled=st.session_state.is_generating
        )
    
    st.markdown("---")
    render_chat(api_client, health_data, model)


if __name__ == "__main__":
    main()
