"""Sidebar components """
import streamlit as st
from typing import List, Dict
from toast import ToastNotification
from session_state import get_current_chat, clear_chat
from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, FIXED_LLM_MODEL

def render_document_card(doc: Dict, api_client):
    """Render document card with selection and delete"""
    doc_name = doc['filename']
    is_selected = st.session_state.selected_document == doc_name
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        if st.button(
            f"{'📘' if is_selected else '📄'} **{doc_name}**",
            key=f"select_{doc_name}",
            use_container_width=True,
            type="primary" if is_selected else "secondary",
            disabled=st.session_state.is_generating
        ):
            st.session_state.selected_document = doc_name
            st.rerun()
    
    with col2:
        if st.button("✕", key=f"delete_{doc_name}", help="Delete", disabled=st.session_state.is_generating):
            status_code, response = api_client.delete_document(doc_name)
            if status_code == 200:
                st.session_state.document_chats.pop(doc_name, None)
                if st.session_state.selected_document == doc_name:
                    st.session_state.selected_document = None
                ToastNotification.show(f"Deleted {doc_name}", "success")
                st.rerun()
            else:
                ToastNotification.show(response.get('message', 'Delete failed'), "error")
    
    if is_selected:
        st.caption(f"📊 {doc['chunks']} chunks • {doc['size']:,} bytes • {doc['type'].upper()}")
        if msg_count := len(st.session_state.document_chats.get(doc_name, [])):
            st.caption(f"💬 {msg_count} messages")

def render_sidebar(api_client):
    """Render sidebar"""
    with st.sidebar:
        documents = api_client.get_documents()
        if documents:
            st.info(f"📊 {len(documents)} document(s) loaded")
        
        st.subheader("📖 Your Documents")
        for doc in documents:
            render_document_card(doc, api_client)
        if not documents:
            st.info("💡 No documents yet. Upload below!")
        
        st.markdown("---")
        st.subheader("📤 Upload Documents")
        st.caption(f"🤖 Using model: **{FIXED_LLM_MODEL}**")
        
        if st.session_state.selected_document and get_current_chat():
            st.markdown("---")
            if st.button("💬 Clear Chat", use_container_width=True, disabled=st.session_state.is_generating):
                clear_chat()
                st.rerun()
