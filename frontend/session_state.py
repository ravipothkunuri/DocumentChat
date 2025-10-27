"""Session state management"""
import streamlit as st
from typing import List, Dict

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'document_chats': {},
        'selected_document': None,
        'uploader_key': 0,
        'pending_toasts': [],
        'last_uploaded_files': [],
        'is_generating': False,
        'stop_generation': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_current_chat() -> List[Dict]:
    """Get chat history for selected document"""
    doc = st.session_state.selected_document
    if doc and doc not in st.session_state.document_chats:
        st.session_state.document_chats[doc] = []
    return st.session_state.document_chats.get(doc, [])

def add_message(message: Dict):
    """Add message to current chat"""
    if doc := st.session_state.selected_document:
        st.session_state.document_chats.setdefault(doc, []).append(message)

def clear_chat():
    """Clear current chat history"""
    if doc := st.session_state.selected_document:
        st.session_state.document_chats[doc] = []
