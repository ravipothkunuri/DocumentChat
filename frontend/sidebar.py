"""
Clean and modern sidebar with document management
"""
import streamlit as st
from typing import List, Dict
from toast import ToastNotification
from session_state import (
    get_current_chat, clear_chat, load_conversation, 
    delete_conversation
)
from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, DEFAULT_LLM_MODEL
from datetime import datetime


def render_document_card(doc: Dict, api_client):
    """Render clean document card"""
    doc_name = doc['filename']
    is_selected = st.session_state.selected_document == doc_name
    
    with st.container():
        col1, col2 = st.columns([4, 1])
        
        with col1:
            if st.button(
                f"{'📘' if is_selected else '📄'} {doc_name}",
                key=f"select_{doc_name}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
                disabled=st.session_state.is_generating,
                help=f"{doc['chunks']} chunks • {doc['size'] / 1024:.1f} KB"
            ):
                st.session_state.selected_document = doc_name
                st.rerun()
        
        with col2:
            if st.button("🗑️", key=f"delete_{doc_name}", 
                       help="Delete",
                       disabled=st.session_state.is_generating):
                status_code, response = api_client.delete_document(doc_name)
                if status_code == 200:
                    if doc_name in st.session_state.document_chats:
                        del st.session_state.document_chats[doc_name]
                    if st.session_state.selected_document == doc_name:
                        st.session_state.selected_document = None
                    
                    ToastNotification.show(f"Deleted {doc_name}", "success")
                    st.rerun()
                else:
                    ToastNotification.show(f"{response.get('message', 'Failed')}", "error")


def render_conversation_history():
    """Render conversation history"""
    if not st.session_state.conversation_history:
        st.info("No saved conversations")
        return
    
    st.caption(f"{len(st.session_state.conversation_history)} conversation(s)")
    
    for conv in st.session_state.conversation_history:
        is_current = st.session_state.selected_conversation == conv['id']
        
        col1, col2 = st.columns([4, 1])
        
        with col1:
            dt = datetime.fromisoformat(conv['timestamp'])
            time_str = dt.strftime('%m/%d %H:%M')
            
            if st.button(
                f"{'📘' if is_current else '📄'} {conv['title'][:20]}...\n{time_str}",
                key=f"conv_{conv['id']}",
                use_container_width=True,
                type="primary" if is_current else "secondary"
            ):
                load_conversation(conv['id'])
                st.rerun()
        
        with col2:
            if st.button("🗑️", key=f"del_conv_{conv['id']}", help="Delete"):
                delete_conversation(conv['id'])
                if st.session_state.selected_conversation == conv['id']:
                    st.session_state.selected_conversation = None
                ToastNotification.show("Deleted", "success")
                st.rerun()


def upload_files(files: List, api_client):
    """Handle file upload"""
    success_count = 0
    uploaded_names = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(files):
        status_text.text(f"Uploading {i + 1}/{len(files)}: {file.name}")
        
        status_code, response = api_client.upload_file(file)
        
        if status_code == 200:
            ToastNotification.show(f"✓ {file.name}", "success")
            success_count += 1
            uploaded_names.append(file.name)
        else:
            ToastNotification.show(f"✗ {file.name}: {response.get('message', 'Failed')}", "error")
        
        progress_bar.progress((i + 1) / len(files))
    
    status_text.empty()
    progress_bar.empty()
    
    if success_count > 0:
        st.success(f"✅ Uploaded {success_count}/{len(files)} file(s)")
    
    if uploaded_names and not st.session_state.selected_document:
        st.session_state.selected_document = uploaded_names[0]
    
    st.session_state.uploader_key += 1
    st.rerun()


def render_model_selector(api_client):
    """Render model selection"""
    st.subheader("🤖 Model")
    
    models_data = api_client.get_models()
    current_config = models_data.get('current_config', {})
    ollama_models = models_data.get('ollama', {})
    
    llm_models = ollama_models.get('llm_models', [DEFAULT_LLM_MODEL])
    current_model = st.session_state.get('current_model', 
                                         current_config.get('model', DEFAULT_LLM_MODEL))
    
    if current_model not in llm_models and llm_models:
        current_model = llm_models[0]
    
    selected_model = st.selectbox(
        "Chat Model",
        options=llm_models,
        index=llm_models.index(current_model) if current_model in llm_models else 0,
        disabled=st.session_state.is_generating,
        key="model_selector",
        label_visibility="collapsed"
    )
    
    if selected_model != st.session_state.get('current_model'):
        st.session_state.current_model = selected_model
        ToastNotification.show(f"Model: {selected_model}", "success")
        st.rerun()


def render_sidebar(api_client):
    """Render clean sidebar"""
    with st.sidebar:
        render_model_selector(api_client)
        st.markdown("---")
        
        tab1, tab2 = st.tabs(["📚 Documents", "💬 History"])
        
        with tab1:
            documents = api_client.get_documents()
            
            if documents:
                st.caption(f"{len(documents)} document(s)")
                for doc in documents:
                    render_document_card(doc, api_client)
            else:
                st.info("No documents yet")
            
            st.markdown("---")
            st.subheader("📤 Upload")
            
            uploaded_files = st.file_uploader(
                "Choose files",
                type=ALLOWED_EXTENSIONS,
                accept_multiple_files=True,
                help=f"{', '.join(ALLOWED_EXTENSIONS).upper()} (max {MAX_FILE_SIZE_MB}MB)",
                key=f"uploader_{st.session_state.uploader_key}",
                disabled=st.session_state.is_generating,
                label_visibility="collapsed"
            )
            
            if uploaded_files:
                if 'last_uploaded_files' not in st.session_state:
                    st.session_state.last_uploaded_files = []
                
                current_file_names = [f.name for f in uploaded_files]
                
                if current_file_names != st.session_state.last_uploaded_files:
                    st.session_state.last_uploaded_files = current_file_names
                    upload_files(uploaded_files, api_client)
        
        with tab2:
            render_conversation_history()
        
        if st.session_state.selected_document and get_current_chat():
            st.markdown("---")
            if st.button("🗑️ Clear Chat", use_container_width=True, 
                        disabled=st.session_state.is_generating):
                clear_chat()
                st.rerun()
