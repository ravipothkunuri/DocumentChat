"""
Sidebar components for document management
"""
import streamlit as st
from typing import List, Dict
from toast import ToastNotification
from session_state import get_current_chat, clear_chat
from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, DEFAULT_LLM_MODEL


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
        if st.button("✕", key=f"delete_{doc_name}", 
                   help="Delete document",
                   type="secondary",
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
                ToastNotification.show(f"{response.get('message', 'Delete failed')}", "error")
    
    if is_selected:
        st.caption(f"📊 {doc['chunks']} chunks • {doc['size']:,} bytes • {doc['type'].upper()}")
        if doc_name in st.session_state.document_chats:
            msg_count = len(st.session_state.document_chats[doc_name])
            if msg_count > 0:
                st.caption(f"💬 {msg_count} messages")


def upload_files(files: List, api_client):
    """Handle file upload"""
    success_count = 0
    uploaded_names = []
    progress = st.progress(0)
    
    for i, file in enumerate(files):
        status_code, response = api_client.upload_file(file)
        
        if status_code == 200:
            ToastNotification.show(f"{file.name}: {response.get('chunks', 0)} chunks", "success")
            success_count += 1
            uploaded_names.append(file.name)
        else:
            ToastNotification.show(f"{file.name}: {response.get('message', 'Failed')}", "error")
        
        progress.progress((i + 1) / len(files))
    
    if uploaded_names and not st.session_state.selected_document:
        st.session_state.selected_document = uploaded_names[0]
    
    st.session_state.uploader_key += 1
    st.rerun()


def render_model_selector(api_client):
    """Render model selection dropdown"""
    st.subheader("🤖 Model Settings")
    
    # Get available models
    models_data = api_client.get_models()
    
    # Get current configuration
    current_config = models_data.get('current_config', {})
    ollama_models = models_data.get('ollama', {})
    
    # LLM Model selection
    llm_models = ollama_models.get('llm_models', [DEFAULT_LLM_MODEL])
    current_model = st.session_state.get('current_model', 
                                         current_config.get('model', DEFAULT_LLM_MODEL))
    
    # Ensure current model is in the list
    if current_model not in llm_models and llm_models:
        current_model = llm_models[0]
    
    selected_model = st.selectbox(
        "💬 Chat Model",
        options=llm_models,
        index=llm_models.index(current_model) if current_model in llm_models else 0,
        help="Select the language model for chat responses",
        disabled=st.session_state.is_generating,
        key="model_selector"
    )
    
    # Update session state if changed
    if selected_model != st.session_state.get('current_model'):
        st.session_state.current_model = selected_model
        ToastNotification.show(f"Model changed to {selected_model}", "success")
        st.rerun()
    
    # Embedding Model info (read-only)
    embedding_model = current_config.get('embedding_model', 'nomic-embed-text')
    st.caption(f"🔢 Embedding: **{embedding_model}**")


def render_sidebar(api_client):
    """Render sidebar"""
    with st.sidebar:        
        documents = api_client.get_documents()
        if documents:
            st.info(f"📊 {len(documents)} document(s) loaded")
        st.subheader("📖 Your Documents")
        
        if documents:
            for doc in documents:
                render_document_card(doc, api_client)
        else:
            st.info("💡 No documents yet. Upload below!")
        
        st.markdown("---")
        st.subheader("📤 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=ALLOWED_EXTENSIONS,
            accept_multiple_files=True,
            help=f"Supported: {', '.join(ALLOWED_EXTENSIONS).upper()} (max {MAX_FILE_SIZE_MB}MB)",
            key=f"uploader_{st.session_state.uploader_key}",
            disabled=st.session_state.is_generating
        )
        
        # Auto-process files when uploaded
        if uploaded_files:
            if 'last_uploaded_files' not in st.session_state:
                st.session_state.last_uploaded_files = []
            
            current_file_names = [f.name for f in uploaded_files]
            
            if current_file_names != st.session_state.last_uploaded_files:
                st.session_state.last_uploaded_files = current_file_names
                upload_files(uploaded_files, api_client)
        
        # Model selector section
        st.markdown("---")
        render_model_selector(api_client)
        
        if st.session_state.selected_document and get_current_chat():
            st.markdown("---")
            if st.button("💬 Clear Chat", use_container_width=True, 
                        disabled=st.session_state.is_generating):
                clear_chat()
                st.rerun()
