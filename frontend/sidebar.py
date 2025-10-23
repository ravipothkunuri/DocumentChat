"""
Enhanced sidebar with document preview and conversation history
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


def render_document_preview_dropdown(doc: Dict):
    """Render document preview in dropdown"""
    with st.expander(f"📄 {doc['filename']}", expanded=False):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"""
            **Size:** {doc['size'] / 1024:.1f} KB  
            **Type:** {doc['type'].upper()}  
            **Chunks:** {doc['chunks']}  
            **Uploaded:** {datetime.fromisoformat(doc['uploaded_at']).strftime('%Y-%m-%d %H:%M')}
            """)
        
        with col2:
            is_selected = st.session_state.selected_document == doc['filename']
            if st.button(
                "✓ Selected" if is_selected else "Select",
                key=f"select_dropdown_{doc['filename']}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
                disabled=st.session_state.is_generating or is_selected
            ):
                st.session_state.selected_document = doc['filename']
                st.rerun()


def render_document_card(doc: Dict, api_client):
    """Render compact document card with dropdown menu"""
    doc_name = doc['filename']
    is_selected = st.session_state.selected_document == doc_name
    
    # Single card with all information
    with st.container():
        col1, col2, col3 = st.columns([5, 1, 1])
        
        with col1:
            # Document name button
            if st.button(
                f"{'📘' if is_selected else '📄'} {doc_name}",
                key=f"select_{doc_name}",
                use_container_width=True,
                type="primary" if is_selected else "secondary",
                disabled=st.session_state.is_generating
            ):
                st.session_state.selected_document = doc_name
                st.rerun()
        
        with col2:
            # Info dropdown
            if st.button("ℹ️", key=f"info_{doc_name}",
                       help="Document info",
                       disabled=st.session_state.is_generating):
                if st.session_state.get('show_doc_info') == doc_name:
                    st.session_state.show_doc_info = None
                else:
                    st.session_state.show_doc_info = doc_name
                st.rerun()
        
        with col3:
            # Delete button
            if st.button("🗑️", key=f"delete_{doc_name}", 
                       help="Delete document",
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
        
        # Show info dropdown if toggled
        if st.session_state.get('show_doc_info') == doc_name:
            st.markdown(f"""
            <div style="padding: 0.5rem; background: rgba(102, 126, 234, 0.05); border-radius: 8px; margin-top: 0.5rem;">
            <small><strong>Size:</strong> {doc['size'] / 1024:.1f} KB<br>
            <strong>Type:</strong> {doc['type'].upper()}<br>
            <strong>Chunks:</strong> {doc['chunks']}<br>
            <strong>Uploaded:</strong> {datetime.fromisoformat(doc['uploaded_at']).strftime('%Y-%m-%d %H:%M')}</small>
            </div>
            """, unsafe_allow_html=True)
        
        # Show message count if selected
        if is_selected and doc_name in st.session_state.document_chats:
            msg_count = len(st.session_state.document_chats[doc_name])
            if msg_count > 0:
                st.caption(f"💬 {msg_count} messages")
        
        st.markdown("---")


def render_conversation_history():
    """Render conversation history sidebar"""
    if not st.session_state.conversation_history:
        st.info("💬 No saved conversations yet")
        return
    
    st.subheader(f"💬 History ({len(st.session_state.conversation_history)})")
    
    for conv in st.session_state.conversation_history:
        is_current = st.session_state.selected_conversation == conv['id']
        
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                dt = datetime.fromisoformat(conv['timestamp'])
                time_str = dt.strftime('%m/%d %H:%M')
                
                if st.button(
                    f"{'📘' if is_current else '📄'} {conv['title']}\n{time_str} • {conv['document']}",
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
                    ToastNotification.show("Conversation deleted", "success")
                    st.rerun()
            
            st.markdown("---")


def upload_files(files: List, api_client):
    """Handle file upload"""
    success_count = 0
    uploaded_names = []
    
    # Create progress bar and status text placeholders
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(files):
        # Show current file being processed
        status_text.text(f"Uploading {i + 1}/{len(files)}: {file.name}")
        
        status_code, response = api_client.upload_file(file)
        
        if status_code == 200:
            ToastNotification.show(f"{file.name}: {response.get('chunks', 0)} chunks", "success")
            success_count += 1
            uploaded_names.append(file.name)
        else:
            ToastNotification.show(f"{file.name}: {response.get('message', 'Failed')}", "error")
        
        # Update progress
        progress_bar.progress((i + 1) / len(files))
    
    # Clear status and progress after completion
    status_text.empty()
    progress_bar.empty()
    
    # Show completion message
    if success_count > 0:
        st.success(f"✅ Uploaded {success_count}/{len(files)} file(s) successfully")
    
    if uploaded_names and not st.session_state.selected_document:
        st.session_state.selected_document = uploaded_names[0]
    
    st.session_state.uploader_key += 1
    st.rerun()


def render_model_selector(api_client):
    """Render model selection dropdown"""
    st.subheader("🤖 Model Settings")
    
    models_data = api_client.get_models()
    current_config = models_data.get('current_config', {})
    ollama_models = models_data.get('ollama', {})
    
    llm_models = ollama_models.get('llm_models', [DEFAULT_LLM_MODEL])
    current_model = st.session_state.get('current_model', 
                                         current_config.get('model', DEFAULT_LLM_MODEL))
    
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
    
    if selected_model != st.session_state.get('current_model'):
        st.session_state.current_model = selected_model
        ToastNotification.show(f"Model changed to {selected_model}", "success")
        st.rerun()
    
    embedding_model = current_config.get('embedding_model', 'nomic-embed-text')
    st.caption(f"📊 Embedding: **{embedding_model}**")


def render_sidebar(api_client):
    """Render enhanced sidebar"""
    with st.sidebar:
        # Model selector at the top
        render_model_selector(api_client)
        st.markdown("---")
        
        # Toggle between documents and conversation history
        tab1, tab2 = st.tabs(["📚 Documents", "💬 History"])
        
        with tab1:
            documents = api_client.get_documents()
            
            if documents:
                st.info(f"📊 {len(documents)} document(s) loaded")
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
            
            if uploaded_files:
                if 'last_uploaded_files' not in st.session_state:
                    st.session_state.last_uploaded_files = []
                
                current_file_names = [f.name for f in uploaded_files]
                
                if current_file_names != st.session_state.last_uploaded_files:
                    st.session_state.last_uploaded_files = current_file_names
                    upload_files(uploaded_files, api_client)
        
        with tab2:
            render_conversation_history()
        
        # Clear chat button at the bottom
        if st.session_state.selected_document and get_current_chat():
            st.markdown("---")
            if st.button("🗑️ Clear Chat", use_container_width=True, 
                        disabled=st.session_state.is_generating):
                clear_chat()
                st.rerun()
