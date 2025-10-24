"""
Clean and modern sidebar with document management
"""
import streamlit as st
from typing import List, Dict
from toast import ToastNotification
from session_state import get_current_chat, clear_chat
from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, DEFAULT_LLM_MODEL
from datetime import datetime
from document_modal import show_document_overview
from conversation_service import ConversationService


def render_document_card(doc: Dict, api_client):
    """Render document card with expander and vertical buttons"""
    doc_name = doc['filename']
    is_selected = st.session_state.selected_document == doc_name
    
    # Create expander for document - auto-expands when selected
    with st.expander(
        f"{'✅' if is_selected else '📄'} {doc_name}",
        expanded=is_selected
    ):
        # Document info
        st.caption(f"{doc['chunks']} chunks • {doc['size'] / 1024:.1f} KB")
        
        # Select/Open Chat button
        if st.button(
            "💬 Open Chat" if not is_selected else "💬 Chat Active",
            key=f"select_{doc_name}",
            use_container_width=True,
            type="primary" if is_selected else "secondary",
            disabled=st.session_state.is_generating or is_selected
        ):
            st.session_state.selected_document = doc_name
            ToastNotification.show(f"Selected: {doc_name}", "success")
            st.rerun()
        
        # Overview button (vertical)
        if st.button("📊 Overview", key=f"overview_{doc_name}", 
                   use_container_width=True,
                   disabled=st.session_state.is_generating):
            status_code, details = api_client.get_document_details(doc_name)
            if status_code == 200:
                show_document_overview(details)
            else:
                ToastNotification.show("Failed to load details", "error")
        
        # Delete button (vertical)
        if st.button("🗑️ Delete", key=f"delete_{doc_name}", 
                   use_container_width=True,
                   disabled=st.session_state.is_generating):
            status_code, response = api_client.delete_document(doc_name)
            if status_code == 200:
                if doc_name in st.session_state.document_chats:
                    del st.session_state.document_chats[doc_name]
                ConversationService.delete_conversation(doc_name)
                if st.session_state.selected_document == doc_name:
                    st.session_state.selected_document = None
                
                ToastNotification.show(f"Deleted {doc_name}", "success")
                st.rerun()
            else:
                ToastNotification.show(f"{response.get('message', 'Failed')}", "error")


def upload_files(files: List, api_client):
    """Handle file upload with validation"""
    success_count = 0
    uploaded_names = []
    skipped_count = 0
    total_files = len(files)
    
    with st.status(f"Uploading {total_files} file(s)...", expanded=True) as status:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(files):
            # Show starting progress for this file
            base_progress = (i / total_files)
            file_progress_range = 1 / total_files
            
            status_text.write(f"📤 Processing **{file.name}** ({i + 1}/{total_files})")
            progress_bar.progress(base_progress)
            
            # Check for empty files
            file.seek(0, 2)  # Seek to end
            file_size = file.tell()
            file.seek(0)  # Reset to beginning
            
            if file_size == 0:
                status_text.write(f"⚠️ Skipped {file.name} (empty file)")
                ToastNotification.show(f"✗ {file.name}: File is empty", "error")
                skipped_count += 1
                progress_bar.progress(base_progress + file_progress_range)
                continue
            
            # Show upload in progress (25% of file's progress range)
            progress_bar.progress(base_progress + file_progress_range * 0.25)
            status_text.write(f"⏫ Uploading **{file.name}**...")
            
            # Show processing (50% of file's progress range)
            progress_bar.progress(base_progress + file_progress_range * 0.5)
            
            status_code, response = api_client.upload_file(file)
            
            # Show finalizing (75% of file's progress range)
            progress_bar.progress(base_progress + file_progress_range * 0.75)
            
            if status_code == 200:
                status_text.write(f"✅ {file.name} uploaded successfully")
                ToastNotification.show(f"✓ {file.name}", "success")
                success_count += 1
                uploaded_names.append(file.name)
            else:
                error_msg = response.get('message', 'Failed')
                status_text.write(f"❌ {file.name}: {error_msg}")
                ToastNotification.show(f"✗ {file.name}: {error_msg}", "error")
            
            # Complete this file's progress
            progress_bar.progress(base_progress + file_progress_range)
        
        # Complete progress
        progress_bar.progress(1.0)
        
        # Update status
        if success_count == total_files:
            status.update(label=f"✅ All {total_files} file(s) uploaded!", state="complete", expanded=False)
        elif success_count > 0:
            status.update(label=f"⚠️ Uploaded {success_count}/{total_files} file(s)", state="complete", expanded=False)
        else:
            status.update(label=f"❌ Upload failed", state="error", expanded=True)
    
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
    """Render clean sidebar with separate sections"""
    with st.sidebar:
        render_model_selector(api_client)
        st.markdown("---")
        
        # Documents Section
        st.subheader("📚 Documents")
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
        
        # Clear Chat button
        if st.session_state.selected_document and get_current_chat():
            st.markdown("---")
            if st.button("🗑️ Clear Chat", use_container_width=True, 
                        disabled=st.session_state.is_generating):
                clear_chat()
                ToastNotification.show("Chat cleared", "success")
                st.rerun()
