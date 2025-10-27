"""Sidebar components - FIXED with file upload"""
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
    """Render sidebar with file upload"""
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
        
        # FIXED: Added the actual file uploader
        uploaded_files = st.file_uploader(
            "Choose files",
            type=ALLOWED_EXTENSIONS,
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
            disabled=st.session_state.is_generating,
            label_visibility="collapsed"
        )
        
        # Handle file uploads
        if uploaded_files:
            # Check if these are new files (not already processed)
            current_file_names = [f.name for f in uploaded_files]
            
            if current_file_names != st.session_state.last_uploaded_files:
                st.session_state.last_uploaded_files = current_file_names
                
                for uploaded_file in uploaded_files:
                    # Check file size
                    file_size_mb = uploaded_file.size / (1024 * 1024)
                    if file_size_mb > MAX_FILE_SIZE_MB:
                        ToastNotification.show(
                            f"❌ {uploaded_file.name} exceeds {MAX_FILE_SIZE_MB}MB limit",
                            "error"
                        )
                        continue
                    
                    # Upload file
                    with st.spinner(f"Uploading {uploaded_file.name}..."):
                        status_code, response = api_client.upload_file(uploaded_file)
                        
                        if status_code == 200:
                            ToastNotification.show(
                                f"✅ {uploaded_file.name} uploaded successfully",
                                "success"
                            )
                            # Auto-select the newly uploaded document
                            st.session_state.selected_document = uploaded_file.name
                        else:
                            error_msg = response.get('message', 'Upload failed')
                            ToastNotification.show(f"❌ {error_msg}", "error")
                
                # Increment uploader key to reset the widget
                st.session_state.uploader_key += 1
                st.rerun()
        
        # Display file requirements
        st.caption(f"📝 Formats: {', '.join(ALLOWED_EXTENSIONS).upper()}")
        st.caption(f"📏 Max size: {MAX_FILE_SIZE_MB}MB per file")
        
        # Clear chat button
        if st.session_state.selected_document and get_current_chat():
            st.markdown("---")
            if st.button("💬 Clear Chat", use_container_width=True, disabled=st.session_state.is_generating):
                clear_chat()
                st.rerun()
