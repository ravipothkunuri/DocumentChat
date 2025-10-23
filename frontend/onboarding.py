"""
Enhanced onboarding experience for first-time users
"""
import streamlit as st
from typing import Dict


def render_onboarding():
    """Render interactive onboarding tour"""
    
    if not st.session_state.get('show_onboarding', True):
        return
    
    # Full-screen onboarding modal
    with st.container():
        st.markdown("""
        <div style="
            padding: 2rem;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            border-radius: 16px;
            border: 2px solid rgba(102, 126, 234, 0.3);
            margin-bottom: 2rem;
        ">
        """, unsafe_allow_html=True)
        
        st.markdown("# 👋 Welcome to RAG Chat!")
        st.markdown("### Your AI-powered document assistant")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            #### 📤 Step 1: Upload
            Upload your documents (PDF, TXT, DOCX) using the sidebar.
            
            Maximum file size: 20MB  
            Multiple files supported
            """)
        
        with col2:
            st.markdown("""
            #### 💬 Step 2: Chat
            Ask questions about your documents naturally.
            
            Use suggested questions  
            Get AI-powered answers
            """)
        
        with col3:
            st.markdown("""
            #### 📥 Step 3: Export
            Save your conversation history for later.
            
            Export as JSON or Markdown  
            Review past conversations
            """)
        
        st.markdown("---")
        
        # Feature highlights
        st.markdown("### ✨ Key Features")
        
        feat_col1, feat_col2, feat_col3, feat_col4 = st.columns(4)
        
        with feat_col1:
            st.markdown("""
            **💡 Smart Suggestions**  
            Get relevant question suggestions based on your documents
            """)
        
        with feat_col2:
            st.markdown("""
            **📚 Source Citations**  
            See exactly which parts of documents were used
            """)
        
        with feat_col3:
            st.markdown("""
            **ℹ️ Document Info**  
            Quick preview of document details and metadata
            """)
        
        with feat_col4:
            st.markdown("""
            **🗂️ History**  
            Access and restore previous conversations
            """)
        
        st.markdown("---")
        
        # Quick tips
        with st.expander("💡 Quick Tips", expanded=True):
            st.markdown("""
            - **Multiple Documents**: Upload multiple files to build a comprehensive knowledge base
            - **Stop Generation**: Click the ⏹️ button to stop a response mid-generation
            - **Model Selection**: Choose different AI models from the sidebar
            - **Export Anytime**: Download your chat history at any point
            - **Conversation History**: Save important chats to revisit later
            """)
        
        # Action buttons
        col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 1])
        
        with col_btn1:
            if st.button("🚀 Get Started", type="primary", use_container_width=True):
                st.session_state.show_onboarding = False
                st.rerun()
        
        with col_btn2:
            if st.button("📖 View Documentation", use_container_width=True):
                st.info("💡 Documentation: Check the sidebar for upload options and model settings")
        
        with col_btn3:
            if st.button("✕", help="Close", use_container_width=True):
                st.session_state.show_onboarding = False
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)


def render_quick_start_card():
    """Render a compact quick-start card for users with no documents"""
    st.markdown("""
    <div style="
        padding: 1.5rem;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        border-radius: 12px;
        border: 2px solid rgba(102, 126, 234, 0.2);
        margin-bottom: 1rem;
    ">
    """, unsafe_allow_html=True)
    
    st.markdown("### 🚀 Quick Start Guide")
    
    steps_col1, steps_col2 = st.columns(2)
    
    with steps_col1:
        st.markdown("""
        **1. Upload Documents** 📤  
        Click the file uploader in the sidebar
        
        **2. Select a Document** 📄  
        Click on any uploaded document to begin
        """)
    
    with steps_col2:
        st.markdown("""
        **3. Ask Questions** 💭  
        Type or use suggested questions
        
        **4. Export & Save** 📥  
        Download or save your conversations
        """)
    
    if st.button("❓ Show Full Tutorial", use_container_width=True):
        st.session_state.show_onboarding = True
        st.rerun()
    
    st.markdown("</div>", unsafe_allow_html=True)


def should_show_onboarding() -> bool:
    """Determine if onboarding should be shown"""
    # Show if explicitly set to True or on first visit
    return st.session_state.get('show_onboarding', True)
