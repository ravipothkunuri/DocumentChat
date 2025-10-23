"""
Concise onboarding experience for first-time users
"""
import streamlit as st


def render_onboarding():
    """Render streamlined onboarding"""
    
    # Determine the message based on context
    is_first_time = st.session_state.get('show_onboarding', True)
    has_selected = st.session_state.selected_document is not None
    
    with st.container():
        st.markdown("""
        <div style="
            padding: 1.5rem;
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            border-radius: 12px;
            border: 2px solid rgba(102, 126, 234, 0.3);
            margin-bottom: 1.5rem;
        ">
        """, unsafe_allow_html=True)
        
        if is_first_time:
            st.markdown("# 👋 Welcome to RAG Chat")
            st.markdown("Chat with your documents using AI")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("""
                **📤 Upload**  
                Add PDFs, TXT, or DOCX files via sidebar (max 20MB)
                """)
            
            with col2:
                st.markdown("""
                **💬 Chat**  
                Ask questions about your documents naturally
                """)
            
            with col3:
                st.markdown("""
                **📚 Sources**  
                See which document sections were used
                """)
            
            st.markdown("---")
            
            col_btn1, col_btn2 = st.columns([3, 1])
            
            with col_btn1:
                if st.button("🚀 Get Started", type="primary", use_container_width=True):
                    st.session_state.show_onboarding = False
                    st.rerun()
            
            with col_btn2:
                if st.button("✕", help="Close", use_container_width=True):
                    st.session_state.show_onboarding = False
                    st.rerun()
        else:
            # Show when user has documents but none selected
            st.markdown("### 📄 Select a Document to Start Chatting")
            st.markdown("Choose a document from the sidebar to begin asking questions.")
        
        st.markdown("</div>", unsafe_allow_html=True)


def render_quick_start_card():
    """Compact quick-start card for users with no documents"""
    st.info("""
    **🚀 Quick Start:**  
    1. Upload documents using the sidebar  
    2. Click a document to select it  
    3. Ask questions in the chat below
    """)
