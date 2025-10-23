"""
Chat interface components - Claude-style with clean send/stop toggle
"""
import streamlit as st
from datetime import datetime
from typing import Dict
from session_state import get_current_chat, add_message
from toast import ToastNotification


def handle_stop_generation():
    """Callback for stop button"""
    st.session_state.stop_generation = True


def render_custom_chat_input():
    """Render custom chat input with integrated send/stop button"""
    
    # Use columns to create the input layout
    col1, col2 = st.columns([10, 1])
    
    with col1:
        user_input = st.text_area(
            "Message",
            key="custom_chat_input",
            placeholder=f"💭 Ask about {st.session_state.selected_document}...",
            label_visibility="collapsed",
            height=80
        )
    
    with col2:
        if st.session_state.is_generating:
            # Show stop button during generation
            if st.button(
                "⬛",
                key="stop_btn_inline",
                help="Stop generation",
                use_container_width=True
            ):
                handle_stop_generation()
                st.rerun()
        else:
            # Show send button normally
            if st.button(
                "➤",
                key="send_btn_inline",
                help="Send message (Ctrl+Enter)",
                use_container_width=True
            ):
                if user_input and user_input.strip():
                    return user_input.strip()
    
    return None


def render_chat(api_client, health_data: Dict, model: str):
    """Render chat interface"""
    
    if health_data and health_data.get('document_count', 0) == 0:
        st.info("👋 **Welcome!** Upload documents to start.")
        with st.expander("📖 Quick Start", expanded=True):
            st.markdown("""
            1. **Upload** 📤 - Add PDF, TXT, or DOCX files
            2. **Select** 💬 - Click any document
            3. **Ask** 💭 - Type your question
            4. **Get Answers** 🎯 - AI-powered responses
            """)
        return
    
    if not st.session_state.selected_document:
        st.warning("📄 **Select a document** to start.")
        return
    
    if health_data:
        ollama = health_data.get('ollama_status', {})
        if not ollama.get('available'):
            ToastNotification.show("Ollama unavailable", "warning")
    
    # Display chat history
    for msg in get_current_chat():
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("stopped"):
                st.caption("⚠️ Generation was stopped")
    
    # Process pending query if exists
    if st.session_state.pending_query and st.session_state.is_generating:
        prompt = st.session_state.pending_query
        model_to_use = st.session_state.pending_model or model
        
        # Generate response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_placeholder.markdown(
                '<div class="loading-dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>',
                unsafe_allow_html=True
            )
            
            response = ""
            sources = []
            stopped = False
            error_occurred = False
            
            try:
                for data in api_client.query_stream(prompt, model=model_to_use):
                    if st.session_state.stop_generation:
                        stopped = True
                        if response:
                            response += "\n\n*[Generation stopped by user]*"
                        else:
                            response = "*[Generation stopped before content was generated]*"
                        response_placeholder.markdown(response)
                        break
                    
                    if data.get('type') == 'metadata':
                        sources = data.get('sources', [])
                    elif data.get('type') == 'content':
                        response += data.get('content', '')
                        response_placeholder.markdown(response + "▌")
                    elif data.get('type') == 'done':
                        response_placeholder.markdown(response)
                    elif data.get('type') == 'error':
                        error_msg = data.get('message', 'Unknown error')
                        error = f"❌ Error: {error_msg}"
                        response_placeholder.error(error)
                        response = error
                        error_occurred = True
                        break
                
                if response and not error_occurred:
                    response_placeholder.markdown(response)
                
                add_message({
                    "role": "assistant",
                    "content": response if response else "*[No response generated]*",
                    "sources": sources,
                    "timestamp": datetime.now().isoformat(),
                    "stopped": stopped
                })
                
                if stopped:
                    ToastNotification.show("Generation stopped", "warning")
            
            except Exception as e:
                error = f"❌ Error: {str(e)}"
                response_placeholder.error(error)
                add_message({
                    "role": "assistant",
                    "content": error,
                    "sources": [],
                    "timestamp": datetime.now().isoformat()
                })
                ToastNotification.show(f"Error: {str(e)}", "error")
            
            finally:
                st.session_state.pending_query = None
                st.session_state.pending_model = None
                st.session_state.is_generating = False
                st.session_state.stop_generation = False
                # Clear the text area
                if 'custom_chat_input' in st.session_state:
                    st.session_state.custom_chat_input = ""
                st.rerun()
    
    # Custom chat input with integrated send/stop button
    st.markdown("---")
    
    # Status indicator (only when generating)
    if st.session_state.is_generating:
        st.markdown(
            '<p style="text-align: center; color: #ef4444; font-size: 0.875rem; margin-bottom: 0.5rem;">'
            '<span style="display: inline-block; width: 8px; height: 8px; background: #ef4444; border-radius: 50%; margin-right: 6px; animation: pulse 2s infinite;"></span>'
            '<strong>Generating...</strong> Click ⬛ to stop'
            '</p>', 
            unsafe_allow_html=True
        )
    
    prompt = render_custom_chat_input()
    
    if prompt:
        # Add user message
        add_message({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })
        
        # Start generation
        st.session_state.pending_query = prompt
        st.session_state.pending_model = model
        st.session_state.is_generating = True
        st.session_state.stop_generation = False
        st.rerun()
