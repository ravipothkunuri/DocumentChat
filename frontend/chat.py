"""
Chat interface components - Using Streamlit's native chat input with manual stop control
"""
import streamlit as st
from datetime import datetime
from typing import Dict
from session_state import get_current_chat, add_message
from toast import ToastNotification


def render_chat(api_client, health_data: Dict, model: str):
    """Render chat interface with native Streamlit components"""
    
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
    
    # Use native Streamlit chat input
    prompt = st.chat_input(
        f"💭 Ask about {st.session_state.selected_document}...",
        disabled=st.session_state.is_generating
    )
    
    # Handle new prompt
    if prompt and not st.session_state.is_generating:
        # Add user message
        add_message({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })
        
        # Start generation
        st.session_state.is_generating = True
        st.session_state.stop_generation = False
        st.rerun()
    
    # Process generation if in progress
    if st.session_state.is_generating:
        # Get the last user message
        chat_history = get_current_chat()
        if chat_history and chat_history[-1]["role"] == "user":
            user_prompt = chat_history[-1]["content"]
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                # Create columns for response and stop button
                col1, col2 = st.columns([6, 1])
                
                with col1:
                    response_placeholder = st.empty()
                
                with col2:
                    stop_button_placeholder = st.empty()
                    if stop_button_placeholder.button("⏹️", key="stop_inline", help="Stop generation", use_container_width=True):
                        st.session_state.stop_generation = True
                        st.rerun()
                
                response = ""
                sources = []
                stopped = False
                error_occurred = False
                
                try:
                    for data in api_client.query_stream(user_prompt, model=model):
                        # Check stop flag
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
                    
                    # Clear stop button after generation completes
                    stop_button_placeholder.empty()
                    
                    # Save assistant message
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
                    stop_button_placeholder.empty()  # Clear stop button on error
                    add_message({
                        "role": "assistant",
                        "content": error,
                        "sources": [],
                        "timestamp": datetime.now().isoformat()
                    })
                    ToastNotification.show(f"Error: {str(e)}", "error")
                
                finally:
                    # Always reset state
                    st.session_state.is_generating = False
                    st.session_state.stop_generation = False
                    st.rerun()
