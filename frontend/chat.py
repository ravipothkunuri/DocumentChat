"""
Chat interface components - Improved stop functionality
"""
import streamlit as st
from datetime import datetime
from typing import Dict
from session_state import get_current_chat, add_message
from toast import ToastNotification


def handle_stop_generation():
    """Callback for stop button - sets flag to interrupt generation"""
    st.session_state.stop_generation = True


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
        
        # Generate response (user message already in chat history)
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
                    # Check for stop signal FIRST before processing data
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
                        # Show cursor while generating
                        response_placeholder.markdown(response + "▌")
                    elif data.get('type') == 'done':
                        # Remove cursor on completion
                        response_placeholder.markdown(response)
                    elif data.get('type') == 'error':
                        error_msg = data.get('message', 'Unknown error')
                        error = f"❌ Error: {error_msg}"
                        response_placeholder.error(error)
                        response = error
                        error_occurred = True
                        break
                
                # Final rendering (if not already done)
                if response and not error_occurred:
                    response_placeholder.markdown(response)
                
                # Add assistant response to chat history
                add_message({
                    "role": "assistant",
                    "content": response if response else "*[No response generated]*",
                    "sources": sources,
                    "timestamp": datetime.now().isoformat(),
                    "stopped": stopped
                })
                
                # Show toast notification if stopped
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
                # CRITICAL: Clean up all state flags
                st.session_state.pending_query = None
                st.session_state.pending_model = None
                st.session_state.is_generating = False
                st.session_state.stop_generation = False
                # Rerun to update UI (show chat input again)
                st.rerun()
    
    # Chat input area - conditional rendering based on generation state
    if st.session_state.is_generating:
        # Show stop button when generating with custom styling
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.button(
                "🛑 Stop Generation",
                key="stop_btn",
                use_container_width=True,
                type="secondary",
                on_click=handle_stop_generation,
                help="Click to stop the current response"
            )
    else:
        # Show chat input when not generating
        if prompt := st.chat_input(
            f"💭 Ask about {st.session_state.selected_document}...",
            key="chat_input"
        ):
            # Add user message to chat history
            add_message({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            })
            
            # Store query and initiate generation
            st.session_state.pending_query = prompt
            st.session_state.pending_model = model
            st.session_state.is_generating = True
            st.session_state.stop_generation = False
            
            # Rerun to trigger generation and show stop button
            st.rerun()
