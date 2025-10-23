"""
Chat interface components
"""
import streamlit as st
from datetime import datetime
from typing import Dict
from session_state import get_current_chat, add_message
from toast import ToastNotification


def handle_stop_generation():
    """Callback for stop button"""
    st.session_state.stop_generation = True
    # Don't clear is_generating here - let the stream handler do it


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
            
            try:
                for data in api_client.query_stream(prompt, model=model_to_use):
                    # Check for stop signal
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
                        error = f"❌ Error: {data.get('message', 'Unknown')}"
                        response_placeholder.error(error)
                        response = error
                
                # Final rendering
                if response:
                    response_placeholder.markdown(response)
                
                # Add to chat history
                add_message({
                    "role": "assistant",
                    "content": response if response else "*[No response generated]*",
                    "sources": sources,
                    "timestamp": datetime.now().isoformat(),
                    "stopped": stopped
                })
            
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
                # Clear pending query and reset state
                st.session_state.pending_query = None
                st.session_state.pending_model = None
                st.session_state.is_generating = False
                st.session_state.stop_generation = False
                st.rerun()
    
    # Chat input area - conditional rendering
    if st.session_state.is_generating:
        # Show stop button
        if st.button(
            "🛑 Stop Generation",
            key="stop_btn",
            use_container_width=True,
            type="secondary",
            on_click=handle_stop_generation
        ):
            pass  # Handler is in on_click
    else:
        # Show chat input (send button)
        if prompt := st.chat_input(
            f"💭 Ask about {st.session_state.selected_document}...",
            key="chat_input"
        ):
            # Store query and start generation
            add_message({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            })
            st.session_state.pending_query = prompt
            st.session_state.pending_model = model
            st.session_state.is_generating = True
            st.session_state.stop_generation = False
            st.rerun()