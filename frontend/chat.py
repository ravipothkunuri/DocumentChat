"""Async chat interface"""
import streamlit as st
import asyncio
import random
from datetime import datetime
from typing import Dict, Optional
from session_state import get_current_chat, add_message
from toast import ToastNotification
from export_utils import export_to_json, export_to_markdown

THINKING_MESSAGES = [
    "🤔 Analyzing document...",
    "💭 Thinking...",
    "📖 Reading through content...",
    "🔍 Searching for answers...",
    "⚡ Processing your question...",
    "🧠 Understanding the context...",
]

def render_chat(api_client, health_data: Optional[Dict] = None):
    """Render async chat interface"""
    
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
    
    # Export dropdown in header
    chat_history = get_current_chat()
    if chat_history:
        col1, col2 = st.columns([4, 1])
        with col2:
            export_format = st.selectbox(
                "Export",
                options=["Select format", "JSON", "Markdown"],
                key="export_format",
                label_visibility="collapsed"
            )
            
            if export_format != "Select format":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                doc_name = st.session_state.selected_document.replace('.pdf', '')
                
                if export_format == "JSON":
                    content = export_to_json(chat_history, st.session_state.selected_document)
                    filename = f"{doc_name}_chat_{timestamp}.json"
                else:  # Markdown
                    content = export_to_markdown(chat_history, st.session_state.selected_document)
                    filename = f"{doc_name}_chat_{timestamp}.md"
                
                # Create download button
                st.download_button(
                    label=f"⬇️ Download {export_format}",
                    data=content,
                    file_name=filename,
                    mime="application/json" if export_format == "JSON" else "text/markdown",
                    key=f"download_{export_format}_{timestamp}",
                    use_container_width=True
                )
    
    # Display chat history
    chat_history = get_current_chat()
    messages_to_display = chat_history[:-1] if st.session_state.is_generating else chat_history
    
    for msg in messages_to_display:
        role = msg["role"]
        avatar = "👤" if role == "user" else "🤖"
        
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])
            
            timestamp = msg.get("timestamp", "")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    time_str = dt.strftime("%I:%M %p")
                    st.caption(f"🕐 {time_str}")
                except:
                    pass
            
            if msg.get("stopped"):
                st.caption("⚠️ Generation was stopped")
    
    # Chat input
    prompt = st.chat_input(
        f"💭 Ask about {st.session_state.selected_document}...",
        disabled=st.session_state.is_generating
    )
    
    # Handle new prompt
    if prompt and not st.session_state.is_generating:
        add_message({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })
        st.session_state.is_generating = True
        st.session_state.stop_generation = False
        st.rerun()
    
    # Process generation asynchronously
    if st.session_state.is_generating:
        chat_history = get_current_chat()
        if chat_history and chat_history[-1]["role"] == "user":
            user_prompt = chat_history[-1]["content"]
            last_msg = chat_history[-1]
            
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_prompt)
                timestamp = last_msg.get("timestamp", "")
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        time_str = dt.strftime("%I:%M %p")
                        st.caption(f"🕐 {time_str}")
                    except:
                        pass
            
            with st.chat_message("assistant", avatar="🤖"):
                thinking_placeholder = st.empty()
                thinking_message = f"*{random.choice(THINKING_MESSAGES)}*"
                thinking_placeholder.markdown(thinking_message)
                
                col1, col2 = st.columns([6, 1])
                
                with col1:
                    response_placeholder = st.empty()
                
                with col2:
                    if st.button("⏹️", key="stop_inline", help="Stop", use_container_width=True):
                        st.session_state.stop_generation = True
                        st.rerun()
                
                # Run async streaming
                response, stopped = asyncio.run(
                    process_stream(api_client, user_prompt, thinking_placeholder, response_placeholder)
                )
                
                add_message({
                    "role": "assistant",
                    "content": response or "*[No response generated]*",
                    "timestamp": datetime.now().isoformat(),
                    "stopped": stopped
                })
                
                if stopped:
                    ToastNotification.show("Generation stopped", "warning")
                
                st.session_state.is_generating = False
                st.session_state.stop_generation = False
                st.rerun()

async def process_stream(api_client, prompt: str, thinking_placeholder, response_placeholder) -> tuple[str, bool]:
    """Process async stream with cancellation support"""
    response = ""
    stopped = False
    
    try:
        async for data in api_client.query_stream(prompt):
            if st.session_state.stop_generation:
                stopped = True
                thinking_placeholder.empty()
                response += "\n\n*[Interrupted by user]*" if response else "*[Interrupted]*"
                response_placeholder.markdown(response)
                break
            
            if data.get('type') == 'metadata':
                thinking_placeholder.empty()
            elif data.get('type') == 'content':
                thinking_placeholder.empty()
                response += data.get('content', '')
                response_placeholder.markdown(response + "▌")
            elif data.get('type') == 'done':
                response_placeholder.markdown(response)
            elif data.get('type') == 'error':
                thinking_placeholder.empty()
                error = f"❌ Error: {data.get('message', 'Unknown error')}"
                response_placeholder.error(error)
                response = error
                break
        
        thinking_placeholder.empty()
        if response:
            response_placeholder.markdown(response)
            
    except Exception as e:
        thinking_placeholder.empty()
        error = f"❌ Error: {str(e)}"
        response_placeholder.error(error)
        response = error
        ToastNotification.show(f"Error: {str(e)}", "error")
    
    return response, stopped
