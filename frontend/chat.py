"""
Async Chat Interface with Improved Export Component

Key features:
- Real-time streaming AI responses
- Per-document chat history
- Stop generation anytime
- Compact export controls (no wasted space!)
"""

import streamlit as st
import asyncio
import random
from datetime import datetime
from typing import Dict, Optional
from utils import get_current_chat, add_message, ToastNotification, export_to_json, export_to_markdown

THINKING_MESSAGES = [
    "🤔 Analyzing document...",
    "💭 Thinking...",
    "📖 Reading through content...",
    "🔍 Searching for answers...",
    "⚡ Processing your question...",
    "🧠 Understanding the context...",
]

def render_chat(api_client, health_data: Optional[Dict] = None):
    """Render the chat interface with compact export controls."""
    
    # Show welcome message if no documents
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
    
    # Warn if no document selected
    if not st.session_state.selected_document:
        st.warning("📄 **Select a document** to start chatting.")
        return
    
    # Check Ollama availability
    if health_data:
        ollama = health_data.get('ollama_status', {})
        if not ollama.get('available'):
            ToastNotification.show("Ollama unavailable", "warning")
    
    # ========================================================================
    # EXPORT SECTION - COMPACT LAYOUT (NO WASTED SPACE!)
    # ========================================================================
    
    chat_history = get_current_chat()
    if chat_history:
        # Compact columns: two buttons on left, empty space on right
        col1, col2, _ = st.columns([1.5, 1.5, 7])
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_name = st.session_state.selected_document.replace('.pdf', '').replace('.txt', '').replace('.docx', '')
        is_streaming = st.session_state.is_generating
        with col1:
            # JSON export button
            json_content = export_to_json(chat_history, st.session_state.selected_document)
            json_size = len(json_content.encode('utf-8')) / 1024  # Size in KB
            
            st.download_button(
                label="📄 Export JSON",
                data=json_content,
                file_name=f"{doc_name}_chat_{timestamp}.json",
                mime="application/json",
                use_container_width=True,
                type="secondary",
                disabled=is_streaming,
                help=f"Download {len(chat_history)} messages as JSON ({json_size:.1f} KB)"
            )
        
        with col2:
            # Markdown export button
            md_content = export_to_markdown(chat_history, st.session_state.selected_document)
            md_size = len(md_content.encode('utf-8')) / 1024  # Size in KB
            
            st.download_button(
                label="📝 Export MD",
                data=md_content,
                file_name=f"{doc_name}_chat_{timestamp}.md",
                mime="text/markdown",
                use_container_width=True,
                type="secondary",
                disabled=is_streaming,
                help=f"Download {len(chat_history)} messages as Markdown ({md_size:.1f} KB)"
            )
    
    # ========================================================================
    # DISPLAY CHAT HISTORY
    # ========================================================================
    
    # Don't show the last message if we're currently generating
    messages_to_display = chat_history[:-1] if st.session_state.is_generating else chat_history
    
    for msg in messages_to_display:
        role = msg["role"]
        avatar = "👤" if role == "user" else "🤖"
        
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])
            
            # Show timestamp if available
            timestamp = msg.get("timestamp", "")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    time_str = dt.strftime("%I:%M %p")
                    st.caption(f"🕐 {time_str}")
                except:
                    pass
            
            # Show if generation was stopped
            if msg.get("stopped"):
                st.caption("⚠️ Generation was stopped")
    
    # ========================================================================
    # CHAT INPUT
    # ========================================================================
    
    prompt = st.chat_input(
        f"💭 Ask about {st.session_state.selected_document}...",
        disabled=st.session_state.is_generating
    )
    
    # Handle new user message
    if prompt and not st.session_state.is_generating:
        add_message({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })
        st.session_state.is_generating = True
        st.session_state.stop_generation = False
        st.rerun()
    
    # ========================================================================
    # GENERATE AI RESPONSE
    # ========================================================================
    
    if st.session_state.is_generating:
        chat_history = get_current_chat()
        
        # Make sure we have a user message to respond to
        if chat_history and chat_history[-1]["role"] == "user":
            user_prompt = chat_history[-1]["content"]
            last_msg = chat_history[-1]
            
            # Display the user's message
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
            
            # Generate AI response
            with st.chat_message("assistant", avatar="🤖"):
                # Thinking indicator
                thinking_placeholder = st.empty()
                thinking_message = f"*{random.choice(THINKING_MESSAGES)}*"
                thinking_placeholder.markdown(thinking_message)
                
                # Create columns for response and stop button
                col1, col2 = st.columns([6, 1])
                
                with col1:
                    response_placeholder = st.empty()
                
                with col2:
                    if st.button("⏹️", key="stop_inline", help="Stop generation", use_container_width=True):
                        st.session_state.stop_generation = True
                        st.rerun()
                
                # Run async streaming
                response, stopped = asyncio.run(
                    process_stream(api_client, user_prompt, thinking_placeholder, response_placeholder)
                )
                
                # Save the assistant's response
                add_message({
                    "role": "assistant",
                    "content": response or "*[No response generated]*",
                    "timestamp": datetime.now().isoformat(),
                    "stopped": stopped
                })
                
                if stopped:
                    ToastNotification.show("Generation stopped", "warning")
                
                # Reset generation state
                st.session_state.is_generating = False
                st.session_state.stop_generation = False
                st.rerun()


async def process_stream(api_client, prompt: str, thinking_placeholder, response_placeholder) -> tuple[str, bool]:
    """
    Process the AI response stream asynchronously.
    
    Handles:
    - Real-time token streaming
    - User cancellation
    - Error handling
    
    Returns:
        (response_text, was_stopped)
    """
    response = ""
    stopped = False
    
    try:
        async for data in api_client.query_stream(prompt):
            # Check if user clicked stop
            if st.session_state.stop_generation:
                stopped = True
                thinking_placeholder.empty()
                response += "\n\n*[Interrupted by user]*" if response else "*[Interrupted]*"
                response_placeholder.markdown(response)
                break
            
            # Handle different message types
            if data.get('type') == 'content':
                thinking_placeholder.empty()
                response += data.get('content', '')
                response_placeholder.markdown(response + "▌")  # Blinking cursor effect
                
            elif data.get('type') == 'done':
                response_placeholder.markdown(response)
                
            elif data.get('type') == 'error':
                thinking_placeholder.empty()
                error = f"❌ Error: {data.get('message', 'Unknown error')}"
                response_placeholder.error(error)
                response = error
                break
        
        # Clean up thinking indicator
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
