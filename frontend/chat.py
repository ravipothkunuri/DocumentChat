"""
Enhanced chat interface with export, suggestions, and better citations
"""
import streamlit as st
import random
from datetime import datetime
from typing import Dict
from session_state import (
    get_current_chat, add_message, export_chat_json, 
    export_chat_markdown, get_suggested_questions, save_conversation
)
from toast import ToastNotification

# Random thinking messages
THINKING_MESSAGES = [
    "🤔 Analyzing document...",
    "💭 Thinking...",
    "📖 Reading through content...",
    "🔍 Searching for answers...",
    "⚡ Processing your question...",
    "🧠 Understanding the context...",
    "📚 Consulting the documents...",
    "🔎 Finding relevant information...",
    "💡 Gathering insights...",
    "🎯 Locating the answer...",
    "📝 Reviewing the content...",
    "🌟 Working on it...",
    "⏳ Just a moment...",
    "🚀 Generating response...",
    "🔮 Exploring the knowledge base..."
]


def render_export_options():
    """Render chat export options"""
    chat_history = get_current_chat()
    
    if not chat_history:
        return
    
    with st.expander("📥 Export Chat", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            json_export = export_chat_json(chat_history)
            st.download_button(
                label="📄 Export as JSON",
                data=json_export,
                file_name=f"chat_{st.session_state.selected_document}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col2:
            md_export = export_chat_markdown(chat_history)
            st.download_button(
                label="📝 Export as Markdown",
                data=md_export,
                file_name=f"chat_{st.session_state.selected_document}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                use_container_width=True
            )
        
        # Save to conversation history
        if st.button("💾 Save to History", use_container_width=True):
            save_conversation()
            ToastNotification.show("Conversation saved to history", "success")
            st.rerun()


def render_suggested_questions(api_client, model: str):
    """Render suggested question buttons"""
    chat_history = get_current_chat()
    
    # Only show suggestions if chat is empty or after assistant response
    if len(chat_history) == 0 or (len(chat_history) > 0 and chat_history[-1]['role'] == 'assistant'):
        suggestions = get_suggested_questions(st.session_state.selected_document)
        
        if suggestions:
            st.markdown("#### 💡 Suggested Questions")
            
            # Display in a grid
            cols = st.columns(2)
            for idx, question in enumerate(suggestions[:4]):
                with cols[idx % 2]:
                    if st.button(
                        question,
                        key=f"suggest_{idx}",
                        use_container_width=True,
                        disabled=st.session_state.is_generating
                    ):
                        # Trigger question
                        st.session_state.suggested_question = question
                        st.rerun()


def render_source_citations(sources: list, similarity_scores: list = None):
    """Render enhanced source citations with scores"""
    if not sources:
        return
    
    unique_sources = list(dict.fromkeys(sources))  # Preserve order, remove duplicates
    
    with st.expander(f"📚 Sources ({len(unique_sources)})", expanded=False):
        for idx, source in enumerate(unique_sources):
            # Count how many times this source appears
            count = sources.count(source)
            
            # Get average similarity score for this source
            if similarity_scores:
                source_indices = [i for i, s in enumerate(sources) if s == source]
                avg_score = sum(similarity_scores[i] for i in source_indices) / len(source_indices)
                score_display = f"{avg_score:.2%}"
                
                # Color based on score
                if avg_score >= 0.7:
                    color = "🟢"
                elif avg_score >= 0.5:
                    color = "🟡"
                else:
                    color = "🔴"
                
                st.markdown(f"{color} **{source}** (Relevance: {score_display}, Used: {count}x)")
            else:
                st.markdown(f"📄 **{source}** (Used: {count}x)")


def render_chat(api_client, health_data: Dict, model: str):
    """Render enhanced chat interface"""
    
    # Check for suggested question trigger
    if hasattr(st.session_state, 'suggested_question'):
        prompt = st.session_state.suggested_question
        del st.session_state.suggested_question
        
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
    
    if health_data and health_data.get('document_count', 0) == 0:
        st.info("👋 **Welcome!** Upload documents to start.")
        with st.expander("📖 Quick Start", expanded=True):
            st.markdown("""
            1. **Upload** 📤 - Add PDF, TXT, or DOCX files
            2. **Select** 💬 - Click any document
            3. **Ask** 💭 - Type your question or use suggestions
            4. **Get Answers** 🎯 - AI-powered responses
            5. **Export** 📥 - Save your chat history
            """)
        return
    
    if not st.session_state.selected_document:
        st.warning("📄 **Select a document** to start.")
        return
    
    if health_data:
        ollama = health_data.get('ollama_status', {})
        if not ollama.get('available'):
            ToastNotification.show("Ollama unavailable", "warning")
    
    # Export options at the top
    render_export_options()
    
    # Display chat history
    chat_history = get_current_chat()
    messages_to_display = chat_history[:-1] if st.session_state.is_generating else chat_history
    
    for msg in messages_to_display:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            # Show sources for assistant messages
            if msg["role"] == "assistant" and msg.get("sources"):
                render_source_citations(
                    msg.get("sources", []),
                    msg.get("similarity_scores")
                )
            
            if msg.get("stopped"):
                st.caption("⚠️ Generation was stopped")
    
    # Suggested questions
    if not st.session_state.is_generating:
        render_suggested_questions(api_client, model)
    
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
    
    # Process generation
    if st.session_state.is_generating:
        chat_history = get_current_chat()
        if chat_history and chat_history[-1]["role"] == "user":
            user_prompt = chat_history[-1]["content"]
            
            with st.chat_message("user"):
                st.markdown(user_prompt)
            
            with st.chat_message("assistant"):
                thinking_placeholder = st.empty()
                thinking_message = f"*{random.choice(THINKING_MESSAGES)}*"
                thinking_placeholder.markdown(thinking_message)
                
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
                similarity_scores = []
                stopped = False
                error_occurred = False
                
                try:
                    for data in api_client.query_stream(user_prompt, model=model):
                        if st.session_state.stop_generation:
                            stopped = True
                            thinking_placeholder.empty()
                            if response:
                                response += "\n\n*[Interrupted by user]*"
                            else:
                                response = "*[Interrupted before content was generated]*"
                            response_placeholder.markdown(response)
                            break
                        
                        if data.get('type') == 'metadata':
                            sources = data.get('sources', [])
                            similarity_scores = data.get('similarity_scores', [])
                            thinking_placeholder.empty()
                        elif data.get('type') == 'content':
                            thinking_placeholder.empty()
                            response += data.get('content', '')
                            response_placeholder.markdown(response + "▌")
                        elif data.get('type') == 'done':
                            response_placeholder.markdown(response)
                        elif data.get('type') == 'error':
                            thinking_placeholder.empty()
                            error_msg = data.get('message', 'Unknown error')
                            error = f"❌ Error: {error_msg}"
                            response_placeholder.error(error)
                            response = error
                            error_occurred = True
                            break
                    
                    thinking_placeholder.empty()
                    if response and not error_occurred:
                        response_placeholder.markdown(response)
                    
                    stop_button_placeholder.empty()
                    
                    # Show sources immediately after response
                    if sources and not error_occurred:
                        render_source_citations(sources, similarity_scores)
                    
                    # Save message
                    add_message({
                        "role": "assistant",
                        "content": response if response else "*[No response generated]*",
                        "sources": sources,
                        "similarity_scores": similarity_scores,
                        "timestamp": datetime.now().isoformat(),
                        "stopped": stopped
                    })
                    
                    if stopped:
                        ToastNotification.show("Generation stopped", "warning")
                        
                except Exception as e:
                    thinking_placeholder.empty()
                    error = f"❌ Error: {str(e)}"
                    response_placeholder.error(error)
                    stop_button_placeholder.empty()
                    add_message({
                        "role": "assistant",
                        "content": error,
                        "sources": [],
                        "timestamp": datetime.now().isoformat()
                    })
                    ToastNotification.show(f"Error: {str(e)}", "error")
                
                finally:
                    st.session_state.is_generating = False
                    st.session_state.stop_generation = False
                    st.rerun()
