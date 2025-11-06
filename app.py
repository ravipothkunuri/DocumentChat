"""DocumentChat Frontend - Interactive Streamlit Interface"""
import json
import asyncio
import random
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import streamlit as st
from configuration import API_BASE_URL, MAX_FILE_SIZE_MB, UI_ALLOWED_EXTENSIONS, LLM_MODEL, THINKING_MESSAGES
from frontend.api_client import APIClient
from frontend.persistence import (
    save_chat_history_to_local,
    load_chat_history_from_local,
    delete_chat_history,
    export_chat_as_json,
)
from frontend.notifications import ToastNotification
from frontend.styling import apply_custom_css

## Persistence helpers imported from frontend.persistence

def export_chat_as_markdown(doc_name: str) -> str:
    try:
        messages = load_chat_history_from_local(doc_name) or st.session_state.document_chats.get(doc_name, [])

        lines = [
            f"# Chat Conversation: {doc_name}",
            f"\n**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n**Messages:** {len(messages)}",
            "\n---\n",
        ]

        for i, msg in enumerate(messages, 1):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")
            stopped = msg.get("stopped", False)

            role_display = "👤 **User**" if role == "user" else "🤖 **Assistant**"
            lines.append(f"\n## Message {i}: {role_display}\n")

            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    lines.append(f"*Time: {dt.strftime('%I:%M %p')}*\n")
                except Exception:
                    pass

            lines.append(f"\n{content}\n")
            if stopped:
                lines.append("\n*⚠️ Generation was stopped by user*\n")
            lines.append("\n---\n")

        return "".join(lines)
    except Exception:
        return "# Export Error\nCould not export chat history."

# Session State
def init_session_state() -> None:
    defaults = {
        'document_chats': {},
        'selected_document': None,
        'uploader_key': 0,
        'pending_toasts': [],
        'last_uploaded_files': [],
        'is_generating': False,
        'stop_generation': False,
        'persistence_loaded': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_current_chat() -> List[Dict]:
    doc = st.session_state.selected_document
    if doc:
        if doc not in st.session_state.document_chats:
            saved_chat = load_chat_history_from_local(doc)
            if saved_chat:
                st.session_state.document_chats[doc] = saved_chat
            else:
                st.session_state.document_chats[doc] = []
    return st.session_state.document_chats.get(doc, [])

def add_message(message: Dict) -> None:
    if doc := st.session_state.selected_document:
        st.session_state.document_chats.setdefault(doc, []).append(message)
        save_chat_history_to_local(doc, st.session_state.document_chats[doc])

def clear_chat() -> None:
    if doc := st.session_state.selected_document:
        st.session_state.document_chats[doc] = []
        save_chat_history_to_local(doc, [])

def get_ui_state() -> Dict[str, bool]:
    return {"disabled": st.session_state.is_generating}

## Notifications now imported from frontend.notifications

## Styling now imported from frontend.styling

# UI Components
def render_document_card(doc: Dict, api_client: APIClient) -> None:
    doc_name = doc['filename']
    is_selected = st.session_state.selected_document == doc_name
    col1, col2 = st.columns([6, 1])

    with col1:
        if st.button(f"{'📘' if is_selected else '📄'} **{doc_name}**", key=f"select_{doc_name}",
                     use_container_width=True, type="primary" if is_selected else "secondary", **get_ui_state()):
            st.session_state.selected_document = doc_name
            saved_chat = load_chat_history_from_local(doc_name)
            if saved_chat:
                st.session_state.document_chats[doc_name] = saved_chat
            st.rerun()

    with col2:
        if st.button("✕", key=f"delete_{doc_name}", help="Delete document", **get_ui_state()):
            status_code, response = api_client.delete_document(doc_name)
            if status_code == 200:
                st.session_state.document_chats.pop(doc_name, None)
                delete_chat_history(doc_name)
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

def render_sidebar(api_client: APIClient) -> None:
    with st.sidebar:
        documents = api_client.get_documents()

        if documents:
            st.info(f"📊 {len(documents)} document(s) loaded")
            st.subheader("📖 Your Documents")
            for doc in documents:
                render_document_card(doc, api_client)
            if documents:
                st.caption(f"🤖 Using model: **{LLM_MODEL}**")
        else:
            st.info("💡 No documents yet. Upload below!")

        st.markdown("---")
        st.subheader("📤 Upload Documents")

        uploaded_files = st.file_uploader("Choose files", type=UI_ALLOWED_EXTENSIONS, accept_multiple_files=True,
                                         key=f"uploader_{st.session_state.uploader_key}", 
                                         label_visibility="collapsed", **get_ui_state())

        if uploaded_files:
            current_file_names = [f.name for f in uploaded_files]
            if current_file_names != st.session_state.last_uploaded_files:
                st.session_state.last_uploaded_files = current_file_names
                for uploaded_file in uploaded_files:
                    file_size_mb = uploaded_file.size / (1024 * 1024)
                    if file_size_mb > MAX_FILE_SIZE_MB:
                        ToastNotification.show(f"{uploaded_file.name} exceeds {MAX_FILE_SIZE_MB}MB limit", "error")
                        continue

                    with st.spinner(f"Uploading {uploaded_file.name}..."):
                        status_code, response = api_client.upload_file(uploaded_file)
                        if status_code == 200:
                            ToastNotification.show(f"{uploaded_file.name} uploaded successfully", "success")
                            st.session_state.selected_document = uploaded_file.name
                        else:
                            ToastNotification.show(f"{response.get('message', 'Upload failed')}", "error")
                st.session_state.uploader_key += 1
                st.rerun()

        with st.expander("ℹ️ Upload Requirements", expanded=False):
            st.caption(f"**Formats:** {', '.join(UI_ALLOWED_EXTENSIONS).upper()}")
            st.caption(f"**Max size:** {MAX_FILE_SIZE_MB} MB per file")
            st.caption(f"**Multiple files:** Supported")

        if st.session_state.selected_document and get_current_chat():
            st.markdown("---")
            if st.button("💬 Clear Chat", use_container_width=True, **get_ui_state()):
                clear_chat()
                st.rerun()

async def process_stream(api_client: APIClient, prompt: str, response_placeholder) -> Tuple[str, bool]:
    response = ""
    stopped = False
    try:
        response_placeholder.markdown(f"*{random.choice(THINKING_MESSAGES)}... *")
        async for data in api_client.query_stream(prompt, model=LLM_MODEL):
            if st.session_state.stop_generation:
                stopped = True
                response += "\n\n*[Interrupted by user]*" if response else "*[Interrupted]*"
                response_placeholder.markdown(response)
                break

            if data.get('type') == 'content':
                response += data.get('content', '')
                response_placeholder.markdown(response + "▌")
            elif data.get('type') == 'done':
                response_placeholder.markdown(response)
            elif data.get('type') == 'error':
                error = f"❌ Error: {data.get('message', 'Unknown error')}"
                response_placeholder.error(error)
                response = error
                break

        if response:
            response_placeholder.markdown(response)
    except Exception as e:
        error = f"❌ Error: {str(e)}"
        response_placeholder.error(error)
        response = error
        ToastNotification.show(f"Error: {str(e)}", "error")

    return response, stopped

def render_export_buttons(doc_name: str) -> None:
    col1, col2, _ = st.columns([1.5, 1.5, 7])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_doc_name = doc_name.replace('.pdf', '').replace('.txt', '').replace('.docx', '')
    is_streaming = st.session_state.is_generating

    with col1:
        json_content = export_chat_as_json(doc_name, st.session_state.document_chats.get(doc_name, []))
        json_size = len(json_content.encode('utf-8')) / 1024
        st.download_button(
            label="📄 Export JSON",
            data=json_content,
            file_name=f"{clean_doc_name}_chat_{timestamp}.json",
            mime="application/json",
            use_container_width=True,
            type="secondary",
            disabled=is_streaming,
            help=f"Download chat as JSON ({json_size:.1f} KB)"
        )

    with col2:
        md_content = export_chat_as_markdown(doc_name)
        md_size = len(md_content.encode('utf-8')) / 1024
        st.download_button(
            label="📝 Export MD",
            data=md_content,
            file_name=f"{clean_doc_name}_chat_{timestamp}.md",
            mime="text/markdown",
            use_container_width=True,
            type="secondary",
            disabled=is_streaming,
            help=f"Download chat as Markdown ({md_size:.1f} KB)"
        )

def render_chat_history(messages: List[Dict]) -> None:
    for msg in messages:
        role = msg["role"]
        avatar = "👤" if role == "user" else "🤖"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg["content"])
            if timestamp := msg.get("timestamp", ""):
                try:
                    dt = datetime.fromisoformat(timestamp)
                    st.caption(f"🕒 {dt.strftime('%I:%M %p')}")
                except:
                    pass
            if msg.get("stopped"):
                st.caption("⚠️ Generation was stopped")

def render_chat(api_client: APIClient, health_data: Optional[Dict] = None) -> None:
    if health_data and health_data.get('document_count', 0) == 0:
        st.info("👋 **Welcome!** Upload documents to start chatting.")
        with st.expander("📖 Quick Start Guide", expanded=True):
            st.markdown("""
                1. **Upload** 📤 - Add PDF, TXT, or DOCX files using the sidebar
                2. **Select** 💬 - Click on any uploaded document to open it
                3. **Ask** 💭 - Type your questions in the chat input
                4. **Get Answers** 🎯 - Receive AI-powered responses based on your documents
            """)
        return

    if not st.session_state.selected_document:
        st.warning("📄 **Select a document** from the sidebar to start chatting.")
        return

    if health_data:
        ollama = health_data.get('ollama_status', {})
        if not ollama.get('available'):
            ToastNotification.show("Ollama service unavailable", "warning")

    chat_history = get_current_chat()

    if chat_history:
        render_export_buttons(st.session_state.selected_document)

    messages_to_display = chat_history[:-1] if st.session_state.is_generating else chat_history
    render_chat_history(messages_to_display)

    prompt = st.chat_input(f"💭 Ask about {st.session_state.selected_document}...", **get_ui_state())

    if prompt and not st.session_state.is_generating:
        add_message({"role": "user", "content": prompt, "timestamp": datetime.now().isoformat()})
        st.session_state.is_generating = True
        st.session_state.stop_generation = False
        st.rerun()

    if st.session_state.is_generating:
        chat_history = get_current_chat()
        if chat_history and chat_history[-1]["role"] == "user":
            user_prompt = chat_history[-1]["content"]
            last_msg = chat_history[-1]

            with st.chat_message("user", avatar="👤"):
                st.markdown(user_prompt)
                if timestamp := last_msg.get("timestamp", ""):
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        st.caption(f"🕒 {dt.strftime('%I:%M %p')}")
                    except:
                        pass

            with st.chat_message("assistant", avatar="🤖"):
                col1, col2 = st.columns([6, 1])
                with col1:
                    st.markdown('<div class="response-box">', unsafe_allow_html=True)
                    response_placeholder = st.empty()
                    st.markdown('</div>', unsafe_allow_html=True)
                with col2:
                    st.markdown('<div class="stop-col">', unsafe_allow_html=True)
                    if st.button("⏹️", key="stop_inline", help="Stop generation", use_container_width=False):
                        st.session_state.stop_generation = True
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)

                response, stopped = asyncio.run(
                    process_stream(api_client, user_prompt, response_placeholder)
                )

                add_message({
                    "role": "assistant",
                    "content": response or "*[No response generated]*",
                    "timestamp": datetime.now().isoformat(),
                    "stopped": stopped
                })

                if stopped:
                    ToastNotification.show("Generation stopped by user", "warning")

                st.session_state.is_generating = False
                st.session_state.stop_generation = False
                st.rerun()

# Main Application
def main():
    st.set_page_config(page_title="Chat With Documents using AI", page_icon="📚", 
                      layout="wide", initial_sidebar_state="auto")
    init_session_state()
    apply_custom_css()

    api_client = APIClient(API_BASE_URL)
    is_healthy, health_data = api_client.health_check()

    if not is_healthy:
        st.error("❌ Backend service unavailable. Please start the FastAPI server.")
        st.stop()

    ToastNotification.render_pending()
    st.markdown('<h1 style="text-align: center;">📚 Chat With Documents</h1>', unsafe_allow_html=True)

    render_sidebar(api_client)
    render_chat(api_client, health_data)

if __name__ == "__main__":
    main()
