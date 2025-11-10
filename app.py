"""DocumentChat Frontend - Interactive Streamlit Interface"""
import json
import asyncio
import random
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, AsyncIterator, Tuple
import httpx
import streamlit as st
from configuration import FALLBACK_BASE_URL, MAX_FILE_SIZE_MB, UI_ALLOWED_EXTENSIONS, DEFAULT_MODEL, THINKING_MESSAGES

API_BASE_URL = os.environ.get("OLLAMA_BASE_URL", FALLBACK_BASE_URL)
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", DEFAULT_MODEL)

# Create persistence directory
CHAT_HISTORY_DIR = Path("chat_history")
CHAT_HISTORY_DIR.mkdir(exist_ok=True)

# API Client
class APIClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.sync_client = httpx.Client(timeout=60.0)
        self.async_client = httpx.AsyncClient(timeout=60.0)

    def _make_request(self, method: str, endpoint: str, timeout: int = 10, **kwargs) -> Tuple[int, Dict]:
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.sync_client.request(method, url, timeout=timeout, **kwargs)
            data = response.json() if response.content else {}
            return response.status_code, data
        except json.JSONDecodeError:
            return response.status_code, {"message": "Invalid JSON response"}
        except httpx.RequestError as e:
            return 500, {"message": f"Connection error: {str(e)}"}

    def health_check(self) -> Tuple[bool, Optional[Dict]]:
        try:
            response = self.sync_client.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200, response.json() if response.is_success else None
        except httpx.RequestError:
            return False, None

    def get_documents(self) -> List[Dict]:
        status_code, data = self._make_request('GET', '/documents')
        return data if status_code == 200 else []

    def upload_file(self, file) -> Tuple[int, Dict]:
        try:
            files = {"file": (file.name, file, file.type)}
            response = self.sync_client.post(f"{self.base_url}/upload", files=files, timeout=60)
            return response.status_code, response.json() if response.content else {}
        except Exception as e:
            return 500, {"message": f"Upload failed: {str(e)}"}

    def delete_document(self, filename: str) -> Tuple[int, Dict]:
        return self._make_request('DELETE', f'/documents/{filename}', timeout=30)

    async def query_stream(self, question: str, top_k: int = 4) -> AsyncIterator[Dict]:
        try:
            payload = {"question": question, "stream": True, "top_k": top_k, "model": DEFAULT_MODEL}
            async with self.async_client.stream('POST', f"{self.base_url}/query", json=payload, timeout=120.0) as response:
                if response.status_code == 200:
                    async for line in response.aiter_lines():
                        if line and line.startswith('data: '):
                            try:
                                yield json.loads(line[6:])
                            except json.JSONDecodeError:
                                continue
                else:
                    yield {"type": "error", "message": f"Query failed with status {response.status_code}"}
        except httpx.ReadTimeout:
            yield {"type": "error", "message": "Request timed out"}
        except Exception as e:
            yield {"type": "error", "message": str(e)}

    def __del__(self):
        try:
            self.sync_client.close()
        except:
            pass

# File-based Persistence Functions
def get_safe_filename(doc_name: str) -> str:
    """Convert document name to safe filename"""
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in doc_name)

def save_chat_history_to_local(doc_name: str, data: List[Dict]):
    """Save chat history to local JSON file"""
    try:
        safe_filename = get_safe_filename(doc_name)
        file_path = CHAT_HISTORY_DIR / f"{safe_filename}_chat.json"

        chat_data = {
            "document": doc_name,
            "last_updated": datetime.now().isoformat(),
            "message_count": len(data),
            "messages": data
        }

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

def load_chat_history_from_local(doc_name: str) -> Optional[List[Dict]]:
    """Load chat history from local JSON file"""
    try:
        safe_filename = get_safe_filename(doc_name)
        file_path = CHAT_HISTORY_DIR / f"{safe_filename}_chat.json"

        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                chat_data = json.load(f)
                return chat_data.get("messages", [])
    except Exception:
        pass
    return None

def delete_chat_history(doc_name: str):
    """Delete chat history file"""
    try:
        safe_filename = get_safe_filename(doc_name)
        file_path = CHAT_HISTORY_DIR / f"{safe_filename}_chat.json"
        if file_path.exists():
            file_path.unlink()
    except Exception:
        pass

# Optimized Export Functions - Reuse Persistence Files
def export_chat_as_json(doc_name: str) -> str:
    """Export chat by reading the persisted JSON file directly"""
    try:
        safe_filename = get_safe_filename(doc_name)
        file_path = CHAT_HISTORY_DIR / f"{safe_filename}_chat.json"

        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            messages = st.session_state.document_chats.get(doc_name, [])
            export_data = {
                "document": doc_name,
                "exported_at": datetime.now().isoformat(),
                "message_count": len(messages),
                "messages": messages
            }
            return json.dumps(export_data, indent=2, ensure_ascii=False)
    except Exception:
        return "{}"

def export_chat_as_markdown(doc_name: str) -> str:
    """Export chat as Markdown by reading from persisted file"""
    try:
        safe_filename = get_safe_filename(doc_name)
        file_path = CHAT_HISTORY_DIR / f"{safe_filename}_chat.json"

        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                chat_data = json.load(f)
                messages = chat_data.get("messages", [])
        else:
            messages = st.session_state.document_chats.get(doc_name, [])

        lines = [f"# Chat Conversation: {doc_name}",
                 f"\n**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                 f"\n**Messages:** {len(messages)}", "\n---\n"]

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
                except:
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

# Notifications
class ToastNotification:
    ICONS = {"success": "✅", "error": "❌", "warning": "⚠️", "info": "ℹ️"}

    @staticmethod
    def show(message: str, toast_type: str = "info") -> None:
        if 'pending_toasts' not in st.session_state:
            st.session_state.pending_toasts = []
        st.session_state.pending_toasts.append({'message': message, 'type': toast_type})

    @staticmethod
    def render_pending() -> None:
        if 'pending_toasts' not in st.session_state or not st.session_state.pending_toasts:
            return
        for toast in st.session_state.pending_toasts:
            icon = ToastNotification.ICONS.get(toast['type'], "ℹ️")
            st.toast(f"{toast['message']}", icon=icon)
        st.session_state.pending_toasts = []

# UI Styling
def apply_custom_css() -> None:
    st.markdown("""
        <style>
        .stChatMessage {border-radius: 10px; padding: 10px; margin-bottom: 10px;}
        button[kind="primary"] {background-color: #4CAF50;}
        .stButton button {transition: all 0.3s ease;}
        .stButton button:hover {transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2);}
        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

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
                st.caption(f"🤖 Using model: **{DEFAULT_MODEL}**")
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

async def process_stream(api_client: APIClient, prompt: str, thinking_placeholder, response_placeholder) -> Tuple[str, bool]:
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

            if data.get('type') == 'content':
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

def render_export_buttons(doc_name: str) -> None:
    col1, col2, _ = st.columns([1.5, 1.5, 7])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_doc_name = doc_name.replace('.pdf', '').replace('.txt', '').replace('.docx', '')
    is_streaming = st.session_state.is_generating

    with col1:
        json_content = export_chat_as_json(doc_name)
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
                thinking_placeholder = st.empty()
                thinking_message = f"*{random.choice(THINKING_MESSAGES)}... *"
                thinking_placeholder.markdown(thinking_message)

                col1, col2 = st.columns([6, 1])
                with col1:
                    response_placeholder = st.empty()
                with col2:
                    if st.button("⏹️", key="stop_inline", help="Stop generation", use_container_width=False):
                        st.session_state.stop_generation = True
                        st.rerun()

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
