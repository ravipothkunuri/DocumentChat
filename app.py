import streamlit as st
import requests
import json
import time
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MAX_FILE_SIZE_MB = 20
ALLOWED_EXTENSIONS = ['pdf', 'txt', 'docx']

# ============================================================================
# API CLIENT
# ============================================================================

class RAGAPIClient:
    """Centralized API client for all backend interactions"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
    
    def _request(self, method: str, endpoint: str, timeout: int = 10, **kwargs) -> Tuple[int, Dict]:
        """Unified request handler with error handling."""
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.session.request(method, url, timeout=timeout, **kwargs)
            data = response.json() if response.content else {}
            return response.status_code, data
        except json.JSONDecodeError:
            return response.status_code, {"message": "Invalid response from server"}
        except requests.exceptions.RequestException as e:
            return 500, {"message": f"Connection error: {str(e)}"}
    
    def health_check(self) -> Tuple[bool, Optional[Dict]]:
        """Check backend health"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200, response.json() if response.ok else None
        except requests.exceptions.RequestException:
            return False, None
    
    def get_models(self) -> Dict:
        """Get available models"""
        status_code, data = self._request('GET', '/models')
        return data if status_code == 200 else {
            "ollama": {
                "llm_models": ["phi3", "llama3", "mistral", "deepseek-r1"],
                "embedding_models": ["nomic-embed-text"]
            },
            "current_config": {"model": "phi3", "embedding_model": "nomic-embed-text"}
        }
    
    def get_documents(self) -> List[Dict]:
        """Get list of uploaded documents"""
        status_code, data = self._request('GET', '/documents')
        return data if status_code == 200 else []
    
    def upload_file(self, file) -> Tuple[int, Dict]:
        """Upload a file to the backend"""
        try:
            files = {"file": (file.name, file, file.type)}
            response = self.session.post(f"{self.base_url}/upload", files=files, timeout=60)
            return response.status_code, response.json() if response.content else {}
        except Exception as e:
            return 500, {"message": f"Upload error: {str(e)}"}
    
    def delete_document(self, filename: str) -> Tuple[int, Dict]:
        """Delete a specific document"""
        return self._request('DELETE', f'/documents/{filename}', timeout=30)
    
    def query_stream(self, question: str, top_k: int = 4, model: str = None):
        """Stream query response"""
        try:
            payload = {"question": question, "stream": True, "top_k": top_k}
            if model:
                payload["model"] = model
            
            response = self.session.post(
                f"{self.base_url}/query",
                json=payload,
                stream=True,
                timeout=60
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        line_text = line.decode('utf-8')
                        if line_text.startswith('data: '):
                            yield json.loads(line_text[6:])
            else:
                yield {"type": "error", "message": f"Query failed with status {response.status_code}"}
        except Exception as e:
            yield {"type": "error", "message": str(e)}

# ============================================================================
# TOAST COMPONENT
# ============================================================================

class ToastNotification:
    """Independent toast notification system using session state flags"""
    
    @staticmethod
    def show(message: str, toast_type: str = "info"):
        """Queue a toast notification to be displayed after rerun"""
        if 'pending_toasts' not in st.session_state:
            st.session_state.pending_toasts = []
        
        st.session_state.pending_toasts.append({
            'message': message,
            'type': toast_type
        })
    
    @staticmethod
    def render_pending():
        """Render all pending toasts and clear the queue"""
        if 'pending_toasts' not in st.session_state or not st.session_state.pending_toasts:
            return
        
        icon_map = {
            "success": "✅",
            "error": "❌",
            "warning": "⚠️",
            "info": "ℹ️"
        }
        
        for toast in st.session_state.pending_toasts:
            icon = icon_map.get(toast['type'], "ℹ️")
            st.toast(f"{toast['message']}", icon=icon)
        
        # Clear the queue
        st.session_state.pending_toasts = []

# ============================================================================
# UI STYLING
# ============================================================================

def apply_custom_css():
    """Apply custom CSS - Respects system theme"""
    st.markdown("""
    <style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 0.5rem 0;
    }
    
    /* Enhanced button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-1px);
    }
    
    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Sidebar - Fixed width */
    section[data-testid="stSidebar"] {
        width: 320px !important;
        min-width: 320px !important;
        max-width: 320px !important;
    }
    
    section[data-testid="stSidebar"] > div {
        width: 320px !important;
    }
    
    /* Delete button - Centered */
    button[key*="delete_"] {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 2px solid rgba(239, 68, 68, 0.5) !important;
        color: #ef4444 !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        padding: 0 !important;
        min-height: 40px !important;
        max-height: 40px !important;
        width: 45px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    button[key*="delete_"]:hover {
        background: rgba(239, 68, 68, 0.2) !important;
        border-color: #ef4444 !important;
        color: #dc2626 !important;
        transform: scale(1.05);
    }
    
    /* Stop button styling */
    button[key="stop_generation_btn"] {
        background: rgba(239, 68, 68, 0.15) !important;
        border: 2px solid #ef4444 !important;
        color: #ef4444 !important;
        font-weight: 600 !important;
        animation: pulse-red 2s ease-in-out infinite;
    }
    
    button[key="stop_generation_btn"]:hover {
        background: rgba(239, 68, 68, 0.25) !important;
        border-color: #dc2626 !important;
        color: #dc2626 !important;
        transform: scale(1.02);
    }
    
    @keyframes pulse-red {
        0%, 100% {
            box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4);
        }
        50% {
            box-shadow: 0 0 0 8px rgba(239, 68, 68, 0);
        }
    }
    
    /* Chat input styling */
    .stChatInput > div {
        border-radius: 12px;
    }
    
    /* Chat input container */
    .chat-input-container {
        position: relative;
        margin-top: 1rem;
    }
    
    /* Loading animation */
    .loading-dots {
        display: inline-flex;
        gap: 8px;
        align-items: center;
        padding: 20px;
    }
    
    .loading-dots .dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        animation: bounce 1.4s infinite ease-in-out both;
    }
    
    .loading-dots .dot:nth-child(1) {
        animation-delay: -0.32s;
    }
    
    .loading-dots .dot:nth-child(2) {
        animation-delay: -0.16s;
    }
    
    @keyframes bounce {
        0%, 80%, 100% {
            transform: scale(0);
            opacity: 0.5;
        }
        40% {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    /* Generation indicator */
    .generation-indicator {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 8px 16px;
        background: rgba(102, 126, 234, 0.1);
        border-radius: 8px;
        margin-bottom: 12px;
        font-size: 0.9rem;
        color: #667eea;
        font-weight: 500;
    }
    
    .generation-indicator .spinner {
        width: 16px;
        height: 16px;
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-top-color: #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Responsive design */
    @media screen and (max-width: 768px) {
        .main-header { 
            font-size: 1.8rem; 
        }
        
        .stButton button { 
            width: 100% !important; 
            min-height: 44px !important; 
        }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================

def init_session_state():
    """Initialize session state variables"""
    defaults = {
        'document_chats': {},
        'selected_document': None,
        'uploader_key': 0,
        'pending_toasts': [],
        'last_uploaded_files': [],
        'is_generating': False,
        'stop_generation': False,
        'current_generation_id': None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_current_chat() -> List[Dict]:
    """Get chat history for selected document"""
    doc = st.session_state.selected_document
    if doc:
        if doc not in st.session_state.document_chats:
            st.session_state.document_chats[doc] = []
        return st.session_state.document_chats[doc]
    return []

def add_message(message: Dict):
    """Add message to current chat"""
    doc = st.session_state.selected_document
    if doc:
        if doc not in st.session_state.document_chats:
            st.session_state.document_chats[doc] = []
        st.session_state.document_chats[doc].append(message)

def clear_chat():
    """Clear current chat history"""
    doc = st.session_state.selected_document
    if doc:
        st.session_state.document_chats[doc] = []

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_document_card(doc: Dict, api_client: RAGAPIClient):
    """Render document card with selection and delete"""
    doc_name = doc['filename']
    is_selected = st.session_state.selected_document == doc_name
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        if st.button(
            f"{'📘' if is_selected else '📄'} **{doc_name}**",
            key=f"select_{doc_name}",
            use_container_width=True,
            type="primary" if is_selected else "secondary",
            disabled=st.session_state.is_generating
        ):
            st.session_state.selected_document = doc_name
            st.rerun()
    
    with col2:
        if st.button("✕", key=f"delete_{doc_name}", 
                   help="Delete document",
                   type="secondary",
                   disabled=st.session_state.is_generating):
            status_code, response = api_client.delete_document(doc_name)
            if status_code == 200:
                if doc_name in st.session_state.document_chats:
                    del st.session_state.document_chats[doc_name]
                if st.session_state.selected_document == doc_name:
                    st.session_state.selected_document = None
                
                ToastNotification.show(f"Deleted {doc_name}", "success")
                st.rerun()
            else:
                ToastNotification.show(f"{response.get('message', 'Delete failed')}", "error")
    
    if is_selected:
        st.caption(f"📊 {doc['chunks']} chunks • {doc['size']:,} bytes • {doc['type'].upper()}")
        if doc_name in st.session_state.document_chats:
            msg_count = len(st.session_state.document_chats[doc_name])
            if msg_count > 0:
                st.caption(f"💬 {msg_count} messages")

def upload_files(files: List, api_client: RAGAPIClient):
    """Handle file upload"""
    success_count = 0
    uploaded_names = []
    progress = st.progress(0)
    
    for i, file in enumerate(files):
        status_code, response = api_client.upload_file(file)
        
        if status_code == 200:
            ToastNotification.show(f"{file.name}: {response.get('chunks', 0)} chunks", "success")
            success_count += 1
            uploaded_names.append(file.name)
        else:
            ToastNotification.show(f"{file.name}: {response.get('message', 'Failed')}", "error")
        
        progress.progress((i + 1) / len(files))
    
    if uploaded_names and not st.session_state.selected_document:
        st.session_state.selected_document = uploaded_names[0]
    
    st.session_state.uploader_key += 1
    st.rerun()

# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar(api_client: RAGAPIClient):
    """Render sidebar"""
    with st.sidebar:        
        documents = api_client.get_documents()
        if documents:
            st.info(f"📊 {len(documents)} document(s) loaded")
        st.subheader("📖 Your Documents")
        
        if documents:
            for doc in documents:
                render_document_card(doc, api_client)
        else:
            st.info("💡 No documents yet. Upload below!")
        
        st.markdown("---")
        st.subheader("📤 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=ALLOWED_EXTENSIONS,
            accept_multiple_files=True,
            help=f"Supported: {', '.join(ALLOWED_EXTENSIONS).upper()} (max {MAX_FILE_SIZE_MB}MB)",
            key=f"uploader_{st.session_state.uploader_key}",
            disabled=st.session_state.is_generating
        )
        
        # Auto-process files when uploaded
        if uploaded_files:
            if 'last_uploaded_files' not in st.session_state:
                st.session_state.last_uploaded_files = []
            
            current_file_names = [f.name for f in uploaded_files]
            
            if current_file_names != st.session_state.last_uploaded_files:
                st.session_state.last_uploaded_files = current_file_names
                upload_files(uploaded_files, api_client)
        
        if st.session_state.selected_document and get_current_chat():
            st.markdown("---")
            if st.button("💬 Clear Chat", use_container_width=True, 
                        disabled=st.session_state.is_generating):
                clear_chat()
                st.rerun()

# ============================================================================
# CHAT INTERFACE
# ============================================================================

def render_chat(api_client: RAGAPIClient, health_data: Dict, model: str):
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
    
    # If generating, show response in real-time
    if st.session_state.is_generating:
        # Get the pending question from the last user message
        chat_history = get_current_chat()
        if chat_history and chat_history[-1]["role"] == "user":
            prompt = chat_history[-1]["content"]
            
            # Show user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate assistant response
            with st.chat_message("assistant"):
                placeholder = st.empty()
                placeholder.markdown(
                    '<div class="loading-dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>',
                    unsafe_allow_html=True
                )
                
                response = ""
                sources = []
                stopped = False
                generation_id = st.session_state.current_generation_id
                
                try:
                    for data in api_client.query_stream(prompt, model=model):
                        # Check for stop signal
                        if st.session_state.stop_generation:
                            stopped = True
                            if response:
                                response += "\n\n*[Generation stopped by user]*"
                            else:
                                response = "*[Generation stopped before content was generated]*"
                            placeholder.markdown(response)
                            break
                        
                        if data.get('type') == 'metadata':
                            sources = data.get('sources', [])
                        elif data.get('type') == 'content':
                            response += data.get('content', '')
                            # Show typing cursor
                            placeholder.markdown(response + "▌")
                        elif data.get('type') == 'done':
                            placeholder.markdown(response)
                        elif data.get('type') == 'error':
                            error = f"❌ Error: {data.get('message', 'Unknown')}"
                            placeholder.error(error)
                            response = error
                    
                    # Final rendering without cursor
                    if response and not stopped:
                        placeholder.markdown(response)
                    
                    # Add message to chat history
                    add_message({
                        "role": "assistant",
                        "content": response if response else "*[No response generated]*",
                        "sources": sources,
                        "timestamp": datetime.now().isoformat(),
                        "stopped": stopped,
                        "generation_id": generation_id
                    })
                    
                    if stopped:
                        ToastNotification.show("Generation stopped successfully", "info")
                
                except Exception as e:
                    error = f"❌ Error: {str(e)}"
                    placeholder.error(error)
                    add_message({
                        "role": "assistant",
                        "content": error,
                        "sources": [],
                        "timestamp": datetime.now().isoformat(),
                        "generation_id": generation_id
                    })
                    ToastNotification.show(f"Error: {str(e)}", "error")
                
                finally:
                    # Reset generation state and rerun to show send button again
                    st.session_state.is_generating = False
                    st.session_state.stop_generation = False
                    st.session_state.current_generation_id = None
                    st.rerun()
    
    # Input area - Show stop button during generation, chat input otherwise
    if st.session_state.is_generating:
        # Generation indicator
        st.markdown(
            '<div class="generation-indicator">'
            '<div class="spinner"></div>'
            '<span>Generating response...</span>'
            '</div>',
            unsafe_allow_html=True
        )
        
        # Stop button replaces chat input
        if st.button(
            "🛑 Stop Generation",
            key="stop_generation_btn",
            use_container_width=True,
            type="secondary"
        ):
            st.session_state.stop_generation = True
            ToastNotification.show("Stopping generation...", "info")
            st.rerun()
    else:
        # Regular chat input (send button)
        if prompt := st.chat_input(
            f"💭 Ask about {st.session_state.selected_document}...",
            key="chat_input"
        ):
            # Start generation
            st.session_state.stop_generation = False
            st.session_state.is_generating = True
            st.session_state.current_generation_id = datetime.now().isoformat()
            handle_query(prompt, api_client, model)

def handle_query(prompt: str, api_client: RAGAPIClient, model: str):
    """Handle chat query with stop capability"""
    generation_id = st.session_state.current_generation_id
    
    # Add user message
    add_message({
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().isoformat()
    })
    
    # Display user message immediately
    st.rerun()

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main application"""
    st.set_page_config(
        page_title="RAG Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="auto"
    )
    
    apply_custom_css()
    init_session_state()
    
    # Render pending toasts at the start of each render
    ToastNotification.render_pending()
    
    api_client = RAGAPIClient(API_BASE_URL)
    
    health_ok, health_data = api_client.health_check()
    
    if not health_ok:
        st.error("🔴 **Backend Offline**")
        st.markdown("Start the backend:\n```\npython rag_backend.py\n```")
        return
    
    render_sidebar(api_client)
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        title = st.session_state.selected_document or "Chat with Documents"
        st.markdown(f'<h2 style="margin-bottom: 0;">💬 {title}</h2>', unsafe_allow_html=True)
    
    with col2:
        models_data = api_client.get_models()
        llm_models = models_data.get('ollama', {}).get('llm_models', ['phi3'])
        current = health_data.get('configuration', {}).get('model', 'phi3') if health_data else 'phi3'
        
        model = st.selectbox(
            "🤖 Model",
            options=llm_models,
            index=llm_models.index(current) if current in llm_models else 0,
            label_visibility="collapsed",
            disabled=st.session_state.is_generating
        )
    
    st.markdown("---")
    render_chat(api_client, health_data, model)

if __name__ == "__main__":
    main()
