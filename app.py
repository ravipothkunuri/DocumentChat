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
# TOAST NOTIFICATIONS
# ============================================================================

def show_toast(message: str, type: str = "info"):
    """Display toast notification independent of sidebar"""
    toast_colors = {
        "success": ("#d4edda", "#155724", "#28a745"),
        "error": ("#f8d7da", "#721c24", "#dc3545"),
        "warning": ("#fff3cd", "#856404", "#ffc107"),
        "info": ("#d1ecf1", "#0c5460", "#17a2b8")
    }
    
    bg_color, text_color, border_color = toast_colors.get(type, toast_colors["info"])
    
    toast_html = f"""
    <div id="toast-notification" style="
        position: fixed;
        top: 20px;
        right: 20px;
        background: {bg_color};
        color: {text_color};
        padding: 16px 24px;
        border-radius: 12px;
        border-left: 4px solid {border_color};
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        z-index: 9999;
        animation: slideInRight 0.3s ease-out, fadeOut 0.3s ease-in 2.7s;
        min-width: 250px;
        max-width: 400px;
        font-weight: 500;
    ">
        {message}
    </div>
    <style>
        @keyframes slideInRight {{
            from {{ transform: translateX(400px); opacity: 0; }}
            to {{ transform: translateX(0); opacity: 1; }}
        }}
        @keyframes fadeOut {{
            from {{ opacity: 1; }}
            to {{ opacity: 0; }}
        }}
    </style>
    <script>
        setTimeout(function() {{
            var toast = document.getElementById('toast-notification');
            if (toast) toast.remove();
        }}, 3000);
    </script>
    """
    st.markdown(toast_html, unsafe_allow_html=True)

# ============================================================================
# UI STYLING
# ============================================================================

def apply_custom_css():
    """Apply custom CSS - Light mode only"""
    st.markdown("""
    <style>
    /* Force light mode */
    [data-theme="dark"] {
        filter: invert(0) !important;
    }
    
    body, .stApp {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    
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
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border-radius: 8px;
        font-weight: 500;
        border: 2px solid transparent;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
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
    
    /* Secondary button */
    .stButton > button[kind="secondary"] {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        color: #495057;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: #e9ecef;
        border-color: #dee2e6;
    }
    
    /* Document card styling */
    .doc-card {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.75rem;
        border: 2px solid rgba(0, 0, 0, 0.1);
        background: rgba(248, 249, 250, 0.5);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    .doc-card:hover {
        background-color: rgba(233, 236, 239, 0.8);
        border-color: rgba(0, 0, 0, 0.15);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
        transform: translateY(-2px);
    }
    
    .doc-card-selected {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.15) 0%, rgba(76, 175, 80, 0.25) 100%);
        border: 2px solid #4caf50;
        box-shadow: 0 4px 16px rgba(76, 175, 80, 0.3);
        transform: translateY(-2px);
    }
    
    .doc-card-selected:hover {
        background: linear-gradient(135deg, rgba(76, 175, 80, 0.25) 0%, rgba(76, 175, 80, 0.35) 100%);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.35);
    }
    
    /* Sidebar - Fixed width for proper button alignment */
    section[data-testid="stSidebar"] {
        width: 320px !important;
        min-width: 320px !important;
        max-width: 320px !important;
        background: #ffffff;
        border-right: 1px solid rgba(0, 0, 0, 0.1);
    }
    
    section[data-testid="stSidebar"] > div {
        width: 320px !important;
        background: #ffffff;
    }
    
    /* Delete button - Perfectly centered with fixed width */
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
        transform: scale(1.08) !important;
    }
    
    /* File uploader styling */
    .stFileUploader {
        border: 2px dashed #cbd5e0;
        border-radius: 12px;
        padding: 1.5rem;
        background: #f7fafc;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: #667eea;
        background: #edf2f7;
    }
    
    /* Chat input styling */
    .stChatInput > div {
        border-radius: 12px;
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .stChatInput > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Alert boxes */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid;
        padding: 1rem 1.25rem;
        animation: slideIn 0.3s ease-out;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(-10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        border-radius: 8px;
        background: rgba(248, 249, 250, 0.8);
        border: 1px solid rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(233, 236, 239, 0.9);
        border-color: rgba(0, 0, 0, 0.2);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Select box */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Divider */
    hr {
        margin: 1.5rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #e9ecef 50%, transparent 100%);
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        animation: messageSlideIn 0.3s ease-out;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    @keyframes messageSlideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
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
    
    /* Caption styling */
    .stCaption {
        color: #6c757d;
        font-size: 0.875rem;
    }
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
        'uploader_key': 0
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
            type="primary" if is_selected else "secondary"
        ):
            st.session_state.selected_document = doc_name
            st.rerun()
    
    with col2:
        if st.button("✕", key=f"delete_{doc_name}", 
                   help="Delete document",
                   type="secondary"):
            status_code, response = api_client.delete_document(doc_name)
            if status_code == 200:
                # Clean up
                if doc_name in st.session_state.document_chats:
                    del st.session_state.document_chats[doc_name]
                if st.session_state.selected_document == doc_name:
                    st.session_state.selected_document = None
                
                show_toast(f"✅ Deleted {doc_name}", "success")
                time.sleep(0.5)
                st.rerun()
            else:
                show_toast(f"❌ {response.get('message', 'Delete failed')}", "error")
    
    if is_selected:
        st.caption(f"📊 {doc['chunks']} chunks • {doc['size']:,} bytes • {doc['type'].upper()}")
        if doc_name in st.session_state.document_chats:
            msg_count = len(st.session_state.document_chats[doc_name])
            if msg_count > 0:
                st.caption(f"💬 {msg_count} messages")
    
    st.markdown("---")

def upload_files(files: List, api_client: RAGAPIClient):
    """Handle file upload"""
    success_count = 0
    total = len(files)
    uploaded_names = []
    
    progress = st.progress(0)
    status = st.empty()
    
    for i, file in enumerate(files):
        status.text(f"⏳ Uploading {file.name}...")
        
        status_code, response = api_client.upload_file(file)
        
        if status_code == 200:
            show_toast(f"✅ {file.name}: {response.get('chunks', 0)} chunks", "success")
            success_count += 1
            uploaded_names.append(file.name)
        else:
            show_toast(f"❌ {file.name}: {response.get('message', 'Failed')}", "error")
        
        progress.progress((i + 1) / total)
    
    status.text(f"✨ Complete: {success_count}/{total}")
    
    # Auto-select first uploaded document
    if uploaded_names and not st.session_state.selected_document:
        st.session_state.selected_document = uploaded_names[0]
    
    st.session_state.uploader_key += 1
    time.sleep(1)
    st.rerun()

# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar(api_client: RAGAPIClient):
    """Render sidebar"""
    with st.sidebar:
        st.markdown('<h1 class="main-header">📚 RAG Assistant</h1>', unsafe_allow_html=True)
        
        documents = api_client.get_documents()
        if documents:
            st.info(f"📊 {len(documents)} document(s) loaded")
        
        st.markdown("---")
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
            key=f"uploader_{st.session_state.uploader_key}"
        )
        
        if uploaded_files:
            if st.button("🚀 Process Files", type="primary", use_container_width=True):
                upload_files(uploaded_files, api_client)
        
        if st.session_state.selected_document and get_current_chat():
            st.markdown("---")
            if st.button("💬 Clear Chat", use_container_width=True):
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
    
    # Check Ollama status
    if health_data:
        ollama = health_data.get('ollama_status', {})
        if not ollama.get('available'):
            show_toast("⚠️ Ollama unavailable", "warning")
    
    # Display chat history
    for msg in get_current_chat():
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # Chat input
    if prompt := st.chat_input(f"💭 Ask about {st.session_state.selected_document}..."):
        handle_query(prompt, api_client, model)

def handle_query(prompt: str, api_client: RAGAPIClient, model: str):
    """Handle chat query"""
    # Add user message
    add_message({
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().isoformat()
    })
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown(
            '<div class="loading-dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>',
            unsafe_allow_html=True
        )
        
        response = ""
        sources = []
        
        try:
            for data in api_client.query_stream(prompt, model=model):
                if data.get('type') == 'metadata':
                    sources = data.get('sources', [])
                elif data.get('type') == 'content':
                    response += data.get('content', '')
                    placeholder.markdown(response + "▌")
                elif data.get('type') == 'done':
                    placeholder.markdown(response)
                elif data.get('type') == 'error':
                    error = f"❌ Error: {data.get('message', 'Unknown')}"
                    placeholder.error(error)
                    response = error
            
            add_message({
                "role": "assistant",
                "content": response,
                "sources": sources,
                "timestamp": datetime.now().isoformat()
            })
        
        except Exception as e:
            error = f"❌ Error: {str(e)}"
            placeholder.error(error)
            add_message({
                "role": "assistant",
                "content": error,
                "sources": [],
                "timestamp": datetime.now().isoformat()
            })

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
    api_client = RAGAPIClient(API_BASE_URL)
    
    health_ok, health_data = api_client.health_check()
    
    if not health_ok:
        st.error("🔴 **Backend Offline**")
        st.markdown("Start the backend:\n```\npython rag_backend.py\n```")
        return
    
    render_sidebar(api_client)
    
    # Header
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
            label_visibility="collapsed"
        )
    
    st.markdown("---")
    render_chat(api_client, health_data, model)

if __name__ == "__main__":
    main()
