import streamlit as st
import requests
import json
import time
from typing import List, Dict, Any, Optional, Tuple
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
    
    def configure_system(self, config: Dict) -> Tuple[int, Dict]:
        """Update system configuration"""
        return self._request('POST', '/configure', json=config, timeout=30)
    
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
# UI STYLING
# ============================================================================

def apply_custom_css():
    """Apply custom CSS for better UI"""
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
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .status-success { background-color: #d4edda; color: #155724; }
    .status-error { background-color: #f8d7da; color: #721c24; }
    .status-info { background-color: #d1ecf1; color: #0c5460; }
    
    /* Dark mode support */
    [data-theme="dark"] {
        --bg-primary: #1e1e1e;
        --bg-secondary: #2d2d2d;
        --text-primary: #e0e0e0;
        --border-color: #404040;
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
    
    /* Primary button enhancement */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Secondary button enhancement */
    .stButton > button[kind="secondary"] {
        background: #f8f9fa;
        border: 2px solid #e9ecef;
        color: #495057;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: #e9ecef;
        border-color: #dee2e6;
    }
    
    /* Document card styling - dark mode compatible with uniform shadow */
    .doc-card {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.75rem;
        border: 2px solid rgba(0, 0, 0, 0.1);
        background: rgba(255, 255, 255, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    .doc-card:hover {
        background-color: rgba(0, 0, 0, 0.02);
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
    
    /* Delete button styling - perfectly centered */
    button[key*="delete_"] {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 2px solid rgba(239, 68, 68, 0.5) !important;
        color: #ef4444 !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        padding: 0 !important;
        min-height: 40px !important;
        max-height: 40px !important;
        width: 100% !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
    }
    
    button[key*="delete_"] > div {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100% !important;
    }
    
    button[key*="delete_"]:hover {
        background: rgba(239, 68, 68, 0.2) !important;
        border-color: #ef4444 !important;
        color: #dc2626 !important;
        transform: scale(1.08) !important;
    }
    
    /* Remove clear all button */
    button[key="clear_all_btn"] {
        display: none !important;
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
    
    /* Info/Warning/Success boxes */
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
    
    /* Success message enhancement */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left-color: #28a745;
    }
    
    /* Error message enhancement */
    .stError {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left-color: #dc3545;
    }
    
    /* Warning message enhancement */
    .stWarning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffe69c 100%);
        border-left-color: #ffc107;
    }
    
    /* Info message enhancement */
    .stInfo {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        border-left-color: #17a2b8;
    }
    
    /* Expander styling with uniform shadow */
    .streamlit-expanderHeader {
        border-radius: 8px;
        background: rgba(128, 128, 128, 0.05);
        border: 1px solid rgba(128, 128, 128, 0.2);
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(128, 128, 128, 0.1);
        border-color: rgba(128, 128, 128, 0.3);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #e9ecef;
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Sidebar styling - dark mode compatible */
    section[data-testid="stSidebar"] {
        background: transparent;
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }
    
    section[data-testid="stSidebar"] > div {
        background: transparent;
    }
    
    /* Sidebar content styling for better dark mode */
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stButton {
        color: inherit;
    }
    
    /* Info boxes in sidebar - dark mode compatible */
    section[data-testid="stSidebar"] .stAlert {
        background: rgba(102, 126, 234, 0.1);
        border-left-color: #667eea;
        color: inherit;
    }
    
    /* Divider styling */
    hr {
        margin: 1.5rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent 0%, #e9ecef 50%, transparent 100%);
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
        
        div[data-testid="metric-container"] { 
            min-width: 100% !important; 
        }
        
        .doc-card {
            padding: 0.75rem;
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
    
    /* Chat message styling with uniform shadow */
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
    
    /* Bouncing balls loader animation */
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
    
    /* Subheader styling */
    .stSubheader {
        color: #2d3748;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def init_session_state():
    """Initialize session state variables"""
    if 'document_chats' not in st.session_state:
        st.session_state.document_chats = {}
    
    if 'selected_document' not in st.session_state:
        st.session_state.selected_document = None
    
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0

def get_current_chat_history() -> List[Dict]:
    """Get chat history for currently selected document"""
    if st.session_state.selected_document:
        doc_name = st.session_state.selected_document
        if doc_name not in st.session_state.document_chats:
            st.session_state.document_chats[doc_name] = []
        return st.session_state.document_chats[doc_name]
    return []

def add_message_to_current_chat(message: Dict):
    """Add message to current document's chat history"""
    if st.session_state.selected_document:
        doc_name = st.session_state.selected_document
        if doc_name not in st.session_state.document_chats:
            st.session_state.document_chats[doc_name] = []
        st.session_state.document_chats[doc_name].append(message)

def clear_current_chat():
    """Clear chat history for current document"""
    if st.session_state.selected_document:
        doc_name = st.session_state.selected_document
        st.session_state.document_chats[doc_name] = []

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_document_card(doc: Dict, api_client: RAGAPIClient):
    """Render a document card with actions"""
    doc_name = doc['filename']
    is_selected = st.session_state.selected_document == doc_name
    
    # Create a container with custom styling
    card_class = "doc-card doc-card-selected" if is_selected else "doc-card"
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        # Make the entire card clickable
        if st.button(
            f"{'📘' if is_selected else '📄'} **{doc_name}**",
            key=f"select_{doc_name}",
            use_container_width=True,
            type="primary" if is_selected else "secondary"
        ):
            st.session_state.selected_document = doc_name
            st.rerun()
    
    with col2:
        # Delete button with X
        if st.button("✕", key=f"delete_{doc_name}", 
                   help="Delete document",
                   use_container_width=True,
                   type="secondary"):
            status_code, response = api_client.delete_document(doc_name)
            if status_code == 200:
                # Clean up chat history for deleted document
                if doc_name in st.session_state.document_chats:
                    del st.session_state.document_chats[doc_name]
                
                # Clear selection if this was the selected document
                if st.session_state.selected_document == doc_name:
                    st.session_state.selected_document = None
                
                st.success(f"✅ Deleted")
                time.sleep(0.5)
                st.rerun()
            else:
                st.error(f"❌ {response.get('message', 'Failed')}")
    
    # Show details in small text below
    if is_selected:
        st.caption(f"📊 {doc['chunks']} chunks • {doc['size']:,} bytes • {doc['type'].upper()}")
        if doc_name in st.session_state.document_chats:
            msg_count = len(st.session_state.document_chats[doc_name])
            if msg_count > 0:
                st.caption(f"💬 {msg_count} messages")
    
    st.markdown("---")


def upload_files(files: List, api_client: RAGAPIClient):
    """Handle file upload with progress tracking"""
    success_count = 0
    total_files = len(files)
    uploaded_doc_names = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(files):
        status_text.text(f"⏳ Uploading {file.name}...")
        
        status_code, response = api_client.upload_file(file)
        
        if status_code == 200:
            st.success(f"✅ {file.name}: {response.get('chunks', 0)} chunks created")
            success_count += 1
            uploaded_doc_names.append(file.name)
        else:
            st.error(f"❌ {file.name}: {response.get('message', 'Failed')}")
        
        progress_bar.progress((i + 1) / total_files)
    
    status_text.text(f"✨ Complete: {success_count}/{total_files} uploaded")
    
    # Auto-select the first uploaded document if any succeeded
    if success_count > 0 and uploaded_doc_names and not st.session_state.selected_document:
        st.session_state.selected_document = uploaded_doc_names[0]
    
    # Clear the file uploader by incrementing its key (always, even on failure)
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0
    st.session_state.uploader_key += 1
    
    time.sleep(1.5)
    st.rerun()

# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar(api_client: RAGAPIClient):
    """Render sidebar with document management"""
    with st.sidebar:
        st.markdown('<h1 class="main-header">📚 RAG Assistant</h1>', unsafe_allow_html=True)
        
        documents = api_client.get_documents()
        if documents:
            st.info(f"📊 {len(documents)} document(s) loaded")
        
        st.markdown("---")
        
        # Section header
        st.subheader("📖 Your Documents")
        
        if documents:
            for doc in documents:
                render_document_card(doc, api_client)
        else:
            st.info("💡 No documents yet. Upload below to get started!")
        
        st.markdown("---")
        st.subheader("📤 Upload Documents")
        
        # Use dynamic key to reset file uploader after successful upload
        uploaded_files = st.file_uploader(
            "Choose files",
            type=ALLOWED_EXTENSIONS,
            accept_multiple_files=True,
            help=f"Supported: {', '.join(ALLOWED_EXTENSIONS).upper()} (max {MAX_FILE_SIZE_MB}MB each)",
            key=f"file_uploader_{st.session_state.uploader_key}"
        )
        
        # Show process button only when files are selected
        if uploaded_files:
            if st.button("🚀 Process Files", type="primary", use_container_width=True):
                upload_files(uploaded_files, api_client)
        
        # Clear chat button for selected document
        if st.session_state.selected_document and get_current_chat_history():
            st.markdown("---")
            if st.button("💬 Clear Current Chat", use_container_width=True):
                clear_current_chat()
                st.rerun()

# ============================================================================
# CHAT INTERFACE
# ============================================================================

def render_chat_interface(api_client: RAGAPIClient, health_data: Dict, selected_model: str):
    """Render main chat interface"""
    
    if health_data and health_data.get('document_count', 0) == 0:
        st.info("👋 **Welcome!** Upload documents from the sidebar to start chatting.")
        
        with st.expander("📖 Quick Start Guide", expanded=True):
            st.markdown("""
            ### Getting Started
            
            1. **Upload Documents** 📤 - Click the sidebar and upload PDF, TXT, or DOCX files
            2. **Select Document** 💬 - Click the chat icon on any document to select it
            3. **Ask Questions** 💭 - Type your question in the chat input below
            4. **Get Answers** 🎯 - Receive AI-powered answers with source citations
            5. **Switch Documents** 🔄 - Each document has its own independent chat history
            """)
        return
    
    if not st.session_state.selected_document:
        st.warning("📄 **Please select a document** from the sidebar to start chatting.")
        return
    
    # Show connection status in a compact way
    ollama_status = health_data.get('ollama_status', {}) if health_data else {}
    if not ollama_status.get('available'):
        st.error("✗ Ollama Unavailable")
    
    # Get chat history for current document
    chat_history = get_current_chat_history()
    
    chat_container = st.container()
    with chat_container:
        for message in chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
    if prompt := st.chat_input(f"💭 Ask about {st.session_state.selected_document}..."):
        handle_chat_input(prompt, api_client, selected_model)


def handle_chat_input(prompt: str, api_client: RAGAPIClient, model: str):
    """Handle user chat input"""
    message = {
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().isoformat()
    }
    add_message_to_current_chat(message)
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Show loading animation
        message_placeholder.markdown(
            '<div class="loading-dots"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>',
            unsafe_allow_html=True
        )
        
        full_response = ""
        sources = []
        endpoint_type = None
        
        try:
            for data in api_client.query_stream(prompt, model=model):
                if data.get('type') == 'metadata':
                    sources = data.get('sources', [])
                    endpoint_type = data.get('endpoint_type')
                elif data.get('type') == 'content':
                    full_response += data.get('content', '')
                    message_placeholder.markdown(full_response + "▌")
                elif data.get('type') == 'done':
                    message_placeholder.markdown(full_response)
                elif data.get('type') == 'error':
                    error_msg = f"❌ Error: {data.get('message', 'Unknown error')}"
                    message_placeholder.error(error_msg)
                    full_response = error_msg
            
            assistant_message = {
                "role": "assistant",
                "content": full_response,
                "sources": sources,
                "endpoint_type": endpoint_type,
                "timestamp": datetime.now().isoformat()
            }
            add_message_to_current_chat(assistant_message)
        
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            message_placeholder.error(error_msg)
            error_message = {
                "role": "assistant",
                "content": error_msg,
                "sources": [],
                "timestamp": datetime.now().isoformat()
            }
            add_message_to_current_chat(error_message)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    st.set_page_config(
        page_title="RAG Assistant - Document Chat",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={'About': "# RAG Assistant\nChat with your documents using AI"}
    )
    
    apply_custom_css()
    init_session_state()
    api_client = RAGAPIClient(API_BASE_URL)
    
    health_status, health_data = api_client.health_check()
    
    if not health_status:
        st.error("🔴 **Backend Offline**")
        st.markdown("The RAG backend service is not running. Please start it with:\n```\npython rag_backend.py\n```")
        return
    
    render_sidebar(api_client)
    
    # Header with document name and model dropdown
    col1, col2 = st.columns([4, 1])
    
    with col1:
        if st.session_state.selected_document:
            st.markdown(f'<h2 style="margin-bottom: 0;">💬 {st.session_state.selected_document}</h2>', unsafe_allow_html=True)
        else:
            st.markdown('<h2 style="margin-bottom: 0;">💬 Chat with Your Documents</h2>', unsafe_allow_html=True)
    
    with col2:
        # Get available models
        models_data = api_client.get_models()
        llm_models = models_data.get('ollama', {}).get('llm_models', ['phi3', 'llama3', 'mistral', 'deepseek-r1'])
        current_config = health_data.get('configuration', {}) if health_data else {}
        current_model = current_config.get('model', 'phi3')
        
        # Model selection dropdown
        selected_model = st.selectbox(
            "🤖 Model",
            options=llm_models,
            index=llm_models.index(current_model) if current_model in llm_models else 0,
            key="model_selector",
            label_visibility="collapsed"
        )
    
    st.markdown("---")
    
    render_chat_interface(api_client, health_data, selected_model)


if __name__ == "__main__":
    main()
