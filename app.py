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
    
    def clear_all_documents(self) -> Tuple[int, Dict]:
        """Clear all documents"""
        return self._request('DELETE', '/clear', timeout=30)
    
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
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
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
    
    .doc-card {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .doc-card:hover {
        background-color: #f8f9fa;
        cursor: pointer;
    }
    
    .doc-card-selected {
        background-color: #d4edda;
        border: 2px solid #28a745;
        box-shadow: 0 2px 4px rgba(40, 167, 69, 0.2);
    }
    
    .section-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    @media screen and (max-width: 768px) {
        .main-header { font-size: 1.8rem; }
        .stButton button { width: 100% !important; min-height: 44px !important; }
        div[data-testid="metric-container"] { min-width: 100% !important; }
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
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
    
    if 'confirm_clear' not in st.session_state:
        st.session_state.confirm_clear = False
    
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
    
    if success_count > 0:
        # Auto-select the first uploaded document
        if uploaded_doc_names and not st.session_state.selected_document:
            st.session_state.selected_document = uploaded_doc_names[0]
        
        # Clear the file uploader by incrementing its key
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
        
        # Section header with Clear All button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("📖 Your Documents")
        with col2:
            if documents:
                if st.button("🗑️", key="clear_all_btn", help="Clear all documents", use_container_width=True):
                    if st.session_state.get('confirm_clear', False):
                        with st.spinner("Clearing..."):
                            status_code, response = api_client.clear_all_documents()
                        if status_code == 200:
                            # Clear all chat histories
                            st.session_state.document_chats = {}
                            st.session_state.selected_document = None
                            st.success("✅ Cleared!")
                            st.session_state.confirm_clear = False
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(f"❌ {response.get('message')}")
                    else:
                        st.session_state.confirm_clear = True
                        st.warning("⚠️ Click again")
        
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
    
    ollama_status = health_data.get('ollama_status', {}) if health_data else {}
    if ollama_status.get('available'):
        st.success(f"✓ Ollama Connected | 📘 Chatting with: **{st.session_state.selected_document}**")
    else:
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
    
    # Header with model dropdown
    col1, col2 = st.columns([4, 1])
    
    with col1:
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
