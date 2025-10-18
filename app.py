import streamlit as st
import requests
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
MAX_FILE_SIZE_MB = 20
ALLOWED_EXTENSIONS = ['pdf', 'txt', 'docx']

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Document:
    """Document model for better type handling"""
    filename: str
    type: str
    size: int
    chunks: int
    uploaded_at: str
    
    @property
    def size_formatted(self) -> str:
        """Format size in human-readable format"""
        size = self.size
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

@dataclass
class ChatMessage:
    """Chat message model"""
    role: str
    content: str
    sources: List[str] = None
    timestamp: str = None
    endpoint_type: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.sources is None:
            self.sources = []

# ============================================================================
# API CLIENT
# ============================================================================

class RAGAPIClient:
    """Centralized API client for all backend interactions"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.session = requests.Session()
    
    def _handle_response(self, response: requests.Response) -> Tuple[int, Dict]:
        """Centralized response handling"""
        try:
            data = response.json() if response.content else {}
            return response.status_code, data
        except json.JSONDecodeError:
            return response.status_code, {"message": "Invalid response from server"}
    
    def health_check(self) -> Tuple[bool, Optional[Dict]]:
        """Check backend health"""
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200, response.json() if response.ok else None
        except requests.exceptions.RequestException:
            return False, None
    
    def get_models(self) -> Dict:
        """Get available models"""
        try:
            response = self.session.get(f"{self.base_url}/models", timeout=10)
            if response.ok:
                return response.json()
        except requests.exceptions.RequestException:
            pass
        return {
            "llm_models": ["phi3", "llama3", "mistral", "deepseek-r1"], 
            "embedding_models": ["nomic-embed-text"],
            "current_config": {
                "model": "phi3",
                "embedding_model": "nomic-embed-text"
            }
        }
    
    def get_documents(self) -> List[Dict]:
        """Get list of uploaded documents"""
        try:
            response = self.session.get(f"{self.base_url}/documents", timeout=10)
            if response.ok:
                return response.json()
        except requests.exceptions.RequestException:
            pass
        return []
    
    def get_stats(self) -> Dict:
        """Get system statistics"""
        try:
            response = self.session.get(f"{self.base_url}/stats", timeout=10)
            if response.ok:
                return response.json()
        except requests.exceptions.RequestException:
            pass
        return {}
    
    def upload_file(self, file) -> Tuple[int, Dict]:
        """Upload a file to the backend"""
        try:
            files = {"file": (file.name, file, file.type)}
            response = self.session.post(
                f"{self.base_url}/upload", 
                files=files, 
                timeout=60
            )
            return self._handle_response(response)
        except requests.exceptions.RequestException as e:
            return 500, {"message": f"Connection error: {str(e)}"}
    
    def delete_document(self, filename: str) -> Tuple[int, Dict]:
        """Delete a specific document"""
        try:
            response = self.session.delete(
                f"{self.base_url}/documents/{filename}", 
                timeout=30
            )
            return self._handle_response(response)
        except requests.exceptions.RequestException as e:
            return 500, {"message": f"Connection error: {str(e)}"}
    
    def clear_all_documents(self) -> Tuple[int, Dict]:
        """Clear all documents"""
        try:
            response = self.session.delete(f"{self.base_url}/clear", timeout=30)
            return self._handle_response(response)
        except requests.exceptions.RequestException as e:
            return 500, {"message": f"Connection error: {str(e)}"}
    
    def configure_system(self, config: Dict) -> Tuple[int, Dict]:
        """Update system configuration"""
        try:
            response = self.session.post(
                f"{self.base_url}/configure", 
                json=config, 
                timeout=30
            )
            return self._handle_response(response)
        except requests.exceptions.RequestException as e:
            return 500, {"message": f"Connection error: {str(e)}"}
    
    def rebuild_vectors(self) -> Tuple[int, Dict]:
        """Rebuild vector store"""
        try:
            response = self.session.post(
                f"{self.base_url}/rebuild-vectors", 
                timeout=120
            )
            return self._handle_response(response)
        except requests.exceptions.RequestException as e:
            return 500, {"message": f"Connection error: {str(e)}"}
    
    def query_stream(self, question: str, top_k: int = 4, model: str = None, temperature: float = None):
        """Stream query response"""
        try:
            payload = {
                "question": question, 
                "stream": True, 
                "top_k": top_k
            }
            
            if model:
                payload["model"] = model
            if temperature is not None:
                payload["temperature"] = temperature
            
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
    /* Base theme improvements */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 600;
    }
    
    .status-success {
        background-color: #d4edda;
        color: #155724;
    }
    
    .status-error {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    .status-info {
        background-color: #d1ecf1;
        color: #0c5460;
    }
    
    /* Document cards */
    .doc-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1rem;
        border-radius: 0.75rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Chat improvements */
    .stChatMessage {
        background-color: transparent;
    }
    
    /* Mobile responsive styles */
    @media screen and (max-width: 768px) {
        .main-header {
            font-size: 1.8rem;
            margin-bottom: 1rem;
        }
        
        .stButton button {
            width: 100% !important;
            margin: 0.25rem 0 !important;
            min-height: 44px !important;
            padding: 0.75rem 1rem !important;
        }
        
        .row-widget.stHorizontalBlock {
            flex-direction: column !important;
        }
        
        .stTextInput, .stTextArea, .stFileUploader {
            width: 100% !important;
        }
        
        div[data-testid="metric-container"] {
            min-width: 100% !important;
            margin-bottom: 1rem;
        }
        
        section[data-testid="stSidebar"] {
            width: 100% !important;
        }
        
        .stSelectbox, .stSlider, .stNumberInput {
            width: 100% !important;
            margin-bottom: 1rem !important;
        }
    }
    
    /* Tablet styles */
    @media screen and (min-width: 769px) and (max-width: 1024px) {
        .main-header {
            font-size: 2rem;
        }
        
        div[data-testid="column"] {
            padding: 0 0.5rem !important;
        }
    }
    
    /* Touch-friendly sizing */
    @media (hover: none) and (pointer: coarse) {
        button, a, input, select {
            min-height: 44px;
        }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# UI COMPONENTS
# ============================================================================

class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def render_document_card(doc: Dict, api_client: RAGAPIClient):
        """Render a document card with actions"""
        with st.expander(f"📄 {doc['filename']}", expanded=False):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.caption(f"**Type:** {doc['type'].upper()}")
                st.caption(f"**Size:** {doc['size']:,} bytes")
                st.caption(f"**Chunks:** {doc['chunks']}")
                st.caption(f"**Uploaded:** {doc['uploaded_at'][:19]}")
            
            with col2:
                if st.button("🗑️", key=f"delete_{doc['filename']}", 
                           help="Delete document", use_container_width=True):
                    status_code, response = api_client.delete_document(doc['filename'])
                    if status_code == 200:
                        st.success(f"✅ Deleted")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error(f"❌ {response.get('message', 'Failed')}")
    
    @staticmethod
    def render_metrics_dashboard(health_data: Dict, stats_data: Dict):
        """Render system metrics dashboard"""
        if not health_data:
            return
        
        config = health_data.get('configuration', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📚 Documents", 
                health_data.get('document_count', 0),
                help="Total documents uploaded"
            )
        
        with col2:
            st.metric(
                "🧩 Chunks", 
                health_data.get('total_chunks', 0),
                help="Total text chunks indexed"
            )
        
        with col3:
            st.metric(
                "💬 Queries", 
                stats_data.get('total_queries', 0),
                help="Total queries processed"
            )
        
        with col4:
            avg_chunks = stats_data.get('average_chunks_per_document', 0)
            st.metric(
                "📊 Avg Chunks/Doc", 
                f"{avg_chunks:.1f}",
                help="Average chunks per document"
            )
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**🤖 LLM Model**\n\n`{config.get('model', 'N/A')}`")
        
        with col2:
            st.info(f"**🔢 Embedding Model**\n\n`{config.get('embedding_model', 'N/A')}`")
        
        with col3:
            st.info(f"**🌡️ Temperature**\n\n`{config.get('temperature', 0.7)}`")

# ============================================================================
# DIALOGS
# ============================================================================

@st.dialog("⚙️ System Settings", width="large")
def settings_dialog(api_client: RAGAPIClient):
    """Settings modal dialog"""
    models_data = api_client.get_models()
    health_status, health_data = api_client.health_check()
    
    # Get current configuration from health endpoint
    current_config = {}
    if health_data:
        current_config = health_data.get('configuration', {})
    else:
        current_config = models_data.get('current_config', {})
    
    st.subheader("🤖 Model Configuration")
    
    with st.form("config_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            ollama_models = models_data.get('ollama', [])
            llm_models = ollama_models.get('llm_models', ['phi3', 'llama3', 'mistral', 'deepseek-r1'])
            current_llm = current_config.get('model', 'phi3')                        
            # Find index of current model
            try:
                llm_index = llm_models.index(current_llm)
            except ValueError:
                llm_index = 0
            
            llm_model = st.selectbox(
                "LLM Model",
                options=llm_models,
                index=llm_index,
                help="Model for generating answers. DeepSeek-R1 uses intelligent endpoint detection."
            )
        
        with col2:
            embedding_models = models_data.get('embedding_models', ['nomic-embed-text'])
            current_embedding = current_config.get('embedding_model', 'nomic-embed-text')
            
            # Find index of current embedding model
            try:
                embed_index = embedding_models.index(current_embedding)
            except ValueError:
                embed_index = 0
            
            embedding_model = st.selectbox(
                "Embedding Model",
                options=embedding_models,
                index=embed_index,
                help="Model for document embeddings"
            )
        
        st.markdown("---")
        st.subheader("📝 Text Processing")
        
        col1, col2 = st.columns(2)
        
        with col1:
            chunk_size = st.number_input(
                "Chunk Size", 
                min_value=100, 
                max_value=2000, 
                value=current_config.get('chunk_size', 1000), 
                step=50,
                help="Size of text chunks for processing"
            )
            
            temperature = st.slider(
                "Temperature", 
                0.0, 
                2.0, 
                float(current_config.get('temperature', 0.7)), 
                step=0.1,
                help="Controls response randomness (0=focused, 2=creative)"
            )
        
        with col2:
            chunk_overlap = st.number_input(
                "Chunk Overlap", 
                min_value=0, 
                max_value=500, 
                value=current_config.get('chunk_overlap', 200), 
                step=25,
                help="Overlap between consecutive chunks"
            )
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            submitted = st.form_submit_button(
                "💾 Update Configuration", 
                type="primary", 
                use_container_width=True
            )
        with col2:
            apply_and_rebuild = st.form_submit_button(
                "💾 Update & Rebuild",
                use_container_width=True
            )
        with col3:
            cancel = st.form_submit_button(
                "❌ Cancel",
                use_container_width=True
            )
    
    if cancel:
        st.rerun()
    
    if submitted or apply_and_rebuild:
        config = {
            "model": llm_model,
            "embedding_model": embedding_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "temperature": temperature
        }
        
        with st.spinner("Updating configuration..."):
            status_code, response = api_client.configure_system(config)
        
        if status_code == 200:
            st.success("✅ Configuration updated successfully!")
            
            changed_fields = response.get('changed_fields', [])
            if changed_fields:
                st.info(f"📝 Changed fields: {', '.join(changed_fields)}")
            
            # Show warning if embedding model changed
            embedding_changed = 'embedding_model' in changed_fields
            if embedding_changed:
                st.warning("⚠️ Embedding model changed! You should rebuild vectors for existing documents.")
            
            # Auto-rebuild if requested or if embedding changed and user chose that option
            if apply_and_rebuild or (embedding_changed and apply_and_rebuild):
                with st.spinner("Rebuilding vectors... This may take a while."):
                    rebuild_status, rebuild_response = api_client.rebuild_vectors()
                    
                    if rebuild_status == 200:
                        st.success("✅ Vectors rebuilt successfully!")
                        results = rebuild_response.get('results', {})
                        
                        success_count = sum(1 for r in results.values() if r.get('success'))
                        st.info(f"📊 Rebuilt {success_count}/{len(results)} documents")
                        
                        with st.expander("View Details"):
                            for filename, result in results.items():
                                if result.get('success'):
                                    st.write(f"✅ {filename}: {result.get('chunks')} chunks")
                                else:
                                    st.write(f"❌ {filename}: {result.get('error')}")
                    else:
                        st.error(f"❌ Rebuild failed: {rebuild_response.get('message', 'Unknown error')}")
            
            time.sleep(1.5)
            st.rerun()
        else:
            st.error(f"❌ Configuration update failed: {response.get('message', 'Unknown error')}")

# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar(api_client: RAGAPIClient):
    """Render sidebar with document management"""
    with st.sidebar:
        st.markdown('<h1 class="main-header">📚 RAG Assistant</h1>', unsafe_allow_html=True)
        
        # Quick stats
        documents = api_client.get_documents()
        if documents:
            st.info(f"📊 {len(documents)} document(s) loaded")
        
        st.markdown("---")
        
        # Document list
        st.subheader("📁 Your Documents")
        
        if documents:
            for doc in documents:
                UIComponents.render_document_card(doc, api_client)
            
            # Bulk actions
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🗑️ Clear All", use_container_width=True, type="secondary"):
                    if st.session_state.get('confirm_clear', False):
                        with st.spinner("Clearing..."):
                            status_code, response = api_client.clear_all_documents()
                        if status_code == 200:
                            st.success("✅ Cleared!")
                            st.session_state.confirm_clear = False
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error(f"❌ {response.get('message')}")
                    else:
                        st.session_state.confirm_clear = True
                        st.warning("⚠️ Click again to confirm")
            
            with col2:
                if st.button("🔄 Rebuild", use_container_width=True):
                    with st.spinner("Rebuilding..."):
                        status_code, response = api_client.rebuild_vectors()
                    if status_code == 200:
                        st.success("✅ Done!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("❌ Failed")
        else:
            st.info("💡 No documents yet. Upload below to get started!")
        
        # File uploader
        st.markdown("---")
        st.subheader("📤 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=ALLOWED_EXTENSIONS,
            accept_multiple_files=True,
            help=f"Supported: {', '.join(ALLOWED_EXTENSIONS).upper()} (max {MAX_FILE_SIZE_MB}MB each)",
            key="file_uploader"
        )
        
        if uploaded_files:
            if st.button("🚀 Process Files", type="primary", use_container_width=True):
                upload_files(uploaded_files, api_client)
        
        # Clear chat
        if st.session_state.get('chat_history') and st.session_state.chat_history:
            st.markdown("---")
            if st.button("💬 Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

def upload_files(files: List, api_client: RAGAPIClient):
    """Handle file upload with progress tracking"""
    success_count = 0
    total_files = len(files)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, file in enumerate(files):
        status_text.text(f"⏳ Uploading {file.name}...")
        
        status_code, response = api_client.upload_file(file)
        
        if status_code == 200:
            st.success(f"✅ {file.name}: {response.get('chunks', 0)} chunks created")
            success_count += 1
        else:
            st.error(f"❌ {file.name}: {response.get('message', 'Failed')}")
        
        progress_bar.progress((i + 1) / total_files)
    
    status_text.text(f"✨ Complete: {success_count}/{total_files} uploaded")
    
    if success_count > 0:
        st.balloons()
        time.sleep(1.5)
        st.rerun()

# ============================================================================
# CHAT INTERFACE
# ============================================================================

def render_chat_interface(api_client: RAGAPIClient, health_data: Dict):
    """Render main chat interface"""
    
    # Check if documents exist
    if health_data and health_data.get('document_count', 0) == 0:
        st.info("👋 **Welcome!** Upload documents from the sidebar to start chatting.")
        
        # Show quick start guide
        with st.expander("📖 Quick Start Guide", expanded=True):
            st.markdown("""
            ### Getting Started
            
            1. **Upload Documents** 📤  
               Click the sidebar and upload PDF, TXT, or DOCX files
            
            2. **Ask Questions** 💬  
               Type your question in the chat input below
            
            3. **Get Answers** 🎯  
               Receive AI-powered answers with source citations
            
            4. **Configure Settings** ⚙️  
               Click the settings button to customize models and parameters
            
            ### Supported Models
            
            - **DeepSeek-R1**: Advanced reasoning model with automatic endpoint detection
            - **Llama 3**: Fast and efficient general-purpose model
            - **Phi-3**: Compact model optimized for quick responses
            - **Mistral**: Balanced performance and quality
            """)
        return
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if message["role"] == "assistant":
                    # Show endpoint info if available
                    if message.get("endpoint_type"):
                        endpoint_badge = f'<span class="status-badge status-info">Endpoint: {message["endpoint_type"]}</span>'
                        st.markdown(endpoint_badge, unsafe_allow_html=True)
                    
                    # Show sources
                    if message.get("sources"):
                        with st.expander("📚 View Sources"):
                            unique_sources = list(set(message["sources"]))
                            for i, source in enumerate(unique_sources, 1):
                                st.markdown(f"{i}. `{source}`")
    
    # Chat input
    if prompt := st.chat_input("💭 Ask your documents anything...", key="chat_input"):
        handle_chat_input(prompt, api_client)

def handle_chat_input(prompt: str, api_client: RAGAPIClient):
    """Handle user chat input"""
    # Add user message
    st.session_state.chat_history.append({
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now().isoformat()
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response with streaming
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        sources = []
        endpoint_type = None
        model_used = None
        
        try:
            for data in api_client.query_stream(prompt):
                if data.get('type') == 'metadata':
                    sources = data.get('sources', [])
                    endpoint_type = data.get('endpoint_type')
                    model_used = data.get('model_used')
                elif data.get('type') == 'content':
                    full_response += data.get('content', '')
                    message_placeholder.markdown(full_response + "▌")
                elif data.get('type') == 'done':
                    message_placeholder.markdown(full_response)
                elif data.get('type') == 'error':
                    error_msg = f"❌ Error: {data.get('message', 'Unknown error')}"
                    message_placeholder.error(error_msg)
                    full_response = error_msg
            
            # Add to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": full_response,
                "sources": sources,
                "endpoint_type": endpoint_type,
                "model_used": model_used,
                "timestamp": datetime.now().isoformat()
            })
            
            # Show endpoint info
            if endpoint_type:
                endpoint_badge = f'<span class="status-badge status-info">Endpoint: {endpoint_type}</span>'
                st.markdown(endpoint_badge, unsafe_allow_html=True)
            
            # Show sources
            if sources:
                with st.expander("📚 View Sources"):
                    unique_sources = list(set(sources))
                    for i, source in enumerate(unique_sources, 1):
                        st.markdown(f"{i}. `{source}`")
        
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": error_msg,
                "sources": [],
                "timestamp": datetime.now().isoformat()
            })

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point"""
    
    # Page config
    st.set_page_config(
        page_title="RAG Assistant - Document Chat",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            'About': "# RAG Assistant\nChat with your documents using AI with intelligent endpoint detection"
        }
    )
    
    # Apply custom styling
    apply_custom_css()
    
    # Initialize API client
    api_client = RAGAPIClient(API_BASE_URL)
    
    # Check backend health
    health_status, health_data = api_client.health_check()
    
    if not health_status:
        st.error("🔴 **Backend Offline**")
        st.markdown("""
        The RAG backend service is not running. Please start it with:
        ```
        python rag_backend.py
        ```
        """)
        return
    
    # Render sidebar
    render_sidebar(api_client)
    
    # Main content area
    col1, col2, col3 = st.columns([6, 1, 1])
    
    with col1:
        st.markdown('<h2 style="margin-bottom: 0;">💬 Chat with Your Documents</h2>', unsafe_allow_html=True)
    
    with col2:
        if st.button("📊", help="Dashboard", use_container_width=True):
            st.session_state.show_dashboard = not st.session_state.get('show_dashboard', False)
    
    with col3:
        if st.button("⚙️", help="Settings", use_container_width=True):
            settings_dialog(api_client)
    
    # Show dashboard if toggled
    if st.session_state.get('show_dashboard', False):
        st.markdown("---")
        st.markdown("### 📊 System Dashboard")
        stats_data = api_client.get_stats()
        UIComponents.render_metrics_dashboard(health_data, stats_data)
        st.markdown("---")
    
    # Render chat interface
    render_chat_interface(api_client, health_data)
    
    # Show connection status in footer
    if health_data:
        ollama_status = health_data.get('ollama_status', {})
        if ollama_status.get('available'):
            status_msg = f'<span class="status-badge status-success">✓ Connected to Ollama</span>'
        else:
            status_msg = f'<span class="status-badge status-error">✗ Ollama Unavailable</span>'
        
        st.markdown(f'<div style="text-align: center; margin-top: 2rem;">{status_msg}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
