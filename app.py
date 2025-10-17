import streamlit as st
import requests
import json
import time
from typing import List, Dict, Any
import os

# Backend API URL
API_BASE_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="LangChain Ollama RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="auto"
)

# Custom CSS for responsive mobile-friendly styling
st.markdown("""
<style>
/* Base styles */
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.status-success {
    color: #28a745;
    font-weight: bold;
}
.status-error {
    color: #dc3545;
    font-weight: bold;
}
.metric-container {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}

/* Mobile responsive styles */
@media screen and (max-width: 768px) {
    /* Adjust header for mobile */
    .main-header {
        font-size: 1.8rem;
        margin-bottom: 1rem;
    }
    
    /* Make buttons full width on mobile */
    .stButton button {
        width: 100% !important;
        margin: 0.25rem 0 !important;
    }
    
    /* Stack columns vertically on mobile */
    .row-widget.stHorizontalBlock {
        flex-direction: column !important;
    }
    
    /* Better text input sizing */
    .stTextInput, .stTextArea {
        width: 100% !important;
    }
    
    /* Adjust metrics display */
    div[data-testid="metric-container"] {
        min-width: 100% !important;
        margin-bottom: 1rem;
    }
    
    /* Better file uploader */
    .stFileUploader {
        width: 100% !important;
    }
    
    /* Sidebar adjustments */
    section[data-testid="stSidebar"] {
        width: 100% !important;
    }
    
    /* Form elements */
    .stSelectbox, .stSlider, .stNumberInput {
        width: 100% !important;
        margin-bottom: 1rem !important;
    }
    
    /* Expander full width */
    .streamlit-expanderHeader {
        width: 100% !important;
    }
}

/* Tablet responsive styles */
@media screen and (min-width: 769px) and (max-width: 1024px) {
    .main-header {
        font-size: 2rem;
    }
    
    div[data-testid="column"] {
        padding: 0 0.5rem !important;
    }
}

/* Touch-friendly button sizing */
@media (hover: none) and (pointer: coarse) {
    .stButton button {
        min-height: 44px !important;
        padding: 0.75rem 1rem !important;
    }
    
    /* Larger tap targets for mobile */
    button, a, input, select {
        min-height: 44px;
    }
}
</style>
""", unsafe_allow_html=True)

def check_backend_health():
    """Check if the backend is running and healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except requests.exceptions.RequestException:
        return False, None

def get_models():
    """Get available models from backend."""
    try:
        response = requests.get(f"{API_BASE_URL}/models", timeout=10)
        if response.status_code == 200:
            return response.json()
        return {"llm_models": [], "embedding_models": [], "current_llm": "llama3", "current_embedding": "nomic-embed-text"}
    except requests.exceptions.RequestException:
        return {"llm_models": [], "embedding_models": [], "current_llm": "llama3", "current_embedding": "nomic-embed-text"}

def get_documents():
    """Get list of uploaded documents."""
    try:
        response = requests.get(f"{API_BASE_URL}/documents", timeout=10)
        if response.status_code == 200:
            return response.json()
        return []
    except requests.exceptions.RequestException:
        return []


def upload_file(file):
    """Upload a file to the backend."""
    try:
        files = {"file": (file.name, file, file.type)}
        response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=60)
        return response.status_code, response.json() if response.status_code in [200, 400] else {"message": "Upload failed"}
    except requests.exceptions.RequestException as e:
        return 500, {"message": f"Connection error: {str(e)}"}


def delete_document(filename):
    """Delete a specific document."""
    try:
        response = requests.delete(f"{API_BASE_URL}/documents/{filename}", timeout=30)
        return response.status_code, response.json() if response.status_code in [200, 404] else {"message": "Delete failed"}
    except requests.exceptions.RequestException as e:
        return 500, {"message": f"Connection error: {str(e)}"}

def clear_all_documents():
    """Clear all documents."""
    try:
        response = requests.delete(f"{API_BASE_URL}/clear", timeout=30)
        return response.status_code, response.json() if response.status_code == 200 else {"message": "Clear failed"}
    except requests.exceptions.RequestException as e:
        return 500, {"message": f"Connection error: {str(e)}"}

def configure_system(config):
    """Update system configuration."""
    try:
        response = requests.post(f"{API_BASE_URL}/configure", json=config, timeout=30)
        return response.status_code, response.json() if response.status_code in [200, 400] else {"message": "Configuration failed"}
    except requests.exceptions.RequestException as e:
        return 500, {"message": f"Connection error: {str(e)}"}

def preview_document(filename, num_chunks=3):
    """Preview document chunks."""
    try:
        response = requests.get(f"{API_BASE_URL}/documents/{filename}/preview", params={"num_chunks": num_chunks}, timeout=30)
        return response.status_code, response.json() if response.status_code in [200, 404] else {"message": "Preview failed"}
    except requests.exceptions.RequestException as e:
        return 500, {"message": f"Connection error: {str(e)}"}

def rebuild_vectors():
    """Rebuild vector store."""
    try:
        response = requests.post(f"{API_BASE_URL}/rebuild-vectors", timeout=120)
        return response.status_code, response.json() if response.status_code == 200 else {"message": "Rebuild failed"}
    except requests.exceptions.RequestException as e:
        return 500, {"message": f"Connection error: {str(e)}"}

# Main app
def main():
    # Check backend health
    health_status, health_data = check_backend_health()
    
    if not health_status:
        st.error("🔴 Backend is not running! Please start the FastAPI backend on port 8000.")
        st.code("python rag_backend.py")
        return
    
    # Hamburger menu navigation
    with st.sidebar:
        st.markdown('<h1 class="main-header">📚 RAG Assistant</h1>', unsafe_allow_html=True)
        st.markdown("---")
        
        # Initialize session state for page
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "📋 Dashboard"
        
        # Menu options
        if st.button("📋 Dashboard", use_container_width=True):
            st.session_state.current_page = "📋 Dashboard"
        
        if st.button("📤 Upload Documents", use_container_width=True):
            st.session_state.current_page = "📤 Upload Documents"
        
        if st.button("⚙️ Configuration", use_container_width=True):
            st.session_state.current_page = "⚙️ Configuration"
    
    # Display selected page
    if st.session_state.current_page == "📋 Dashboard":
        dashboard_page(health_data)
    elif st.session_state.current_page == "📤 Upload Documents":
        upload_page()
    elif st.session_state.current_page == "⚙️ Configuration":
        config_page()

def dashboard_page(health_data):
    st.header("Dashboard")
    
    # Health status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if health_data and health_data.get('status') == 'healthy':
            st.markdown('<p class="status-success">🟢 System Healthy</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="status-error">🔴 System Issues</p>', unsafe_allow_html=True)
    
    with col2:
        if health_data:
            doc_count = health_data.get('document_count', 0)
            st.metric("Documents", doc_count)
    
    with col3:
        if health_data:
            query_count = health_data.get('total_queries', 0)
            st.metric("Total Queries", query_count)
    
    # System information
    if health_data:
        st.subheader("System Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Current Configuration:**")
            config = health_data.get('configuration', {})
            st.write(f"• LLM Model: {config.get('model', 'N/A')}")
            st.write(f"• Embedding Model: {config.get('embedding_model', 'N/A')}")
            st.write(f"• Chunk Size: {config.get('chunk_size', 'N/A')}")
            st.write(f"• Temperature: {config.get('temperature', 'N/A')}")
        
        with col2:
            st.write("**Ollama Status:**")
            ollama_status = health_data.get('ollama_status', {})
            if ollama_status.get('available'):
                st.markdown('<p class="status-success">✅ Ollama Available</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p class="status-error">❌ Ollama Issues</p>', unsafe_allow_html=True)
    
    # Recent documents
    st.subheader("Recent Documents")
    documents = get_documents()
    if documents:
        # Show last 5 documents
        recent_docs = documents[-5:] if len(documents) > 5 else documents
        for doc in reversed(recent_docs):
            with st.expander(f"📄 {doc['filename']} ({doc['type'].upper()})"):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**Size:** {doc['size']:,} bytes")
                with col2:
                    st.write(f"**Chunks:** {doc['chunks']}")
                with col3:
                    st.write(f"**Uploaded:** {doc['uploaded_at']}")
    else:
        st.info("No documents uploaded yet. Go to the Upload Documents page to get started!")

def upload_page():
    st.header("Upload Documents")
    
    st.write("Upload PDF, TXT, or DOCX files to add them to your knowledge base.")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'txt', 'docx'],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, DOCX"
    )
    
    if uploaded_files:
        if st.button("Upload Files", type="primary"):
            success_count = 0
            total_files = len(uploaded_files)
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, file in enumerate(uploaded_files):
                status_text.text(f"Uploading {file.name}...")
                
                status_code, response = upload_file(file)
                
                if status_code == 200:
                    st.success(f"✅ {file.name}: {response['message']} ({response['chunks']} chunks)")
                    success_count += 1
                else:
                    st.error(f"❌ {file.name}: {response.get('message', 'Upload failed')}")
                
                progress_bar.progress((i + 1) / total_files)
            
            status_text.text(f"Upload complete: {success_count}/{total_files} files processed successfully")
            
            if success_count > 0:
                st.balloons()
    
    # Show current documents
    st.subheader("Current Documents")
    documents = get_documents()
    
    if documents:
        for doc in documents:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.write(f"📄 **{doc['filename']}** ({doc['type'].upper()})")
                    st.caption(f"{doc['size']:,} bytes • {doc['chunks']} chunks • {doc['uploaded_at']}")
                
                with col2:
                    if st.button("👁️", key=f"preview_{doc['filename']}", help="Preview"):
                        st.session_state[f"preview_{doc['filename']}"] = True
                
                with col3:
                    if st.button("🗑️", key=f"delete_{doc['filename']}", help="Delete", type="secondary"):
                        status_code, response = delete_document(doc['filename'])
                        if status_code == 200:
                            st.success(f"Deleted {doc['filename']}")
                            st.rerun()
                        else:
                            st.error(f"Failed to delete: {response.get('message', 'Unknown error')}")
                
                # Preview functionality
                if st.session_state.get(f"preview_{doc['filename']}", False):
                    status_code, preview_data = preview_document(doc['filename'])
                    if status_code == 200:
                        st.write("**Preview:**")
                        for chunk in preview_data.get('chunks', []):
                            st.text_area(
                                f"Chunk {chunk['chunk_id']} ({chunk['length']} chars)",
                                chunk['content'],
                                height=100,
                                key=f"preview_chunk_{doc['filename']}_{chunk['chunk_id']}"
                            )
                    else:
                        st.error("Failed to load preview")
                    
                    if st.button("Close Preview", key=f"close_preview_{doc['filename']}"):
                        st.session_state[f"preview_{doc['filename']}"] = False
                        st.rerun()
                
                st.divider()
        
        # Clear all button
        if st.button("🗑️ Clear All Documents", type="secondary"):
            if st.session_state.get('confirm_clear_all', False):
                status_code, response = clear_all_documents()
                if status_code == 200:
                    st.success("All documents cleared!")
                    st.session_state['confirm_clear_all'] = False
                    st.rerun()
                else:
                    st.error(f"Failed to clear: {response.get('message', 'Unknown error')}")
            else:
                st.session_state['confirm_clear_all'] = True
                st.warning("Click again to confirm deletion of ALL documents")
                st.rerun()
    else:
        st.info("No documents uploaded yet.")

def config_page():
    st.header("Configuration")
    
    # Get current models and config
    models_data = get_models()
    
    st.subheader("Model Configuration")
    
    with st.form("config_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            llm_model = st.selectbox(
                "LLM Model",
                options=models_data.get('llm_models', ['llama3']),
                index=0,
                help="Model used for generating answers"
            )
        
        with col2:
            embedding_model = st.selectbox(
                "Embedding Model",
                options=models_data.get('embedding_models', ['nomic-embed-text']),
                index=0,
                help="Model used for generating document embeddings"
            )
        
        st.subheader("Text Processing")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=1000, step=50)
        
        with col2:
            chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=200, step=25)
        
        with col3:
            temperature = st.slider("Temperature", 0.0, 2.0, 0.7, step=0.1)
        
        submitted = st.form_submit_button("Update Configuration", type="primary")
    
    if submitted:
        config = {
            "model": llm_model,
            "embedding_model": embedding_model,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "temperature": temperature
        }
        
        status_code, response = configure_system(config)
        
        if status_code == 200:
            st.success("Configuration updated successfully!")
            
            changed_fields = response.get('changed_fields', [])
            if changed_fields:
                st.info(f"Changed fields: {', '.join(changed_fields)}")
            
            if 'embedding_model' in changed_fields:
                st.warning("⚠️ Embedding model changed! You may need to rebuild vectors for existing documents.")
                
                if st.button("🔄 Rebuild Vectors Now"):
                    with st.spinner("Rebuilding vectors..."):
                        rebuild_status, rebuild_response = rebuild_vectors()
                        
                        if rebuild_status == 200:
                            st.success("Vectors rebuilt successfully!")
                            results = rebuild_response.get('results', {})
                            for filename, result in results.items():
                                if result['success']:
                                    st.write(f"✅ {filename}: {result['chunks']} chunks")
                                else:
                                    st.write(f"❌ {filename}: {result['error']}")
                        else:
                            st.error(f"Rebuild failed: {rebuild_response.get('message', 'Unknown error')}")
        else:
            st.error(f"Configuration update failed: {response.get('message', 'Unknown error')}")
    
    # Current configuration display
    st.subheader("Current Configuration")
    current_config = {
        "LLM Model": models_data.get('current_llm', 'N/A'),
        "Embedding Model": models_data.get('current_embedding', 'N/A')
    }
    
    for key, value in current_config.items():
        st.write(f"**{key}:** {value}")

# Run the app
if __name__ == "__main__":
    main()
