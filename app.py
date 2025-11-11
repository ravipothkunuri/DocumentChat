"""
DocumentChat Frontend - Interactive Streamlit Interface

This module provides a user-friendly web interface for the DocumentChat system,
allowing users to upload documents, select them for querying, and engage in
conversational question-answering powered by AI.

Key Features:
    - Document upload with drag-and-drop support
    - Multi-format support (PDF, TXT, DOCX)
    - Real-time streaming chat responses
    - Persistent chat history per document
    - Chat export to JSON and Markdown
    - Stop generation capability
    - Toast notifications for user feedback
    - Responsive sidebar navigation

Architecture:
    - Streamlit for UI rendering and state management
    - Async HTTP client for backend communication
    - File-based persistence for chat history
    - Session state for real-time updates

Environment Variables:
    - OLLAMA_BASE_URL: Backend API URL (default: http://localhost:8000)
    - OLLAMA_CHAT_MODEL: Model name for display

Author: Your Name
Version: 1.0.0
"""

import json
import asyncio
import random
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, AsyncIterator, Tuple

import httpx
import streamlit as st
from configuration import (
    FALLBACK_BASE_URL, MAX_FILE_SIZE_MB, UI_ALLOWED_EXTENSIONS,
    DEFAULT_MODEL, THINKING_MESSAGES
)

# Configuration from environment
API_BASE_URL = os.environ.get("OLLAMA_BASE_URL", FALLBACK_BASE_URL)
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", DEFAULT_MODEL)

# Create persistence directory for chat history
CHAT_HISTORY_DIR = Path("chat_history")
CHAT_HISTORY_DIR.mkdir(exist_ok=True)


# =============================================================================
# API Client - Backend Communication
# =============================================================================

class APIClient:
    """
    HTTP client for communicating with the DocumentChat backend.
    
    Provides methods for all backend API operations including document
    management and query streaming. Handles both sync and async requests.
    
    Attributes:
        base_url: Base URL of the backend API
        sync_client: Synchronous HTTP client for blocking operations
        async_client: Asynchronous HTTP client for streaming
        
    Example:
        client = APIClient("http://localhost:8000")
        is_healthy, data = client.health_check()
        docs = client.get_documents()
        status, response = client.upload_file(file)
    """
    
    def __init__(self, base_url: str):
        """
        Initialize the API client.
        
        Args:
            base_url: Backend API base URL
        """
        self.base_url = base_url
        self.sync_client = httpx.Client(timeout=60.0)
        self.async_client = httpx.AsyncClient(timeout=60.0)

    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        timeout: int = 10, 
        **kwargs
    ) -> Tuple[int, Dict]:
        """
        Make a synchronous HTTP request to the backend.
        
        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint path (e.g., '/documents')
            timeout: Request timeout in seconds
            **kwargs: Additional arguments passed to httpx.request
            
        Returns:
            Tuple of (status_code, response_data)
            
        Example:
            status, data = client._make_request('GET', '/documents')
            status, data = client._make_request('POST', '/upload', files=files)
        """
        try:
            url = f"{self.base_url}{endpoint}"
            response = self.sync_client.request(
                method, url, timeout=timeout, **kwargs
            )
            
            # Parse JSON response if available
            data = response.json() if response.content else {}
            return response.status_code, data
            
        except json.JSONDecodeError:
            return response.status_code, {"message": "Invalid JSON response"}
        except httpx.RequestError as e:
            return 500, {"message": f"Connection error: {str(e)}"}

    def health_check(self) -> Tuple[bool, Optional[Dict]]:
        """
        Check if backend service is healthy.
        
        Returns:
            Tuple of (is_healthy, health_data)
            - is_healthy: True if service is responding
            - health_data: Dictionary with service status or None
            
        Example:
            is_healthy, data = client.health_check()
            if is_healthy:
                print(f"Documents: {data['document_count']}")
        """
        try:
            response = self.sync_client.get(
                f"{self.base_url}/health", 
                timeout=5
            )
            is_success = response.status_code == 200
            data = response.json() if is_success else None
            return is_success, data
        except httpx.RequestError:
            return False, None

    def get_documents(self) -> List[Dict]:
        """
        Retrieve list of all uploaded documents.
        
        Returns:
            List of document metadata dictionaries, or empty list on failure
            
        Example:
            docs = client.get_documents()
            for doc in docs:
                print(f"{doc['filename']}: {doc['chunks']} chunks")
        """
        status_code, data = self._make_request('GET', '/documents')
        return data if status_code == 200 else []

    def upload_file(self, file) -> Tuple[int, Dict]:
        """
        Upload a file to the backend for processing.
        
        Args:
            file: Streamlit UploadedFile object
            
        Returns:
            Tuple of (status_code, response_data)
            
        Example:
            uploaded_file = st.file_uploader("Choose file")
            if uploaded_file:
                status, response = client.upload_file(uploaded_file)
                if status == 200:
                    st.success(f"Uploaded: {response['chunks']} chunks")
        """
        try:
            files = {"file": (file.name, file, file.type)}
            response = self.sync_client.post(
                f"{self.base_url}/upload",
                files=files,
                timeout=60
            )
            data = response.json() if response.content else {}
            return response.status_code, data
        except Exception as e:
            return 500, {"message": f"Upload failed: {str(e)}"}

    def delete_document(self, filename: str) -> Tuple[int, Dict]:
        """
        Delete a document from the backend.
        
        Args:
            filename: Name of the document to delete
            
        Returns:
            Tuple of (status_code, response_data)
            
        Example:
            status, response = client.delete_document("report.pdf")
            if status == 200:
                st.success("Document deleted")
        """
        return self._make_request(
            'DELETE', 
            f'/documents/{filename}', 
            timeout=30
        )

    async def query_stream(
        self, 
        question: str, 
        top_k: int = 4
    ) -> AsyncIterator[Dict]:
        """
        Stream query response from the backend.
        
        Yields dictionaries with different types:
        - metadata: Initial response with sources and scores
        - content: Text chunks of the generated response
        - done: Completion signal with processing time
        - error: Error information
        
        Args:
            question: User's question
            top_k: Number of similar chunks to retrieve
            
        Yields:
            Dictionary with 'type' and relevant data
            
        Example:
            async for data in client.query_stream("What is AI?"):
                if data['type'] == 'content':
                    print(data['content'], end='')
                elif data['type'] == 'done':
                    print(f"\nTime: {data['processing_time']}s")
        """
        try:
            payload = {
                "question": question,
                "stream": True,
                "top_k": top_k,
                "model": DEFAULT_MODEL
            }
            
            # Open streaming connection
            async with self.async_client.stream(
                'POST',
                f"{self.base_url}/query",
                json=payload,
                timeout=120.0
            ) as response:
                if response.status_code == 200:
                    # Process Server-Sent Events
                    async for line in response.aiter_lines():
                        if line and line.startswith('data: '):
                            try:
                                # Parse JSON data after "data: " prefix
                                yield json.loads(line[6:])
                            except json.JSONDecodeError:
                                continue
                else:
                    yield {
                        "type": "error",
                        "message": f"Query failed with status {response.status_code}"
                    }
        except httpx.ReadTimeout:
            yield {"type": "error", "message": "Request timed out"}
        except Exception as e:
            yield {"type": "error", "message": str(e)}

    def __del__(self):
        """
        Cleanup HTTP clients on object destruction.
        
        Ensures proper resource cleanup when client is garbage collected.
        """
        try:
            self.sync_client.close()
        except:
            pass


# =============================================================================
# File-based Persistence Functions
# =============================================================================

def get_safe_filename(doc_name: str) -> str:
    """
    Convert document name to filesystem-safe filename.
    
    Replaces special characters with underscores to prevent
    filesystem issues.
    
    Args:
        doc_name: Original document name
        
    Returns:
        Safe filename string
        
    Example:
        safe = get_safe_filename("my doc (2024).pdf")
        # Returns: "my_doc__2024_.pdf"
    """
    return "".join(
        c if c.isalnum() or c in "._-" else "_" 
        for c in doc_name
    )


def save_chat_history_to_local(doc_name: str, data: List[Dict]):
    """
    Persist chat history to local JSON file.
    
    Saves complete chat history with metadata for a specific document.
    Creates or updates the corresponding JSON file.
    
    Args:
        doc_name: Document name
        data: List of message dictionaries
        
    File Structure:
        {
            "document": "report.pdf",
            "last_updated": "2024-01-01T12:00:00",
            "message_count": 10,
            "messages": [...]
        }
        
    Example:
        messages = [
            {"role": "user", "content": "What is AI?"},
            {"role": "assistant", "content": "AI is..."}
        ]
        save_chat_history_to_local("doc.pdf", messages)
    """
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
        # Silently fail to avoid disrupting user experience
        pass


def load_chat_history_from_local(doc_name: str) -> Optional[List[Dict]]:
    """
    Load chat history from local JSON file.
    
    Args:
        doc_name: Document name
        
    Returns:
        List of message dictionaries, or None if file doesn't exist
        
    Example:
        messages = load_chat_history_from_local("doc.pdf")
        if messages:
            st.write(f"Loaded {len(messages)} messages")
    """
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
    """
    Delete chat history file for a document.
    
    Args:
        doc_name: Document name
        
    Example:
        delete_chat_history("old_doc.pdf")
    """
    try:
        safe_filename = get_safe_filename(doc_name)
        file_path = CHAT_HISTORY_DIR / f"{safe_filename}_chat.json"
        if file_path.exists():
            file_path.unlink()
    except Exception:
        pass


# =============================================================================
# Export Functions
# =============================================================================

def export_chat_as_json(doc_name: str) -> str:
    """
    Export chat history as formatted JSON string.
    
    Reads from persisted file if available, otherwise uses session state.
    
    Args:
        doc_name: Document name
        
    Returns:
        JSON string with chat history
        
    Example:
        json_str = export_chat_as_json("doc.pdf")
        st.download_button("Download", json_str, file_name="chat.json")
    """
    try:
        safe_filename = get_safe_filename(doc_name)
        file_path = CHAT_HISTORY_DIR / f"{safe_filename}_chat.json"

        if file_path.exists():
            # Read directly from file
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # Fallback to session state
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
    """
    Export chat history as formatted Markdown string.
    
    Creates a readable markdown document with:
    - Document header
    - Export metadata
    - Formatted messages with timestamps
    - Role indicators (User/Assistant)
    
    Args:
        doc_name: Document name
        
    Returns:
        Markdown-formatted string
        
    Example:
        md_str = export_chat_as_markdown("doc.pdf")
        st.download_button("Download MD", md_str, file_name="chat.md")
    """
    try:
        safe_filename = get_safe_filename(doc_name)
        file_path = CHAT_HISTORY_DIR / f"{safe_filename}_chat.json"

        # Load messages from file or session state
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                chat_data = json.load(f)
                messages = chat_data.get("messages", [])
        else:
            messages = st.session_state.document_chats.get(doc_name, [])

        # Build markdown document
        lines = [
            f"# Chat Conversation: {doc_name}",
            f"\n**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n**Messages:** {len(messages)}",
            "\n---\n"
        ]

        # Format each message
        for i, msg in enumerate(messages, 1):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            timestamp = msg.get("timestamp", "")
            stopped = msg.get("stopped", False)

            # Role header with emoji
            role_display = "👤 **User**" if role == "user" else "🤖 **Assistant**"
            lines.append(f"\n## Message {i}: {role_display}\n")

            # Timestamp if available
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                    lines.append(f"*Time: {dt.strftime('%I:%M %p')}*\n")
                except:
                    pass

            # Message content
            lines.append(f"\n{content}\n")
            
            # Stopped indicator
            if stopped:
                lines.append("\n*⚠️ Generation was stopped by user*\n")
            
            lines.append("\n---\n")

        return "".join(lines)
    except Exception:
        return "# Export Error\nCould not export chat history."


# =============================================================================
# Session State Management
# =============================================================================

def init_session_state() -> None:
    """
    Initialize Streamlit session state with default values.
    
    Session state keys:
        - document_chats: Dict mapping doc names to chat histories
        - selected_document: Currently selected document name
        - uploader_key: Counter for file uploader widget
        - pending_toasts: Queue of toast notifications
        - last_uploaded_files: Track uploaded files to prevent duplicates
        - is_generating: Whether LLM is currently generating
        - stop_generation: Flag to signal generation stop
        - persistence_loaded: Whether persistence has been initialized
        
    Called once at application startup.
    """
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
    """
    Get chat history for currently selected document.
    
    Automatically loads from disk if not in session state.
    
    Returns:
        List of message dictionaries for current document
        
    Example:
        messages = get_current_chat()
        for msg in messages:
            st.chat_message(msg['role']).write(msg['content'])
    """
    doc = st.session_state.selected_document
    
    if doc:
        # Load from disk if not in memory
        if doc not in st.session_state.document_chats:
            saved_chat = load_chat_history_from_local(doc)
            if saved_chat:
                st.session_state.document_chats[doc] = saved_chat
            else:
                st.session_state.document_chats[doc] = []
    
    return st.session_state.document_chats.get(doc, [])


def add_message(message: Dict) -> None:
    """
    Add a message to current document's chat history.
    
    Automatically persists to disk after adding.
    
    Args:
        message: Message dictionary with at least 'role' and 'content'
        
    Example:
        add_message({
            "role": "user",
            "content": "What is AI?",
            "timestamp": datetime.now().isoformat()
        })
    """
    if doc := st.session_state.selected_document:
        # Add to session state
        st.session_state.document_chats.setdefault(doc, []).append(message)
        
        # Persist to disk
        save_chat_history_to_local(
            doc, 
            st.session_state.document_chats[doc]
        )


def clear_chat() -> None:
    """
    Clear chat history for currently selected document.
    
    Clears both session state and persistent storage.
    
    Example:
        if st.button("Clear Chat"):
            clear_chat()
            st.rerun()
    """
    if doc := st.session_state.selected_document:
        st.session_state.document_chats[doc] = []
        save_chat_history_to_local(doc, [])


def get_ui_state() -> Dict[str, bool]:
    """
    Get UI state for disabling widgets during generation.
    
    Returns:
        Dictionary with 'disabled' key
        
    Example:
        st.button("Upload", **get_ui_state())  # Disabled during generation
    """
    return {"disabled": st.session_state.is_generating}


# =============================================================================
# Toast Notification System
# =============================================================================

class ToastNotification:
    """
    Simple toast notification system for user feedback.
    
    Provides non-blocking notifications that appear at the top of the page.
    Supports different types: success, error, warning, info.
    
    Example:
        ToastNotification.show("File uploaded!", "success")
        ToastNotification.show("Connection error", "error")
        ToastNotification.render_pending()  # Display all pending toasts
    """
    
    ICONS = {
        "success": "✅",
        "error": "❌",
        "warning": "⚠️",
        "info": "ℹ️"
    }

    @staticmethod
    def show(message: str, toast_type: str = "info") -> None:
        """
        Queue a toast notification for display.
        
        Args:
            message: Notification message
            toast_type: Type of notification (success/error/warning/info)
            
        Note:
            Toasts are queued and displayed on next render_pending() call
        """
        if 'pending_toasts' not in st.session_state:
            st.session_state.pending_toasts = []
        
        st.session_state.pending_toasts.append({
            'message': message,
            'type': toast_type
        })

    @staticmethod
    def render_pending() -> None:
        """
        Display all pending toast notifications.
        
        Should be called early in the page render cycle to ensure
        toasts appear before other content.
        
        Clears the pending queue after displaying.
        """
        if 'pending_toasts' not in st.session_state or not st.session_state.pending_toasts:
            return
        
        for toast in st.session_state.pending_toasts:
            icon = ToastNotification.ICONS.get(toast['type'], "ℹ️")
            st.toast(f"{toast['message']}", icon=icon)
        
        # Clear queue
        st.session_state.pending_toasts = []


# =============================================================================
# UI Styling
# =============================================================================

def apply_custom_css() -> None:
    """
    Apply custom CSS styling to the Streamlit app.
    
    Customizations:
        - Rounded chat message containers
        - Primary button styling
        - Hover effects on buttons
        - Hidden footer and menu
        
    Called once during app initialization.
    """
    st.markdown("""
        <style>
        .stChatMessage {
            border-radius: 10px;
            padding: 10px;
            margin-bottom: 10px;
        }
        button[kind="primary"] {
            background-color: #4CAF50;
        }
        .stButton button {
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        footer {
            visibility: hidden;
        }
        #MainMenu {
            visibility: hidden;
        }
        </style>
    """, unsafe_allow_html=True)


# =============================================================================
# UI Components
# =============================================================================

def render_document_card(doc: Dict, api_client: APIClient) -> None:
    """
    Render a document card in the sidebar.
    
    Displays:
        - Document name with icon
        - Select button (changes style when selected)
        - Delete button
        - Document metadata (chunks, size, type)
        - Message count
        
    Args:
        doc: Document metadata dictionary
        api_client: API client for delete operations
        
    Example:
        for doc in documents:
            render_document_card(doc, api_client)
    """
    doc_name = doc['filename']
    is_selected = st.session_state.selected_document == doc_name
    
    col1, col2 = st.columns([6, 1])

    with col1:
        # Select button with conditional styling
        if st.button(
            f"{'📘' if is_selected else '📄'} **{doc_name}**",
            key=f"select_{doc_name}",
            use_container_width=True,
            type="primary" if is_selected else "secondary",
            **get_ui_state()
        ):
            # Select document and load chat history
            st.session_state.selected_document = doc_name
            saved_chat = load_chat_history_from_local(doc_name)
            if saved_chat:
                st.session_state.document_chats[doc_name] = saved_chat
            st.rerun()

    with col2:
        # Delete button
        if st.button(
            "✕",
            key=f"delete_{doc_name}",
            help="Delete document",
            **get_ui_state()
        ):
            status_code, response = api_client.delete_document(doc_name)
            
            if status_code == 200:
                # Clean up local state
                st.session_state.document_chats.pop(doc_name, None)
                delete_chat_history(doc_name)
                
                # Deselect if currently selected
                if st.session_state.selected_document == doc_name:
                    st.session_state.selected_document = None
                
                ToastNotification.show(f"Deleted {doc_name}", "success")
                st.rerun()
            else:
                ToastNotification.show(
                    response.get('message', 'Delete failed'),
                    "error"
                )

    # Show metadata for selected document
    if is_selected:
        st.caption(
            f"📊 {doc['chunks']} chunks • "
            f"{doc['size']:,} bytes • "
            f"{doc['type'].upper()}"
        )
        
        # Show message count if chat exists
        if msg_count := len(st.session_state.document_chats.get(doc_name, [])):
            st.caption(f"💬 {msg_count} messages")


def render_sidebar(api_client: APIClient) -> None:
    """
    Render the complete sidebar with documents and upload functionality.
    
    Sidebar sections:
        1. Document list with cards
        2. Upload section with file uploader
        3. Upload requirements info
        4. Clear chat button (if applicable)
        
    Args:
        api_client: API client for backend operations
        
    Called once per page render.
    """
    with st.sidebar:
        # Get current documents
        documents = api_client.get_documents()

        if documents:
            st.info(f"📊 {len(documents)} document(s) loaded")
            st.subheader("📖 Your Documents")
            
            # Render each document card
            for doc in documents:
                render_document_card(doc, api_client)
            
            if documents:
                st.caption(f"🤖 Using model: **{DEFAULT_MODEL}**")
        else:
            st.info("💡 No documents yet. Upload below!")

        st.markdown("---")
        st.subheader("📤 Upload Documents")

        # File uploader with dynamic key for reset capability
        uploaded_files = st.file_uploader(
            "Choose files",
            type=UI_ALLOWED_EXTENSIONS,
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
            label_visibility="collapsed",
            **get_ui_state()
        )

        # Handle file uploads
        if uploaded_files:
            current_file_names = [f.name for f in uploaded_files]
            
            # Only process if files changed
            if current_file_names != st.session_state.last_uploaded_files:
                st.session_state.last_uploaded_files = current_file_names
                
                for uploaded_file in uploaded_files:
                    # Check file size
                    file_size_mb = uploaded_file.size / (1024 * 1024)
                    if file_size_mb > MAX_FILE_SIZE_MB:
                        ToastNotification.show(
                            f"{uploaded_file.name} exceeds {MAX_FILE_SIZE_MB}MB limit",
                            "error"
                        )
                        continue

                    # Upload file
                    with st.spinner(f"Uploading {uploaded_file.name}..."):
                        status_code, response = api_client.upload_file(uploaded_file)
                        
                        if status_code == 200:
                            ToastNotification.show(
                                f"{uploaded_file.name} uploaded successfully",
                                "success"
                            )
                            # Auto-select newly uploaded document
                            st.session_state.selected_document = uploaded_file.name
                        else:
                            ToastNotification.show(
                                f"{response.get('message', 'Upload failed')}",
                                "error"
                            )
                
                # Reset uploader
                st.session_state.uploader_key += 1
                st.rerun()

        # Upload requirements info
        with st.expander("ℹ️ Upload Requirements", expanded=False):
            st.caption(f"**Formats:** {', '.join(UI_ALLOWED_EXTENSIONS).upper()}")
            st.caption(f"**Max size:** {MAX_FILE_SIZE_MB} MB per file")
            st.caption(f"**Multiple files:** Supported")

        # Clear chat button
        if st.session_state.selected_document and get_current_chat():
            st.markdown("---")
            if st.button("💬 Clear Chat", use_container_width=True, **get_ui_state()):
                clear_chat()
                st.rerun()


# =============================================================================
# Streaming and Chat Rendering
# =============================================================================

async def process_stream(
    api_client: APIClient,
    prompt: str,
    thinking_placeholder,
    response_placeholder
) -> Tuple[str, bool]:
    """
    Process streaming response from backend.
    
    Handles:
        - Displaying thinking message
        - Streaming content chunks
        - User interruption
        - Error handling
        - Final response display
        
    Args:
        api_client: API client instance
        prompt: User's question
        thinking_placeholder: Streamlit placeholder for thinking message
        response_placeholder: Streamlit placeholder for response
        
    Returns:
        Tuple of (response_text, was_stopped)
        
    Example:
        response, stopped = await process_stream(
            api_client,
            "What is AI?",
            thinking_placeholder,
            response_placeholder
        )
    """
    response = ""
    stopped = False
    
    try:
        # Stream response chunks
        async for data in api_client.query_stream(prompt):
            # Check for stop signal
            if st.session_state.stop_generation:
                stopped = True
                thinking_placeholder.empty()
                response += "\n\n*[Interrupted by user]*" if response else "*[Interrupted]*"
                response_placeholder.markdown(response)
                break

            # Handle content chunks
            if data.get('type') == 'content':
                thinking_placeholder.empty()
                response += data.get('content', '')
                # Display with blinking cursor effect
                response_placeholder.markdown(response + "▌")
                
            # Handle completion
            elif data.get('type') == 'done':
                response_placeholder.markdown(response)
                
            # Handle errors
            elif data.get('type') == 'error':
                thinking_placeholder.empty()
                error = f"❌ Error: {data.get('message', 'Unknown error')}"
                response_placeholder.error(error)
                response = error
                break

        # Clear thinking message and finalize response
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
    """
    Render export buttons for JSON and Markdown formats.
    
    Creates download buttons with file size information.
    Buttons are disabled during response generation.
    
    Args:
        doc_name: Name of the document whose chat is being exported
        
    Example:
        render_export_buttons("report.pdf")
        # Shows: [📄 Export JSON] [📝 Export MD]
    """
    col1, col2, _ = st.columns([1.5, 1.5, 7])
    
    # Generate timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Clean document name for filename
    clean_doc_name = doc_name.replace('.pdf', '').replace('.txt', '').replace('.docx', '')
    
    # Check if currently generating
    is_streaming = st.session_state.is_generating

    with col1:
        # JSON export button
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
        # Markdown export button
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
    """
    Render the chat message history.
    
    Displays all messages with appropriate formatting:
    - User messages with 👤 avatar
    - Assistant messages with 🤖 avatar
    - Timestamps formatted as time only
    - Stop indicators for interrupted generations
    
    Args:
        messages: List of message dictionaries
        
    Message Format:
        {
            "role": "user" | "assistant",
            "content": "message text",
            "timestamp": "2024-01-01T12:00:00" (optional),
            "stopped": True (optional, for interrupted messages)
        }
        
    Example:
        messages = get_current_chat()
        render_chat_history(messages)
    """
    for msg in messages:
        role = msg["role"]
        avatar = "👤" if role == "user" else "🤖"
        
        with st.chat_message(role, avatar=avatar):
            # Display message content
            st.markdown(msg["content"])
            
            # Display timestamp if available
            if timestamp := msg.get("timestamp", ""):
                try:
                    dt = datetime.fromisoformat(timestamp)
                    st.caption(f"🕒 {dt.strftime('%I:%M %p')}")
                except:
                    pass
            
            # Display stop indicator if message was interrupted
            if msg.get("stopped"):
                st.caption("⚠️ Generation was stopped")


def render_chat(api_client: APIClient, health_data: Optional[Dict] = None) -> None:
    """
    Render the main chat interface.
    
    This is the central component that handles:
    - Welcome screen (no documents)
    - Document selection prompt
    - Chat history display
    - Message input and processing
    - Streaming response handling
    - Export functionality
    
    Args:
        api_client: API client for backend communication
        health_data: Optional health check data from backend
        
    States Handled:
        1. No documents uploaded: Show welcome message
        2. No document selected: Show selection prompt
        3. Chat ready: Show history and input
        4. Generating response: Show streaming with stop button
        
    Example:
        api_client = APIClient("http://localhost:8000")
        is_healthy, health_data = api_client.health_check()
        render_chat(api_client, health_data)
    """
    # Check if system has no documents
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

    # Check if no document is selected
    if not st.session_state.selected_document:
        st.warning("📄 **Select a document** from the sidebar to start chatting.")
        return

    # Check Ollama service status
    if health_data:
        ollama = health_data.get('ollama_status', {})
        if not ollama.get('available'):
            ToastNotification.show("Ollama service unavailable", "warning")

    # Get chat history for current document
    chat_history = get_current_chat()

    # Show export buttons if chat history exists
    if chat_history:
        render_export_buttons(st.session_state.selected_document)

    # Display chat history (excluding last message if generating)
    messages_to_display = (
        chat_history[:-1] if st.session_state.is_generating 
        else chat_history
    )
    render_chat_history(messages_to_display)

    # Chat input for new questions
    prompt = st.chat_input(
        f"💭 Ask about {st.session_state.selected_document}...",
        **get_ui_state()
    )

    # Handle new user input
    if prompt and not st.session_state.is_generating:
        # Add user message to history
        add_message({
            "role": "user",
            "content": prompt,
            "timestamp": datetime.now().isoformat()
        })
        
        # Set generation flags
        st.session_state.is_generating = True
        st.session_state.stop_generation = False
        st.rerun()

    # Handle ongoing generation
    if st.session_state.is_generating:
        chat_history = get_current_chat()
        
        # Check if last message is from user (waiting for response)
        if chat_history and chat_history[-1]["role"] == "user":
            user_prompt = chat_history[-1]["content"]
            last_msg = chat_history[-1]

            # Display user message
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_prompt)
                
                # Show timestamp
                if timestamp := last_msg.get("timestamp", ""):
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        st.caption(f"🕒 {dt.strftime('%I:%M %p')}")
                    except:
                        pass

            # Display assistant response (streaming)
            with st.chat_message("assistant", avatar="🤖"):
                thinking_placeholder = st.empty()
                
                # Show random thinking message
                thinking_message = f"*{random.choice(THINKING_MESSAGES)}...*"
                thinking_placeholder.markdown(thinking_message)

                # Create columns for response and stop button
                col1, col2 = st.columns([6, 1])
                
                with col1:
                    response_placeholder = st.empty()
                    
                with col2:
                    # Stop generation button
                    if st.button(
                        "⏹️",
                        key="stop_inline",
                        help="Stop generation",
                        use_container_width=False
                    ):
                        st.session_state.stop_generation = True
                        st.rerun()

                # Process streaming response
                response, stopped = asyncio.run(
                    process_stream(
                        api_client,
                        user_prompt,
                        thinking_placeholder,
                        response_placeholder
                    )
                )

                # Add assistant response to history
                add_message({
                    "role": "assistant",
                    "content": response or "*[No response generated]*",
                    "timestamp": datetime.now().isoformat(),
                    "stopped": stopped
                })

                # Show notification if stopped
                if stopped:
                    ToastNotification.show("Generation stopped by user", "warning")

                # Reset generation flags
                st.session_state.is_generating = False
                st.session_state.stop_generation = False
                st.rerun()


# =============================================================================
# Main Application
# =============================================================================

def main():
    """
    Main application entry point.
    
    Orchestrates the entire Streamlit application:
    1. Configure page settings
    2. Initialize session state
    3. Apply custom styling
    4. Check backend health
    5. Render UI components
    
    Page Configuration:
        - Title: "Chat With Documents using AI"
        - Icon: 📚
        - Layout: Wide
        - Sidebar: Auto-collapse on mobile
        
    Application Flow:
        1. Initialize session state (first run only)
        2. Create API client
        3. Check backend health
        4. Display pending toast notifications
        5. Render page title
        6. Render sidebar (documents & upload)
        7. Render main chat interface
        
    Error Handling:
        - Backend unavailable: Show error and stop
        - Ollama unavailable: Show warning but continue
        - Upload errors: Show toast notifications
        - Query errors: Display in chat
        
    Example Usage:
        if __name__ == "__main__":
            main()
            
    To Run:
        streamlit run app.py
    """
    # Configure Streamlit page
    st.set_page_config(
        page_title="Chat With Documents using AI",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="auto"
    )
    
    # Initialize application state
    init_session_state()
    
    # Apply custom CSS styling
    apply_custom_css()

    # Create API client
    api_client = APIClient(API_BASE_URL)
    
    # Check backend health
    is_healthy, health_data = api_client.health_check()

    # Handle backend unavailability
    if not is_healthy:
        st.error("❌ Backend service unavailable. Please start the FastAPI server.")
        st.info("💡 Run: `python backend.py` in a separate terminal")
        st.stop()

    # Display pending toast notifications
    ToastNotification.render_pending()
    
    # Display main title
    st.markdown(
        '<h1 style="text-align: center;">📚 Chat With Documents</h1>',
        unsafe_allow_html=True
    )

    # Render sidebar (documents list and upload)
    render_sidebar(api_client)
    
    # Render main chat interface
    render_chat(api_client, health_data)


if __name__ == "__main__":
    """
    Application entry point.
    
    Run with: streamlit run app.py
    
    Requirements:
        - Backend must be running (python backend.py)
        - Ollama must be running (ollama serve)
        - Required models must be pulled:
          * ollama pull llama3.2
          * ollama pull nomic-embed-text
    """
    main()
