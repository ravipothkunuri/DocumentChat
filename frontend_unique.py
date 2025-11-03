"""
DocumentChat Frontend - State Machine Driven UI with Action Queue
Unique implementation using FSM, Command Pattern, and Reactive Updates
"""
import json
import asyncio
import random
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field

import httpx
import streamlit as st

# Configuration
API_BASE_URL = "http://localhost:8000"
MAX_FILE_SIZE_MB = 20
ALLOWED_EXTENSIONS = ['pdf', 'txt', 'docx']
LLM_MODEL = "llama3.2"
THINKING_MESSAGES = [
    "🤔 Processing your query...", "💭 Consulting documents...", "📖 Analyzing content...",
    "🔍 Searching knowledge base...", "⚡ Generating response...", "🧠 Understanding context..."
]

# ============================================================================
# FINITE STATE MACHINE - Explicit chat flow states
# ============================================================================

class ChatState(Enum):
    """Possible states of chat interface"""
    IDLE = "idle"
    WAITING_INPUT = "waiting_input"
    PROCESSING_QUERY = "processing_query"
    STREAMING_RESPONSE = "streaming_response"
    ERROR = "error"

class ConversationStateMachine:
    """Manages conversation state transitions"""
    
    def __init__(self):
        self.current_state = ChatState.IDLE
        self.transitions = {
            ChatState.IDLE: [ChatState.WAITING_INPUT],
            ChatState.WAITING_INPUT: [ChatState.PROCESSING_QUERY, ChatState.IDLE],
            ChatState.PROCESSING_QUERY: [ChatState.STREAMING_RESPONSE, ChatState.ERROR],
            ChatState.STREAMING_RESPONSE: [ChatState.WAITING_INPUT, ChatState.ERROR, ChatState.IDLE],
            ChatState.ERROR: [ChatState.WAITING_INPUT, ChatState.IDLE]
        }
    
    def can_transition_to(self, new_state: ChatState) -> bool:
        """Check if transition is valid"""
        return new_state in self.transitions.get(self.current_state, [])
    
    def transition_to(self, new_state: ChatState) -> bool:
        """Attempt state transition"""
        if self.can_transition_to(new_state):
            self.current_state = new_state
            return True
        return False
    
    def is_busy(self) -> bool:
        """Check if system is processing"""
        return self.current_state in [ChatState.PROCESSING_QUERY, ChatState.STREAMING_RESPONSE]

# ============================================================================
# ACTION QUEUE SYSTEM - Command pattern for operations
# ============================================================================

@dataclass
class Action:
    """Base action command"""
    action_type: str
    payload: Dict
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class ActionQueue:
    """Queue for managing UI actions"""
    
    def __init__(self):
        self.pending_actions: List[Action] = []
        self.handlers: Dict[str, Callable] = {}
    
    def register_handler(self, action_type: str, handler: Callable):
        """Register action handler"""
        self.handlers[action_type] = handler
    
    def enqueue(self, action: Action):
        """Add action to queue"""
        self.pending_actions.append(action)
    
    def process_all(self):
        """Process all pending actions"""
        while self.pending_actions:
            action = self.pending_actions.pop(0)
            handler = self.handlers.get(action.action_type)
            if handler:
                try:
                    handler(action)
                except Exception as e:
                    st.error(f"Action failed: {e}")

# ============================================================================
# NOTIFICATION SYSTEM - Observer pattern for reactive updates
# ============================================================================

class NotificationLevel(Enum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"

@dataclass
class Notification:
    message: str
    level: NotificationLevel
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

class NotificationManager:
    """Manages user notifications"""
    
    def __init__(self):
        self.notifications: List[Notification] = []
    
    def notify(self, message: str, level: NotificationLevel = NotificationLevel.INFO):
        """Add notification"""
        self.notifications.append(Notification(message, level))
    
    def render_all(self):
        """Display all pending notifications"""
        icons = {
            NotificationLevel.INFO: "ℹ️",
            NotificationLevel.SUCCESS: "✅",
            NotificationLevel.WARNING: "⚠️",
            NotificationLevel.ERROR: "❌"
        }
        for notif in self.notifications:
            st.toast(notif.message, icon=icons.get(notif.level, "ℹ️"))
        self.notifications.clear()

# ============================================================================
# DOCUMENT SESSION MANAGER - Advanced conversation tracking
# ============================================================================

@dataclass
class Message:
    role: str
    content: str
    timestamp: str
    metadata: Dict = field(default_factory=dict)

class DocumentSession:
    """Manages conversation session for a document"""
    
    def __init__(self, document_name: str):
        self.document_name = document_name
        self.messages: List[Message] = []
        self.created_at = datetime.now().isoformat()
        self.message_count = 0
    
    def add_message(self, role: str, content: str, **metadata):
        """Add message to session"""
        msg = Message(
            role=role,
            content=content,
            timestamp=datetime.now().isoformat(),
            metadata=metadata
        )
        self.messages.append(msg)
        self.message_count += 1
    
    def get_messages(self) -> List[Message]:
        """Get all messages"""
        return self.messages
    
    def clear(self):
        """Clear session messages"""
        self.messages.clear()
        self.message_count = 0
    
    def export_to_dict(self) -> Dict:
        """Export session as dictionary"""
        return {
            "document": self.document_name,
            "created_at": self.created_at,
            "message_count": self.message_count,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "metadata": msg.metadata
                }
                for msg in self.messages
            ]
        }

class SessionRegistry:
    """Registry of all document sessions"""
    
    def __init__(self):
        self.sessions: Dict[str, DocumentSession] = {}
    
    def get_or_create(self, document_name: str) -> DocumentSession:
        """Get existing session or create new one"""
        if document_name not in self.sessions:
            self.sessions[document_name] = DocumentSession(document_name)
        return self.sessions[document_name]
    
    def remove(self, document_name: str):
        """Remove session"""
        self.sessions.pop(document_name, None)
    
    def list_active_sessions(self) -> List[str]:
        """Get list of documents with active sessions"""
        return list(self.sessions.keys())

# ============================================================================
# API CLIENT WITH CONNECTION POOLING
# ============================================================================

class BackendAPIClient:
    """HTTP client for backend communication"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.sync_client = httpx.Client(timeout=60.0)
        self.async_client = httpx.AsyncClient(timeout=60.0)
    
    def _execute_request(self, method: str, endpoint: str, **kwargs) -> Tuple[int, Dict]:
        """Execute HTTP request"""
        try:
            response = self.sync_client.request(method, f"{self.base_url}{endpoint}", **kwargs)
            return response.status_code, response.json() if response.content else {}
        except Exception as e:
            return 500, {"message": str(e)}
    
    def check_health(self) -> Tuple[bool, Optional[Dict]]:
        """Health check"""
        try:
            response = self.sync_client.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200, response.json() if response.is_success else None
        except:
            return False, None
    
    def fetch_documents(self) -> List[Dict]:
        """Get all documents"""
        status, data = self._execute_request('GET', '/documents')
        return data if status == 200 else []
    
    def upload_document(self, file) -> Tuple[int, Dict]:
        """Upload file"""
        try:
            files = {"file": (file.name, file, file.type)}
            response = self.sync_client.post(f"{self.base_url}/upload", files=files, timeout=60)
            return response.status_code, response.json() if response.content else {}
        except Exception as e:
            return 500, {"message": str(e)}
    
    def remove_document(self, filename: str) -> Tuple[int, Dict]:
        """Delete document"""
        return self._execute_request('DELETE', f'/documents/{filename}', timeout=30)
    
    async def stream_query(self, question: str, top_k: int = 4):
        """Stream query response"""
        try:
            payload = {"question": question, "stream": True, "top_k": top_k, "model": LLM_MODEL}
            async with self.async_client.stream('POST', f"{self.base_url}/query", json=payload, timeout=120.0) as response:
                if response.status_code == 200:
                    async for line in response.aiter_lines():
                        if line and line.startswith('data: '):
                            try:
                                yield json.loads(line[6:])
                            except:
                                continue
                else:
                    yield {"type": "error", "message": f"Query failed: {response.status_code}"}
        except httpx.ReadTimeout:
            yield {"type": "error", "message": "Request timed out"}
        except Exception as e:
            yield {"type": "error", "message": str(e)}
    
    def cleanup(self):
        """Close connections"""
        try:
            self.sync_client.close()
        except:
            pass

# ============================================================================
# EXPORT STRATEGIES
# ============================================================================

class ExportFormatter:
    """Formats conversation for export"""
    
    @staticmethod
    def to_json(session: DocumentSession) -> str:
        """Export as JSON"""
        return json.dumps(session.export_to_dict(), indent=2, ensure_ascii=False)
    
    @staticmethod
    def to_markdown(session: DocumentSession) -> str:
        """Export as Markdown"""
        lines = [
            f"# Conversation: {session.document_name}",
            f"\n**Created:** {session.created_at}",
            f"**Messages:** {session.message_count}\n",
            "\n---\n"
        ]
        
        for i, msg in enumerate(session.get_messages(), 1):
            role_icon = "👤 User" if msg.role == "user" else "🤖 Assistant"
            lines.append(f"\n## {i}. {role_icon}\n")
            
            try:
                dt = datetime.fromisoformat(msg.timestamp)
                lines.append(f"*{dt.strftime('%I:%M %p')}*\n\n")
            except:
                pass
            
            lines.append(f"{msg.content}\n")
            
            if msg.metadata.get("stopped"):
                lines.append("\n*⚠️ Response interrupted*\n")
            
            lines.append("\n---\n")
        
        return "".join(lines)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def initialize_application_state():
    """Initialize Streamlit session state"""
    if 'state_machine' not in st.session_state:
        st.session_state.state_machine = ConversationStateMachine()
    if 'action_queue' not in st.session_state:
        st.session_state.action_queue = ActionQueue()
    if 'notification_manager' not in st.session_state:
        st.session_state.notification_manager = NotificationManager()
    if 'session_registry' not in st.session_state:
        st.session_state.session_registry = SessionRegistry()
    if 'active_document' not in st.session_state:
        st.session_state.active_document = None
    if 'upload_counter' not in st.session_state:
        st.session_state.upload_counter = 0
    if 'last_upload_cache' not in st.session_state:
        st.session_state.last_upload_cache = []
    if 'stop_requested' not in st.session_state:
        st.session_state.stop_requested = False

# ============================================================================
# UI STYLING
# ============================================================================

def apply_custom_styling():
    """Apply custom CSS"""
    st.markdown("""
    <style>
    .stChatMessage[data-testid="user-message"], .stChatMessage:has([data-testid*="user"]) {
        margin-left: auto !important; margin-right: 0 !important;
    }
    .stChatMessage[data-testid="assistant-message"], .stChatMessage:has([data-testid*="assistant"]) {
        margin-right: auto !important; margin-left: 0 !important;
    }
    button[key*="delete_"] {
        background: rgba(239, 68, 68, 0.1) !important; border: 1px solid rgba(239, 68, 68, 0.5) !important;
        color: #ef4444 !important; font-weight: 600 !important; min-height: 38px !important;
        width: 100% !important; max-width: 42px !important;
    }
    button[key*="delete_"]:hover {
        background: rgba(239, 68, 68, 0.2) !important; border-color: #ef4444 !important;
    }
    button[key="stop_generation"] {
        background: #ef4444 !important; color: white !important; border: none !important;
        font-size: 1.3rem !important; min-height: 40px !important;
    }
    button[key="stop_generation"]:hover { background: #dc2626 !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_document_selector(doc: Dict, api_client: BackendAPIClient):
    """Render single document card"""
    is_active = st.session_state.active_document == doc['filename']
    is_busy = st.session_state.state_machine.is_busy()
    
    col1, col2 = st.columns([6, 1])
    
    with col1:
        if st.button(
            f"{'📘' if is_active else '📄'} **{doc['filename']}**",
            key=f"select_{doc['filename']}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
            disabled=is_busy
        ):
            st.session_state.active_document = doc['filename']
            st.session_state.state_machine.transition_to(ChatState.WAITING_INPUT)
            st.rerun()
    
    with col2:
        if st.button("✕", key=f"delete_{doc['filename']}", help="Delete", disabled=is_busy):
            status, response = api_client.remove_document(doc['filename'])
            if status == 200:
                st.session_state.session_registry.remove(doc['filename'])
                if st.session_state.active_document == doc['filename']:
                    st.session_state.active_document = None
                    st.session_state.state_machine.transition_to(ChatState.IDLE)
                st.session_state.notification_manager.notify(f"Deleted {doc['filename']}", NotificationLevel.SUCCESS)
                st.rerun()
            else:
                st.session_state.notification_manager.notify(response.get('message', 'Delete failed'), NotificationLevel.ERROR)
    
    if is_active:
        st.caption(f"📊 {doc['chunks']} chunks • {doc['size']:,} bytes • {doc['type'].upper()}")
        session = st.session_state.session_registry.get_or_create(doc['filename'])
        if session.message_count > 0:
            st.caption(f"💬 {session.message_count} messages")

def render_sidebar_panel(api_client: BackendAPIClient):
    """Render sidebar with documents and upload"""
    with st.sidebar:
        documents = api_client.fetch_documents()
        
        if documents:
            st.info(f"📊 {len(documents)} document(s) available")
        
        st.subheader("📖 Document Library")
        
        for doc in documents:
            render_document_selector(doc, api_client)
        
        if documents:
            st.caption(f"🤖 Model: **{LLM_MODEL}**")
        else:
            st.info("💡 Upload documents to begin")
        
        st.markdown("---")
        st.subheader("📤 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Select files",
            type=ALLOWED_EXTENSIONS,
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.upload_counter}",
            label_visibility="collapsed",
            disabled=st.session_state.state_machine.is_busy()
        )
        
        if uploaded_files:
            current_names = [f.name for f in uploaded_files]
            if current_names != st.session_state.last_upload_cache:
                st.session_state.last_upload_cache = current_names
                
                for file in uploaded_files:
                    if file.size / (1024 * 1024) > MAX_FILE_SIZE_MB:
                        st.session_state.notification_manager.notify(
                            f"{file.name} exceeds {MAX_FILE_SIZE_MB}MB", NotificationLevel.ERROR
                        )
                        continue
                    
                    with st.spinner(f"Uploading {file.name}..."):
                        status, response = api_client.upload_document(file)
                        if status == 200:
                            st.session_state.notification_manager.notify(
                                f"{file.name} uploaded successfully", NotificationLevel.SUCCESS
                            )
                            st.session_state.active_document = file.name
                        else:
                            st.session_state.notification_manager.notify(
                                response.get('message', 'Upload failed'), NotificationLevel.ERROR
                            )
                
                st.session_state.upload_counter += 1
                st.rerun()
        
        with st.expander("ℹ️ Information", expanded=False):
            st.caption(f"**Formats:** {', '.join(ALLOWED_EXTENSIONS).upper()}")
            st.caption(f"**Max size:** {MAX_FILE_SIZE_MB} MB")
        
        if st.session_state.active_document:
            session = st.session_state.session_registry.get_or_create(st.session_state.active_document)
            if session.message_count > 0:
                st.markdown("---")
                if st.button("💬 Clear Chat", use_container_width=True, 
                           disabled=st.session_state.state_machine.is_busy()):
                    session.clear()
                    st.rerun()

async def execute_query_stream(api_client: BackendAPIClient, question: str, 
                               thinking_placeholder, response_placeholder) -> Tuple[str, bool]:
    """Execute streaming query"""
    response_text = ""
    was_stopped = False
    
    try:
        async for data in api_client.stream_query(question):
            if st.session_state.stop_requested:
                was_stopped = True
                thinking_placeholder.empty()
                response_text += "\n\n*[Stopped by user]*" if response_text else "*[Stopped]*"
                response_placeholder.markdown(response_text)
                break
            
            if data.get('type') == 'content':
                thinking_placeholder.empty()
                response_text += data.get('content', '')
                response_placeholder.markdown(response_text + "▌")
            elif data.get('type') == 'done':
                response_placeholder.markdown(response_text)
            elif data.get('type') == 'error':
                thinking_placeholder.empty()
                error_msg = f"❌ Error: {data.get('message', 'Unknown')}"
                response_placeholder.error(error_msg)
                response_text = error_msg
                break
        
        thinking_placeholder.empty()
        if response_text:
            response_placeholder.markdown(response_text)
            
    except Exception as e:
        thinking_placeholder.empty()
        error_msg = f"❌ Error: {str(e)}"
        response_placeholder.error(error_msg)
        response_text = error_msg
        st.session_state.notification_manager.notify(f"Error: {str(e)}", NotificationLevel.ERROR)
    
    return response_text, was_stopped

def render_export_controls(session: DocumentSession):
    """Render export buttons"""
    col1, col2, _ = st.columns([1.5, 1.5, 7])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_name = session.document_name.rsplit('.', 1)[0]
    is_busy = st.session_state.state_machine.is_busy()
    
    with col1:
        st.download_button(
            "📄 Export JSON",
            ExportFormatter.to_json(session),
            f"{clean_name}_chat_{timestamp}.json",
            "application/json",
            use_container_width=True,
            disabled=is_busy
        )
    
    with col2:
        st.download_button(
            "📝 Export MD",
            ExportFormatter.to_markdown(session),
            f"{clean_name}_chat_{timestamp}.md",
            "text/markdown",
            use_container_width=True,
            disabled=is_busy
        )

def render_conversation_interface(api_client: BackendAPIClient, health_info: Optional[Dict]):
    """Render main chat interface"""
    
    # Check if documents exist
    if health_info and health_info.get('statistics', {}).get('documents', 0) == 0:
        st.info("👋 **Welcome!** Upload documents to start chatting.")
        with st.expander("📖 Getting Started", expanded=True):
            st.markdown("""
            1. **Upload** 📤 - Add PDF, TXT, or DOCX files
            2. **Select** 💬 - Choose a document to chat with
            3. **Ask** 💭 - Type your questions
            4. **Explore** 🎯 - Get AI-powered answers
            """)
        return
    
    # Check if document is selected
    if not st.session_state.active_document:
        st.warning("📄 **Select a document** from the sidebar to begin.")
        return
    
    # Check Ollama status
    if health_info:
        if health_info.get('services', {}).get('ollama') != 'available':
            st.session_state.notification_manager.notify("Ollama service unavailable", NotificationLevel.WARNING)
    
    # Get current session
    session = st.session_state.session_registry.get_or_create(st.session_state.active_document)
    
    # Export controls
    if session.message_count > 0:
        render_export_controls(session)
    
    # Display conversation history
    messages_to_show = session.get_messages()
    if st.session_state.state_machine.current_state == ChatState.STREAMING_RESPONSE:
        messages_to_show = messages_to_show[:-1]
    
    for msg in messages_to_show:
        avatar = "👤" if msg.role == "user" else "🤖"
        with st.chat_message(msg.role, avatar=avatar):
            st.markdown(msg.content)
            try:
                dt = datetime.fromisoformat(msg.timestamp)
                st.caption(f"🕒 {dt.strftime('%I:%M %p')}")
            except:
                pass
            if msg.metadata.get("stopped"):
                st.caption("⚠️ Generation stopped")
    
    # Chat input
    user_input = st.chat_input(
        f"💭 Ask about {st.session_state.active_document}...",
        disabled=st.session_state.state_machine.is_busy()
    )
    
    if user_input and st.session_state.state_machine.current_state == ChatState.WAITING_INPUT:
        session.add_message("user", user_input)
        st.session_state.state_machine.transition_to(ChatState.PROCESSING_QUERY)
        st.session_state.stop_requested = False
        st.rerun()
    
    # Process query if in processing state
    if st.session_state.state_machine.current_state == ChatState.PROCESSING_QUERY:
        messages = session.get_messages()
        if messages and messages[-1].role == "user":
            user_question = messages[-1].content
            last_msg = messages[-1]
            
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_question)
                try:
                    dt = datetime.fromisoformat(last_msg.timestamp)
                    st.caption(f"🕒 {dt.strftime('%I:%M %p')}")
                except:
                    pass
            
            st.session_state.state_machine.transition_to(ChatState.STREAMING_RESPONSE)
            
            with st.chat_message("assistant", avatar="🤖"):
                thinking_ph = st.empty()
                thinking_ph.markdown(f"*{random.choice(THINKING_MESSAGES)}*")
                
                col1, col2 = st.columns([6, 1])
                with col1:
                    response_ph = st.empty()
                with col2:
                    if st.button("⏹️", key="stop_generation", help="Stop", use_container_width=True):
                        st.session_state.stop_requested = True
                        st.rerun()
                
                response, stopped = asyncio.run(
                    execute_query_stream(api_client, user_question, thinking_ph, response_ph)
                )
                
                session.add_message("assistant", response or "*[No response]*", stopped=stopped)
                
                if stopped:
                    st.session_state.notification_manager.notify("Generation stopped", NotificationLevel.WARNING)
                
                st.session_state.state_machine.transition_to(ChatState.WAITING_INPUT)
                st.session_state.stop_requested = False
                st.rerun()

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Application entry point"""
    st.set_page_config(
        page_title="DocumentChat AI System",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="auto"
    )
    
    initialize_application_state()
    apply_custom_styling()
    
    api_client = BackendAPIClient(API_BASE_URL)
    
    is_healthy, health_data = api_client.check_health()
    if not is_healthy:
        st.error("❌ Backend service unavailable. Start the FastAPI server.")
        st.stop()
    
    st.session_state.notification_manager.render_all()
    st.markdown('<h1>📚 <b>DocumentChat AI System</b></h1>', unsafe_allow_html=True)
    
    render_sidebar_panel(api_client)
    render_conversation_interface(api_client, health_data)

if __name__ == "__main__":
    main()
