"""
Frontend Helper Functions

This is where we keep all the UI helper code:
- Managing chat history and app state
- Showing toast notifications
- Exporting conversations in different formats

Basically everything that makes the Streamlit UI work smoothly!
"""

"""
Frontend Helper Functions - WITH CHAT PERSISTENCE

Add this to your existing utils.py (frontend)
"""

import json
import streamlit as st
from datetime import datetime
from pathlib import Path
from typing import List, Dict


# ============================================================================
# CHAT PERSISTENCE
# ============================================================================

CHAT_HISTORY_FILE = Path("config/chat_history.json")

def load_chat_history() -> Dict[str, List[Dict]]:
    """
    Load chat history from disk.
    
    Returns:
        Dictionary mapping document names to their chat histories
    """
    try:
        if CHAT_HISTORY_FILE.exists():
            with open(CHAT_HISTORY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"✅ Loaded chat history: {len(data)} documents")
                return data
        else:
            print("📝 No existing chat history found, starting fresh")
            return {}
    except Exception as e:
        print(f"⚠️ Error loading chat history: {e}")
        return {}


def save_chat_history(chat_data: Dict[str, List[Dict]]):
    """
    Save chat history to disk.
    
    Args:
        chat_data: Dictionary mapping document names to their chat histories
    """
    try:
        # Ensure directory exists
        CHAT_HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        with open(CHAT_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved chat history: {len(chat_data)} documents")
    except Exception as e:
        print(f"❌ Error saving chat history: {e}")


# ============================================================================
# SESSION STATE MANAGEMENT (UPDATED)
# ============================================================================

def init_session_state():
    """
    Set up all the variables our app needs to remember.
    NOW WITH PERSISTENT CHAT HISTORY!
    """
    # Load chat history from disk FIRST
    if 'document_chats' not in st.session_state:
        st.session_state.document_chats = load_chat_history()
    
    defaults = {
        'selected_document': None,
        'uploader_key': 0,
        'pending_toasts': [],
        'last_uploaded_files': [],
        'is_generating': False,
        'stop_generation': False
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_current_chat() -> List[Dict]:
    """
    Get the chat history for whatever document you have open.
    """
    doc = st.session_state.selected_document
    
    # Create an empty chat for new documents
    if doc and doc not in st.session_state.document_chats:
        st.session_state.document_chats[doc] = []
    
    return st.session_state.document_chats.get(doc, [])


def add_message(message: Dict):
    """
    Add a new message to the current chat.
    NOW AUTOMATICALLY SAVES TO DISK!
    """
    if doc := st.session_state.selected_document:
        st.session_state.document_chats.setdefault(doc, []).append(message)
        # Auto-save after each message
        save_chat_history(st.session_state.document_chats)


def clear_chat():
    """
    Wipe out the chat history for the current document.
    NOW PERSISTS THE DELETION!
    """
    if doc := st.session_state.selected_document:
        st.session_state.document_chats[doc] = []
        # Save the cleared state
        save_chat_history(st.session_state.document_chats)


def delete_document_chat(document_name: str):
    """
    Delete chat history when a document is deleted.
    Call this from sidebar.py when deleting a document.
    """
    if document_name in st.session_state.document_chats:
        del st.session_state.document_chats[document_name]
        save_chat_history(st.session_state.document_chats)
        print(f"🗑️ Deleted chat history for: {document_name}")


# ============================================================================
# TOAST NOTIFICATION SYSTEM (unchanged)
# ============================================================================

class ToastNotification:
    """Simple notification system for the UI."""
    
    @staticmethod
    def show(message: str, toast_type: str = "info"):
        """Queue up a notification to show the user."""
        if 'pending_toasts' not in st.session_state:
            st.session_state.pending_toasts = []
        
        st.session_state.pending_toasts.append({
            'message': message,
            'type': toast_type
        })
    
    @staticmethod
    def render_pending():
        """Show all queued notifications."""
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
        
        st.session_state.pending_toasts = []


# ============================================================================
# EXPORT UTILITIES (unchanged)
# ============================================================================

def export_to_json(messages: List[Dict], document_name: str) -> str:
    """Export your chat as a JSON file."""
    export_data = {
        "document": document_name,
        "exported_at": datetime.now().isoformat(),
        "message_count": len(messages),
        "conversation": messages
    }
    
    return json.dumps(export_data, indent=2, ensure_ascii=False)


def export_to_markdown(messages: List[Dict], document_name: str) -> str:
    """Export your chat as a readable Markdown file."""
    lines = [
        f"# Chat Conversation: {document_name}",
        f"\n**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n**Messages:** {len(messages)}",
        "\n---\n"
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
                time_str = dt.strftime("%I:%M %p")
                lines.append(f"*Time: {time_str}*\n")
            except (ValueError, TypeError):
                pass
        
        lines.append(f"\n{content}\n")
        
        if stopped:
            lines.append("\n*⚠️ You stopped this response*\n")
        
        lines.append("\n---\n")
    
    return "".join(lines)
