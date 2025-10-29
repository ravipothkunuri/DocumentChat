"""
Frontend Helper Functions

This is where we keep all the UI helper code:
- Managing chat history and app state
- Showing toast notifications
- Exporting conversations in different formats

Basically everything that makes the Streamlit UI work smoothly!
"""

import json
import streamlit as st
from datetime import datetime
from typing import List, Dict


# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def init_session_state():
    """
    Set up all the variables our app needs to remember.
    
    Streamlit "forgets" everything on each interaction, so we use session_state
    to keep track of important stuff like:
    - Which document you're looking at
    - Your chat history for each document
    - Whether the AI is currently typing
    - Notifications to show you
    
    Safe to call this multiple times - it only sets things up once!
    """
    defaults = {
        'document_chats': {},           # Separate chat for each document
        'selected_document': None,       # Which doc you're currently viewing
        'uploader_key': 0,              # Tricks Streamlit into resetting upload widget
        'pending_toasts': [],           # Queue of notifications to show
        'last_uploaded_files': [],      # Prevents duplicate upload processing
        'is_generating': False,         # Is AI currently responding?
        'stop_generation': False        # Did you hit the stop button?
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def get_current_chat() -> List[Dict]:
    """
    Get the chat history for whatever document you have open.
    
    Each message is a dict with:
    - role: "user" or "assistant"
    - content: the actual message text
    - timestamp: when it was sent
    - stopped: (optional) if you interrupted the AI
    
    Returns:
        List of messages, or empty list if no document is selected
    
    Usage:
        messages = get_current_chat()
        for msg in messages:
            print(f"{msg['role']}: {msg['content']}")
    """
    doc = st.session_state.selected_document
    
    # Create an empty chat for new documents
    if doc and doc not in st.session_state.document_chats:
        st.session_state.document_chats[doc] = []
    
    return st.session_state.document_chats.get(doc, [])


def add_message(message: Dict):
    """
    Add a new message to the current chat.
    
    Make sure your message has at least 'role' and 'content' keys!
    The timestamp is nice to include too.
    
    Example:
        add_message({
            "role": "user",
            "content": "What's this document about?",
            "timestamp": datetime.now().isoformat()
        })
    """
    if doc := st.session_state.selected_document:
        st.session_state.document_chats.setdefault(doc, []).append(message)


def clear_chat():
    """
    Wipe out the chat history for the current document.
    
    This can't be undone, so make sure the user really wants to do it!
    
    Usage:
        if st.button("Clear Chat", type="primary"):
            clear_chat()
            st.rerun()
    """
    if doc := st.session_state.selected_document:
        st.session_state.document_chats[doc] = []


# ============================================================================
# TOAST NOTIFICATION SYSTEM
# ============================================================================

class ToastNotification:
    """
    Simple notification system for the UI.
    
    Works like this:
    1. During processing, queue up notifications with show()
    2. At the start of your app, call render_pending() to display them
    
    Notifications auto-dismiss after a few seconds. Perfect for quick feedback
    like "Upload successful!" or "Something went wrong :("
    
    Example:
        # Somewhere in your code
        ToastNotification.show("Saved successfully!", "success")
        
        # At the top of app.py
        ToastNotification.render_pending()
    """
    
    @staticmethod
    def show(message: str, toast_type: str = "info"):
        """
        Queue up a notification to show the user.
        
        Types:
        - "success": Green checkmark (yay!)
        - "error": Red X (oops!)
        - "warning": Yellow warning triangle
        - "info": Blue info icon (FYI)
        
        Example:
            ToastNotification.show("Document uploaded!", "success")
            ToastNotification.show("Oops, try again", "error")
        """
        if 'pending_toasts' not in st.session_state:
            st.session_state.pending_toasts = []
        
        st.session_state.pending_toasts.append({
            'message': message,
            'type': toast_type
        })
    
    @staticmethod
    def render_pending():
        """
        Show all queued notifications.
        
        Call this once at the start of your app (after init_session_state).
        It'll display everything in the queue, then clear it.
        
        Example:
            # In app.py
            init_session_state()
            ToastNotification.render_pending()  # ← Do this!
            st.title("My App")
        """
        if 'pending_toasts' not in st.session_state or not st.session_state.pending_toasts:
            return
        
        # Pick the right emoji for each type
        icon_map = {
            "success": "✅",
            "error": "❌",
            "warning": "⚠️",
            "info": "ℹ️"
        }
        
        # Show all the notifications
        for toast in st.session_state.pending_toasts:
            icon = icon_map.get(toast['type'], "ℹ️")
            st.toast(f"{toast['message']}", icon=icon)
        
        # Clear the queue
        st.session_state.pending_toasts = []


# ============================================================================
# EXPORT UTILITIES
# ============================================================================

def export_to_json(messages: List[Dict], document_name: str) -> str:
    """
    Export your chat as a JSON file.
    
    Great for:
    - Keeping a backup
    - Processing with other tools
    - Sharing with teammates
    
    The JSON includes metadata like when you exported it and how many
    messages there are.
    
    Args:
        messages: Your chat history
        document_name: Which document this chat is about
        
    Returns:
        A nice formatted JSON string
    
    Example:
        json_str = export_to_json(get_current_chat(), "report.pdf")
        # Now you can save it or download it!
    """
    export_data = {
        "document": document_name,
        "exported_at": datetime.now().isoformat(),
        "message_count": len(messages),
        "conversation": messages
    }
    
    # Pretty print with 2-space indents
    return json.dumps(export_data, indent=2, ensure_ascii=False)


def export_to_markdown(messages: List[Dict], document_name: str) -> str:
    """
    Export your chat as a readable Markdown file.
    
    Perfect for:
    - Creating documentation
    - Sharing with non-technical folks
    - Just reading through the conversation
    
    The markdown includes:
    - A nice header with the document name
    - Timestamp for each message
    - Clear indication of who said what
    - Notes about interrupted responses
    
    Args:
        messages: Your chat history
        document_name: Which document this chat is about
        
    Returns:
        Markdown-formatted text
    
    Example:
        md = export_to_markdown(get_current_chat(), "report.pdf")
        with open("chat.md", "w") as f:
            f.write(md)
    """
    lines = [
        f"# Chat Conversation: {document_name}",
        f"\n**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n**Messages:** {len(messages)}",
        "\n---\n"
    ]
    
    # Format each message nicely
    for i, msg in enumerate(messages, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")
        stopped = msg.get("stopped", False)
        
        # Add emoji and formatting based on who's talking
        role_display = "👤 **User**" if role == "user" else "🤖 **Assistant**"
        
        lines.append(f"\n## Message {i}: {role_display}\n")
        
        # Add timestamp if we have one
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp)
                time_str = dt.strftime("%I:%M %p")
                lines.append(f"*Time: {time_str}*\n")
            except (ValueError, TypeError):
                pass  # Skip bad timestamps
        
        # The actual message
        lines.append(f"\n{content}\n")
        
        # Note if you stopped the AI mid-response
        if stopped:
            lines.append("\n*⚠️ You stopped this response*\n")
        
        lines.append("\n---\n")
    
    return "".join(lines)
