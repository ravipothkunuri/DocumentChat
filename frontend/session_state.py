"""
Enhanced session state management with conversation history support
"""
import streamlit as st
from typing import List, Dict
from datetime import datetime
import json


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
        'show_onboarding': True,
        'conversation_history': [],
        'selected_conversation': None,
        'show_doc_info': None,  # Changed from show_document_preview
        'suggested_questions': {},
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


def save_conversation():
    """Save current conversation to history"""
    doc = st.session_state.selected_document
    if doc and doc in st.session_state.document_chats:
        chat = st.session_state.document_chats[doc]
        if len(chat) > 0:
            conversation = {
                'id': f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'document': doc,
                'timestamp': datetime.now().isoformat(),
                'messages': chat.copy(),
                'title': chat[0]['content'][:50] + "..." if chat else "Untitled"
            }
            st.session_state.conversation_history.insert(0, conversation)
            # Keep only last 50 conversations
            st.session_state.conversation_history = st.session_state.conversation_history[:50]


def load_conversation(conversation_id: str):
    """Load a saved conversation"""
    for conv in st.session_state.conversation_history:
        if conv['id'] == conversation_id:
            st.session_state.selected_document = conv['document']
            st.session_state.document_chats[conv['document']] = conv['messages'].copy()
            st.session_state.selected_conversation = conversation_id
            return True
    return False


def delete_conversation(conversation_id: str):
    """Delete a saved conversation"""
    st.session_state.conversation_history = [
        conv for conv in st.session_state.conversation_history 
        if conv['id'] != conversation_id
    ]


def export_chat_json(chat_history: List[Dict]) -> str:
    """Export chat history as JSON"""
    export_data = {
        'document': st.session_state.selected_document,
        'exported_at': datetime.now().isoformat(),
        'messages': chat_history
    }
    return json.dumps(export_data, indent=2)


def export_chat_markdown(chat_history: List[Dict]) -> str:
    """Export chat history as Markdown"""
    lines = [
        f"# Chat History: {st.session_state.selected_document}",
        f"**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        ""
    ]
    
    for msg in chat_history:
        role = "**You:**" if msg['role'] == 'user' else "**Assistant:**"
        timestamp = msg.get('timestamp', '')
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp)
                timestamp = dt.strftime('%H:%M:%S')
            except:
                timestamp = ''
        
        lines.append(f"### {role} {f'*({timestamp})*' if timestamp else ''}")
        lines.append("")
        lines.append(msg['content'])
        
        if msg.get('sources'):
            lines.append("")
            lines.append(f"*Sources: {', '.join(set(msg['sources']))}*")
        
        lines.append("")
        lines.append("---")
        lines.append("")
    
    return "\n".join(lines)


def get_suggested_questions(document_name: str) -> List[str]:
    """Get suggested questions for a document"""
    if document_name in st.session_state.suggested_questions:
        return st.session_state.suggested_questions[document_name]
    
    # Default suggestions based on document type
    default_suggestions = [
        "What is this document about?",
        "Summarize the key points",
        "What are the main topics covered?",
        "Are there any important conclusions?"
    ]
    
    st.session_state.suggested_questions[document_name] = default_suggestions
    return default_suggestions


def set_suggested_questions(document_name: str, questions: List[str]):
    """Set custom suggested questions for a document"""
    st.session_state.suggested_questions[document_name] = questions
