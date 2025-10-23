"""
Toast notification system
"""
import streamlit as st


class ToastNotification:
    """Independent toast notification system using session state flags"""
    
    @staticmethod
    def show(message: str, toast_type: str = "info"):
        """Queue a toast notification to be displayed after rerun"""
        if 'pending_toasts' not in st.session_state:
            st.session_state.pending_toasts = []
        
        st.session_state.pending_toasts.append({
            'message': message,
            'type': toast_type
        })
    
    @staticmethod
    def render_pending():
        """Render all pending toasts and clear the queue"""
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
        
        # Clear the queue
        st.session_state.pending_toasts = []
