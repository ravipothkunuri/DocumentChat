"""Toast-style notifications for Streamlit UI."""
import streamlit as st


class ToastNotification:
    ICONS = {"success": "✅", "error": "❌", "warning": "⚠️", "info": "ℹ️"}

    @staticmethod
    def show(message: str, toast_type: str = "info") -> None:
        if 'pending_toasts' not in st.session_state:
            st.session_state.pending_toasts = []
        st.session_state.pending_toasts.append({'message': message, 'type': toast_type})

    @staticmethod
    def render_pending() -> None:
        if 'pending_toasts' not in st.session_state or not st.session_state.pending_toasts:
            return
        for toast in st.session_state.pending_toasts:
            icon = ToastNotification.ICONS.get(toast['type'], "ℹ️")
            st.toast(f"{toast['message']}", icon=icon)
        st.session_state.pending_toasts = []


