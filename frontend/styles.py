"""
Minimal CSS styling for the RAG application
Relies on Streamlit's built-in components for most styling
"""
import streamlit as st


def apply_custom_css():
    """Apply minimal custom CSS - relies on Streamlit defaults"""
    st.markdown("""
    <style>
    /* Chat message alignment - user messages on right, assistant on left */
    .stChatMessage[data-testid="user-message"],
    .stChatMessage:has([data-testid*="user"]) {
        margin-left: auto !important;
        margin-right: 0 !important;
    }
    
    .stChatMessage[data-testid="assistant-message"],
    .stChatMessage:has([data-testid*="assistant"]) {
        margin-right: auto !important;
        margin-left: 0 !important;
    }
    
    /* Delete button styling - compact red button */
    button[key*="delete_"] {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.5) !important;
        color: #ef4444 !important;
        font-weight: 600 !important;
        min-height: 38px !important;
        width: 100% !important;
        max-width: 42px !important;
    }
    
    button[key*="delete_"]:hover {
        background: rgba(239, 68, 68, 0.2) !important;
        border-color: #ef4444 !important;
    }
    
    /* Stop button styling - red stop button */
    button[key="stop_inline"] {
        background: #ef4444 !important;
        color: white !important;
        border: none !important;
        font-size: 1.3rem !important;
        min-height: 40px !important;
    }
    
    button[key="stop_inline"]:hover {
        background: #dc2626 !important;
    }
    
    /* Optional: Hide Streamlit branding for cleaner interface */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
