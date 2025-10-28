"""
Relies on Streamlit's built-in components for most styling
"""
import streamlit as st


def apply_custom_css():
    """Apply minimal custom CSS - relies on Streamlit defaults"""
    st.markdown("""
    <style>
    /* Wider sidebar for better document list display */
    [data-testid="stSidebar"] {
        min-width: 380px !important;
        max-width: 380px !important;
    }
    
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
    
    /* Delete button styling - compact red button with proper sizing */
    button[key*="delete_"] {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.5) !important;
        color: #ef4444 !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        min-height: 38px !important;
        width: 100% !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    button[key*="delete_"]:hover {
        background: rgba(239, 68, 68, 0.2) !important;
        border-color: #ef4444 !important;
        transform: scale(1.05) !important;
        transition: all 0.2s ease !important;
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
    
    /* Improve document selection buttons */
    button[key*="select_"] {
        text-align: left !important;
        justify-content: flex-start !important;
        padding: 0.5rem 0.75rem !important;
    }
    
    /* Optional: Hide Streamlit branding for cleaner interface */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
