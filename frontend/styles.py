"""
Custom CSS styling for the RAG application - FIXED VERSION
"""
import streamlit as st


def apply_custom_css():
    """Apply custom CSS - Respects system theme with fixed animations"""
    st.markdown("""
    <style>
    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 0.5rem 0;
    }
    
    /* FIXED: Sidebar toggle button - Moved down to avoid Streamlit header overlap */
    button[kind="header"] {
        position: fixed !important;
        top: 3.5rem !important;  /* Changed from 0.75rem to avoid overlap */
        left: 0.75rem !important;
        z-index: 999 !important;
        background: rgba(102, 126, 234, 0.1) !important;
        border: 2px solid rgba(102, 126, 234, 0.3) !important;
        border-radius: 8px !important;
        padding: 0.5rem !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    button[kind="header"]:hover {
        background: rgba(102, 126, 234, 0.2) !important;
        border-color: rgba(102, 126, 234, 0.5) !important;
        transform: scale(1.05) !important;
    }
    
    /* Enhanced button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transform: translateY(-1px);
    }
    
    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: white;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* FIXED: Sidebar - Smooth animations with proper timing */
    section[data-testid="stSidebar"] {
        width: 320px !important;
        min-width: 320px !important;
        max-width: 320px !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        overflow: hidden !important;
    }
    
    section[data-testid="stSidebar"] > div {
        width: 320px !important;
        padding-top: 4rem !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        overflow-y: auto !important;
        overflow-x: hidden !important;
    }
    
    /* FIXED: Collapsed sidebar - Cleaner animation */
    section[data-testid="stSidebar"][aria-expanded="false"] {
        width: 0 !important;
        min-width: 0 !important;
        max-width: 0 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    section[data-testid="stSidebar"][aria-expanded="false"] > div {
        width: 0 !important;
        min-width: 0 !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        opacity: 0 !important;
        transform: translateX(-100%) !important;
    }
    
    /* FIXED: Main content area - Smooth transition when sidebar toggles */
    .main .block-container {
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        max-width: 100% !important;
    }
    
    /* FIXED: Prevent content jump during sidebar animation */
    [data-testid="stAppViewContainer"] {
        transition: padding-left 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    
    /* Delete button - Centered */
    button[key*="delete_"] {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 2px solid rgba(239, 68, 68, 0.5) !important;
        color: #ef4444 !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        padding: 0 !important;
        min-height: 40px !important;
        max-height: 40px !important;
        width: 45px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    button[key*="delete_"]:hover {
        background: rgba(239, 68, 68, 0.2) !important;
        border-color: #ef4444 !important;
        color: #dc2626 !important;
        transform: scale(1.05);
    }
    
    /* Inline stop button - Next to generating message */
    button[key="stop_inline"] {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 1.5rem !important;
        min-height: 45px !important;
        padding: 0 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.3) !important;
        animation: pulse-stop 2s infinite !important;
    }
    
    button[key="stop_inline"]:hover {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
        transform: scale(1.1) !important;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.5) !important;
    }
    
    @keyframes pulse-stop {
        0%, 100% { 
            box-shadow: 0 2px 8px rgba(239, 68, 68, 0.3);
        }
        50% { 
            box-shadow: 0 4px 12px rgba(239, 68, 68, 0.5);
        }
    }
    
    /* Pulse animation for status indicator */
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
            transform: scale(1);
        }
        50% {
            opacity: 0.5;
            transform: scale(1.1);
        }
    }
    
    /* Loading animation */
    .loading-dots {
        display: inline-flex;
        gap: 8px;
        align-items: center;
        padding: 20px;
    }
    
    .loading-dots .dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        animation: bounce 1.4s infinite ease-in-out both;
    }
    
    .loading-dots .dot:nth-child(1) {
        animation-delay: -0.32s;
    }
    
    .loading-dots .dot:nth-child(2) {
        animation-delay: -0.16s;
    }
    
    @keyframes bounce {
        0%, 80%, 100% {
            transform: scale(0);
            opacity: 0.5;
        }
        40% {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    /* Document card hover effect */
    button[key*="select_"] {
        transition: all 0.2s ease !important;
    }
    
    button[key*="select_"]:hover {
        transform: translateX(4px) !important;
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px !important;
        padding: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* FIXED: Ensure Streamlit header buttons stay visible and don't overlap */
    [data-testid="stToolbar"] {
        z-index: 1000 !important;
        background: transparent !important;
    }
    
    /* FIXED: Streamlit's header container */
    header[data-testid="stHeader"] {
        z-index: 1000 !important;
        background: transparent !important;
    }
    
    /* Responsive design */
    @media screen and (max-width: 768px) {
        .main-header { 
            font-size: 1.8rem; 
        }
        
        .stButton button { 
            width: 100% !important; 
            min-height: 44px !important; 
        }
        
        section[data-testid="stSidebar"] {
            width: 280px !important;
            min-width: 280px !important;
        }
        
        section[data-testid="stSidebar"] > div {
            width: 280px !important;
        }
        
        /* FIXED: Mobile - Move toggle button even lower if needed */
        button[kind="header"] {
            top: 4rem !important;
        }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.05);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(102, 126, 234, 0.3);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(102, 126, 234, 0.5);
    }
    </style>
    """, unsafe_allow_html=True)
