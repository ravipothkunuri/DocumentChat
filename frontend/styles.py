"""
Custom CSS styling for the RAG application
"""
import streamlit as st


def apply_custom_css():
    """Apply custom CSS - Respects system theme"""
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
    
    /* STOP BUTTON - Highly visible and animated */
    .stop-button-container {
        margin: 1rem 0;
        padding: 1rem;
        background: rgba(239, 68, 68, 0.05);
        border-radius: 12px;
        border: 2px dashed rgba(239, 68, 68, 0.3);
    }
    
    button[key="stop_btn"] {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        border: 2px solid #ef4444 !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        padding: 0.75rem 1.5rem !important;
        min-height: 50px !important;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4) !important;
        animation: pulse-stop 2s infinite !important;
    }
    
    button[key="stop_btn"]:hover {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
        border-color: #dc2626 !important;
        transform: scale(1.05) !important;
        box-shadow: 0 6px 20px rgba(239, 68, 68, 0.6) !important;
    }
    
    @keyframes pulse-stop {
        0%, 100% { 
            opacity: 1;
            box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
        }
        50% { 
            opacity: 0.85;
            box-shadow: 0 4px 20px rgba(239, 68, 68, 0.6);
        }
    }
    
    /* Sidebar - Fixed width */
    section[data-testid="stSidebar"] {
        width: 320px !important;
        min-width: 320px !important;
        max-width: 320px !important;
    }
    
    section[data-testid="stSidebar"] > div {
        width: 320px !important;
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
    
    /* Custom chat input styling (Claude-like) */
    .stTextArea > div > div > textarea {
        border-radius: 12px !important;
        border: 2px solid #e5e7eb !important;
        font-size: 1rem !important;
        padding: 12px 16px !important;
        transition: all 0.2s ease !important;
        resize: none !important;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
        outline: none !important;
    }
    
    /* Send button (inline with input) - Modern gradient design */
    button[key="send_btn_inline"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 2rem !important;
        height: 80px !important;
        min-width: 60px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.25) !important;
        cursor: pointer !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    /* Send button hover effect with shine */
    button[key="send_btn_inline"]:hover {
        transform: translateY(-2px) scale(1.05) !important;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4) !important;
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
    }
    
    button[key="send_btn_inline"]:active {
        transform: translateY(0) scale(0.98) !important;
        box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Send button ripple effect */
    button[key="send_btn_inline"]::before {
        content: '' !important;
        position: absolute !important;
        top: 50% !important;
        left: 50% !important;
        width: 0 !important;
        height: 0 !important;
        border-radius: 50% !important;
        background: rgba(255, 255, 255, 0.5) !important;
        transform: translate(-50%, -50%) !important;
        transition: width 0.6s, height 0.6s !important;
    }
    
    button[key="send_btn_inline"]:hover::before {
        width: 100px !important;
        height: 100px !important;
    }
    
    /* Stop button (inline with input) - Danger style */
    button[key="stop_btn_inline"] {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-size: 1.8rem !important;
        height: 80px !important;
        min-width: 60px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4) !important;
        animation: pulse-red 2s infinite !important;
        cursor: pointer !important;
    }
    
    button[key="stop_btn_inline"]:hover {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
        transform: translateY(-2px) scale(1.05) !important;
        box-shadow: 0 8px 20px rgba(239, 68, 68, 0.6) !important;
    }
    
    button[key="stop_btn_inline"]:active {
        transform: translateY(0) scale(0.98) !important;
    }
    
    @keyframes pulse-red {
        0%, 100% { 
            box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4);
        }
        50% { 
            box-shadow: 0 6px 16px rgba(239, 68, 68, 0.6);
        }
    }
    
    /* Status indicator pulse animation */
    @keyframes pulse {
        0%, 100% {
            opacity: 1;
            transform: scale(1);
        }
        50% {
            opacity: 0.6;
            transform: scale(1.2);
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
    
    /* Responsive design */
    @media screen and (max-width: 768px) {
        .main-header { 
            font-size: 1.8rem; 
        }
        
        .stButton button { 
            width: 100% !important; 
            min-height: 44px !important; 
        }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
