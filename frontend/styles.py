"""
Enhanced custom CSS styling with fixed sidebar behavior
"""
import streamlit as st


def apply_custom_css():
    """Apply enhanced custom CSS - Respects system theme"""
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
    
    /* Suggested question buttons */
    button[key^="suggest_"] {
        background: rgba(102, 126, 234, 0.05) !important;
        border: 2px solid rgba(102, 126, 234, 0.2) !important;
        border-radius: 12px !important;
        padding: 0.75rem !important;
        text-align: left !important;
        font-size: 0.9rem !important;
        min-height: 60px !important;
    }
    
    button[key^="suggest_"]:hover {
        background: rgba(102, 126, 234, 0.1) !important;
        border-color: rgba(102, 126, 234, 0.4) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2) !important;
    }
    
    /* Info button */
    button[key^="info_"] {
        background: rgba(59, 130, 246, 0.1) !important;
        border: 2px solid rgba(59, 130, 246, 0.3) !important;
        color: #3b82f6 !important;
        font-size: 1.1rem !important;
        padding: 0 !important;
        min-height: 40px !important;
        max-height: 40px !important;
        width: 45px !important;
    }
    
    button[key^="info_"]:hover {
        background: rgba(59, 130, 246, 0.2) !important;
        border-color: #3b82f6 !important;
    }
    
    /* Delete button in conversation history - Compact */
    button[key*="del_conv_"] {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 2px solid rgba(239, 68, 68, 0.5) !important;
        color: #ef4444 !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
        padding: 0 !important;
        min-height: 40px !important;
        max-height: 40px !important;
        width: 100% !important;
    }
    
    button[key*="del_conv_"]:hover {
        background: rgba(239, 68, 68, 0.2) !important;
        border-color: #ef4444 !important;
        color: #dc2626 !important;
        transform: scale(1.05);
    }
    
    /* Inline stop button */
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
    
    /* Download button styling */
    .stDownloadButton > button {
        background: rgba(34, 197, 94, 0.1) !important;
        border: 2px solid rgba(34, 197, 94, 0.3) !important;
        color: #22c55e !important;
    }
    
    .stDownloadButton > button:hover {
        background: rgba(34, 197, 94, 0.2) !important;
        border-color: #22c55e !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(102, 126, 234, 0.05) !important;
        border-radius: 8px !important;
        border: 1px solid rgba(102, 126, 234, 0.2) !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(102, 126, 234, 0.1) !important;
        border-color: rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Source citation styling */
    .streamlit-expanderContent {
        background: rgba(0, 0, 0, 0.02) !important;
        border-radius: 0 0 8px 8px !important;
        padding: 1rem !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 0.5rem 1rem;
        background: rgba(102, 126, 234, 0.05);
        border: 2px solid transparent;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(102, 126, 234, 0.1) !important;
        border-color: rgba(102, 126, 234, 0.3) !important;
    }
    
    /* Conversation history buttons */
    button[key^="conv_"] {
        text-align: left !important;
        white-space: pre-line !important;
        font-size: 0.85rem !important;
        min-height: 60px !important;
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
    
    /* Document card styling */
    button[key*="select_"] {
        transition: all 0.2s ease !important;
        position: relative !important;
        text-align: left !important;
    }
    
    button[key*="select_"]:hover {
        transform: translateX(4px) !important;
    }
    
    /* Selected document highlight */
    button[key*="select_"][kind="primary"] {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%) !important;
        border-left: 4px solid #667eea !important;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.2) !important;
    }
    
    button[key*="select_"][kind="primary"]::before {
        content: '' !important;
        position: absolute !important;
        left: 0 !important;
        top: 0 !important;
        bottom: 0 !important;
        width: 4px !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Action buttons (Overview, Delete) */
    button[key*="overview_"], button[key*="delete_"] {
        font-size: 0.85rem !important;
        padding: 0.4rem 0.6rem !important;
        min-height: 36px !important;
        margin-top: 0.25rem !important;
        border-radius: 6px !important;
    }
    
    button[key*="overview_"] {
        background: rgba(59, 130, 246, 0.08) !important;
        border: 1.5px solid rgba(59, 130, 246, 0.25) !important;
        color: #3b82f6 !important;
    }
    
    button[key*="overview_"]:hover {
        background: rgba(59, 130, 246, 0.15) !important;
        border-color: #3b82f6 !important;
    }
    
    button[key*="delete_"] {
        background: rgba(239, 68, 68, 0.08) !important;
        border: 1.5px solid rgba(239, 68, 68, 0.25) !important;
        color: #ef4444 !important;
        font-weight: 600 !important;
    }
    
    button[key*="delete_"]:hover {
        background: rgba(239, 68, 68, 0.15) !important;
        border-color: #ef4444 !important;
        color: #dc2626 !important;
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px !important;
        padding: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Caption styling */
    .caption {
        font-size: 0.85rem;
        opacity: 0.7;
        margin-top: 0.25rem;
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
