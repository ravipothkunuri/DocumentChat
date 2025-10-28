"""
Custom CSS styling for the RAG application - WORKING VERSION
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
    
    /* Sidebar - Smooth animations */
    section[data-testid="stSidebar"] {
        width: 320px !important;
        min-width: 320px !important;
        max-width: 320px !important;
        transition: width 0.3s ease, min-width 0.3s ease, max-width 0.3s ease !important;
    }
    
    section[data-testid="stSidebar"] > div {
        width: 320px !important;
        transition: width 0.3s ease, opacity 0.3s ease !important;
    }
    
    /* Collapsed sidebar */
    section[data-testid="stSidebar"][aria-expanded="false"] {
        width: 0 !important;
        min-width: 0 !important;
        max-width: 0 !important;
    }
    
    section[data-testid="stSidebar"][aria-expanded="false"] > div {
        width: 0 !important;
        opacity: 0 !important;
    }
    
    /* Main content area */
    .main .block-container {
        transition: padding-left 0.3s ease !important;
        max-width: 100% !important;
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
    
    /* Chat container */
    .main .block-container {
        background: #f8f9fa !important;
    }
    
    /* Chat message styling - Modern speech bubbles */
    .stChatMessage {
        border-radius: 18px !important;
        padding: 0.875rem 1.125rem !important;
        margin-bottom: 0.75rem !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08) !important;
        animation: slideIn 0.25s ease-out !important;
        max-width: 65% !important;
        position: relative !important;
        border: none !important;
    }
    
    /* User messages - Right aligned with blue bubble */
    .stChatMessage[data-testid="user-message"],
    .stChatMessage:has([data-testid*="user"]) {
        background: #4A9FF5 !important;
        color: white !important;
        margin-left: auto !important;
        margin-right: 0 !important;
        border-bottom-right-radius: 4px !important;
    }
    
    /* Speech bubble tail for user messages - points right toward avatar */
    .stChatMessage[data-testid="user-message"]::after,
    .stChatMessage:has([data-testid*="user"])::after {
        content: "" !important;
        position: absolute !important;
        bottom: 8px !important;
        right: -8px !important;
        width: 0 !important;
        height: 0 !important;
        border-style: solid !important;
        border-width: 8px 0 8px 10px !important;
        border-color: transparent transparent transparent #4A9FF5 !important;
    }
    
    .stChatMessage[data-testid="user-message"] p,
    .stChatMessage:has([data-testid*="user"]) p {
        color: white !important;
        margin: 0 !important;
    }
    
    .stChatMessage[data-testid="user-message"] .stMarkdown,
    .stChatMessage:has([data-testid*="user"]) .stMarkdown {
        color: white !important;
    }
    
    /* Assistant messages - Left aligned with light gray bubble */
    .stChatMessage[data-testid="assistant-message"],
    .stChatMessage:has([data-testid*="assistant"]) {
        background: #E8EAED !important;
        color: #202124 !important;
        margin-right: auto !important;
        margin-left: 0 !important;
        border-bottom-left-radius: 4px !important;
    }
    
    /* Speech bubble tail for assistant messages - points left toward avatar */
    .stChatMessage[data-testid="assistant-message"]::before,
    .stChatMessage:has([data-testid*="assistant"])::before {
        content: "" !important;
        position: absolute !important;
        bottom: 8px !important;
        left: -8px !important;
        width: 0 !important;
        height: 0 !important;
        border-style: solid !important;
        border-width: 8px 10px 8px 0 !important;
        border-color: transparent #E8EAED transparent transparent !important;
    }
    
    .stChatMessage[data-testid="assistant-message"] p,
    .stChatMessage:has([data-testid*="assistant"]) p {
        color: #202124 !important;
        margin: 0 !important;
    }
    
    /* Timestamp styling */
    .stChatMessage .stCaption {
        opacity: 0.65 !important;
        font-size: 0.7rem !important;
        margin-top: 0.375rem !important;
        font-weight: 400 !important;
    }
    
    /* Avatar positioning */
    .stChatMessage .stAvatar {
        width: 32px !important;
        height: 32px !important;
    }
    
    /* User message avatar on right */
    .stChatMessage[data-testid="user-message"] .stAvatar,
    .stChatMessage:has([data-testid*="user"]) .stAvatar {
        order: 2 !important;
        margin-left: 0.625rem !important;
        margin-right: 0 !important;
    }
    
    /* Assistant message avatar on left */
    .stChatMessage[data-testid="assistant-message"] .stAvatar,
    .stChatMessage:has([data-testid*="assistant"]) .stAvatar {
        margin-right: 0.625rem !important;
        margin-left: 0 !important;
    }
    
    /* Chat input styling */
    .stChatInput {
        border-radius: 24px !important;
        background: white !important;
    }
    
    .stChatInput > div {
        border-radius: 24px !important;
        border: 1px solid #E0E0E0 !important;
        background: white !important;
        transition: all 0.2s ease !important;
    }
    
    .stChatInput > div:focus-within {
        border-color: #4A9FF5 !important;
        box-shadow: 0 0 0 2px rgba(74, 159, 245, 0.1) !important;
    }
    
    .stChatInput input {
        color: #202124 !important;
    }
    
    .stChatInput input::placeholder {
        color: #9AA0A6 !important;
    }
    
    /* Message slide-in animation */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(8px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
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
        
        section[data-testid="stSidebar"] {
            width: 280px !important;
            min-width: 280px !important;
        }
        
        section[data-testid="stSidebar"] > div {
            width: 280px !important;
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
