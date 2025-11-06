"""Styling helpers for Streamlit components and global CSS injection."""
import streamlit as st


def apply_custom_css() -> None:
    st.markdown(
        """
        <style>
        .stChatMessage {border-radius: 10px; padding: 10px; margin-bottom: 10px;}
        button[kind="primary"] {background-color: #4CAF50;}
        .stButton button {transition: all 0.3s ease;}
        .stButton button:hover {transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2);}
        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}

        # /* Keep assistant response area height stable during streaming */
        # .response-box { min-height: 160px; max-height: 320px; overflow-y: auto; }
        # .stop-col { min-height: 160px; display: flex; align-items: flex-start; }
        </style>
        """,
        unsafe_allow_html=True,
    )


