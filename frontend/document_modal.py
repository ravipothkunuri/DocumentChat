"""
Document overview modal component
"""
import streamlit as st
from typing import Dict, Optional
from datetime import datetime


@st.dialog("Document Overview")
def show_document_overview(document_details: Dict):
    """Display document overview in a modal"""
    
    st.markdown(f"### 📄 {document_details.get('filename', 'Unknown')}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("File Type", document_details.get('type', 'N/A').upper())
        st.metric("File Size", f"{document_details.get('size_mb', 0):.2f} MB")
        st.metric("Total Chunks", document_details.get('chunks', 0))
    
    with col2:
        st.metric("Status", document_details.get('status', 'N/A').title())
        st.metric("Avg Chunk Length", f"{document_details.get('average_chunk_length', 0):.0f} chars")
        
        uploaded_at = document_details.get('uploaded_at', '')
        if uploaded_at:
            try:
                dt = datetime.fromisoformat(uploaded_at)
                st.metric("Uploaded", dt.strftime('%b %d, %Y %H:%M'))
            except:
                st.metric("Uploaded", uploaded_at)
    
    st.markdown("---")
    
    st.markdown("**📊 Chunk Statistics**")
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    
    with stats_col1:
        st.write(f"**Min:** {document_details.get('min_chunk_length', 0)} chars")
    with stats_col2:
        st.write(f"**Max:** {document_details.get('max_chunk_length', 0)} chars")
    with stats_col3:
        st.write(f"**Total:** {document_details.get('total_chunk_length', 0)} chars")
    
    st.markdown("---")
    
    preview = document_details.get('chunk_preview', '')
    if preview:
        st.markdown("**📝 Content Preview**")
        st.text_area(
            "First chunk preview",
            preview,
            height=150,
            disabled=True,
            label_visibility="collapsed"
        )
    
    if st.button("Close", type="primary", use_container_width=True):
        st.rerun()
