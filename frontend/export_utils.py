"""Export utilities for chat conversations"""
import json
from datetime import datetime
from typing import List, Dict


def export_to_json(messages: List[Dict], document_name: str) -> str:
    """Export chat messages to JSON format"""
    export_data = {
        "document": document_name,
        "exported_at": datetime.now().isoformat(),
        "message_count": len(messages),
        "conversation": messages
    }
    return json.dumps(export_data, indent=2, ensure_ascii=False)


def export_to_markdown(messages: List[Dict], document_name: str) -> str:
    """Export chat messages to Markdown format"""
    lines = [
        f"# Chat Conversation: {document_name}",
        f"\n**Exported:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n**Messages:** {len(messages)}",
        "\n---\n"
    ]
    
    for i, msg in enumerate(messages, 1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")
        stopped = msg.get("stopped", False)
        
        # Format role
        role_display = "👤 **User**" if role == "user" else "🤖 **Assistant**"
        
        lines.append(f"\n## Message {i}: {role_display}\n")
        
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp)
                time_str = dt.strftime("%I:%M %p")
                lines.append(f"*Time: {time_str}*\n")
            except:
                pass
        
        lines.append(f"\n{content}\n")
        
        if stopped:
            lines.append("\n*⚠️ Generation was stopped*\n")
        
        lines.append("\n---\n")
    
    return "".join(lines)


def create_download_link(content: str, filename: str, file_type: str) -> str:
    """Create a download link for the content"""
    import base64
    
    b64_content = base64.b64encode(content.encode()).decode()
    
    if file_type == "json":
        mime_type = "application/json"
    else:  # markdown
        mime_type = "text/markdown"
    
    return f'<a href="data:{mime_type};base64,{b64_content}" download="{filename}" style="text-decoration:none;">⬇️ Download {file_type.upper()}</a>'
