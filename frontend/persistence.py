"""Local file-based persistence for chat histories and exports."""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


CHAT_HISTORY_DIR = Path("chat_history")
CHAT_HISTORY_DIR.mkdir(exist_ok=True)


def get_safe_filename(doc_name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in doc_name)


def save_chat_history_to_local(doc_name: str, data: List[Dict]):
    try:
        safe_filename = get_safe_filename(doc_name)
        file_path = CHAT_HISTORY_DIR / f"{safe_filename}_chat.json"
        chat_data = {
            "document": doc_name,
            "last_updated": datetime.now().isoformat(),
            "message_count": len(data),
            "messages": data,
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)
    except Exception:
        pass


def load_chat_history_from_local(doc_name: str) -> Optional[List[Dict]]:
    try:
        safe_filename = get_safe_filename(doc_name)
        file_path = CHAT_HISTORY_DIR / f"{safe_filename}_chat.json"
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                chat_data = json.load(f)
                return chat_data.get("messages", [])
    except Exception:
        pass
    return None


def delete_chat_history(doc_name: str):
    try:
        safe_filename = get_safe_filename(doc_name)
        file_path = CHAT_HISTORY_DIR / f"{safe_filename}_chat.json"
        if file_path.exists():
            file_path.unlink()
    except Exception:
        pass


def export_chat_as_json(doc_name: str, messages: List[Dict]) -> str:
    try:
        safe_filename = get_safe_filename(doc_name)
        file_path = CHAT_HISTORY_DIR / f"{safe_filename}_chat.json"
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            export_data = {
                "document": doc_name,
                "exported_at": datetime.now().isoformat(),
                "message_count": len(messages),
                "messages": messages,
            }
            return json.dumps(export_data, indent=2, ensure_ascii=False)
    except Exception:
        return "{}"


