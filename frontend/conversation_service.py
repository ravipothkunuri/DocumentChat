"""
Conversation persistence service for automatic file-based chat storage
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

CONVERSATIONS_DIR = Path("vector_data/conversations")
CONVERSATIONS_DIR.mkdir(parents=True, exist_ok=True)


class ConversationService:
    """Service for managing conversation persistence"""
    
    @staticmethod
    def _get_conversation_path(document_name: str) -> Path:
        """Get the file path for a document's conversation"""
        safe_name = document_name.replace('/', '_').replace('\\', '_')
        return CONVERSATIONS_DIR / f"{safe_name}.json"
    
    @staticmethod
    def load_conversation(document_name: str) -> List[Dict]:
        """Load conversation for a specific document"""
        if not document_name:
            return []
        
        conversation_path = ConversationService._get_conversation_path(document_name)
        
        if not conversation_path.exists():
            logger.debug(f"No saved conversation for {document_name}")
            return []
        
        try:
            with open(conversation_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            messages = data.get('messages', [])
            logger.info(f"Loaded {len(messages)} messages for {document_name}")
            return messages
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in conversation file for {document_name}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error loading conversation for {document_name}: {e}")
            return []
    
    @staticmethod
    def save_conversation(document_name: str, messages: List[Dict]) -> bool:
        """Save conversation for a specific document"""
        if not document_name:
            return False
        
        conversation_path = ConversationService._get_conversation_path(document_name)
        
        try:
            data = {
                'document': document_name,
                'last_updated': datetime.now().isoformat(),
                'message_count': len(messages),
                'messages': messages
            }
            
            with open(conversation_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Saved {len(messages)} messages for {document_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving conversation for {document_name}: {e}")
            return False
    
    @staticmethod
    def delete_conversation(document_name: str) -> bool:
        """Delete conversation file for a specific document"""
        if not document_name:
            return False
        
        conversation_path = ConversationService._get_conversation_path(document_name)
        
        try:
            if conversation_path.exists():
                conversation_path.unlink()
                logger.info(f"Deleted conversation for {document_name}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting conversation for {document_name}: {e}")
            return False
    
    @staticmethod
    def list_all_conversations() -> List[Dict]:
        """List all saved conversations"""
        conversations = []
        
        try:
            for conversation_file in CONVERSATIONS_DIR.glob("*.json"):
                try:
                    with open(conversation_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    conversations.append({
                        'document': data.get('document', conversation_file.stem),
                        'last_updated': data.get('last_updated', ''),
                        'message_count': data.get('message_count', 0)
                    })
                except Exception as e:
                    logger.warning(f"Error reading {conversation_file}: {e}")
                    continue
        
        except Exception as e:
            logger.error(f"Error listing conversations: {e}")
        
        return conversations
