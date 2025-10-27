# Backend Structure - Modular Organization

## 📁 Directory Structure

```
backend/
├── __init__.py                 (3 lines)   - Package initialization
├── main.py                     (40 lines)  - FastAPI app entry point
├── config.py                   (22 lines)  - Configuration constants
├── models.py                   (27 lines)  - Pydantic models
├── routes.py                   (240 lines) - API endpoints
├── ollama_client.py            (60 lines)  - Async Ollama LLM
├── config_manager.py           (45 lines)  - Config persistence
├── metadata_manager.py         (50 lines)  - Document metadata
├── model_manager.py            (35 lines)  - Model lifecycle
├── document_processor.py       (40 lines)  - Document loading/splitting
└── utils.py                    (35 lines)  - Utility functions

vector_store.py                 (unchanged) - Vector operations
```

## 📊 File Breakdown

### Core Files (597 lines total)

| File | Lines | Responsibility |
|------|-------|---------------|
| `main.py` | 40 | FastAPI app, CORS, lifespan |
| `routes.py` | 240 | All API endpoints |
| `config.py` | 22 | Constants & paths |
| `models.py` | 27 | Pydantic schemas |
| **Subtotal** | **329** | **Core API** |

### LLM & Models (95 lines total)

| File | Lines | Responsibility |
|------|-------|---------------|
| `ollama_client.py` | 60 | Async Ollama streaming |
| `model_manager.py` | 35 | Model initialization |
| **Subtotal** | **95** | **LLM Management** |

### Data Management (135 lines total)

| File | Lines | Responsibility |
|------|-------|---------------|
| `config_manager.py` | 45 | Config persistence |
| `metadata_manager.py` | 50 | Document metadata |
| `document_processor.py` | 40 | Document loading |
| **Subtotal** | **135** | **Data Layer** |

### Utilities (38 lines total)

| File | Lines | Responsibility |
|------|-------|---------------|
| `utils.py` | 35 | Helper functions |
| `__init__.py` | 3 | Package init |
| **Subtotal** | **38** | **Utilities** |

---

## 🎯 Separation of Concerns

### 1. **Entry Point** - `main.py`
- Creates FastAPI app
- Configures CORS
- Manages application lifespan
- Includes routes

### 2. **API Layer** - `routes.py`
- All HTTP endpoints
- Request/response handling
- Business logic orchestration
- Error handling

### 3. **Configuration** - `config.py`
- Environment variables
- File paths
- Constants
- Directory initialization

### 4. **Data Models** - `models.py`
- Request schemas
- Response schemas
- Type validation

### 5. **LLM Integration**
- **`ollama_client.py`** - Async Ollama communication
- **`model_manager.py`** - Model lifecycle & caching

### 6. **Data Layer**
- **`config_manager.py`** - App configuration
- **`metadata_manager.py`** - Document tracking
- **`document_processor.py`** - File loading & chunking

### 7. **Utilities** - `utils.py`
- Text cleaning
- Health checks
- File validation

---

## 🚀 Running the Backend

### Option 1: Run as Module
```bash
python -m backend.main
```

### Option 2: Run Directly
```bash
python backend/main.py
```

### Option 3: Using Uvicorn
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

---

## 🔄 Import Structure

### routes.py imports:
```python
from backend.models import QueryRequest, DocumentUploadResponse
from backend.config import UPLOAD_DIR, FIXED_MODEL
from backend.utils import check_ollama_health, validate_file
from backend.config_manager import ConfigManager
from backend.metadata_manager import MetadataManager
from backend.model_manager import ModelManager
from backend.document_processor import DocumentProcessor
from vector_store import VectorStore
```

### main.py imports:
```python
from backend.routes import router, get_model_manager
```

---

## 📝 Key Design Principles

### Single Responsibility
- Each file has ONE clear purpose
- No circular dependencies
- Easy to test independently

### Manager Pattern
- `ConfigManager` - Configuration
- `MetadataManager` - Document tracking
- `ModelManager` - LLM & embeddings

### Async First
- All routes are async
- Streaming uses async generators
- Proper resource cleanup

### Centralized State
- All managers instantiated in `routes.py`
- Shared across all endpoints
- Lifecycle managed by FastAPI

---

## 🔧 Migration from Monolithic File

### Before (1 file - ~450 lines)
```
rag_backend.py
```

### After (11 files - ~597 lines)
```
backend/
├── main.py          (40)   ← Entry point
├── routes.py        (240)  ← API endpoints
├── config.py        (22)   ← Configuration
├── models.py        (27)   ← Schemas
├── ollama_client.py (60)   ← Async LLM
├── config_manager.py (45)  ← Config persistence
├── metadata_manager.py (50) ← Document tracking
├── model_manager.py (35)   ← Model lifecycle
├── document_processor.py (40) ← File processing
├── utils.py         (35)   ← Helpers
└── __init__.py      (3)    ← Package init
```

---

## ✅ Benefits of Modular Structure

1. **Better Organization** - Easy to find code
2. **Easier Testing** - Test individual components
3. **Clear Dependencies** - Explicit imports
4. **Team Collaboration** - Work on separate files
5. **Code Reusability** - Import specific managers
6. **Maintainability** - Smaller, focused files
7. **Scalability** - Add new features easily

---

## 🎨 Extending the Backend

### Adding a New Endpoint
1. Add route function in `routes.py`
2. Add Pydantic model in `models.py` (if needed)
3. Use existing managers for logic

### Adding a New Feature
1. Create new manager file (e.g., `cache_manager.py`)
2. Initialize in `routes.py`
3. Use in relevant endpoints

### Example: Adding Redis Cache
```python
# backend/cache_manager.py
class CacheManager:
    def __init__(self):
        self.redis = Redis()
    
    def get(self, key: str):
        ...

# backend/routes.py
cache_manager = CacheManager()

@router.get("/cached-query")
async def cached_query():
    cached = cache_manager.get("key")
    ...
```

---

## 🔍 File Purposes Quick Reference

| Need to... | Edit this file |
|------------|----------------|
| Add API endpoint | `routes.py` |
| Change model | `config.py` → `FIXED_MODEL` |
| Modify streaming logic | `ollama_client.py` |
| Change chunk settings | `config_manager.py` → `DEFAULT_CONFIG` |
| Add validation | `models.py` or `utils.py` |
| Initialize new manager | `routes.py` |
| Change paths | `config.py` |
| Add startup logic | `main.py` → `lifespan` |

---

## 🎯 Summary

**Total Lines**: ~597 (vs 450 monolithic)
**Total Files**: 11 (vs 1 monolithic)
**Average Lines/File**: ~54

The modular structure adds ~150 lines but provides:
- ✅ Clear separation of concerns
- ✅ Better code organization
- ✅ Easier maintenance
- ✅ Better testability
- ✅ Team-friendly structure
