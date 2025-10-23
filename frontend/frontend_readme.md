# RAG Assistant - Project Structure

This application has been modularized into separate files for better maintainability and clarity.

## File Structure

```
rag-assistant/
├── app.py                 # Main application entry point
├── config.py             # Configuration settings
├── api_client.py         # Backend API client
├── session_state.py      # Session state management
├── toast.py              # Toast notification system
├── styles.py             # Custom CSS styling
├── sidebar.py            # Sidebar components
├── chat.py               # Chat interface
└── README.md             # This file
```

## Module Descriptions

### `config.py`
- API configuration (base URL)
- File upload settings (max size, allowed extensions)
- UI settings (sidebar width, default values)
- Model defaults (LLM and embedding models)

### `api_client.py`
- `RAGAPIClient` class for all backend interactions
- Methods for health checks, document management, queries
- Streaming query support with error handling

### `session_state.py`
- Session state initialization
- Chat history management per document
- Helper functions for adding/clearing messages

### `toast.py`
- `ToastNotification` class for user feedback
- Queue-based notification system
- Icon mapping for different notification types

### `styles.py`
- Custom CSS for UI components
- Button styling and animations
- Loading indicators
- Responsive design rules

### `sidebar.py`
- Document list rendering
- Document selection and deletion
- File upload handling
- Auto-processing of uploaded files

### `chat.py`
- Chat message display
- Query processing with streaming responses
- Stop generation functionality
- Real-time response updates with typing cursor

### `app.py`
- Main application orchestration
- Page configuration
- Component integration
- Model selection UI

## Key Features

### 1. **Modular Architecture**
Each module has a single responsibility, making the code easier to maintain and test.

### 2. **Stop Generation**
Users can stop ongoing generation by clicking the stop button that replaces the send button during generation.

### 3. **Real-time Streaming**
Responses stream in real-time with a typing cursor indicator.

### 4. **Document Management**
Easy document upload, selection, and deletion with visual feedback.

### 5. **Toast Notifications**
Non-intrusive notifications for user actions and system events.

### 6. **Per-Document Chat History**
Each document maintains its own chat history, allowing users to switch between documents without losing context.

## How It Works

### Chat Flow

```
User sends message → Add to chat history → Store in pending_query
    ↓
Rerun → Display stop button + Start streaming
    ↓
Stream response with live updates (typing cursor)
    ↓
Complete or Stop → Clear pending_query → Rerun → Show send button
```

### Stop Generation Flow

```
Generation in progress → User clicks Stop
    ↓
Set stop_generation = True
    ↓
Stream loop detects stop signal → Interrupt gracefully
    ↓
Add partial response to chat → Reset state → Show send button
```

## Running the Application

### Prerequisites
- Python 3.8+
- Streamlit
- Backend API running (see backend documentation)

### Installation

```bash
# Install dependencies
pip install streamlit requests

# Set environment variables (optional)
export API_BASE_URL="http://localhost:8000"
```

### Running

```bash
# Run the application
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Configuration

### Environment Variables

- `API_BASE_URL`: Backend API URL (default: `http://localhost:8000`)

### Customization

To customize the application:

1. **Change UI colors**: Edit gradient colors in `styles.py`
2. **Adjust file limits**: Modify `MAX_FILE_SIZE_MB` in `config.py`
3. **Add file types**: Update `ALLOWED_EXTENSIONS` in `config.py`
4. **Change sidebar width**: Modify `SIDEBAR_WIDTH` in `config.py`

## Development Tips

### Adding New Features

1. **New API endpoint**: Add method to `RAGAPIClient` in `api_client.py`
2. **New UI component**: Create function in appropriate module (`sidebar.py`, `chat.py`)
3. **New session state**: Add to defaults in `session_state.py`
4. **New styling**: Add CSS to `styles.py`

### Debugging

- Check browser console for JavaScript errors
- Use Streamlit's built-in debugging: `streamlit run app.py --logger.level=debug`
- Add `st.write()` statements to inspect session state

### Testing

```python
# Test API client
from api_client import RAGAPIClient
client = RAGAPIClient("http://localhost:8000")
health_ok, data = client.health_check()
print(f"Health: {health_ok}, Data: {data}")

# Test session state
from session_state import init_session_state, add_message
init_session_state()
add_message({"role": "user", "content": "Test"})
```

## Troubleshooting

### Common Issues

**Backend Offline Error**
- Ensure backend is running: `python rag_backend.py`
- Check `API_BASE_URL` is correct

**Stop Button Not Working**
- Verify `is_generating` state is properly set
- Check for exceptions in stream loop

**Chat Not Updating**
- Ensure `st.rerun()` is called after state changes
- Check `pending_query` is cleared in finally block

**File Upload Fails**
- Verify file size is under `MAX_FILE_SIZE_MB`
- Check file extension is in `ALLOWED_EXTENSIONS`
- Ensure backend `/upload` endpoint is working

## Architecture Benefits

### Separation of Concerns
- **Config**: All settings in one place
- **API**: Backend communication isolated
- **State**: Centralized state management
- **UI**: Separate presentation logic

### Maintainability
- Easy to locate and fix bugs
- Changes to one module don't affect others
- Clear dependencies between modules

### Testability
- Each module can be tested independently
- Mock API client for UI testing
- Test state management separately

### Scalability
- Easy to add new features
- Can split modules further if needed
- Clear pattern for new components

## Future Enhancements

Potential improvements:
- Add unit tests for each module
- Implement caching for API responses
- Add export chat history feature
- Support for multiple file uploads simultaneously
- Add search within chat history
- Implement user authentication
- Add dark/light theme toggle

## Contributing

When adding new features:
1. Follow the existing module structure
2. Add docstrings to all functions
3. Update this README with new features
4. Test thoroughly before committing

## License

[Your License Here]

## Support

For issues or questions:
- Check the troubleshooting section
- Review module documentation
- Contact: [Your Contact Info]