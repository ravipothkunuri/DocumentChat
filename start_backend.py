"""Start FastAPI backend server"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host="localhost",
        port=8000,
        log_level="info"
    )
