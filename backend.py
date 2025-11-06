"""DocumentChat Backend - API entrypoint"""
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from configuration import UPLOAD_DIR, VECTOR_DIR
from api.routes import router as api_router, model_manager


UPLOAD_DIR.mkdir(exist_ok=True)
VECTOR_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Starting DocumentChat Application")
    yield
    logger.info("📻 Shutting down gracefully")
    await model_manager.cleanup()


app = FastAPI(title="DocumentChat Backend API", version="1.0.0", description="AI-powered document understanding and question answering", lifespan=lifespan)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.include_router(api_router)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)