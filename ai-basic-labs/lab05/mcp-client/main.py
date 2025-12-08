"""Main application entry point - ARI Processing Server"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.config import settings
from app.core.logging import setup_logging
from app.infrastructure.mcp.mcp_service import mcp_service
from app.routers.api import router as api_router

# Setup logging
setup_logging()
logger = logging.getLogger("app.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    # Startup
    logger.info("Starting ARI Processing Server...")
    try:
        # Initialize MCP service
        await mcp_service.initialize()
        logger.info("MCP service initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize MCP service: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ARI Processing Server...")
    try:
        await mcp_service.shutdown()
        logger.info("MCP service shutdown completed")
    except Exception as e:
        logger.error(f"Error during service shutdown: {e}")

# Create FastAPI app
app = FastAPI(
    title="ARI Processing Server",
    version=settings.app_version,
    lifespan=lifespan,
    debug=settings.debug
)

# Add CORS middleware (NextJS 웹 UI 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],  # NextJS 개발 서버
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration"""
    return {"status": "healthy", "service": "ari-processing-server"}

# Static files (다운로드용)
from fastapi.staticfiles import StaticFiles
import os

static_downloads_dir = os.path.join(os.path.dirname(__file__), 'static', 'downloads')
os.makedirs(static_downloads_dir, exist_ok=True)
app.mount("/downloads", StaticFiles(directory=static_downloads_dir), name="downloads")

# Include routers
app.include_router(api_router, prefix="/api")

# Development server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        limit_concurrency=1000,  # 동시 연결 수 제한
        timeout_keep_alive=300,  # Keep-Alive 타임아웃 (기본: 5초)
        h11_max_incomplete_event_size=100 * 1024 * 1024  # 100MB - HTTP request body 크기 제한
    )
