"""Logging configuration for the application"""
import logging
import sys
from typing import Dict, Any
from app.config import settings

def setup_logging() -> None:
    """Configure application logging"""
    
    # Create formatter
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Set specific logger levels - 모든 라이브러리 로그를 WARNING 이상으로
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.ERROR)
    logging.getLogger("httpx").setLevel(logging.ERROR)
    logging.getLogger("mcp").setLevel(logging.ERROR)
    logging.getLogger("openai").setLevel(logging.ERROR)
    
    # 앱 로그는 INFO 레벨로 설정 (디버깅을 위해)
    logging.getLogger("app").setLevel(logging.INFO)
    
    # RAG 관련 로거들 명시적으로 INFO 레벨 설정
    logging.getLogger("app.presentation.api.rag.rag_router").setLevel(logging.INFO)
    logging.getLogger("app.application.rag.rag_service").setLevel(logging.INFO)
    logging.getLogger("app.infrastructure.vectordb.qdrant_service").setLevel(logging.INFO)
    logging.getLogger("app.infrastructure.search.opensearch_service").setLevel(logging.INFO)
    logging.getLogger("app.infrastructure.llm.llm_service").setLevel(logging.INFO)

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name"""
    return logging.getLogger(f"app.{name}")

class LoggerMixin:
    """Mixin class to add logging capability to any class"""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        return get_logger(self.__class__.__name__)