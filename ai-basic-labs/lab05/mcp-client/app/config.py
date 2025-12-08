"""Configuration management for ARI Processing Client"""
from pydantic_settings import BaseSettings
from typing import List
import os
from pathlib import Path
from dotenv import load_dotenv

# lab05 디렉토리의 .env 파일 로드
# lab05/mcp-client/app/config.py → mcp-client → lab05
lab05_root = Path(__file__).parent.parent.parent
env_path = lab05_root / ".env"

if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ .env 파일 로드 성공: {env_path}")
else:
    print(f"⚠️  .env 파일이 {env_path}에 없습니다.")
    print(f"   lab05/.env 파일을 생성해주세요.")
    # 환경 변수에서 직접 로드 시도
    load_dotenv()

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    log_level: str = "info"
    
    # MCP Server Configuration
    mcp_server_url: str = "http://127.0.0.1:4200/my-custom-path/"
    mcp_connection_timeout: int = 30
    mcp_retry_attempts: int = 3
    
    # CORS Configuration
    cors_origins: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000"]
    cors_allow_credentials: bool = True
    
    # Application Configuration
    app_title: str = "ARI Processing Server"
    app_version: str = "1.0.0"
    
    # AI & LLM Configuration - OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    
    # AI & LLM Configuration - Azure OpenAI
    azure_openai_enabled: bool = os.getenv("AZURE_OPENAI_ENABLED", "false").lower() == "true"
    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_openai_api_base: str = os.getenv("AZURE_OPENAI_API_BASE", "")
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    azure_openai_model: str = os.getenv("AZURE_OPENAI_MODEL", "deploy-gpt-4o-01")  # 실제 배포된 모델명
    
    # Vector Database Configuration
    qdrant_host: str = ""
    
    # Search Engine Configuration
    opensearch_host: str = ""
    
    # Database Configuration
    database_url: str = ""
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # 추가 필드 무시

# Global settings instance
settings = Settings()