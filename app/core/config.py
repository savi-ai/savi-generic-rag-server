from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Configuration
    api_title: str = "Savi RAG Backend"
    api_version: str = "1.0.0"
    debug: bool = False
    
    # Database Configuration
    database_url: str = "sqlite:///./rag_database.db"
    
    # Vector Database Configuration
    chroma_persist_directory: str = "./chroma_db"
    
    # LLM Configuration
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    default_llm_provider: str = "ollama"  # ollama, openai, anthropic
    
    # AWS Configuration
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    
    # File Upload Configuration
    upload_dir: str = "./uploads"
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_file_types: list = [".pdf", ".docx", ".txt", ".xlsx"]
    
    # Vectorization Configuration
    default_chunk_size: int = 1000
    default_chunk_overlap: int = 200
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Search Configuration
    default_top_k: int = 5
    
    # LLM Generation Configuration
    default_temperature: float = 0.7
    default_max_tokens: int = 1000
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings() 