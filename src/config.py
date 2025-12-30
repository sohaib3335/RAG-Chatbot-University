"""
Configuration module for RAG Chatbot
Handles environment variables and settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for RAG Chatbot settings"""
    
    # Base paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    # Use the university knowledge base PDF folder
    KNOWLEDGE_BASE_DIR = BASE_DIR / "university_knowledge_base_pdf" / "university_knowledge_base_pdf"
    CHROMA_PERSIST_DIR = BASE_DIR / os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    
    # Model settings
    USE_LOCAL_EMBEDDINGS = os.getenv("USE_LOCAL_EMBEDDINGS", "false").lower() == "true"
    USE_LOCAL_LLM = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
    USE_CLAUDE = os.getenv("USE_CLAUDE", "true").lower() == "true"  # Default to Claude
    
    # Embedding model - using OpenAI's best embedding model
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Better local model
    
    # LLM Models
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4-turbo")  # OpenAI fallback
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")  # Claude Sonnet 4.5
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    
    # Chunking settings
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
    
    # Retrieval settings
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))
    
    # Collection name for ChromaDB
    COLLECTION_NAME = "rag_knowledge_base"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate required configuration"""
        if not cls.USE_LOCAL_EMBEDDINGS and not cls.OPENAI_API_KEY:
            print("Warning: OPENAI_API_KEY not set. Required for embeddings.")
            return False
        if cls.USE_CLAUDE and not cls.ANTHROPIC_API_KEY:
            print("Warning: ANTHROPIC_API_KEY not set. Set it in .env file or set USE_CLAUDE=false.")
            return False
        return True
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)


# Create directories on import
Config.create_directories()
