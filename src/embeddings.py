"""
Embedding Module
Handles creation and management of text embeddings
Supports OpenAI text-embedding-3-large (recommended) and HuggingFace models
"""

from typing import List, Optional
from langchain.schema import Document

from src.config import Config


class EmbeddingManager:
    """
    Embedding Manager class for creating and managing text embeddings.
    Supports both OpenAI and local HuggingFace embeddings.
    """
    
    def __init__(self, use_local: bool = None):
        """
        Initialize EmbeddingManager
        
        Args:
            use_local: Whether to use local embeddings (HuggingFace)
        """
        self.use_local = use_local if use_local is not None else Config.USE_LOCAL_EMBEDDINGS
        self.embeddings = self._initialize_embeddings()
        
    def _initialize_embeddings(self):
        """Initialize the embedding model based on configuration"""
        if self.use_local:
            return self._get_local_embeddings()
        else:
            return self._get_openai_embeddings()
    
    def _get_openai_embeddings(self):
        """Get OpenAI embeddings"""
        from langchain_openai import OpenAIEmbeddings
        
        if not Config.OPENAI_API_KEY:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY in .env file "
                "or set USE_LOCAL_EMBEDDINGS=true to use local models."
            )
        
        print(f"🔌 Using OpenAI embeddings: {Config.EMBEDDING_MODEL}")
        return OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            openai_api_key=Config.OPENAI_API_KEY
        )
    
    def _get_local_embeddings(self):
        """Get local HuggingFace embeddings - using all-mpnet-base-v2 for better quality"""
        from langchain_community.embeddings import HuggingFaceEmbeddings
        
        print(f"🔌 Using local embeddings: {Config.LOCAL_EMBEDDING_MODEL}")
        return HuggingFaceEmbeddings(
            model_name=Config.LOCAL_EMBEDDING_MODEL,  # all-mpnet-base-v2 for better quality
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text
        
        Args:
            text: Query text to embed
            
        Returns:
            List of embedding values
        """
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return self.embeddings.embed_documents(texts)
    
    def get_embeddings(self):
        """Get the underlying embeddings object for use with vector stores"""
        return self.embeddings


if __name__ == "__main__":
    # Test the embedding manager
    print("Testing Embedding Manager...")
    
    try:
        manager = EmbeddingManager(use_local=True)
        
        test_text = "This is a test sentence for embedding."
        embedding = manager.embed_query(test_text)
        
        print(f"Embedding dimension: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the required dependencies installed.")
