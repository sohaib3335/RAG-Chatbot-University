"""
RAG Chain Module
Combines all components into a complete RAG pipeline
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
from langchain.schema import Document

from src.config import Config
from src.document_loader import DocumentLoader, TextSplitter, load_and_split_documents
from src.embeddings import EmbeddingManager
from src.vector_store import VectorStore
from src.llm import LLMManager


class RAGChain:
    """
    RAG Chain class that combines document loading, embedding, 
    vector storage, retrieval, and response generation.
    """
    
    def __init__(
        self,
        use_local_embeddings: bool = None,
        use_local_llm: bool = None,
        persist_directory: Optional[Path] = None
    ):
        """
        Initialize RAG Chain
        
        Args:
            use_local_embeddings: Use local embeddings instead of OpenAI
            use_local_llm: Use local LLM instead of OpenAI
            persist_directory: Directory for vector store persistence
        """
        self.use_local_embeddings = use_local_embeddings if use_local_embeddings is not None else Config.USE_LOCAL_EMBEDDINGS
        self.use_local_llm = use_local_llm if use_local_llm is not None else Config.USE_LOCAL_LLM
        self.persist_directory = persist_directory or Config.CHROMA_PERSIST_DIR
        
        # Initialize components
        self.document_loader = DocumentLoader()
        self.text_splitter = TextSplitter()
        self.embedding_manager = EmbeddingManager(use_local=self.use_local_embeddings)
        self.vector_store = VectorStore(
            persist_directory=self.persist_directory,
            embedding_manager=self.embedding_manager
        )
        self.llm_manager = LLMManager(use_local=self.use_local_llm)
        
        self.is_initialized = False
        
    def ingest_documents(
        self,
        source_path: Optional[Path] = None,
        documents: Optional[List[Document]] = None
    ) -> int:
        """
        Ingest documents into the RAG system
        
        Args:
            source_path: Path to directory or file to ingest
            documents: Pre-loaded documents to ingest
            
        Returns:
            Number of documents ingested
        """
        print("\n" + "="*50)
        print("📥 DOCUMENT INGESTION")
        print("="*50 + "\n")
        
        # Load documents if not provided
        if documents is None:
            if source_path:
                source_path = Path(source_path)
                if source_path.is_file():
                    documents = self.document_loader.load_single_document(str(source_path))
                else:
                    documents = self.document_loader.load_directory(source_path)
            else:
                documents = self.document_loader.load_directory()
        
        if not documents:
            print("⚠️ No documents to ingest")
            return 0
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Create vector store
        self.vector_store.create_from_documents(chunks)
        self.is_initialized = True
        
        print(f"\n✅ Ingestion complete: {len(chunks)} chunks indexed")
        return len(chunks)
    
    def load_existing_store(self) -> bool:
        """
        Load an existing vector store
        
        Returns:
            True if successfully loaded, False otherwise
        """
        self.vector_store.load_existing()
        
        stats = self.vector_store.get_collection_stats()
        if stats.get("count", 0) > 0:
            self.is_initialized = True
            print(f"✅ Loaded existing store with {stats['count']} documents")
            return True
        
        return False
    
    def query(
        self,
        question: str,
        k: int = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query the RAG system
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            return_sources: Whether to include source documents in response
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        if not self.is_initialized:
            return {
                "answer": "The RAG system is not initialized. Please ingest documents first.",
                "sources": [],
                "error": "not_initialized"
            }
        
        k = k or Config.TOP_K_RESULTS
        
        # Retrieve relevant documents
        retrieved_docs = self.vector_store.similarity_search(question, k=k)
        
        if not retrieved_docs:
            return {
                "answer": "No relevant documents found for your query.",
                "sources": [],
                "num_retrieved": 0
            }
        
        # Generate response
        result = self.llm_manager.generate_response_from_documents(question, retrieved_docs)
        
        if not return_sources:
            result.pop("sources", None)
        
        result["num_retrieved"] = len(retrieved_docs)
        return result
    
    def query_with_scores(
        self,
        question: str,
        k: int = None
    ) -> Dict[str, Any]:
        """
        Query with relevance scores
        
        Args:
            question: User's question
            k: Number of documents to retrieve
            
        Returns:
            Dictionary with answer, sources with scores, and metadata
        """
        if not self.is_initialized:
            return {
                "answer": "The RAG system is not initialized.",
                "sources": [],
                "error": "not_initialized"
            }
        
        k = k or Config.TOP_K_RESULTS
        
        # Retrieve with scores
        results_with_scores = self.vector_store.similarity_search_with_score(question, k=k)
        
        if not results_with_scores:
            return {
                "answer": "No relevant documents found.",
                "sources": [],
                "num_retrieved": 0
            }
        
        # Separate documents and scores
        documents = [doc for doc, score in results_with_scores]
        
        # Generate response
        result = self.llm_manager.generate_response_from_documents(question, documents)
        
        # Add scores to sources
        result["sources_with_scores"] = [
            {
                "source": doc.metadata.get("source", "Unknown"),
                "score": float(score),
                "content_preview": doc.page_content[:200] + "..."
            }
            for doc, score in results_with_scores
        ]
        
        return result
    
    def chat(self, question: str) -> str:
        """
        Simple chat interface
        
        Args:
            question: User's question
            
        Returns:
            Answer string
        """
        result = self.query(question, return_sources=False)
        return result.get("answer", "I couldn't generate a response.")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get RAG system statistics
        
        Returns:
            Dictionary with system statistics
        """
        vector_stats = self.vector_store.get_collection_stats()
        
        return {
            "is_initialized": self.is_initialized,
            "use_local_embeddings": self.use_local_embeddings,
            "use_local_llm": self.use_local_llm,
            "embedding_model": Config.EMBEDDING_MODEL if not self.use_local_embeddings else Config.LOCAL_EMBEDDING_MODEL,
            "llm_model": Config.LLM_MODEL,
            "vector_store": vector_stats,
            "chunk_size": Config.CHUNK_SIZE,
            "chunk_overlap": Config.CHUNK_OVERLAP,
            "top_k": Config.TOP_K_RESULTS
        }
    
    def reset(self):
        """Reset the RAG system by deleting the vector store"""
        self.vector_store.delete_collection()
        self.is_initialized = False
        print("🔄 RAG system reset")


def create_rag_chain(
    use_local: bool = False,
    ingest_path: Optional[Path] = None
) -> RAGChain:
    """
    Factory function to create and optionally initialize a RAG chain
    
    Args:
        use_local: Use local models for both embeddings and LLM
        ingest_path: Optional path to ingest documents from
        
    Returns:
        Initialized RAGChain instance
    """
    chain = RAGChain(
        use_local_embeddings=use_local,
        use_local_llm=use_local
    )
    
    # Try to load existing store
    if chain.load_existing_store():
        print("Using existing vector store")
    elif ingest_path:
        chain.ingest_documents(source_path=ingest_path)
    
    return chain


if __name__ == "__main__":
    print("Testing RAG Chain...")
    print("="*50)
    
    # Show configuration
    print(f"Knowledge base: {Config.KNOWLEDGE_BASE_DIR}")
    print(f"Vector store: {Config.CHROMA_PERSIST_DIR}")
    
    # Try to initialize (will work with local models)
    try:
        chain = RAGChain(use_local_embeddings=True, use_local_llm=False)
        stats = chain.get_stats()
        print(f"\nRAG Chain Stats: {stats}")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure you have the required dependencies and API keys configured.")
