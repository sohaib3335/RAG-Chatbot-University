"""
Vector Store Module
Handles storage and retrieval of document embeddings using ChromaDB
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
from langchain.schema import Document

from src.config import Config
from src.embeddings import EmbeddingManager


class VectorStore:
    """
    Vector Store class using ChromaDB for storing and retrieving document embeddings.
    """
    
    def __init__(
        self,
        persist_directory: Optional[Path] = None,
        collection_name: str = None,
        embedding_manager: Optional[EmbeddingManager] = None
    ):
        """
        Initialize VectorStore
        
        Args:
            persist_directory: Directory to persist the vector store
            collection_name: Name of the ChromaDB collection
            embedding_manager: EmbeddingManager instance
        """
        self.persist_directory = str(persist_directory or Config.CHROMA_PERSIST_DIR)
        self.collection_name = collection_name or Config.COLLECTION_NAME
        self.embedding_manager = embedding_manager or EmbeddingManager()
        self.vectorstore = None
        
    def create_from_documents(self, documents: List[Document]) -> "VectorStore":
        """
        Create vector store from documents
        
        Args:
            documents: List of Document objects to index
            
        Returns:
            Self for method chaining
        """
        from langchain_community.vectorstores import Chroma
        
        if not documents:
            print("⚠️ No documents provided to index")
            return self
        
        print(f"📦 Creating vector store with {len(documents)} documents...")
        
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_manager.get_embeddings(),
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )
        
        print(f"✓ Vector store created and persisted to: {self.persist_directory}")
        return self
    
    def load_existing(self) -> "VectorStore":
        """
        Load existing vector store from disk
        
        Returns:
            Self for method chaining
        """
        from langchain_community.vectorstores import Chroma
        
        if not Path(self.persist_directory).exists():
            print(f"⚠️ No existing vector store found at: {self.persist_directory}")
            return self
        
        print(f"📂 Loading vector store from: {self.persist_directory}")
        
        self.vectorstore = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_manager.get_embeddings(),
            collection_name=self.collection_name
        )
        
        print("✓ Vector store loaded successfully")
        return self
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to existing vector store
        
        Args:
            documents: List of Document objects to add
            
        Returns:
            List of document IDs
        """
        if self.vectorstore is None:
            print("⚠️ Vector store not initialized. Creating new one...")
            return self.create_from_documents(documents)
        
        ids = self.vectorstore.add_documents(documents)
        print(f"✓ Added {len(documents)} documents to vector store")
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: int = None,
        filter: Dict[str, Any] = None
    ) -> List[Document]:
        """
        Perform similarity search
        
        Args:
            query: Query string
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of relevant Document objects
        """
        if self.vectorstore is None:
            print("⚠️ Vector store not initialized")
            return []
        
        k = k or Config.TOP_K_RESULTS
        
        results = self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter
        )
        
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = None
    ) -> List[tuple]:
        """
        Perform similarity search with relevance scores
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        if self.vectorstore is None:
            print("⚠️ Vector store not initialized")
            return []
        
        k = k or Config.TOP_K_RESULTS
        
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k
        )
        
        return results
    
    def get_retriever(self, search_kwargs: Dict[str, Any] = None):
        """
        Get a retriever interface for the vector store
        
        Args:
            search_kwargs: Optional search parameters
            
        Returns:
            Retriever object
        """
        if self.vectorstore is None:
            print("⚠️ Vector store not initialized")
            return None
        
        search_kwargs = search_kwargs or {"k": Config.TOP_K_RESULTS}
        
        return self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
    
    def delete_collection(self):
        """Delete the current collection"""
        if self.vectorstore is not None:
            self.vectorstore.delete_collection()
            print("✓ Collection deleted")
            self.vectorstore = None
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the collection
        
        Returns:
            Dictionary with collection statistics
        """
        if self.vectorstore is None:
            return {"status": "not_initialized"}
        
        try:
            collection = self.vectorstore._collection
            return {
                "name": self.collection_name,
                "count": collection.count(),
                "persist_directory": self.persist_directory
            }
        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    # Test the vector store
    print("Testing Vector Store...")
    
    # Create test documents
    test_docs = [
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "test", "topic": "ML"}
        ),
        Document(
            page_content="Deep learning uses neural networks with many layers.",
            metadata={"source": "test", "topic": "DL"}
        ),
        Document(
            page_content="Natural language processing deals with text and speech.",
            metadata={"source": "test", "topic": "NLP"}
        )
    ]
    
    try:
        # Initialize with local embeddings for testing
        embedding_manager = EmbeddingManager(use_local=True)
        vs = VectorStore(embedding_manager=embedding_manager)
        vs.create_from_documents(test_docs)
        
        # Test search
        results = vs.similarity_search("What is machine learning?", k=2)
        print(f"\nSearch results for 'What is machine learning?':")
        for doc in results:
            print(f"  - {doc.page_content}")
            
        # Get stats
        stats = vs.get_collection_stats()
        print(f"\nCollection stats: {stats}")
        
    except Exception as e:
        print(f"Error: {e}")
