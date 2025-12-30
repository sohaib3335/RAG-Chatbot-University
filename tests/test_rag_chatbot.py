"""
RAG Chatbot Test Suite
Unit tests for all components
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain.schema import Document


class TestDocumentLoader:
    """Tests for DocumentLoader class"""
    
    def test_supported_extensions(self):
        """Test that supported extensions are correctly defined"""
        from src.document_loader import DocumentLoader
        
        loader = DocumentLoader()
        extensions = loader.get_supported_extensions()
        
        assert ".txt" in extensions
        assert ".pdf" in extensions
        assert ".docx" in extensions
        assert ".csv" in extensions
        assert ".md" in extensions
    
    def test_load_txt_file(self, tmp_path):
        """Test loading a text file"""
        from src.document_loader import DocumentLoader
        
        # Create a test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("This is a test document.")
        
        loader = DocumentLoader()
        documents = loader.load_single_document(str(test_file))
        
        assert len(documents) > 0
        assert "This is a test document." in documents[0].page_content
        assert documents[0].metadata["file_type"] == ".txt"
    
    def test_unsupported_format(self, tmp_path):
        """Test that unsupported formats raise an error"""
        from src.document_loader import DocumentLoader
        
        test_file = tmp_path / "test.xyz"
        test_file.write_text("test")
        
        loader = DocumentLoader()
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            loader.load_single_document(str(test_file))


class TestTextSplitter:
    """Tests for TextSplitter class"""
    
    def test_split_documents(self):
        """Test document splitting"""
        from src.document_loader import TextSplitter
        
        documents = [
            Document(
                page_content="This is a test document. " * 100,
                metadata={"source": "test"}
            )
        ]
        
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        chunks = splitter.split_documents(documents)
        
        assert len(chunks) > 1
        assert all(len(chunk.page_content) <= 100 + 50 for chunk in chunks)  # Allow some flexibility
    
    def test_empty_documents(self):
        """Test splitting empty document list"""
        from src.document_loader import TextSplitter
        
        splitter = TextSplitter()
        chunks = splitter.split_documents([])
        
        assert chunks == []
    
    def test_metadata_preserved(self):
        """Test that metadata is preserved after splitting"""
        from src.document_loader import TextSplitter
        
        documents = [
            Document(
                page_content="Test content " * 50,
                metadata={"source": "test.txt", "author": "test_author"}
            )
        ]
        
        splitter = TextSplitter(chunk_size=100, chunk_overlap=10)
        chunks = splitter.split_documents(documents)
        
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] == "test.txt"
            assert "chunk_index" in chunk.metadata


class TestEmbeddingManager:
    """Tests for EmbeddingManager class"""
    
    @patch('src.embeddings.HuggingFaceEmbeddings')
    def test_local_embeddings_initialization(self, mock_hf):
        """Test local embeddings initialization"""
        from src.embeddings import EmbeddingManager
        
        mock_hf.return_value = Mock()
        manager = EmbeddingManager(use_local=True)
        
        assert manager.use_local is True
        mock_hf.assert_called_once()
    
    def test_embed_query_returns_list(self):
        """Test that embed_query returns a list of floats"""
        # This would require mocking the embedding model
        pass  # Placeholder for integration test


class TestVectorStore:
    """Tests for VectorStore class"""
    
    def test_initialization(self, tmp_path):
        """Test vector store initialization"""
        from src.vector_store import VectorStore
        from src.embeddings import EmbeddingManager
        
        with patch.object(EmbeddingManager, '_initialize_embeddings') as mock_init:
            mock_init.return_value = Mock()
            
            vs = VectorStore(persist_directory=tmp_path / "test_chroma")
            
            assert vs.vectorstore is None
            assert vs.collection_name is not None
    
    def test_get_collection_stats_not_initialized(self, tmp_path):
        """Test stats when vector store is not initialized"""
        from src.vector_store import VectorStore
        
        with patch('src.embeddings.EmbeddingManager'):
            vs = VectorStore(persist_directory=tmp_path)
            stats = vs.get_collection_stats()
            
            assert stats["status"] == "not_initialized"


class TestLLMManager:
    """Tests for LLMManager class"""
    
    def test_format_documents(self):
        """Test document formatting"""
        from src.llm import LLMManager
        
        documents = [
            Document(page_content="Content 1", metadata={"source": "doc1.txt"}),
            Document(page_content="Content 2", metadata={"source": "doc2.txt"})
        ]
        
        with patch.object(LLMManager, '_initialize_llm'):
            manager = LLMManager()
            formatted = manager._format_documents(documents)
            
            assert "Content 1" in formatted
            assert "Content 2" in formatted
            assert "doc1.txt" in formatted
            assert "doc2.txt" in formatted


class TestRAGChain:
    """Tests for RAGChain class"""
    
    def test_query_not_initialized(self):
        """Test querying when system is not initialized"""
        from src.rag_chain import RAGChain
        
        with patch('src.rag_chain.DocumentLoader'), \
             patch('src.rag_chain.TextSplitter'), \
             patch('src.rag_chain.EmbeddingManager'), \
             patch('src.rag_chain.VectorStore'), \
             patch('src.rag_chain.LLMManager'):
            
            chain = RAGChain()
            chain.is_initialized = False
            
            result = chain.query("Test question")
            
            assert "error" in result
            assert result["error"] == "not_initialized"
    
    def test_get_stats(self):
        """Test getting system statistics"""
        from src.rag_chain import RAGChain
        
        with patch('src.rag_chain.DocumentLoader'), \
             patch('src.rag_chain.TextSplitter'), \
             patch('src.rag_chain.EmbeddingManager'), \
             patch('src.rag_chain.VectorStore') as mock_vs, \
             patch('src.rag_chain.LLMManager'):
            
            mock_vs_instance = Mock()
            mock_vs_instance.get_collection_stats.return_value = {"count": 10}
            mock_vs.return_value = mock_vs_instance
            
            chain = RAGChain()
            stats = chain.get_stats()
            
            assert "is_initialized" in stats
            assert "embedding_model" in stats
            assert "llm_model" in stats


class TestIntegration:
    """Integration tests for the full RAG pipeline"""
    
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        return [
            Document(
                page_content="Machine learning is a subset of artificial intelligence.",
                metadata={"source": "ml.txt"}
            ),
            Document(
                page_content="Deep learning uses neural networks with multiple layers.",
                metadata={"source": "dl.txt"}
            ),
            Document(
                page_content="RAG combines retrieval with generation for better LLM responses.",
                metadata={"source": "rag.txt"}
            )
        ]
    
    def test_load_test_queries(self):
        """Test that test queries file is valid JSON"""
        queries_path = Path(__file__).parent / "test_queries.json"
        
        if queries_path.exists():
            with open(queries_path) as f:
                data = json.load(f)
            
            assert "test_queries" in data
            assert len(data["test_queries"]) > 0
            
            for query in data["test_queries"]:
                assert "id" in query
                assert "query" in query
                assert "category" in query
    
    def test_load_expected_responses(self):
        """Test that expected responses file is valid JSON"""
        responses_path = Path(__file__).parent / "expected_responses.json"
        
        if responses_path.exists():
            with open(responses_path) as f:
                data = json.load(f)
            
            assert "expected_responses" in data
            assert len(data["expected_responses"]) > 0
            
            for response in data["expected_responses"]:
                assert "query_id" in response
                assert "expected_answer_contains" in response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
