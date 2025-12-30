"""
Document Loader Module
Handles loading documents from various file formats (PDF, TXT, DOCX, etc.)
Uses PyMuPDF for better PDF parsing
"""

import os
from pathlib import Path
from typing import List, Optional
from langchain.schema import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyMuPDFLoader,  # Using PyMuPDF for better PDF parsing
    Docx2txtLoader,
    DirectoryLoader,
    CSVLoader,
    UnstructuredMarkdownLoader
)

from src.config import Config


class DocumentLoader:
    """
    Document Loader class for loading documents from various formats.
    Supports: PDF, TXT, DOCX, CSV, MD
    """
    
    # Mapping of file extensions to loader classes
    # Using PyMuPDF for PDFs - better text extraction and layout preservation
    LOADER_MAPPING = {
        ".txt": TextLoader,
        ".pdf": PyMuPDFLoader,  # PyMuPDF for superior PDF parsing
        ".docx": Docx2txtLoader,
        ".csv": CSVLoader,
        ".md": UnstructuredMarkdownLoader,
    }
    
    def __init__(self, knowledge_base_path: Optional[Path] = None):
        """
        Initialize DocumentLoader
        
        Args:
            knowledge_base_path: Path to knowledge base directory
        """
        self.knowledge_base_path = knowledge_base_path or Config.KNOWLEDGE_BASE_DIR
        
    def load_single_document(self, file_path: str) -> List[Document]:
        """
        Load a single document from file path
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects
        """
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        
        if ext not in self.LOADER_MAPPING:
            raise ValueError(f"Unsupported file format: {ext}")
        
        loader_class = self.LOADER_MAPPING[ext]
        
        try:
            loader = loader_class(str(file_path))
            documents = loader.load()
            
            # Add metadata
            for doc in documents:
                doc.metadata["source"] = str(file_path)
                doc.metadata["filename"] = file_path.name
                doc.metadata["file_type"] = ext
                
            print(f"✓ Loaded: {file_path.name} ({len(documents)} documents)")
            return documents
            
        except Exception as e:
            print(f"✗ Error loading {file_path.name}: {str(e)}")
            return []
    
    def load_directory(self, directory_path: Optional[Path] = None, recursive: bool = True) -> List[Document]:
        """
        Load all documents from a directory
        
        Args:
            directory_path: Path to directory (defaults to knowledge_base_path)
            recursive: Whether to search subdirectories recursively
            
        Returns:
            List of all Document objects
        """
        directory_path = directory_path or self.knowledge_base_path
        all_documents = []
        
        if not directory_path.exists():
            print(f"Directory not found: {directory_path}")
            return all_documents
        
        print(f"📂 Scanning directory: {directory_path}")
        
        # Get all files in directory (recursively if specified)
        for ext in self.LOADER_MAPPING.keys():
            # Use ** for recursive search, * for non-recursive
            pattern = f"**/*{ext}" if recursive else f"*{ext}"
            files = list(directory_path.glob(pattern))
            
            if files:
                print(f"  Found {len(files)} {ext} files")
            
            for file_path in files:
                documents = self.load_single_document(str(file_path))
                all_documents.extend(documents)
        
        print(f"\n📚 Total documents loaded: {len(all_documents)}")
        return all_documents
    
    def get_supported_extensions(self) -> List[str]:
        """Get list of supported file extensions"""
        return list(self.LOADER_MAPPING.keys())


class TextSplitter:
    """
    Text Splitter class for chunking documents into smaller pieces
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        separators: List[str] = None
    ):
        """
        Initialize TextSplitter
        
        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            separators: List of separators to use for splitting
        """
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into chunks
        
        Args:
            documents: List of Document objects
            
        Returns:
            List of chunked Document objects
        """
        if not documents:
            return []
        
        chunks = self.splitter.split_documents(documents)
        
        # Add chunk index to metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(chunks)
        
        print(f"📄 Split into {len(chunks)} chunks")
        return chunks


def load_and_split_documents(
    directory_path: Optional[Path] = None,
    chunk_size: int = None,
    chunk_overlap: int = None
) -> List[Document]:
    """
    Convenience function to load and split documents in one call
    
    Args:
        directory_path: Path to knowledge base directory
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunked Document objects
    """
    loader = DocumentLoader(directory_path)
    documents = loader.load_directory()
    
    if not documents:
        return []
    
    splitter = TextSplitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(documents)
    
    return chunks


if __name__ == "__main__":
    # Test the document loader
    print("Testing Document Loader...")
    print(f"Knowledge base path: {Config.KNOWLEDGE_BASE_DIR}")
    
    loader = DocumentLoader()
    print(f"Supported formats: {loader.get_supported_extensions()}")
    
    documents = loader.load_directory()
    
    if documents:
        splitter = TextSplitter()
        chunks = splitter.split_documents(documents)
        print(f"\nSample chunk content:\n{chunks[0].page_content[:200]}...")
