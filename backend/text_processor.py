"""
Text Processor Module
Handles text chunking and preprocessing for document indexing.
"""

from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextProcessor:
    """Handles text chunking and preprocessing."""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None
    ):
        """
        Initialize the TextProcessor.
        
        Args:
            chunk_size: Maximum size of each text chunk.
            chunk_overlap: Number of characters to overlap between chunks.
            separators: List of separators to use for splitting text.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        if separators is None:
            separators = ["\n\n", "\n", ". ", " ", ""]
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators,
            length_function=len
        )
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of Document objects to split.
            
        Returns:
            List of chunked Document objects.
        """
        if not documents:
            return []
        
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk index to metadata
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
        
        return chunks
    
    def split_text(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
        """
        Split raw text into document chunks.
        
        Args:
            text: Raw text to split.
            metadata: Optional metadata to attach to each chunk.
            
        Returns:
            List of Document objects.
        """
        if not text.strip():
            return []
        
        if metadata is None:
            metadata = {}
        
        chunks = self.text_splitter.split_text(text)
        
        documents = []
        for idx, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={**metadata, "chunk_index": idx}
            )
            documents.append(doc)
        
        return documents
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text by cleaning and normalizing.
        
        Args:
            text: Raw text to preprocess.
            
        Returns:
            Cleaned text.
        """
        # Remove excessive whitespace
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Remove excessive spaces within lines
            cleaned_line = ' '.join(line.split())
            if cleaned_line:
                cleaned_lines.append(cleaned_line)
        
        # Join lines, preserving paragraph breaks
        result = '\n'.join(cleaned_lines)
        
        return result
    
    def get_statistics(self, documents: List[Document]) -> dict:
        """
        Get statistics about the processed documents.
        
        Args:
            documents: List of Document objects.
            
        Returns:
            Dictionary containing statistics.
        """
        if not documents:
            return {
                "total_documents": 0,
                "total_characters": 0,
                "average_chunk_size": 0,
                "min_chunk_size": 0,
                "max_chunk_size": 0
            }
        
        chunk_sizes = [len(doc.page_content) for doc in documents]
        
        return {
            "total_documents": len(documents),
            "total_characters": sum(chunk_sizes),
            "average_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes)
        }
