"""
Embeddings Module
Handles generation of text embeddings using HuggingFace models.
"""

from typing import List, Optional
from langchain_community.embeddings import HuggingFaceEmbeddings


class EmbeddingManager:
    """Manages embedding generation using HuggingFace models."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        cache_folder: Optional[str] = None
    ):
        """
        Initialize the EmbeddingManager.
        
        Args:
            model_name: Name of the HuggingFace embedding model.
            device: Device to run the model on ('cpu' or 'cuda').
            cache_folder: Optional folder to cache the model.
        """
        self.model_name = model_name
        self.device = device
        
        model_kwargs = {"device": device}
        encode_kwargs = {"normalize_embeddings": True}
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            cache_folder=cache_folder
        )
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed.
            
        Returns:
            Embedding vector as a list of floats.
        """
        return self.embeddings.embed_query(text)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed.
            
        Returns:
            List of embedding vectors.
        """
        return self.embeddings.embed_documents(texts)
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.
        
        Returns:
            Embedding dimension.
        """
        # Generate a test embedding to get dimension
        test_embedding = self.embed_text("test")
        return len(test_embedding)
    
    def get_embeddings(self) -> HuggingFaceEmbeddings:
        """
        Get the underlying embeddings object for use with vector stores.
        
        Returns:
            HuggingFaceEmbeddings instance.
        """
        return self.embeddings
