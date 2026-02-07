"""
Vector Store Module
Handles FAISS vector store operations for document storage and retrieval.
"""

import os
from typing import List, Optional, Tuple
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class VectorStoreManager:
    """Manages FAISS vector store operations."""
    
    def __init__(
        self,
        embeddings: HuggingFaceEmbeddings,
        store_path: str = "./data/vector_store"
    ):
        """
        Initialize the VectorStoreManager.
        
        Args:
            embeddings: Embeddings model to use.
            store_path: Path to save/load the vector store.
        """
        self.embeddings = embeddings
        self.store_path = store_path
        self.vector_store: Optional[FAISS] = None
        
        # Ensure the store directory exists
        os.makedirs(os.path.dirname(store_path) if os.path.dirname(store_path) else ".", exist_ok=True)
    
    def create_store(self, documents: List[Document]) -> FAISS:
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of Document objects to index.
            
        Returns:
            FAISS vector store instance.
        """
        if not documents:
            raise ValueError("No documents provided to create vector store.")
        
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )
        
        return self.vector_store
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to an existing vector store.
        
        Args:
            documents: List of Document objects to add.
        """
        if self.vector_store is None:
            self.create_store(documents)
        else:
            self.vector_store.add_documents(documents)
    
    def save_store(self, path: Optional[str] = None) -> None:
        """
        Save the vector store to disk.
        
        Args:
            path: Optional path to save to. Uses default path if not provided.
        """
        if self.vector_store is None:
            raise ValueError("No vector store to save. Create one first.")
        
        save_path = path or self.store_path
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        self.vector_store.save_local(save_path)
    
    def load_store(self, path: Optional[str] = None) -> FAISS:
        """
        Load a vector store from disk.
        
        Args:
            path: Optional path to load from. Uses default path if not provided.
            
        Returns:
            Loaded FAISS vector store.
        """
        load_path = path or self.store_path
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Vector store not found at: {load_path}")
        
        self.vector_store = FAISS.load_local(
            load_path,
            self.embeddings,
            allow_dangerous_deserialization=True
        )
        
        return self.vector_store
    
    def store_exists(self, path: Optional[str] = None) -> bool:
        """
        Check if a vector store exists at the given path.
        
        Args:
            path: Optional path to check. Uses default path if not provided.
            
        Returns:
            True if store exists, False otherwise.
        """
        check_path = path or self.store_path
        return os.path.exists(check_path) and os.path.exists(os.path.join(check_path, "index.faiss"))
    
    def similarity_search(
        self,
        query: str,
        k: int = 4
    ) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Query text to search for.
            k: Number of results to return.
            
        Returns:
            List of most similar Document objects.
        """
        if self.vector_store is None:
            raise ValueError("No vector store available. Create or load one first.")
        
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search and return scores.
        
        Args:
            query: Query text to search for.
            k: Number of results to return.
            
        Returns:
            List of tuples containing (Document, score).
        """
        if self.vector_store is None:
            raise ValueError("No vector store available. Create or load one first.")
        
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """
        Get a retriever interface for the vector store.
        
        Args:
            search_kwargs: Optional search parameters.
            
        Returns:
            Retriever instance.
        """
        if self.vector_store is None:
            raise ValueError("No vector store available. Create or load one first.")
        
        if search_kwargs is None:
            search_kwargs = {"k": 4}
        
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def clear_store(self) -> None:
        """Clear the current vector store from memory."""
        self.vector_store = None
    
    def delete_store(self, path: Optional[str] = None) -> None:
        """
        Delete the vector store from disk.
        
        Args:
            path: Optional path to delete. Uses default path if not provided.
        """
        import shutil
        import gc
        
        delete_path = path or self.store_path
        
        # Clear reference to vector store first to release file handles
        self.vector_store = None
        gc.collect()  # Force garbage collection to release file handles
        
        if os.path.exists(delete_path):
            def handle_remove_error(func, path, exc_info):
                """Handle errors during file removal on Windows."""
                import stat
                # Try to change file permissions and retry
                try:
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                except Exception:
                    pass  # Ignore if still can't delete
            
            try:
                shutil.rmtree(delete_path, onerror=handle_remove_error)
            except Exception:
                # If rmtree fails, try to delete files one by one
                try:
                    for root, dirs, files in os.walk(delete_path, topdown=False):
                        for name in files:
                            try:
                                file_path = os.path.join(root, name)
                                os.chmod(file_path, 0o777)
                                os.remove(file_path)
                            except Exception:
                                pass
                        for name in dirs:
                            try:
                                os.rmdir(os.path.join(root, name))
                            except Exception:
                                pass
                    os.rmdir(delete_path)
                except Exception:
                    pass  # Directory might still exist but we tried our best
