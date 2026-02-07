"""
QA Chain Module
Handles the question-answering chain that combines retrieval with Grok LLM.
"""

from typing import List, Optional, Dict, Any
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage

from .llm import GrokLLM


class QAChain:
    """Manages the question-answering pipeline."""
    
    DEFAULT_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions based on the provided context. 

Instructions:
- Answer questions accurately based on the provided context
- If the context doesn't contain enough information to answer the question, say so clearly
- Be concise but thorough in your responses
- Cite relevant parts of the context when appropriate
- If asked about something not in the context, acknowledge that and provide a general response if possible"""
    
    def __init__(
        self,
        llm: GrokLLM,
        retriever: Any,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the QA Chain.
        
        Args:
            llm: Grok LLM instance.
            retriever: Vector store retriever.
            system_prompt: Optional custom system prompt.
        """
        self.llm = llm
        self.retriever = retriever
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.chat_history: List[tuple] = []
    
    def _format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into a context string."""
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            source_info = f"[Source {i}: {source}"
            if page:
                source_info += f", Page {page}"
            source_info += "]"
            context_parts.append(f"{source_info}\n{doc.page_content}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def ask(
        self,
        question: str,
        include_history: bool = True
    ) -> Dict[str, Any]:
        """
        Ask a question and get an answer based on the document context.
        
        Args:
            question: User's question.
            include_history: Whether to include chat history.
            
        Returns:
            Dictionary containing the answer and source documents.
        """
        # Retrieve relevant documents
        context_docs = self.retriever.invoke(question)
        
        # Format context
        context = self._format_context(context_docs)
        
        # Build the prompt with history
        history_text = ""
        if include_history and self.chat_history:
            history_parts = []
            for human_msg, ai_msg in self.chat_history[-5:]:
                history_parts.append(f"User: {human_msg}\nAssistant: {ai_msg}")
            history_text = "\n\nPrevious conversation:\n" + "\n\n".join(history_parts)
        
        full_prompt = f"""{self.system_prompt}

Context from documents:
{context}
{history_text}

User Question: {question}

Please provide a helpful answer based on the context above:"""
        
        # Generate response
        answer = self.llm.generate_response(full_prompt)
        
        # Update chat history
        self.chat_history.append((question, answer))
        
        return {
            "answer": answer,
            "source_documents": context_docs,
            "question": question
        }
    
    def ask_simple(self, question: str) -> str:
        """
        Simple interface to ask a question and get just the answer.
        
        Args:
            question: User's question.
            
        Returns:
            Answer string.
        """
        result = self.ask(question)
        return result["answer"]
    
    def ask_with_context(
        self,
        question: str,
        context_docs: List[Document]
    ) -> str:
        """
        Ask a question with pre-retrieved context documents.
        
        Args:
            question: User's question.
            context_docs: List of context documents.
            
        Returns:
            Answer string.
        """
        # Format context
        context = "\n\n".join([doc.page_content for doc in context_docs])
        
        # Create prompt with context
        full_prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
        
        return self.llm.generate_response(full_prompt)
    
    def clear_history(self) -> None:
        """Clear the chat history."""
        self.chat_history = []
    
    def get_history(self) -> List[tuple]:
        """
        Get the chat history.
        
        Returns:
            List of (question, answer) tuples.
        """
        return self.chat_history.copy()
    
    def format_sources(self, documents: List[Document]) -> str:
        """
        Format source documents for display.
        
        Args:
            documents: List of source documents.
            
        Returns:
            Formatted string of sources.
        """
        if not documents:
            return "No sources found."
        
        sources = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            
            source_info = f"**Source {i}**: {source}"
            if page:
                source_info += f" (Page {page})"
            
            # Add a preview of the content
            preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            source_info += f"\n> {preview}"
            
            sources.append(source_info)
        
        return "\n\n".join(sources)


def create_qa_chain(
    llm: GrokLLM,
    retriever: Any,
    system_prompt: Optional[str] = None
) -> QAChain:
    """
    Factory function to create a QA Chain.
    
    Args:
        llm: Grok LLM instance.
        retriever: Vector store retriever.
        system_prompt: Optional custom system prompt.
        
    Returns:
        Configured QAChain instance.
    """
    return QAChain(
        llm=llm,
        retriever=retriever,
        system_prompt=system_prompt
    )
