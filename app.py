"""
AI Document Search Assistant
Main Streamlit Application
"""

import os
import tempfile
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import backend modules
from backend.document_loader import DocumentLoader
from backend.text_processor import TextProcessor
from backend.embeddings import EmbeddingManager
from backend.vector_store import VectorStoreManager
from backend.llm import GrokLLM
from backend.qa_chain import QAChain

# Page configuration
st.set_page_config(
    page_title="AI Document Search Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for chat interface
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #e3f2fd;
    }
    .chat-message.assistant {
        background-color: #f5f5f5;
    }
    .chat-message .message-content {
        margin-top: 0.5rem;
    }
    .source-box {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
    }
    .stats-box {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vector_store_manager" not in st.session_state:
        st.session_state.vector_store_manager = None
    
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None
    
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    
    if "document_stats" not in st.session_state:
        st.session_state.document_stats = {}
    
    if "llm" not in st.session_state:
        st.session_state.llm = None
    
    if "embedding_manager" not in st.session_state:
        st.session_state.embedding_manager = None


def get_config():
    """Get configuration from environment variables or Streamlit secrets."""
    # Try Streamlit secrets first (for Streamlit Cloud), then fall back to env vars
    def get_secret(key, default=""):
        try:
            return st.secrets.get(key, os.getenv(key, default))
        except Exception:
            return os.getenv(key, default)
    
    return {
        "hf_api_key": get_secret("HF_API_KEY", ""),
        "hf_model": get_secret("HF_MODEL", "microsoft/Phi-3-mini-4k-instruct"),
        "embedding_model": get_secret("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        "vector_store_path": get_secret("VECTOR_STORE_PATH", "./data/vector_store"),
        "chunk_size": int(get_secret("CHUNK_SIZE", "1000")),
        "chunk_overlap": int(get_secret("CHUNK_OVERLAP", "200"))
    }


def initialize_components(api_key: str, config: dict):
    """Initialize the backend components."""
    try:
        # Initialize embedding manager
        if st.session_state.embedding_manager is None:
            with st.spinner("Loading embedding model..."):
                st.session_state.embedding_manager = EmbeddingManager(
                    model_name=config["embedding_model"]
                )
        
        # Initialize LLM
        if st.session_state.llm is None:
            st.session_state.llm = GrokLLM(
                api_key=api_key,
                model_name=config["hf_model"]
            )
        
        # Initialize vector store manager
        if st.session_state.vector_store_manager is None:
            st.session_state.vector_store_manager = VectorStoreManager(
                embeddings=st.session_state.embedding_manager.get_embeddings(),
                store_path=config["vector_store_path"]
            )
            
            # Load existing vector store if available
            if st.session_state.vector_store_manager.store_exists():
                with st.spinner("Loading existing vector store..."):
                    st.session_state.vector_store_manager.load_store()
                    st.session_state.documents_loaded = True
        
        return True
    
    except Exception as e:
        st.error(f"Error initializing components: {str(e)}")
        return False


def process_uploaded_files(uploaded_files, config: dict):
    """Process uploaded files and add them to the vector store."""
    if not uploaded_files:
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_chunks = []
    document_loader = DocumentLoader()
    text_processor = TextProcessor(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"]
    )
    
    total_files = len(uploaded_files)
    
    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"Processing: {uploaded_file.name}")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Load and process document
            documents = document_loader.load_document(tmp_path)
            chunks = text_processor.split_documents(documents)
            
            # Update source metadata to use original filename
            for chunk in chunks:
                chunk.metadata["source"] = uploaded_file.name
            
            all_chunks.extend(chunks)
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            progress_bar.progress((idx + 1) / total_files)
            
        except Exception as e:
            st.warning(f"Error processing {uploaded_file.name}: {str(e)}")
            continue
    
    if all_chunks:
        status_text.text("Creating embeddings and updating vector store...")
        
        # Add to vector store
        if st.session_state.vector_store_manager.vector_store is None:
            st.session_state.vector_store_manager.create_store(all_chunks)
        else:
            st.session_state.vector_store_manager.add_documents(all_chunks)
        
        # Save vector store
        st.session_state.vector_store_manager.save_store()
        
        # Update QA chain
        retriever = st.session_state.vector_store_manager.get_retriever({"k": 4})
        st.session_state.qa_chain = QAChain(
            llm=st.session_state.llm,
            retriever=retriever
        )
        
        # Update stats
        stats = text_processor.get_statistics(all_chunks)
        st.session_state.document_stats = stats
        st.session_state.documents_loaded = True
        
        status_text.text("✅ Documents processed successfully!")
        progress_bar.progress(1.0)
        
        return True
    
    return False


def display_chat_message(role: str, content: str, sources: Optional[List] = None):
    """Display a chat message with optional sources."""
    with st.chat_message(role):
        st.markdown(content)
        
        if sources and role == "assistant":
            with st.expander("📚 View Sources"):
                for i, doc in enumerate(sources, 1):
                    source = doc.metadata.get("source", "Unknown")
                    page = doc.metadata.get("page", "")
                    
                    st.markdown(f"**Source {i}**: {source}" + (f" (Page {page})" if page else ""))
                    st.text(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                    st.divider()


def main():
    """Main application function."""
    initialize_session_state()
    config = get_config()
    
    # Header
    st.title("📚 AI Document Search Assistant")
    st.markdown("Upload documents and ask questions using AI")
    
    # Get API key from config (backend)
    api_key = config["hf_api_key"]
    
    if not api_key:
        st.error("⚠️ HuggingFace API key not configured. Please set HF_API_KEY in the .env file.")
        st.stop()
    
    # Initialize components
    if not initialize_components(api_key, config):
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📄 Document Upload")
        
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, DOCX"
        )
        
        if uploaded_files:
            if st.button("📥 Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    if process_uploaded_files(uploaded_files, config):
                        st.success(f"Processed {len(uploaded_files)} document(s)")
                        st.rerun()
        
        st.divider()
        
        # Document Stats
        if st.session_state.documents_loaded and st.session_state.document_stats:
            st.header("📊 Document Statistics")
            stats = st.session_state.document_stats
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Chunks", stats.get("total_documents", 0))
            with col2:
                st.metric("Avg. Chunk Size", f"{stats.get('average_chunk_size', 0):.0f}")
            
            st.metric("Total Characters", f"{stats.get('total_characters', 0):,}")
        
        st.divider()
        
        # Settings
        st.header("🔧 Settings")
        
        search_k = st.slider(
            "Number of context chunks",
            min_value=1,
            max_value=10,
            value=4,
            help="Number of relevant document chunks to retrieve for each question"
        )
        
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            if st.session_state.qa_chain:
                st.session_state.qa_chain.clear_history()
            st.rerun()
        
        if st.button("🔄 Reset Vector Store"):
            if st.session_state.vector_store_manager:
                st.session_state.vector_store_manager.delete_store()
                st.session_state.vector_store_manager.clear_store()
            st.session_state.qa_chain = None
            st.session_state.documents_loaded = False
            st.session_state.document_stats = {}
            st.session_state.messages = []
            st.success("Vector store reset!")
            st.rerun()
    
    # Main chat area
    if not st.session_state.documents_loaded:
        st.info("👈 Please upload documents using the sidebar to get started.")
        
        # Show example usage
        with st.expander("📖 How to use this app"):
            st.markdown("""
            1. **Upload documents** (PDF, TXT, or DOCX files)
            2. **Click 'Process Documents'** to extract and index the content
            3. **Ask questions** in the chat input below
            4. The AI will search your documents and provide relevant answers
            
            **Tips:**
            - Upload multiple documents at once for better context
            - Ask specific questions for more accurate answers
            - Use the 'View Sources' button to see where answers came from
            """)
    else:
        # Ensure QA chain is initialized
        if st.session_state.qa_chain is None and st.session_state.vector_store_manager.vector_store is not None:
            retriever = st.session_state.vector_store_manager.get_retriever({"k": search_k if 'search_k' in dir() else 4})
            st.session_state.qa_chain = QAChain(
                llm=st.session_state.llm,
                retriever=retriever
            )
        
        # Display chat history
        for message in st.session_state.messages:
            display_chat_message(
                message["role"],
                message["content"],
                message.get("sources")
            )
        
        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            display_chat_message("user", prompt)
            
            # Generate response
            with st.spinner("Searching documents and generating answer..."):
                try:
                    # Update retriever with current search_k value
                    retriever = st.session_state.vector_store_manager.get_retriever({"k": search_k if 'search_k' in dir() else 4})
                    st.session_state.qa_chain.retriever = retriever
                    
                    result = st.session_state.qa_chain.ask(prompt)
                    
                    answer = result["answer"]
                    sources = result["source_documents"]
                    
                    # Add assistant message
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                    display_chat_message("assistant", answer, sources)
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


if __name__ == "__main__":
    main()
