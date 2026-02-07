# 📚 AI Document Search Assistant

A powerful document search assistant built with Streamlit, LangChain, and Grok LLM. Upload documents, ask questions, and get AI-powered answers based on your document content.

## ✨ Features

- **Multi-format Document Support**: Upload PDF, TXT, and DOCX files
- **Intelligent Text Processing**: Automatic text extraction and chunking
- **Semantic Search**: FAISS-powered vector similarity search
- **Grok LLM Integration**: Powered by xAI's Grok for intelligent responses
- **Chat Interface**: Conversational UI with chat history
- **Source Attribution**: View the document sources for each answer
- **Persistent Storage**: Vector store is saved locally for quick reloading

## 📁 Project Structure

```
aidoc/
├── app.py                      # Main Streamlit application
├── backend/
│   ├── __init__.py            # Package initialization
│   ├── document_loader.py     # Document loading (PDF, TXT, DOCX)
│   ├── text_processor.py      # Text chunking and preprocessing
│   ├── embeddings.py          # HuggingFace embedding generation
│   ├── vector_store.py        # FAISS vector store operations
│   ├── llm.py                 # Grok LLM integration
│   └── qa_chain.py            # Question-answering chain
├── data/
│   └── vector_store/          # FAISS index storage
├── uploads/                    # Uploaded documents (optional)
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variables template
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- Grok API key from [xAI Console](https://console.x.ai/)

### Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd aidoc
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   # Copy the example file
   copy .env.example .env    # Windows
   cp .env.example .env      # macOS/Linux
   
   # Edit .env and add your Grok API key
   ```

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

6. **Open your browser** and navigate to `http://localhost:8501`

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GROK_API_KEY` | Your Grok API key (required) | - |
| `GROK_API_BASE` | Grok API base URL | `https://api.x.ai/v1` |
| `GROK_MODEL` | Grok model to use | `grok-beta` |
| `EMBEDDING_MODEL` | HuggingFace embedding model | `all-MiniLM-L6-v2` |
| `VECTOR_STORE_PATH` | Path to store FAISS index | `./data/vector_store` |
| `CHUNK_SIZE` | Text chunk size | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |

### Getting a Grok API Key

1. Visit [xAI Console](https://console.x.ai/)
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key to your `.env` file

## 📖 Usage Guide

### Uploading Documents

1. Click on the file uploader in the sidebar
2. Select one or more documents (PDF, TXT, or DOCX)
3. Click "Process Documents" to extract and index the content
4. Wait for processing to complete

### Asking Questions

1. Type your question in the chat input at the bottom
2. Press Enter or click Send
3. The AI will search your documents and generate an answer
4. Click "View Sources" to see which document sections were used

### Tips for Best Results

- **Be specific**: Ask focused questions for more accurate answers
- **Upload related documents**: Group similar documents together
- **Use natural language**: Ask questions as you would to a human
- **Check sources**: Always verify answers against the source documents

## 🛠️ Technical Details

### Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python
- **AI Framework**: LangChain
- **LLM**: Grok (xAI)
- **Vector Database**: FAISS
- **Embeddings**: HuggingFace Sentence Transformers

### Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Streamlit     │────▶│  Document       │
│   Frontend      │     │  Loader         │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
┌─────────────────┐     ┌─────────────────┐
│   QA Chain      │◀────│  Text           │
│   (LangChain)   │     │  Processor      │
└────────┬────────┘     └────────┬────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│   Grok LLM      │     │  Embedding      │
│   API           │     │  Manager        │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │  FAISS Vector   │
                        │  Store          │
                        └─────────────────┘
```

### How It Works

1. **Document Loading**: Documents are parsed and text is extracted
2. **Text Chunking**: Text is split into overlapping chunks for better context
3. **Embedding Generation**: Each chunk is converted to a vector embedding
4. **Vector Storage**: Embeddings are stored in FAISS for fast retrieval
5. **Query Processing**: User questions are embedded and matched against stored vectors
6. **Response Generation**: Relevant chunks are sent to Grok LLM for answer generation

## 🔧 Troubleshooting

### Common Issues

**"Grok API key is required"**
- Ensure your `.env` file exists and contains a valid `GROK_API_KEY`
- Check that the API key is correct (no extra spaces)

**"Error loading embedding model"**
- First run may take time to download the model
- Ensure you have internet connection
- Try: `pip install --upgrade sentence-transformers`

**"Vector store not found"**
- Upload and process documents first
- Check that `./data/vector_store` directory exists

**PDF extraction issues**
- Ensure the PDF is not password protected
- Try a different PDF if text extraction fails

### Performance Tips

- Use smaller chunk sizes for more precise answers
- Increase chunk overlap for better context
- Limit the number of context chunks for faster responses

## 📄 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

---

Built with ❤️ using Streamlit, LangChain, and Grok
