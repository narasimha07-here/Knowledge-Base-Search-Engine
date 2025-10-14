# Knowledge Base Search Engine

A full-stack RAG (Retrieval-Augmented Generation) search engine that lets users upload documents and search through them using natural language queries. Built with FastAPI backend and Streamlit frontend, deployed on Azure and Streamlit Cloud.

## What It Does

Ever wished you could just ask questions about your documents instead of manually searching through them? This knowledge base search engine does exactly that. Upload your PDFs, text files, or other documents, and then ask questions in plain English. The system uses AI to understand your question, find relevant information from your documents, and give you a synthesized answer.

## Tech Stack

**Backend (FastAPI)**: [API](https://ragapibuild.azurewebsites.net/docs)
- **FastAPI**: Modern Python web framework for building APIs
- **LangChain**: Framework for working with LLMs and document processing
- **ChromaDB**: Vector database for storing document embeddings
- **HuggingFace Transformers**: For text embeddings (BAAI/bge-small-en-v1.5)
- **OpenAI/OpenRouter**: LLM providers for generating responses
- **SQLite**: Database for user data and document metadata
- **Argon2**: Password hashing for secure authentication
- **PyPDF2**: PDF document processing
- **Pydantic**: Data validation and serialization

**Frontend (Streamlit)**:
- **Streamlit**: Python framework for building web applications
- **Requests**: HTTP client for API communication

**Deployment & Infrastructure**:
- **Azure App Service**: Backend API hosting
- **Streamlit Cloud**: Frontend hosting

## Key Features

### Document Management
- Upload multiple document formats (PDF,TXT,CSV,JSON)
- Automatic duplicate detection using file hashing
- Document metadata tracking and storage
- User-specific document collections

### Search Capabilities
- **Semantic Search**: Natural language queries for contextual results
- **Keyword Search**: Traditional BM25-based text matching
- **Deep Search**: Combines keyword and semantic search with adjustable weights
- Real-time search with response time tracking

### User Experience
- User authentication with secure password hashing
- Document upload progress tracking
- Search history with timestamps
- Usage statistics (documents, searches, storage)
- Responsive web interface

### Technical Architecture
- **RAG Pipeline**: Document chunking → Embedding → Vector storage → Retrieval → LLM synthesis
- **Async Processing**: Background document processing for better UX
- **Error Handling**: Comprehensive error handling and user feedback
- **CORS Support**: Cross-origin requests between frontend and backend
- **Scalable Design**: Modular architecture for easy maintenance

## How the RAG System Works

**RAG Pipeline**: Document chunking → Embedding → Vector storage → Retrieval → LLM synthesis

1. **Document upload**: Users upload documents through the Streamlit interface
2. **Text Extraction**: System extracts text from various file formats
3. **Semantic Chunking**: Intelligent document segmentation
4. **Embedding Generation**: Each chunk is converted to vector embeddings using HuggingFace models(BAAI/bge-small-en-v1.5)
5. **Vector Storage**: Embeddings stored in ChromaDB with metadata
6. **Query Processing**: User questions are embedded and searched against the vector database
7. **Retrieval**: Most relevant document chunks are retrieved
8. **LLM Synthesis**: Retrieved chunks are sent to an LLM to generate a coherent answer
9. **Response**: User gets the synthesized answer with source citations

## Getting Started

### Prerequisites
- Python 3.10+
- Virtual environment (recommended)

### Local Development

1. **Clone the repository**
```bash
git clone https://github.com/narasimha07-here/Search-Engine.git
```

2. **Set up backend**
```bash
cd api
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

3. **Environment variables**
Create `.env` file in the api directory:
```
OPENAI_API_KEY=openai_api_key
OPENAI_API_BASE=https://openrouter.ai/api/v1
MODEL=any_model_from_openrouter  # or your preferred model
EMBED_MODEL=BAAI/bge-small-en-v1.5
VECTOR_DIR=./chroma_db
UPLOAD_DIR=./uploads
DB_PATH=./app.db 
ALLOWED_ORIGINS=http://localhost:8501
```

4. **Run backend**
```bash
uvicorn main:app --reload --port 8000
```

5. **Set up frontend**
```bash
cd ../frontend
pip install -r requirements.txt
```

6. **Run frontend**
```bash
streamlit run app.py
```

Visit `http://localhost:8501` to access the application.

## API Documentation

Once the backend is running, visit `http://localhost:8000/docs` for interactive API documentation powered by FastAPI's automatic OpenAPI generation.

### Key Endpoints

- `POST /register` - User registration
- `POST /login` - User authentication
- `POST /upload` - Document upload
- `GET /documents` - List user documents
- `POST /search/semantic` - Semantic search
- `POST /search/hybrid` - Deep search
- `GET /search/history` - Search history
- `GET /stats` - User statistics

