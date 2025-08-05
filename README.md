# Overview

This is a complete multimodal Electrical Grid RAG (Retrieval-Augmented Generation) Chatbot application that allows users to upload various document types (PDFs, Word docs, images, text files, CSVs) related to electrical grids and substations, then ask questions about the content. The system uses vector embeddings to find relevant document chunks and combines them with OpenAI's GPT models to provide contextual answers.

Key Features:
- **Multimodal Document Processing**: Handles PDFs, Word docs, text files, CSVs, and images
- **Intelligent PDF Processing**: Automatically detects pages with minimal text (<10 characters) and treats them as images
- **Image Extraction & Display**: Extracts images from PDFs and displays them in chat responses when relevant
- **OCR + AI Vision**: Uses Tesseract OCR and OpenAI Vision API for comprehensive image analysis
- **Vector-based Retrieval**: Simple TF-IDF based vector search for document similarity
- **Real-time Chat**: WebSocket-based communication with typing indicators
- **Image Modal Viewer**: Click-to-expand image viewing with similarity scores
- **Advanced Search**: Semantic search with keyword filtering and document type filtering
- **Batch Operations**: Bulk document management, batch image analysis, and component extraction
- **Dark/Light Theme**: Toggle between themes with persistent preferences
- **Technical Component Analysis**: AI-powered identification of electrical components in images
- **Document Export**: Export document lists and analysis results

The application features a web-based chat interface with real-time WebSocket communication, document processing capabilities, and a vector database for efficient document retrieval. It's designed to serve as an intelligent assistant for electrical grid and substation documentation with full multimodal capabilities.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Static Web Interface**: Simple HTML/CSS/JavaScript frontend served through FastAPI's static file mounting
- **Real-time Communication**: WebSocket-based chat interface for instant messaging between client and server
- **Responsive Design**: Bootstrap-based UI with sidebar for document upload and main chat area
- **File Upload System**: Multi-format file upload with progress tracking and validation

## Backend Architecture
- **FastAPI Framework**: Asynchronous Python web framework serving as the main application server
- **Service-Oriented Design**: Modular architecture with separate services for different concerns:
  - `DocumentProcessor`: Handles multi-format document parsing and content extraction
  - `VectorStore`: Manages ChromaDB integration for vector embeddings and similarity search
  - `RAGService`: Orchestrates retrieval-augmented generation workflows
  - `OpenAIService`: Handles OpenAI API interactions for completions and image analysis
- **WebSocket Management**: Connection manager for handling multiple concurrent chat sessions
- **Session Management**: Per-session conversation history tracking

## Data Storage and Retrieval
- **Simple Vector Store**: Custom TF-IDF based vector storage with persistent disk storage
- **Text Vectorization**: Scikit-learn TfidfVectorizer for text embedding generation
- **Document Chunking**: Smart text chunking with overlap for optimal retrieval performance
- **Similarity Search**: Cosine similarity-based document retrieval with configurable thresholds
- **Image Storage**: Extracted images stored in static/extracted_images for web display

## Document Processing Pipeline
- **Multi-format Support**: PDF (PyPDF2/pdfplumber), DOCX (python-docx), TXT, CSV, and image formats
- **Intelligent PDF Processing**: Automatic detection of image-heavy pages (<10 characters) for special processing
- **Image Extraction**: Saves extracted PDF pages and uploaded images to static directory for display
- **OCR Integration**: Pytesseract for text extraction from images and image-heavy PDF pages
- **AI Vision Analysis**: OpenAI GPT-4o Vision API for detailed image content analysis
- **Metadata Preservation**: File metadata tracking including source, page numbers, document types, and image URLs
- **Content Chunking**: Intelligent text segmentation with overlap for vector storage

## RAG Implementation
- **Hybrid Approach**: Combines vector similarity search with conversational context
- **Context Assembly**: Aggregates relevant document chunks based on similarity scores
- **Image Integration**: Automatically includes relevant images from PDFs in responses
- **Fallback Strategy**: Direct OpenAI completion when no relevant documents are found
- **Source Attribution**: Tracks and returns document sources with similarity scores
- **Multimodal Responses**: Displays both text answers and related images with click-to-expand modals

# External Dependencies

## AI/ML Services
- **OpenAI API**: GPT-4o model for text generation and multimodal image analysis
- **Scikit-learn**: TF-IDF vectorization for text similarity search

## Document Processing
- **PyPDF2/pdfplumber**: PDF text extraction and parsing
- **python-docx**: Microsoft Word document processing
- **Pytesseract**: OCR for image-to-text conversion
- **Pillow (PIL)**: Image processing and manipulation

## Data Storage
- **Simple Vector Store**: Custom pickle-based persistence for TF-IDF vectors and metadata
- **Local File System**: Document storage, vector database persistence, and extracted image storage

## Web Framework
- **FastAPI**: Asynchronous web framework with WebSocket support
- **Uvicorn**: ASGI server for FastAPI applications

## Frontend Libraries
- **Bootstrap 5.1.3**: CSS framework for responsive design
- **Font Awesome 6.0.0**: Icon library for UI elements

## Python Libraries
- **Pydantic**: Data validation and serialization
- **asyncio**: Asynchronous programming support
- **NumPy**: Numerical computing for vector operations
- **Scikit-learn**: Machine learning library for TF-IDF vectorization

## Environment Requirements
- **OPENAI_API_KEY**: Required environment variable for OpenAI API access
- **Tesseract**: System dependency for OCR functionality