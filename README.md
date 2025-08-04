# ⚡ GridAI - Electrical Engineering Assistant

A clean, modern multimodal RAG system for electrical engineering professionals with advanced AI capabilities and beautiful UI.

## 🚀 Features

- **Multimodal Processing**: Analyze text, images, and PDFs
- **Electrical Engineering Expertise**: Domain-specific analysis
- **Conversation Memory**: Persistent chat history
- **Beautiful Modern UI**: Gradient design with markdown support
- **Real-time Processing**: Live upload progress and loading states

## 🛠 Quick Start

### **Prerequisites**
- Python 3.11+
- OpenAI API Key

### **Installation**

1. **Clone and setup**
```bash
git clone <repository-url>
cd GridAI-1
pip install -r requirements.txt
```

2. **Environment setup**
```bash
echo "OPENAI_API_KEY=your_openai_api_key_here" > .env
```

3. **Run application**
```bash
python main.py
```

4. **Access**
- Open http://localhost:8000
- Upload documents and start chatting!

## 📁 Project Structure

```
GridAI-1/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── models.py            # Pydantic models
│   ├── document_processor.py # Document processing service
│   └── rag_service.py       # RAG system service
├── static/
│   └── index.html           # Frontend interface
├── main.py                  # Application entry point
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## 🎯 Usage

1. **Upload Documents**: Drag & drop PDFs or images
2. **Ask Questions**: Query about electrical systems
3. **Analyze Content**: Ask "what is in this pdf" for uploaded content
4. **Get Expert Analysis**: Receive detailed technical responses

## 🔧 API Endpoints

- `GET /` - Main application interface
- `POST /chat` - Chat with AI assistant
- `POST /upload-progress` - Upload files with progress

## 📊 Features

- **Document Processing**: PDF text/image extraction
- **Image Analysis**: Electrical diagram analysis
- **RAG System**: Intelligent document retrieval
- **Conversation Memory**: Context-aware responses
- **Loading States**: Real-time UI feedback

---

**GridAI** - Clean, modern electrical engineering AI assistant ⚡