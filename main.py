import os
import uuid
import asyncio
import json
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from services.document_processor import DocumentProcessor
from services.vector_store import VectorStore
from services.rag_service import RAGService
from services.openai_service import OpenAIService

app = FastAPI(title="Electrical Grid RAG Chatbot")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize services
document_processor = DocumentProcessor()
vector_store = VectorStore()
openai_service = OpenAIService()
rag_service = RAGService(vector_store, openai_service)

# Store active WebSocket connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        print(f"Attempting to send message via WebSocket...")
        try:
            await websocket.send_text(message)
            print("Message sent successfully via WebSocket")
        except Exception as e:
            print(f"Error sending WebSocket message: {e}")

manager = ConnectionManager()

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None

class UploadResponse(BaseModel):
    success: bool
    message: str
    file_id: Optional[str] = None

@app.get("/")
async def read_root():
    return FileResponse("static/index.html")

@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    """Upload and process documents/images"""
    try:
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        
        # Save uploaded file
        os.makedirs("uploads", exist_ok=True)
        file_path = f"uploads/{file_id}_{file.filename}"
        
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Process the document
        filename = file.filename or "unknown_file"
        await document_processor.process_file(file_path, filename, file_id)
        
        return UploadResponse(
            success=True,
            message=f"Successfully processed {file.filename}",
            file_id=file_id
        )
        
    except Exception as e:
        return UploadResponse(
            success=False,
            message=f"Error processing file: {str(e)}"
        )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await manager.connect(websocket)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            print(f"Received WebSocket data: {data}")
            message_data = json.loads(data)
            print(f"Parsed message data: {message_data}")
            
            user_message = message_data.get("message", "")
            session_id = message_data.get("session_id", "default")
            print(f"User message: '{user_message}', Session ID: '{session_id}'")
            
            if not user_message.strip():
                await manager.send_personal_message(
                    json.dumps({"error": "Empty message"}), 
                    websocket
                )
                continue
            
            # Send typing indicator
            await manager.send_personal_message(
                json.dumps({"typing": True}), 
                websocket
            )
            
            try:
                print(f"Getting RAG response for: '{user_message}'")
                # Get response from RAG service
                response = await rag_service.get_response(user_message, session_id)
                print(f"RAG response: {response}")
                
                # Send response back to client
                response_data = {
                    "message": response["answer"],
                    "sources": response.get("sources", []),
                    "images": response.get("images", []),
                    "typing": False
                }
                response_json = json.dumps(response_data)
                print(f"Sending response to client: {response_json[:200]}...")
                await manager.send_personal_message(response_json, websocket)
                print("Response sent successfully")
                
            except Exception as e:
                print(f"Error in WebSocket handler: {str(e)}")
                await manager.send_personal_message(
                    json.dumps({
                        "error": f"Error generating response: {str(e)}",
                        "typing": False
                    }), 
                    websocket
                )
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        stats = await vector_store.get_stats()
        return {
            "documents_count": stats.get("documents", 0),
            "chunks_count": stats.get("chunks", 0),
            "vector_db_status": "healthy"
        }
    except Exception as e:
        return {
            "error": f"Failed to get stats: {str(e)}",
            "vector_db_status": "unhealthy"
        }

# Advanced API endpoints for enhanced features
@app.get("/api/documents")
async def get_documents():
    """Get list of all uploaded documents"""
    try:
        if vector_store:
            docs = vector_store.get_all_documents()
            return docs
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving documents: {str(e)}")

@app.delete("/api/documents")
async def delete_all_documents():
    """Delete all documents from the vector database"""
    try:
        if vector_store:
            await vector_store.clear_all()
        
        # Clear uploaded files
        upload_dir = "uploads"
        if os.path.exists(upload_dir):
            for filename in os.listdir(upload_dir):
                file_path = os.path.join(upload_dir, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        
        # Clear extracted images
        images_dir = "static/extracted_images"
        if os.path.exists(images_dir):
            for filename in os.listdir(images_dir):
                if filename != ".gitkeep":
                    file_path = os.path.join(images_dir, filename)
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
        
        return {"message": "All documents deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting documents: {str(e)}")

@app.post("/api/reindex")
async def reindex_documents():
    """Rebuild the search index"""
    try:
        if vector_store:
            result = vector_store.rebuild_index()
            if result:
                return {"message": "Search index rebuilt successfully"}
            else:
                return {"message": "No documents to reindex"}
        return {"message": "Vector store not available"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error rebuilding index: {str(e)}")

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("simple_vector_db", exist_ok=True)
    os.makedirs("static/extracted_images", exist_ok=True)
    
    # Initialize vector store
    asyncio.run(vector_store.initialize())
    
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=5000)
