from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import tempfile
import shutil
import uuid
import base64
import json
import time
import logging
import aiofiles
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from .models import ChatRequest, ChatResponse, UploadResponse
from .document_processor import DocumentProcessor
from .rag_service import RAGService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check for OpenAI API key
if not os.getenv("OPENAI_API_KEY"):
    logger.error("OPENAI_API_KEY environment variable not set")
    raise ValueError("Please set OPENAI_API_KEY in your .env file")

# Initialize services
doc_processor = DocumentProcessor()
rag_service = RAGService()

app = FastAPI(title="GridAI - Electrical Engineering Assistant", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        async with aiofiles.open("static/index.html", "r", encoding="utf-8") as html_file:
            content = await html_file.read()
            return HTMLResponse(content=content)
    except Exception as e:
        logger.error(f"Error reading HTML file: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading page")

@app.post("/upload-progress")
async def upload_files_with_progress(files: List[UploadFile] = File(...)):
    async def generate_progress():
        try:
            yield f"data: {json.dumps({'progress': 10, 'message': 'Starting processing...'})}\n\n"
            
            temp_dir = tempfile.mkdtemp()
            extracted_images = []
            all_texts = []
            all_image_summaries = []
            
            for i, file in enumerate(files):
                progress = 20 + (i * 60 // len(files))
                yield f"data: {json.dumps({'progress': progress, 'message': f'Processing {file.filename}...'})}\n\n"
                
                # Save file
                temp_path = Path(temp_dir) / f"{uuid.uuid4().hex}_{file.filename}"
                with open(temp_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                try:
                    if file.filename.lower().endswith('.pdf'):
                        texts, images = doc_processor.extract_pdf_content(str(temp_path))
                        all_texts.extend(texts)
                        
                        # Batch analyze images in groups of 2 to avoid rate limits
                        if images:
                            batch_size = 2
                            for i in range(0, len(images), batch_size):
                                batch = images[i:i+batch_size]
                                try:
                                    summaries = await doc_processor.analyze_images_batch(batch)
                                    all_image_summaries.extend(summaries)
                                    # Add delay between batches to avoid rate limits
                                    if i + batch_size < len(images):
                                        await asyncio.sleep(2)
                                except Exception as e:
                                    logger.error(f"Failed to analyze batch {i//batch_size + 1}: {str(e)}")
                                    all_image_summaries.extend(["Image analysis failed"] * len(batch))
                            extracted_images.extend(images)
                    
                    elif file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_data = await file.read()
                        img_base64 = base64.b64encode(image_data).decode('utf-8')
                        summaries = await doc_processor.analyze_images_batch([img_base64])
                        all_image_summaries.extend(summaries)
                        extracted_images.append(img_base64)
                    
                    elif file.filename.lower().endswith('.docx'):
                        texts = rag_service.process_docx(str(temp_path))
                        all_texts.extend(texts)
                    
                    elif file.filename.lower().endswith(('.xlsx', '.xls')):
                        texts = rag_service.process_excel(str(temp_path))
                        all_texts.extend(texts)
                
                except Exception as e:
                    logger.error(f"Error processing {file.filename}: {str(e)}")
                    continue
                finally:
                    temp_path.unlink(missing_ok=True)
            
            yield f"data: {json.dumps({'progress': 90, 'message': 'Setting up retriever...'})}\n\n"
            
            # Setup RAG
            if all_texts or all_image_summaries:
                rag_service.setup_retriever(all_texts, all_image_summaries, extracted_images)
            
            result = {
                "status": "success",
                "message": f"Successfully processed {len(files)} files",
                "session_id": str(uuid.uuid4()),
                "extracted_images": extracted_images,
                "processed_files": [{
                    "name": file.filename,
                    "images": [] if not extracted_images else (extracted_images if len(files) == 1 else [])
                } for file in files]
            }
            
            yield f"data: {json.dumps({'progress': 100, 'message': 'Complete!', 'result': json.dumps(result)})}\n\n"
            
        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            error_result = {"status": "error", "detail": "Processing failed"}
            yield f"data: {json.dumps({'progress': 0, 'message': 'Error occurred', 'result': json.dumps(error_result)})}\n\n"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    return StreamingResponse(generate_progress(), media_type="text/plain")

@app.post("/remove-image")
async def remove_image(image_id: str):
    """Remove image from storage"""
    try:
        rag_service.remove_image(image_id)
        return {"status": "success", "message": f"Image {image_id} removed"}
    except Exception as e:
        logger.error(f"Error removing image: {str(e)}")
        raise HTTPException(status_code=500, detail="Error removing image")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start_time = time.time()
    
    try:
        logger.info(f"Chat request: {request.session_id}")
        
        # Always use RAG service for consistent vision model analysis
        context_images = request.context_images or []
        if request.image:
            context_images = [request.image]
        
        response = await rag_service.query(
            request.message, 
            request.session_id, 
            context_images=context_images
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return ChatResponse(
            response=response.get("response", response if isinstance(response, str) else ""),
            session_id=request.session_id,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        processing_time = (time.time() - start_time) * 1000
        
        return ChatResponse(
            response="I encountered an error processing your request. Please try again.",
            session_id=request.session_id,
            processing_time=processing_time
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)