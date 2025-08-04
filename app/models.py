from pydantic import BaseModel
from typing import Optional, List

class ChatRequest(BaseModel):
    message: str
    session_id: str
    image: Optional[str] = None
    context_images: Optional[List[str]] = []

class ChatResponse(BaseModel):
    response: str
    session_id: str
    processing_time: float = 0.0

class UploadResponse(BaseModel):
    status: str
    message: str
    session_id: str
    extracted_images: List[str] = []