import fitz
import base64
import tempfile
import os
from pathlib import Path
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self):
        self.model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_tokens=800)
        self.stored_images = {}  # Store images with IDs
    
    def extract_pdf_content(self, file_path: str) -> Tuple[List[Document], List[str]]:
        """Extract text and images from PDF with 10 character threshold"""
        doc = None
        try:
            doc = fitz.open(file_path)
            texts = []
            images = []
            
            logger.info(f"Processing PDF with {doc.page_count} pages")
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                
                # Extract text with chunking
                text = page.get_text("text")
                if text.strip():
                    # Split into chunks for better processing
                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                    chunk_size = 1000
                    current_chunk = ""
                    
                    for line in lines:
                        if len(current_chunk) + len(line) < chunk_size:
                            current_chunk += line + "\n"
                        else:
                            if current_chunk.strip():
                                texts.append(Document(
                                    page_content=current_chunk.strip(),
                                    metadata={"page": page_num + 1}
                                ))
                            current_chunk = line + "\n"
                    
                    if current_chunk.strip():
                        texts.append(Document(
                            page_content=current_chunk.strip(),
                            metadata={"page": page_num + 1}
                        ))
                
                # Extract embedded images
                image_list = page.get_images()
                if image_list:
                    for img_idx, img in enumerate(image_list):
                        try:
                            xref = img[0]
                            pixmap = fitz.Pixmap(doc, xref)
                            if pixmap.n - pixmap.alpha < 4:
                                img_data = pixmap.tobytes("png")
                                img_base64 = base64.b64encode(img_data).decode('utf-8')
                                images.append(img_base64)
                            pixmap = None
                        except Exception:
                            continue
                
                # Auto-detect drawing pages (pages with less than 10 characters)
                if len(text.strip()) < 10:
                    try:
                        mat = fitz.Matrix(2.0, 2.0)
                        pixmap = page.get_pixmap(matrix=mat)
                        img_data = pixmap.tobytes("png")
                        img_base64 = base64.b64encode(img_data).decode('utf-8')
                        images.append(img_base64)
                        pixmap = None
                    except Exception:
                        continue
            
            logger.info(f"Extracted: {len(texts)} text chunks, {len(images)} images")
            return texts, images
            
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise
        finally:
            if doc:
                doc.close()
    
    async def analyze_images_batch(self, images: List[str]) -> List[str]:
        """Analyze multiple images in a single request"""
        try:
            if not images:
                return []
            
            prompt = f"""
Analyze these {len(images)} electrical engineering images briefly:

For each image, provide:
- Key electrical components visible
- Technical specifications if visible
- Brief engineering notes

Format: "IMAGE 1: [analysis]\n\nIMAGE 2: [analysis]\n\n..." Keep each analysis under 100 words.
"""
            
            from langchain_core.messages import HumanMessage
            
            content = [{"type": "text", "text": prompt}]
            
            # Add all images to single request
            for i, image in enumerate(images):
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                })
            
            messages = [HumanMessage(content=content)]
            # Use shorter max_tokens to prevent rate limit issues
            model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_tokens=600)
            response = await model.ainvoke(messages)
            
            # Split response by images
            response_text = response.content
            if f"IMAGE {len(images)}:" in response_text:
                # Split by IMAGE markers
                parts = response_text.split("IMAGE ")
                summaries = []
                for i, part in enumerate(parts[1:], 1):  # Skip first empty part
                    # Remove the number and colon, keep the analysis
                    analysis = part.split(":", 1)[1].strip() if ":" in part else part.strip()
                    # Clean up by removing next image marker
                    if i < len(parts) - 1:
                        next_marker = f"\n\nIMAGE {i+1}"
                        if next_marker in analysis:
                            analysis = analysis.split(next_marker)[0].strip()
                    summaries.append(analysis)
                return summaries
            else:
                # Fallback: return same analysis for all images
                return [response_text] * len(images)
            
        except Exception as e:
            logger.error(f"Error analyzing images batch: {str(e)}")
            return ["Error analyzing image content."] * len(images)
    
    async def analyze_image(self, image_base64: str) -> str:
        """Analyze single image (wrapper for batch method)"""
        summaries = await self.analyze_images_batch([image_base64])
        return summaries[0] if summaries else "Error analyzing image content."