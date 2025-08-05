import os
import io
import asyncio
from typing import List, Dict, Any
import PyPDF2
import pdfplumber
from docx import Document
import csv
from PIL import Image
# import pytesseract  # Removed - using only OpenAI Vision
import base64
import json
from pathlib import Path

from .vector_store import VectorStore
from .openai_service import OpenAIService

class DocumentProcessor:
    def __init__(self):
        self.vector_store = VectorStore()
        self.openai_service = OpenAIService()
        self.images_dir = "static/extracted_images"
        os.makedirs(self.images_dir, exist_ok=True)
        self.supported_formats = {
            '.pdf': self._process_pdf,
            '.docx': self._process_docx,
            '.doc': self._process_docx,
            '.txt': self._process_txt,
            '.csv': self._process_csv,
            '.png': self._process_image,
            '.jpg': self._process_image,
            '.jpeg': self._process_image,
            '.bmp': self._process_image,
            '.gif': self._process_image
        }
        
    async def process_file(self, file_path: str, filename: str, file_id: str):
        """Process uploaded file and store in vector database"""
        try:
            file_ext = os.path.splitext(filename)[1].lower()
            
            if file_ext not in self.supported_formats:
                raise ValueError(f"Unsupported file format: {file_ext}")
            
            # Process the file based on its type
            processor = self.supported_formats[file_ext]
            content_chunks = await processor(file_path, filename, file_id or "unknown")
            
            # Store chunks in vector database
            await self.vector_store.add_documents(
                chunks=content_chunks,
                metadata={
                    'file_id': file_id,
                    'filename': filename,
                    'file_type': file_ext
                }
            )
            
            # Clean up uploaded file
            os.remove(file_path)
            
            return True
            
        except Exception as e:
            print(f"Error processing file {filename}: {str(e)}")
            raise e
    
    async def _process_pdf(self, file_path: str, filename: str, file_id: str = "unknown") -> List[Dict[str, Any]]:
        """Process PDF with intelligent image detection"""
        chunks = []
        
        try:
            with open(file_path, 'rb') as file:
                # First pass: check text content per page
                pdf_reader = PyPDF2.PdfReader(file)
                page_analysis = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text().strip()
                    # If page has less than 10 characters, treat as image
                    is_image_page = len(text) < 10
                    page_analysis.append({
                        'page_num': page_num,
                        'text': text,
                        'is_image': is_image_page
                    })
                
                # Second pass: detailed processing with pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    for i, page_data in enumerate(page_analysis):
                        page = pdf.pages[i]
                        
                        if page_data['is_image']:
                            # Convert page to image and use OCR
                            try:
                                page_image = page.to_image()
                                # Convert to PIL Image
                                pil_image = page_image.original
                                
                                # Save extracted image for display
                                image_filename = f"{file_id}_page_{i+1}.png"
                                image_path = os.path.join(self.images_dir, image_filename)
                                pil_image.save(image_path)
                                image_url = f"/static/extracted_images/{image_filename}"
                                
                                # Skip OCR, use only Vision API
                                ocr_text = "[Using AI Vision Analysis only]"
                                
                                # Use OpenAI Vision for detailed analysis if available
                                try:
                                    img_buffer = io.BytesIO()
                                    pil_image.save(img_buffer, format='PNG')
                                    img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                                    
                                    vision_analysis = await self.openai_service.analyze_image(
                                        img_base64, 
                                        "Analyze this electrical grid document page. Focus on technical diagrams, schematics, equipment specifications, and any text content."
                                    )
                                except Exception as vision_error:
                                    print(f"Vision analysis failed: {vision_error}")
                                    vision_analysis = "[Vision analysis not available - OpenAI API issue]"
                                
                                combined_content = f"OCR Text: {ocr_text}\n\nVision Analysis: {vision_analysis}"
                                
                                chunks.append({
                                    'content': combined_content,
                                    'metadata': {
                                        'page': i + 1,
                                        'type': 'image_page',
                                        'source': filename,
                                        'image_url': image_url,
                                        'has_image': True
                                    }
                                })
                                
                            except Exception as e:
                                print(f"Error processing image page {i + 1}: {str(e)}")
                                # Fallback to basic text
                                chunks.append({
                                    'content': page_data['text'] or f"Image page {i + 1} - processing failed",
                                    'metadata': {
                                        'page': i + 1,
                                        'type': 'image_page_fallback',
                                        'source': filename
                                    }
                                })
                        else:
                            # Regular text processing
                            text = page.extract_text()
                            if text and text.strip():
                                # Split into chunks if text is too long
                                text_chunks = self._chunk_text(text, max_chunk_size=1000)
                                for chunk_idx, chunk in enumerate(text_chunks):
                                    chunks.append({
                                        'content': chunk,
                                        'metadata': {
                                            'page': i + 1,
                                            'chunk': chunk_idx + 1,
                                            'type': 'text_page',
                                            'source': filename
                                        }
                                    })
                
        except Exception as e:
            print(f"Error processing PDF {filename}: {str(e)}")
            raise e
        
        return chunks
    
    async def _process_docx(self, file_path: str, filename: str, file_id: str = "unknown") -> List[Dict[str, Any]]:
        """Process Word documents"""
        chunks = []
        
        try:
            doc = Document(file_path)
            full_text = []
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    full_text.append(paragraph.text)
            
            content = '\n'.join(full_text)
            
            if content.strip():
                text_chunks = self._chunk_text(content, max_chunk_size=1000)
                for chunk_idx, chunk in enumerate(text_chunks):
                    chunks.append({
                        'content': chunk,
                        'metadata': {
                            'chunk': chunk_idx + 1,
                            'type': 'docx',
                            'source': filename
                        }
                    })
                    
        except Exception as e:
            print(f"Error processing DOCX {filename}: {str(e)}")
            raise e
        
        return chunks
    
    async def _process_txt(self, file_path: str, filename: str, file_id: str = "unknown") -> List[Dict[str, Any]]:
        """Process text files"""
        chunks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
            
            if content.strip():
                text_chunks = self._chunk_text(content, max_chunk_size=1000)
                for chunk_idx, chunk in enumerate(text_chunks):
                    chunks.append({
                        'content': chunk,
                        'metadata': {
                            'chunk': chunk_idx + 1,
                            'type': 'txt',
                            'source': filename
                        }
                    })
                    
        except Exception as e:
            print(f"Error processing TXT {filename}: {str(e)}")
            raise e
        
        return chunks
    
    async def _process_csv(self, file_path: str, filename: str, file_id: str = "unknown") -> List[Dict[str, Any]]:
        """Process CSV files"""
        chunks = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                csv_reader = csv.reader(file)
                rows = list(csv_reader)
            
            if rows:
                # Convert CSV to structured text
                header = rows[0] if rows else []
                content_lines = [f"CSV File: {filename}"]
                content_lines.append(f"Headers: {', '.join(header)}")
                content_lines.append("")
                
                for row_idx, row in enumerate(rows[1:], 1):
                    if len(row) == len(header):
                        row_data = []
                        for col_idx, value in enumerate(row):
                            if col_idx < len(header):
                                row_data.append(f"{header[col_idx]}: {value}")
                        content_lines.append(f"Row {row_idx}: {', '.join(row_data)}")
                
                content = '\n'.join(content_lines)
                text_chunks = self._chunk_text(content, max_chunk_size=1000)
                
                for chunk_idx, chunk in enumerate(text_chunks):
                    chunks.append({
                        'content': chunk,
                        'metadata': {
                            'chunk': chunk_idx + 1,
                            'type': 'csv',
                            'source': filename
                        }
                    })
                    
        except Exception as e:
            print(f"Error processing CSV {filename}: {str(e)}")
            raise e
        
        return chunks
    
    async def _process_image(self, file_path: str, filename: str, file_id: str = "unknown") -> List[Dict[str, Any]]:
        """Process image files using OCR and Vision API"""
        chunks = []
        
        try:
            # Open image
            image = Image.open(file_path)
            
            # Save uploaded image for display
            file_id = os.path.splitext(filename)[0]
            image_filename = f"{file_id}_uploaded.png"
            image_path = os.path.join(self.images_dir, image_filename)
            image.save(image_path)
            image_url = f"/static/extracted_images/{image_filename}"
            
            # Skip OCR, use only Vision API
            ocr_text = "[Using AI Vision Analysis only]"
            
            # Convert to base64 for Vision API
            img_buffer = io.BytesIO()
            image.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            # Use OpenAI Vision for analysis if available
            try:
                vision_analysis = await self.openai_service.analyze_image(
                    img_base64,
                    "Analyze this electrical grid related image. Describe any equipment, diagrams, schematics, measurements, or technical specifications visible. Focus on electrical grid and substation components."
                )
            except Exception as vision_error:
                print(f"Vision analysis failed: {vision_error}")
                vision_analysis = "[Vision analysis not available - OpenAI API issue]"
            
            # Combine OCR and Vision results
            combined_content = f"Image: {filename}\n\nOCR Text: {ocr_text}\n\nVision Analysis: {vision_analysis}"
            
            chunks.append({
                'content': combined_content,
                'metadata': {
                    'type': 'image',
                    'source': filename,
                    'image_url': image_url,
                    'has_image': True
                }
            })
            
        except Exception as e:
            print(f"Error processing image {filename}: {str(e)}")
            raise e
        
        return chunks
    
    def _chunk_text(self, text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to break at sentence boundary
            chunk_text = text[start:end]
            last_period = chunk_text.rfind('.')
            last_newline = chunk_text.rfind('\n')
            
            break_point = max(last_period, last_newline)
            
            if break_point > start + max_chunk_size // 2:
                end = start + break_point + 1
            
            chunks.append(text[start:end])
            start = end - overlap
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]
