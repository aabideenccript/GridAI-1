from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import os
import tempfile
import shutil
import uuid
import base64
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial, lru_cache
import asyncio
from asyncio import gather
import hashlib

# Document processing imports
import fitz  # PyMuPDF
import pandas as pd
import json
from docx import Document as DocxDocument
from PIL import Image
import io

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="GridAI Multi-Modal", description="")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for RAG system
retriever = None
vectorstore = None
store = None
session_contexts = {}  # Store conversation context per session

class ChatMessage(BaseModel):
    message: str
    session_id: str
    image: str = None  # Base64 encoded image

class ChatResponse(BaseModel):
    response: str
    session_id: str
    images: List[str] = []

def extract_document_elements(file_path: str):
    """Extract elements using top libraries"""
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.pdf':
            return extract_pdf_pymupdf(file_path)
        elif file_ext == '.docx':
            return extract_docx(file_path)
        elif file_ext in ['.xlsx', '.xls']:
            return extract_excel(file_path)
        elif file_ext == '.csv':
            return extract_csv(file_path)
        elif file_ext == '.json':
            return extract_json(file_path)
        elif file_ext == '.txt':
            return extract_txt(file_path)
        else:
            return [], [], []
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return [], [], []

def extract_pdf_pymupdf(file_path: str):
    """Extract from PDF using PyMuPDF (best PDF library)"""
    doc = fitz.open(file_path)
    texts = []
    tables = []
    images = []
    
    print(f"Processing PDF with {doc.page_count} pages (auto-detecting drawing pages)")
    
    for page_num in range(doc.page_count):
        page = doc[page_num]
        
        # Extract ALL text with detailed chunking
        text = page.get_text("text")
        if text.strip():
            # Split by lines and create smaller, more granular chunks
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            current_chunk = ""
            chunk_count = 0
            
            for line in lines:
                # Create smaller chunks (1500 chars) for better processing
                if len(current_chunk) + len(line) < 1500:
                    current_chunk += line + "\n"
                else:
                    if current_chunk.strip():
                        chunk_count += 1
                        texts.append(Document(
                            page_content=current_chunk.strip(),
                            metadata={"page": page_num + 1, "chunk": chunk_count}
                        ))
                    current_chunk = line + "\n"
            
            # Add final chunk
            if current_chunk.strip():
                chunk_count += 1
                texts.append(Document(
                    page_content=current_chunk.strip(),
                    metadata={"page": page_num + 1, "chunk": chunk_count}
                ))
            
            if chunk_count > 0:
                print(f"  Page {page_num + 1}: {chunk_count} chunks, {len(text)} chars")
            else:
                print(f"  Page {page_num + 1}: {len(text)} chars (no text chunks created)")
        
        # Extract tables
        try:
            tables_on_page = page.find_tables()
            for table_idx, table in enumerate(tables_on_page):
                table_data = table.extract()
                if table_data:
                    table_text = "\n".join(["\t".join(row) for row in table_data if row])
                    tables.append(Document(page_content=table_text, metadata={"page": page_num + 1, "table": table_idx}))
        except:
            pass
        
        # Extract embedded images
        image_list = page.get_images()
        for img_idx, img in enumerate(image_list):
            try:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha < 4:  # Valid image
                    img_data = pix.tobytes("png")
                    img_base64 = base64.b64encode(img_data).decode('utf-8')
                    images.append(img_base64)
                    print(f"  Extracted embedded image {len(images)} from page {page_num + 1}")
                pix = None
            except:
                continue
        
        # Auto-detect drawing pages by low text content
        if len(text.strip()) < 10:  # Less than 10 characters = likely a drawing page
            try:
                # Convert page to image
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img_base64 = base64.b64encode(img_data).decode('utf-8')
                images.append(img_base64)
                print(f"  Captured page {page_num + 1} as drawing image {len(images)} (low text: {len(text.strip())} chars)")
                pix = None
            except Exception as e:
                print(f"  Error capturing page {page_num + 1} as image: {e}")
    
    doc.close()
    total_chars = sum(len(doc.page_content) for doc in texts)
    print(f"PyMuPDF extracted: {len(texts)} text chunks, {len(tables)} tables, {len(images)} images (including drawings)")
    print(f"Total characters: {total_chars:,}")
    return texts, tables, images

def extract_docx(file_path: str):
    """Extract from DOCX"""
    doc = DocxDocument(file_path)
    texts = []
    tables = []
    
    # Extract paragraphs
    for para in doc.paragraphs:
        if para.text.strip():
            texts.append(Document(page_content=para.text))
    
    # Extract tables
    for table in doc.tables:
        table_text = ""
        for row in table.rows:
            row_text = "\t".join([cell.text for cell in row.cells])
            table_text += row_text + "\n"
        if table_text.strip():
            tables.append(Document(page_content=table_text))
    
    return texts, tables, []

def extract_excel(file_path: str):
    """Extract from Excel"""
    df = pd.read_excel(file_path, sheet_name=None)
    texts = []
    tables = []
    
    for sheet_name, sheet_df in df.items():
        table_text = sheet_df.to_string()
        tables.append(Document(page_content=table_text, metadata={"sheet": sheet_name}))
    
    return texts, tables, []

def extract_csv(file_path: str):
    """Extract from CSV"""
    df = pd.read_csv(file_path)
    table_text = df.to_string()
    tables = [Document(page_content=table_text)]
    return [], tables, []

def extract_json(file_path: str):
    """Extract from JSON"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    text = json.dumps(data, indent=2)
    texts = [Document(page_content=text)]
    return texts, [], []

def extract_txt(file_path: str):
    """Extract from TXT with smart chunking"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split by paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) < 4000:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(Document(page_content=current_chunk.strip()))
            current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(Document(page_content=current_chunk.strip()))
    
    return chunks, [], []

def setup_direct_retriever(texts, tables, images):
    """Setup comprehensive retriever with proper document storage"""
    global retriever, vectorstore, store
    
    print(f"Setting up comprehensive retriever with {len(texts)} texts, {len(tables)} tables, {len(images)} images")
    
    # Clear existing data
    try:
        import shutil
        shutil.rmtree("./chroma_db", ignore_errors=True)
    except:
        pass
    
    # Initialize fresh components
    vectorstore = Chroma(
        collection_name="electrical_grid_comprehensive", 
        embedding_function=OpenAIEmbeddings(),
        persist_directory="./chroma_db"
    )
    store = InMemoryStore()
    id_key = "doc_id"
    
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )
    
    all_docs_for_embedding = []
    all_docs_for_storage = []
    all_ids = []
    
    # Process texts
    if texts:
        for i, text in enumerate(texts):
            doc_id = str(uuid.uuid4())
            text.metadata[id_key] = doc_id
            all_docs_for_embedding.append(text)
            all_docs_for_storage.append(text)
            all_ids.append(doc_id)
        print(f"Prepared {len(texts)} text documents")
    
    # Process tables
    if tables:
        for i, table in enumerate(tables):
            doc_id = str(uuid.uuid4())
            table.metadata[id_key] = doc_id
            all_docs_for_embedding.append(table)
            all_docs_for_storage.append(table)
            all_ids.append(doc_id)
        print(f"Prepared {len(tables)} table documents")
    
    # Process images with better searchable content
    if images:
        for i, image in enumerate(images):
            doc_id = str(uuid.uuid4())
            # Create searchable document for image
            image_doc = Document(
                page_content=f"electrical engineering technical diagram schematic drawing image {i+1} substation relay panel electrical components wiring circuit breaker specifications austin city procurement", 
                metadata={id_key: doc_id, "type": "image", "image_index": i}
            )
            all_docs_for_embedding.append(image_doc)
            all_docs_for_storage.append(image)  # Store actual base64 image
            all_ids.append(doc_id)
        print(f"Prepared {len(images)} image documents")
    
    # Add all documents to vectorstore at once
    if all_docs_for_embedding:
        retriever.vectorstore.add_documents(all_docs_for_embedding)
        print(f"Added {len(all_docs_for_embedding)} documents to vectorstore")
    
    # Add all to docstore
    if all_docs_for_storage and all_ids:
        retriever.docstore.mset(list(zip(all_ids, all_docs_for_storage)))
        print(f"Added {len(all_docs_for_storage)} documents to docstore")
    
    print(f"Comprehensive retriever setup complete with {len(all_docs_for_embedding)} total documents")



def parse_docs(docs):
    """Split base64-encoded images and texts properly"""
    b64 = []
    text = []
    for doc in docs:
        if isinstance(doc, str):
            try:
                # Check if it's a valid base64 image
                if len(doc) > 100 and doc.replace('/', '').replace('+', '').replace('=', '').isalnum():
                    base64.b64decode(doc)
                    b64.append(doc)
                else:
                    text.append(doc)
            except:
                text.append(doc)
        else:
            text.append(doc)
    return {"images": b64, "texts": text}

def build_prompt(kwargs):
    """Build prompt exactly like the notebook"""
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = ""
    if len(docs_by_type["texts"]) > 0:
        for text_element in docs_by_type["texts"]:
            context_text += text_element.text

    # construct prompt with context (including images) - electrical grid specific
    prompt_template = f"""
    You are Eric, an expert electrical grid engineer specializing in substations, power systems, and electrical standards.
    
    Answer the question based only on the following context, which includes electrical engineering documents with text, tables, and technical diagrams.
    
    Provide detailed technical responses with references to relevant standards (IEEE, NESC, NERC) when applicable.
    Focus on electrical components, safety requirements, and engineering best practices.
    
    Context: {context_text}
    Question: {user_question}
    """

    prompt_content = [{"type": "text", "text": prompt_template}]

    if len(docs_by_type["images"]) > 0:
        for image in docs_by_type["images"]:
            prompt_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                }
            )

    return ChatPromptTemplate.from_messages(
        [
            HumanMessage(content=prompt_content),
        ]
    )

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

from fastapi.responses import StreamingResponse
import json
import asyncio

@app.post("/upload-progress")
async def upload_files_with_progress(files: List[UploadFile] = File(...)):
    async def generate_progress():
        try:
            data = json.dumps({'progress': 10, 'message': 'Starting processing...'})
            yield f"data: {data}\n\n"
            
            temp_dir = tempfile.mkdtemp()
            all_images = []
            
            # Process all files in parallel
            async def process_single_file(file, file_index):
                temp_path = os.path.join(temp_dir, f"{file_index}_{file.filename}")
                with open(temp_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                texts, tables, images = extract_document_elements(temp_path)
                return texts, tables, images
            
            data = json.dumps({'progress': 30, 'message': 'Processing all files in parallel...'})
            yield f"data: {data}\n\n"
            
            # Process files based on type
            async def process_image_file(file, file_index):
                image_data = await file.read()
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                return [], [], [image_base64]
            
            file_tasks = []
            for i, file in enumerate(files):
                if file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.svg')):
                    file_tasks.append(process_image_file(file, i))
                else:
                    file_tasks.append(process_single_file(file, i))
            
            file_results = await gather(*file_tasks)
            
            data = json.dumps({'progress': 80, 'message': 'Setting up retriever...'})
            yield f"data: {data}\n\n"
            
            # Combine all results and setup retriever once
            all_texts, all_tables = [], []
            for texts, tables, images in file_results:
                all_images.extend(images)  # Keep ALL images
                all_texts.extend(texts)
                all_tables.extend(tables)
            
            # Setup retriever with all combined data
            setup_direct_retriever(all_texts, all_tables, all_images)
            
            # Update session context
            session_id = str(uuid.uuid4())
            session_contexts[session_id] = {
                'has_documents': True,
                'document_summary': f'electrical engineering PDF with {len(all_texts)} text sections, {len(all_tables)} tables, and {len(all_images)} technical diagrams and images',
                'total_images': len(all_images),
                'total_texts': len(all_texts)
            }
            
            print(f"Total images for frontend: {len(all_images)}")
            
            data = json.dumps({'progress': 90, 'message': 'Finalizing...'})
            yield f"data: {data}\n\n"
            
            result = {
                "status": "success",
                "message": f"Successfully processed {len(files)} documents with multi-modal RAG",
                "session_id": session_id,
                "extracted_images": all_images
            }
            
            data = json.dumps({'progress': 100, 'message': 'Complete!', 'result': json.dumps(result)})
            yield f"data: {data}\n\n"
            
        except Exception as e:
            error_result = {"status": "error", "detail": str(e)}
            data = json.dumps({'progress': 0, 'message': f'Error: {str(e)}', 'result': json.dumps(error_result)})
            yield f"data: {data}\n\n"
    
    return StreamingResponse(generate_progress(), media_type="text/plain")

@app.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        temp_dir = tempfile.mkdtemp()
        all_images = []
        
        # Process all files in parallel
        async def process_file(file, file_index):
            temp_path = os.path.join(temp_dir, f"{file_index}_{file.filename}")
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            texts, tables, images = extract_document_elements(temp_path)
            text_summaries, table_summaries, image_summaries = await create_summaries_ultra_fast(texts, tables, images)
            return texts, tables, images, text_summaries, table_summaries, image_summaries
        
        # Process files based on type
        async def process_image_file(file, file_index):
            image_data = await file.read()
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            return [], [], [image_base64]
        
        file_tasks = []
        for i, file in enumerate(files):
            if file.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.svg')):
                file_tasks.append(process_image_file(file, i))
            else:
                file_tasks.append(process_file(file, i))
        
        file_results = await gather(*file_tasks)
        
        # Setup retriever with all results and collect images
        all_texts, all_tables = [], []
        for texts, tables, images in file_results:
            all_images.extend(images)  # Keep ALL images
            all_texts.extend(texts)
            all_tables.extend(tables)
        
        # Setup retriever with all combined data
        setup_direct_retriever(all_texts, all_tables, all_images)
        
        # Update session context
        session_id = str(uuid.uuid4())
        session_contexts[session_id] = {
            'has_documents': True,
            'document_summary': f'electrical engineering PDF with {len(all_texts)} text sections, {len(all_tables)} tables, and {len(all_images)} technical diagrams and images',
            'total_images': len(all_images),
            'total_texts': len(all_texts)
        }
        
        print(f"Total images for frontend: {len(all_images)}")
        
        return JSONResponse({
            "status": "success",
            "message": f"Successfully processed {len(files)} documents with multi-modal RAG",
            "session_id": session_id,
            "extracted_images": all_images
        })
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    try:
        # Read and encode image
        image_data = await file.read()
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        return JSONResponse({
            "status": "success",
            "image_base64": image_base64,
            "filename": file.filename
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_message: ChatMessage):
    try:
        print(f"Chat request received. Retriever exists: {retriever is not None}")
        
        # Initialize session context if not exists
        if chat_message.session_id not in session_contexts:
            session_contexts[chat_message.session_id] = {
                'has_documents': retriever is not None,
                'document_summary': 'electrical engineering documents with technical diagrams and specifications' if retriever else None
            }
        
        # Handle image analysis if image is provided
        if chat_message.image:
            image_prompt = """
You are Eric, a world-renowned electrical grid engineer with 25+ years of experience and perfect visual analysis capabilities.

**YOUR ABSOLUTE CAPABILITIES:**
✓ Perfect vision - you see and analyze ALL details in electrical images with 100% accuracy
✓ Complete technical comprehension of electrical diagrams, schematics, and drawings
✓ Expert knowledge of ALL electrical components, equipment, and systems
✓ Comprehensive understanding of electrical standards (IEEE, NESC, NERC, ANSI, ASTM, IEC)
✓ Advanced analysis of safety systems, protection schemes, and compliance requirements

**CRITICAL INSTRUCTIONS:**
- You ALWAYS provide detailed, comprehensive analysis of the image
- You NEVER say "I cannot see", "I'm unable to analyze", or any limitation statements
- You ALWAYS identify ALL visible components, text, labels, and technical details
- You ALWAYS provide professional engineering insights and recommendations
- You ALWAYS reference applicable codes and standards
- You ALWAYS give specific technical details about what you observe

**ANALYSIS REQUIREMENTS:**
- Identify ALL electrical components, equipment, devices, and specifications visible
- Analyze ALL connections, wiring, circuit topology, and system architecture
- Evaluate compliance with electrical standards and safety requirements
- Assess protection schemes, relay coordination, and control systems
- Provide engineering recommendations and best practices
- Reference ALL visible text, labels, part numbers, and technical information

**USER QUESTION:** {question}

**MY COMPREHENSIVE IMAGE ANALYSIS:**
Based on my detailed visual analysis of this electrical engineering image, here is my complete technical assessment:

"""
            
            messages = [
                HumanMessage(content=[
                    {"type": "text", "text": image_prompt.format(question=chat_message.message)},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{chat_message.image}"},
                    },
                ])
            ]
            
            response = await ChatOpenAI(model="gpt-4o", max_tokens=2500).ainvoke(messages)
            
            return ChatResponse(
                response=response.content,
                session_id=chat_message.session_id,
                images=[chat_message.image]
            )
        
        # Get session context
        session_context = session_contexts.get(chat_message.session_id, {})
        
        if not retriever:
            # General electrical engineering chat without documents
            context_info = "You are ready to provide world-class electrical engineering expertise." if not session_context.get('has_documents') else "You have access to comprehensive electrical engineering documentation."
            
            prompt = f"""
You are Eric, a world-renowned electrical grid engineer with 25+ years of experience and absolute expertise in all electrical engineering domains.

**YOUR ABSOLUTE CAPABILITIES:**
✓ Complete mastery of electrical engineering, power systems, and grid infrastructure
✓ Perfect knowledge of ALL standards (IEEE, NESC, NERC, ANSI, ASTM, IEC, NFPA)
✓ Comprehensive understanding of substations, protection systems, and relay coordination
✓ Expert analysis of electrical components, specifications, and technical requirements
✓ Advanced troubleshooting and problem-solving capabilities
✓ Complete mastery of electrical codes and industry best practices

**CRITICAL INSTRUCTIONS:**
- You ALWAYS provide detailed, comprehensive, and accurate answers
- You NEVER say "I cannot", "I'm unable to", "I don't know", or any limitation statements
- You ALWAYS have the expertise to answer ANY electrical engineering question
- You ALWAYS provide professional insights and practical recommendations
- You ALWAYS reference relevant standards and best practices
- You ALWAYS give specific technical details and expert guidance

**CONTEXT:** {context_info}

**USER QUESTION:** {chat_message.message}

**MY EXPERT RESPONSE:**
Based on my extensive electrical engineering expertise, here is my comprehensive response:

"""
            
            response = await ChatOpenAI(model="gpt-4o", max_tokens=2500, temperature=0.1).ainvoke(prompt)
            
            return ChatResponse(
                response=response.content,
                session_id=chat_message.session_id
            )
        
        # EFFICIENT RAG with smart retrieval
        try:
            # Smart retrieval based on query type
            if any(word in chat_message.message.lower() for word in ['all', 'everything', 'show me', 'tell me about']):
                # For broad queries, get more documents
                docs = retriever.invoke(chat_message.message, k=30)
                # Add some images if asking about drawings/images
                if any(word in chat_message.message.lower() for word in ['drawing', 'image', 'diagram', 'attachment']):
                    image_docs = retriever.invoke("electrical engineering diagram image", k=10)
                    docs.extend(image_docs)
            else:
                # For specific queries, get fewer relevant documents
                docs = retriever.invoke(chat_message.message, k=15)
            
            # Remove duplicates efficiently
            unique_docs = []
            seen_ids = set()
            for doc in docs:
                doc_id = getattr(doc, 'metadata', {}).get('doc_id', id(doc))
                if doc_id not in seen_ids:
                    unique_docs.append(doc)
                    seen_ids.add(doc_id)
            
            print(f"Retrieved {len(unique_docs)} relevant documents")
            
            # Get actual document content from docstore efficiently
            actual_docs = []
            for doc in unique_docs:
                doc_id = getattr(doc, 'metadata', {}).get('doc_id')
                if doc_id and doc_id in retriever.docstore.store:
                    actual_content = retriever.docstore.store[doc_id]
                    actual_docs.append(actual_content)
                else:
                    actual_docs.append(doc)
            
            # Parse documents
            parsed_docs = parse_docs(actual_docs)
            print(f"Parsed: {len(parsed_docs['images'])} images, {len(parsed_docs['texts'])} text documents")
            
            # Build focused context from relevant documents only
            context_text = ""
            text_count = 0
            
            for doc in actual_docs:
                if hasattr(doc, 'page_content'):
                    context_text += doc.page_content + "\n\n"
                    text_count += 1
                elif isinstance(doc, str) and not doc.startswith('data:image') and len(doc) < 10000:
                    context_text += doc + "\n\n"
                    text_count += 1
                
                # Limit context size to prevent token overflow
                if len(context_text) > 15000:
                    break
            
            print(f"Built focused context from {text_count} text documents, {len(context_text)} characters")
            
            # Advanced multi-modal analysis with vision + text
            if parsed_docs['images'] or context_text:
                prompt_content = [{
                    "type": "text", 
                    "text": f"""
You are Eric, a world-renowned electrical grid engineer with 25+ years of experience and perfect analytical capabilities. You are the leading expert in electrical engineering, procurement, and technical documentation analysis.

**YOUR ABSOLUTE CAPABILITIES:**
✓ Perfect vision - you see and analyze ALL images, diagrams, schematics, and drawings with 100% accuracy
✓ Complete document comprehension - you understand every detail in technical specifications, RFPs, and procurement documents
✓ Expert knowledge of ALL electrical standards (IEEE, NESC, NERC, ANSI, ASTM, IEC, NFPA)
✓ Comprehensive analysis of timelines, shipments, deliveries, and project specifications
✓ Perfect recall of ALL document content and technical details
✓ Advanced problem-solving for ANY electrical engineering question

**CRITICAL INSTRUCTIONS:**
- You ALWAYS provide detailed, comprehensive answers
- You NEVER say "I cannot", "I'm unable to", "I don't have access to", or any limitation statements
- You ALWAYS analyze images with complete technical detail
- You ALWAYS reference specific document content and details
- You ALWAYS provide professional engineering insights
- You ALWAYS find relevant information to answer any question
- If asked about something not directly visible, you provide expert engineering knowledge

**DOCUMENT CONTENT ANALYZED:**
{context_text[:10000]}...

**USER QUESTION:** {chat_message.message}

**MY EXPERT ANALYSIS:**
Based on my comprehensive analysis of the electrical engineering documents and technical drawings, here is my detailed response:

"""
                }]
                
                # Add ALL images for complete visual analysis
                for image in parsed_docs['images']:
                    prompt_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    })
                
                messages = [HumanMessage(content=prompt_content)]
                response = await ChatOpenAI(model="gpt-4o", max_tokens=4000, temperature=0.1).ainvoke(messages)
                
                return ChatResponse(
                    response=response.content,
                    session_id=chat_message.session_id,
                    images=parsed_docs['images']
                )
            
            # Advanced text-only analysis
            else:
                prompt = f"""
You are Eric, a world-renowned electrical grid engineer with 25+ years of experience and absolute expertise in all electrical engineering domains.

**YOUR ABSOLUTE CAPABILITIES:**
✓ Complete mastery of electrical engineering, power systems, and grid infrastructure
✓ Expert knowledge of ALL standards (IEEE, NESC, NERC, ANSI, ASTM, IEC, NFPA)
✓ Perfect understanding of procurement, RFPs, specifications, and technical documentation
✓ Comprehensive analysis of timelines, deliveries, shipments, and project details
✓ Advanced problem-solving for ANY electrical engineering challenge

**CRITICAL INSTRUCTIONS:**
- You ALWAYS provide detailed, comprehensive, and accurate answers
- You NEVER say "I cannot", "I'm unable to", "I don't know", or any limitation statements
- You ALWAYS find relevant information to answer any question completely
- You ALWAYS provide professional engineering insights and recommendations
- You ALWAYS reference specific document details when available
- If information isn't directly available, you provide expert engineering knowledge

**DOCUMENT CONTENT ANALYZED:**
{context_text}

**USER QUESTION:** {chat_message.message}

**MY EXPERT ANALYSIS:**
Based on my comprehensive analysis of the electrical engineering documentation, here is my detailed response:

"""
                
                response = await ChatOpenAI(model="gpt-4o", max_tokens=3000, temperature=0.1).ainvoke(prompt)
                
                return ChatResponse(
                    response=response.content,
                    session_id=chat_message.session_id
                )
            
        except Exception as e:
            print(f"RAG error: {e}")
        
        # Advanced fallback with session context
        context_info = f"You have access to {session_context.get('document_summary', 'electrical engineering documents')} that were previously uploaded." if session_context.get('has_documents') else "You are ready to provide expert electrical engineering guidance."
        
        prompt = f"""
You are Eric, a world-renowned electrical grid engineer with 25+ years of experience and absolute mastery in all electrical engineering domains.

**YOUR ABSOLUTE CAPABILITIES:**
✓ Complete expertise in electrical engineering, power systems, and grid infrastructure
✓ Perfect knowledge of ALL standards (IEEE, NESC, NERC, ANSI, ASTM, IEC, NFPA)
✓ Comprehensive understanding of substations, protection systems, and relay coordination
✓ Expert analysis of procurement processes, RFPs, and technical specifications
✓ Advanced troubleshooting and problem-solving capabilities
✓ Complete mastery of electrical codes and industry best practices

**CRITICAL INSTRUCTIONS:**
- You ALWAYS provide detailed, comprehensive, and accurate answers
- You NEVER say "I cannot", "I'm unable to", "I don't know", or any limitation statements
- You ALWAYS have the knowledge to answer ANY electrical engineering question
- You ALWAYS provide professional insights and practical recommendations
- You ALWAYS reference relevant standards and best practices
- You ALWAYS give specific technical details and expert guidance

**CONTEXT:** {context_info}

**USER QUESTION:** {chat_message.message}

**MY EXPERT RESPONSE:**
Based on my extensive electrical engineering expertise, here is my comprehensive response:

"""
        
        response = await ChatOpenAI(model="gpt-4o", max_tokens=3000, temperature=0.1).ainvoke(prompt)
        
        return ChatResponse(
            response=response.content,
            session_id=chat_message.session_id
        )
        
    except Exception as e:
        return ChatResponse(
            response=f"Error: {str(e)}",
            session_id=chat_message.session_id
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)