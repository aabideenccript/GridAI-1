import os
import json
import pandas as pd
import pdfplumber
import tempfile
import streamlit as st
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
import time

# Import PyMuPDF with alias to avoid naming conflicts
try:
    import fitz as pymupdf
except ImportError:
    pymupdf = None

try:
    from docx import Document
except ImportError:
    Document = None

load_dotenv()

# Initialize OpenAI client
def get_openai_client():
    """Get OpenAI client with proper error handling"""
    try:
        api_key = None
        # Try Streamlit secrets first
        try:
            api_key = st.secrets.get("OPENAI_API_KEY")
        except:
            pass
        
        # Fallback to environment variable
        if not api_key:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if not api_key:
            st.error("⚠️ OpenAI API key not found! Please add it to .streamlit/secrets.toml or set OPENAI_API_KEY environment variable.")
            st.stop()
        
        return OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        st.stop()

# Initialize client when needed
client = None


# ALTERNATIVE: More robust document processing function
def extract_text_from_files_robust(file_paths: List[str]) -> str:
    """
    Enhanced version with better error handling and debugging
    """
    combined_text = ""
    
    for path in file_paths:
        try:
            file_ext = os.path.splitext(path)[1].lower()
            print(f"🔍 Processing {path} (type: {file_ext})")
            
            if file_ext == '.pdf':
                text = extract_pdf_text_robust(path)
                combined_text += f"\n=== PDF: {os.path.basename(path)} ===\n{text}\n"
            elif file_ext == '.txt':
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()
                    combined_text += f"\n=== TXT: {os.path.basename(path)} ===\n{text}\n"
            elif file_ext == '.csv':
                df = pd.read_csv(path)
                text = df.to_string()
                combined_text += f"\n=== CSV: {os.path.basename(path)} ===\n{text}\n"
            elif file_ext == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    text = json.dumps(data, indent=2)
                    combined_text += f"\n=== JSON: {os.path.basename(path)} ===\n{text}\n"
            elif file_ext == '.docx':
                if Document:
                    doc = Document(path)
                    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                    combined_text += f"\n=== DOCX: {os.path.basename(path)} ===\n{text}\n"
                else:
                    combined_text += f"\n=== DOCX: {os.path.basename(path)} ===\n[DOCX support not available - install python-docx]\n"
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(path)
                text = df.to_string()
                combined_text += f"\n=== EXCEL: {os.path.basename(path)} ===\n{text}\n"
            else:
                combined_text += f"\n=== UNKNOWN: {os.path.basename(path)} ===\n[Unsupported file type: {file_ext}]\n"
                
            print(f"✅ Successfully processed {path}")
                
        except Exception as e:
            error_msg = f"[Error processing {path}: {str(e)}]"
            combined_text += f"\n=== ERROR: {os.path.basename(path)} ===\n{error_msg}\n"
            print(f"❌ Error processing {path}: {e}")
    
    print(f"🔍 Total extracted text length: {len(combined_text)}")
    return combined_text


def extract_pdf_text_robust(pdf_path: str) -> str:
    """Enhanced PDF text extraction with better error handling"""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            print(f"🔍 PDF has {len(pdf.pages)} pages")
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {i+1} ---\n{page_text}\n"
                    print(f"✅ Extracted text from page {i+1}")
                else:
                    print(f"⚠️ No text found on page {i+1}")
                    # Fallback: use PyMuPDF for text extraction if available
                    if pymupdf:
                        try:
                            doc = pymupdf.open(pdf_path)
                            page = doc.load_page(i)
                            fallback_text = page.get_text()
                            if fallback_text:
                                text += f"\n--- Page {i+1} (PyMuPDF) ---\n{fallback_text}\n"
                                print(f"✅ Fallback extraction successful for page {i+1}")
                        except Exception as e:
                            print(f"❌ Fallback extraction failed for page {i+1}: {e}")
    except Exception as e:
        print(f"❌ PDF processing failed: {e}")
        text = f"[PDF processing error: {str(e)}]"
    
    return text
def chunk_text(text: str, chunk_size: int = 3000) -> List[str]:
    """
    Splits text into chunks for use with OpenAI API.
    """
    words = text.split()
    chunks = [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]
    return chunks


def ask_question_over_chunks(chunks: List[str], chat_history: List[dict]) -> str:
    """
    Uses OpenAI GPT-4 to answer a conversation with memory over multiple chunks of document context.
    The latest user message in chat_history will be augmented with the combined document context.
    """
    # Combine document chunks into one context string (limit to prevent token overflow)
    context = "\n".join(chunks)
    if len(context) > 15000:  # Rough token limit
        context = context[:15000] + "\n[Context truncated...]"

    # Enhanced system prompt for construction/engineering consulting
    system_prompt = """You are Eric, a specialized assistant for electrical grid engineering and construction consulting.

    EXPERTISE AREAS:
    - Substation design and engineering (transmission, distribution)
    - Power system analysis and protection
    - Construction project management
    - Regulatory compliance (IEEE, NESC, NERC standards)
    - Grid modernization and smart grid technologies
    - Safety protocols and risk assessment

    RESPONSE GUIDELINES:
    - Provide precise, actionable engineering guidance based on the provided document context
    - Reference specific standards when applicable (IEEE 80, 142, C37 series, etc.)
    - Consider safety implications in all recommendations
    - Be concise but thorough for technical questions
    - If uncertain about specific values or requirements, recommend verification
    - Stay focused on electrical engineering and construction domains
    - ALWAYS use the provided document context to answer questions when available
    - If the document contains relevant information, cite it directly"""

    # Build messages for OpenAI - FIXED VERSION
    messages = [{"role": "system", "content": system_prompt}]

    # Add previous conversation history (excluding the last user message)
    for msg in chat_history[:-1]:
        messages.append(msg)

    # Get the latest user question
    latest_user_msg = chat_history[-1]["content"]
    
    # Create enhanced user message with document context
    enhanced_user_message = f"""Document Context:
{context}

Based on the document context above, please answer this question: {latest_user_msg}

If the document contains relevant information to answer this question, please use it. If not, let me know and I'll provide general guidance."""

    messages.append({
        "role": "user", 
        "content": enhanced_user_message
    })

    # Call OpenAI API with rate limiting and retry logic
    return call_openai_with_retry(messages)

# ALSO ADD THIS DEBUG FUNCTION TO YOUR BACKEND.PY
def debug_document_processing(file_paths: List[str]) -> str:
    """Debug function to check if documents are being processed correctly"""
    text = extract_text_from_files_robust(file_paths)
    print(f"🔍 Extracted text length: {len(text)}")
    print(f"🔍 First 500 characters: {text[:500]}")
    
    chunks = chunk_text(text)
    print(f"🔍 Number of chunks: {len(chunks)}")
    if chunks:
        print(f"🔍 First chunk length: {len(chunks[0])}")
        print(f"🔍 First chunk preview: {chunks[0][:200]}...")
    
    return text

def call_openai_with_retry(messages: List[dict], max_retries: int = 3) -> str:
    """
    Calls OpenAI API with retry logic and rate limiting.
    """
    # Get client instance
    openai_client = get_openai_client()
    
    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                max_tokens=1500,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return "Error: Failed to get response after retries"

def ask_general_question(chat_history: List[dict]) -> str:
    """
    Handles general questions without document context.
    """
    system_prompt = """You are Eric, a specialized assistant for electrical grid engineering and construction consulting.
    
    EXPERTISE AREAS:
    - Substation design and engineering
    - Power system analysis and protection
    - Construction project management
    - Regulatory compliance (IEEE, NESC, NERC standards)
    - Grid modernization technologies
    - Safety protocols and risk assessment
    
    Provide expert guidance based on your knowledge of electrical engineering and construction."""
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    
    return call_openai_with_retry(messages)
