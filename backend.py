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

# Import docx with error handling
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


def extract_text_from_files(file_paths: List[str]) -> str:
    """
    Extracts and concatenates text from various file types.
    Supports: PDF, TXT, CSV, JSON, DOCX, XLSX
    """
    combined_text = ""
    
    for path in file_paths:
        try:
            file_ext = os.path.splitext(path)[1].lower()
            
            if file_ext == '.pdf':
                combined_text += extract_pdf_text(path)
            elif file_ext == '.txt':
                with open(path, 'r', encoding='utf-8') as f:
                    combined_text += f.read() + "\n"
            elif file_ext == '.csv':
                df = pd.read_csv(path)
                combined_text += df.to_string() + "\n"
            elif file_ext == '.json':
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    combined_text += json.dumps(data, indent=2) + "\n"
            elif file_ext == '.docx':
                if Document:
                    doc = Document(path)
                    for paragraph in doc.paragraphs:
                        combined_text += paragraph.text + "\n"
                else:
                    combined_text += f"[DOCX support not available - install python-docx]\n"
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(path)
                combined_text += df.to_string() + "\n"
            else:
                combined_text += f"[Unsupported file type: {file_ext}]\n"
                
        except Exception as e:
            combined_text += f"[Error processing {path}: {str(e)}]\n"
    
    return combined_text

def extract_pdf_text(pdf_path: str) -> str:
    """
    Extracts text from a single PDF file.
    """
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                # Fallback: use PyMuPDF for text extraction if available
                if pymupdf:
                    doc = pymupdf.open(pdf_path)
                    page = doc.load_page(i)
                    text += page.get_text() + "\n"
                else:
                    text += "[PyMuPDF not available for fallback extraction]\n"
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
    - Provide precise, actionable engineering guidance
    - Reference specific standards when applicable (IEEE 80, 142, C37 series, etc.)
    - Consider safety implications in all recommendations
    - Be concise but thorough for technical questions
    - If uncertain about specific values or requirements, recommend verification
    - Stay focused on electrical engineering and construction domains"""

    # Build messages for OpenAI
    messages = [{"role": "system", "content": system_prompt}]

    # Add all messages except the latest user message
    for msg in chat_history[:-1]:
        messages.append(msg)

    # Augment the latest user message with document context
    latest_user_msg = chat_history[-1]
    messages.append({
        "role": "user",
        "content": f"Document Context:\n{context}\n\nQuestion: {latest_user_msg['content']}"
    })

    # Call OpenAI API with rate limiting and retry logic
    return call_openai_with_retry(messages)


def call_openai_with_retry(messages: List[dict], max_retries: int = 3) -> str:
    """
    Calls OpenAI API with retry logic and rate limiting.
    """
    # Get client instance
    openai_client = get_openai_client()
    
    for attempt in range(max_retries):
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.2,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                time.sleep(wait_time)
                continue
            return f"[Error from OpenAI after {max_retries} attempts: {str(e)}]"

def ask_general_question(chat_history: List[dict]) -> str:
    """
    Handles general questions without document context using enhanced prompts.
    """
    system_prompt = """You are Eric, a specialized assistant for electrical grid engineering and construction consulting.

    EXPERTISE AREAS:
    - Substation design and engineering (transmission, distribution)
    - Power system analysis and protection
    - Construction project management
    - Regulatory compliance (IEEE, NESC, NERC standards)
    - Grid modernization and smart grid technologies
    - Safety protocols and risk assessment

    KNOWLEDGE BASE:
    - IEEE Std 80: Substation grounding design
    - IEEE Std 142: Grounding of industrial and commercial power systems
    - IEEE C37 series: Power system protection and control
    - NESC: Safety standards for electric supply and communication lines
    - Transmission substations: 115kV to 765kV operations
    - Distribution systems: 4kV to 69kV operations

    RESPONSE GUIDELINES:
    - Provide precise, actionable engineering guidance
    - Reference specific standards when applicable
    - Consider safety implications in all recommendations
    - Be concise but thorough for technical questions
    - If uncertain about specific values, recommend verification with standards
    - Stay focused on electrical engineering and construction domains
    - Multiple file types are supported for document analysis (PDF, DOCX, XLSX, CSV, TXT, JSON)"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)

    return call_openai_with_retry(messages)
