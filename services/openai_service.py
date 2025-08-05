import os
import json
import base64
from typing import Optional, Dict, Any
from openai import OpenAI

class OpenAIService:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("Warning: OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
    async def get_completion(self, prompt: str, model: str = "gpt-4o", max_tokens: Optional[int] = 800) -> str:
        """Get text completion from OpenAI"""
        if not self.client:
            raise Exception("OpenAI client not initialized. Please check your API key.")
        
        try:
            response = self.client.chat.completions.create(
                model=model,  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            return response.choices[0].message.content or ""
            
        except Exception as e:
            print(f"Error getting OpenAI completion: {str(e)}")
            raise e
    
    async def analyze_image(self, base64_image: str, prompt: Optional[str] = None) -> str:
        """Analyze image using OpenAI Vision model"""
        if not self.client:
            raise Exception("OpenAI client not initialized. Please check your API key.")
        
        try:
            default_prompt = "Analyze this image in detail and describe its key elements, context, and any notable aspects."
            analysis_prompt = prompt or default_prompt
            
            response = self.client.chat.completions.create(
                model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": analysis_prompt,
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                max_tokens=500,
            )
            
            return response.choices[0].message.content or ""
            
        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            raise e
    
    async def get_structured_completion(self, prompt: str, model: str = "gpt-4o") -> Dict[str, Any]:
        """Get structured JSON response from OpenAI"""
        if not self.client:
            raise Exception("OpenAI client not initialized. Please check your API key.")
        
        try:
            response = self.client.chat.completions.create(
                model=model,  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
                messages=[
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content or "{}"
            result = json.loads(content)
            return result
            
        except Exception as e:
            print(f"Error getting structured completion: {str(e)}")
            raise e
    
    async def generate_embeddings(self, text: str, model: str = "text-embedding-ada-002") -> list:
        """Generate embeddings for text"""
        if not self.client:
            raise Exception("OpenAI client not initialized. Please check your API key.")
        
        try:
            response = self.client.embeddings.create(
                input=text,
                model=model
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            print(f"Error generating embeddings: {str(e)}")
            raise e
