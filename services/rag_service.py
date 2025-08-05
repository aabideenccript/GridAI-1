import asyncio
from typing import Dict, Any, List, Optional
from .vector_store import VectorStore
from .openai_service import OpenAIService

class RAGService:
    def __init__(self, vector_store: VectorStore, openai_service: OpenAIService):
        self.vector_store = vector_store
        self.openai_service = openai_service
        self.conversation_history = {}  # Store conversation history by session
        
    async def get_response(self, query: str, session_id: str = "default") -> Dict[str, Any]:
        """Get response using RAG approach"""
        try:
            # Initialize conversation history for new sessions
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []
            
            # Check if this is an image analysis query
            image_query_keywords = ['image', 'see this', 'what do you see', 'analyze', 'picture', 'photo', 'uploaded', 'electrical equipment', 'tell me about this']
            is_image_query = any(keyword in query.lower() for keyword in image_query_keywords)
            
            # Search for relevant documents
            search_results = await self.vector_store.similarity_search(query, k=5)
            
            # Prepare context from search results
            context_docs = []
            sources = []
            images = []
            
            for result in search_results:
                if result['similarity_score'] > 0.1:  # Lower threshold for better image matching
                    context_docs.append(result['content'])
                    source_info = {
                        'filename': result['metadata'].get('filename', 'Unknown'),
                        'page': result['metadata'].get('page'),
                        'type': result['metadata'].get('type', 'document'),
                        'similarity': round(result['similarity_score'], 3)
                    }
                    
                    # Check if this source has an associated image
                    if result['metadata'].get('has_image') and result['metadata'].get('image_url'):
                        source_info['image_url'] = result['metadata']['image_url']
                        images.append({
                            'url': result['metadata']['image_url'],
                            'caption': f"From {result['metadata'].get('filename', 'Unknown')} - {result['metadata'].get('type', 'document')}",
                            'similarity': round(result['similarity_score'], 3)
                        })
                    
                    sources.append(source_info)
            
            # Generate response based on query type and available context  
            if is_image_query:
                # For image queries, look for ALL images in the database regardless of similarity
                all_images = []
                for metadata in self.vector_store.metadata:
                    if metadata.get('has_image') and metadata.get('image_url'):
                        all_images.append({
                            'url': metadata['image_url'],
                            'caption': f"From {metadata.get('filename', 'Unknown')}",
                            'similarity': 1.0  # Set high similarity for direct image queries
                        })
                
                if all_images:
                    # Use image analysis for visual queries
                    response = await self._generate_image_analysis_response(query, all_images, context_docs, session_id)
                    # Update images in the response
                    images = all_images
                elif context_docs:
                    # Use RAG approach with document context
                    response = await self._generate_rag_response(query, context_docs, session_id)
                else:
                    # Fallback to general OpenAI response
                    response = await self._generate_fallback_response(query, session_id)
                    sources = []
            elif context_docs:
                # Use RAG approach with document context
                response = await self._generate_rag_response(query, context_docs, session_id)
            else:
                # Fallback to general OpenAI response
                response = await self._generate_fallback_response(query, session_id)
                sources = []
            
            # Update conversation history
            self.conversation_history[session_id].append({
                'user': query,
                'assistant': response
            })
            
            # Keep only last 10 exchanges to manage memory
            if len(self.conversation_history[session_id]) > 10:
                self.conversation_history[session_id] = self.conversation_history[session_id][-10:]
            
            return {
                'answer': response,
                'sources': sources,
                'images': images,
                'has_context': len(context_docs) > 0
            }
            
        except Exception as e:
            print(f"Error generating RAG response: {str(e)}")
            return {
                'answer': f"I apologize, but I encountered an error while processing your question: {str(e)}",
                'sources': [],
                'images': [],
                'has_context': False
            }
    
    async def _generate_rag_response(self, query: str, context_docs: List[str], session_id: str) -> str:
        """Generate response using document context"""
        try:
            # Prepare context
            context = "\n\n".join([f"Document {i+1}:\n{doc}" for i, doc in enumerate(context_docs)])
            
            # Get conversation history
            history = self.conversation_history.get(session_id, [])
            
            # Prepare conversation context
            conversation_context = ""
            if history:
                recent_history = history[-3:]  # Last 3 exchanges
                for exchange in recent_history:
                    conversation_context += f"Human: {exchange['user']}\nAssistant: {exchange['assistant']}\n\n"
            
            # Create system prompt
            system_prompt = """You are an expert electrical grid and substation engineer. You have access to technical documents and your role is to provide accurate, detailed answers about electrical grid systems, substations, power transmission, distribution, and related electrical engineering topics.

Guidelines:
1. Use the provided document context to answer questions accurately
2. If the documents contain relevant information, prioritize that over general knowledge
3. Be specific about technical details, specifications, and procedures
4. If asked about safety, always emphasize proper safety protocols
5. When discussing equipment, mention specific models, ratings, or specifications if available in the documents
6. If the context doesn't fully answer the question, combine document information with your electrical engineering knowledge
7. Always maintain technical accuracy and cite specific information when possible

Context from uploaded documents:
{context}

Recent conversation:
{conversation_context}

Current question: {query}

Provide a comprehensive, technically accurate response based on the document context and your electrical engineering expertise."""

            prompt = system_prompt.format(
                context=context,
                conversation_context=conversation_context,
                query=query
            )
            
            response = await self.openai_service.get_completion(
                prompt=prompt,
                model="gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
            )
            
            return response
            
        except Exception as e:
            print(f"Error generating RAG response: {str(e)}")
            raise e
    
    async def _generate_fallback_response(self, query: str, session_id: str) -> str:
        """Generate response without document context (fallback to OpenAI)"""
        try:
            # Get conversation history
            history = self.conversation_history.get(session_id, [])
            
            # Prepare conversation context
            conversation_context = ""
            if history:
                recent_history = history[-3:]  # Last 3 exchanges
                for exchange in recent_history:
                    conversation_context += f"Human: {exchange['user']}\nAssistant: {exchange['assistant']}\n\n"
            
            system_prompt = """You are an expert electrical grid and substation engineer. You provide accurate, detailed answers about electrical grid systems, substations, power transmission, distribution, and related electrical engineering topics.

Note: No specific documents were found in the knowledge base for this question, so you're providing answers based on your general electrical engineering knowledge.

Guidelines:
1. Provide technically accurate information about electrical grid and substation topics
2. Include specific technical details, standards, and best practices where relevant
3. Always emphasize safety considerations and proper procedures
4. If you're not certain about specific technical details, acknowledge the limitation
5. Suggest consulting specific standards, manuals, or experts when appropriate

Recent conversation:
{conversation_context}

Current question: {query}

Provide a comprehensive response based on your electrical engineering knowledge."""

            prompt = system_prompt.format(
                conversation_context=conversation_context,
                query=query
            )
            
            response = await self.openai_service.get_completion(
                prompt=prompt,
                model="gpt-4o"  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
            )
            
            # Add note about fallback
            response += "\n\n*Note: This response is based on general electrical engineering knowledge as no specific documents in the knowledge base matched your query.*"
            
            return response
            
        except Exception as e:
            print(f"Error generating fallback response: {str(e)}")
            raise e
    
    def clear_conversation_history(self, session_id: str):
        """Clear conversation history for a session"""
        if session_id in self.conversation_history:
            del self.conversation_history[session_id]
    
    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a session"""
        return self.conversation_history.get(session_id, [])
    
    async def _generate_image_analysis_response(self, query: str, images: List[Dict], context_docs: List[str], session_id: str) -> str:
        """Generate response using image analysis"""
        try:
            # If we have images with base64 data or URLs, analyze them
            analysis_results = []
            
            for img in images:
                try:
                    import os
                    import base64
                    
                    # Extract filename from URL (assumes format /static/extracted_images/filename)
                    if img.get('url'):
                        # Convert relative URL to file path
                        image_path = img['url'].replace('/static/', 'static/')
                        if os.path.exists(image_path):
                            with open(image_path, 'rb') as f:
                                image_data = base64.b64encode(f.read()).decode()
                            
                            # Create electrical equipment analysis prompt
                            analysis_prompt = f"""Analyze this electrical equipment image in detail. Focus on:
1. Identify all electrical components visible (transformers, circuit breakers, switches, transmission lines, etc.)
2. Describe the type of electrical installation (substation, power plant, distribution center, etc.)
3. Note any safety equipment or features
4. Describe the voltage level if apparent from the equipment size/design
5. Mention any technical specifications or ratings visible
6. Assess the condition and age of equipment if possible

User's question: {query}

Provide a comprehensive technical analysis suitable for electrical engineering professionals."""
                            
                            analysis = await self.openai_service.analyze_image(image_data, analysis_prompt)
                            analysis_results.append(f"Analysis of {img.get('caption', 'uploaded image')}:\n{analysis}")
                        
                except Exception as e:
                    print(f"Error analyzing image: {e}")
                    analysis_results.append(f"Could not analyze image {img.get('caption', 'unknown')}: {str(e)}")
            
            if analysis_results:
                # Combine image analysis with document context if available
                if context_docs:
                    combined_prompt = f"""Based on the image analysis and available technical documents, provide a comprehensive response.

Image Analysis Results:
{chr(10).join(analysis_results)}

Related Document Context:
{chr(10).join(context_docs[:2])}

User Question: {query}

Provide a detailed technical response combining the visual analysis with the document information."""
                    
                    return await self.openai_service.get_completion(combined_prompt)
                else:
                    return "\n\n".join(analysis_results)
            else:
                return "I couldn't analyze the uploaded images. Please make sure the images are properly uploaded and try again."
                
        except Exception as e:
            print(f"Error in image analysis response: {e}")
            return f"I encountered an error while analyzing the image: {str(e)}"
