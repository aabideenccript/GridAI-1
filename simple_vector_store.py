import json
import os
import asyncio
from typing import List, Dict, Any, Optional
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SimpleVectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.documents = []
        self.vectors = None
        self.metadata = []
        self.db_path = "simple_vector_db"
        
    async def initialize(self):
        """Initialize the simple vector store"""
        try:
            os.makedirs(self.db_path, exist_ok=True)
            await self.load_data()
            print("Simple vector store initialized successfully")
        except Exception as e:
            print(f"Error initializing simple vector store: {str(e)}")
            raise e
    
    async def add_documents(self, chunks: List[Dict[str, Any]], metadata: Dict[str, Any]):
        """Add document chunks to vector store"""
        try:
            for i, chunk in enumerate(chunks):
                # Combine chunk metadata with file metadata
                combined_metadata = {**metadata, **chunk['metadata']}
                combined_metadata['chunk_id'] = f"{metadata.get('file_id', 'unknown')}_{i}"
                
                self.documents.append(chunk['content'])
                self.metadata.append(combined_metadata)
            
            # Recompute vectors
            if self.documents:
                self.vectors = self.vectorizer.fit_transform(self.documents)
            
            await self.save_data()
            print(f"Added {len(chunks)} chunks to simple vector store")
            
        except Exception as e:
            print(f"Error adding documents to simple vector store: {str(e)}")
            raise e
    
    async def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        try:
            if not self.documents or self.vectors is None:
                return []
            
            # Transform query
            query_vector = self.vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.vectors)[0]
            
            # Get top k results
            top_indices = np.argsort(similarities)[::-1][:k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum similarity threshold
                    results.append({
                        'content': self.documents[idx],
                        'metadata': self.metadata[idx],
                        'similarity_score': float(similarities[idx])
                    })
            
            return results
            
        except Exception as e:
            print(f"Error performing similarity search: {str(e)}")
            return []
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            unique_files = set()
            for meta in self.metadata:
                if 'filename' in meta:
                    unique_files.add(meta['filename'])
            
            return {
                'chunks': len(self.documents),
                'documents': len(unique_files),
                'status': 'healthy'
            }
            
        except Exception as e:
            print(f"Error getting simple vector store stats: {str(e)}")
            return {
                'chunks': 0,
                'documents': 0,
                'status': 'error',
                'error': str(e)
            }
    
    async def save_data(self):
        """Save data to disk"""
        try:
            data = {
                'documents': self.documents,
                'metadata': self.metadata,
                'vectorizer': self.vectorizer,
                'vectors': self.vectors
            }
            
            with open(os.path.join(self.db_path, 'data.pkl'), 'wb') as f:
                pickle.dump(data, f)
                
        except Exception as e:
            print(f"Error saving data: {str(e)}")
    
    async def load_data(self):
        """Load data from disk"""
        try:
            data_file = os.path.join(self.db_path, 'data.pkl')
            if os.path.exists(data_file):
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                
                self.documents = data.get('documents', [])
                self.metadata = data.get('metadata', [])
                self.vectorizer = data.get('vectorizer', TfidfVectorizer(max_features=1000, stop_words='english'))
                self.vectors = data.get('vectors')
                
                print(f"Loaded {len(self.documents)} documents from storage")
                
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            # Initialize empty if loading fails
            self.documents = []
            self.metadata = []
            self.vectors = None
    
    async def delete_document(self, file_id: str):
        """Delete all chunks for a specific document"""
        try:
            indices_to_remove = []
            for i, meta in enumerate(self.metadata):
                if meta.get('file_id') == file_id:
                    indices_to_remove.append(i)
            
            # Remove in reverse order to maintain indices
            for idx in reversed(indices_to_remove):
                del self.documents[idx]
                del self.metadata[idx]
            
            # Recompute vectors if documents remain
            if self.documents:
                self.vectors = self.vectorizer.fit_transform(self.documents)
            else:
                self.vectors = None
            
            await self.save_data()
            print(f"Deleted {len(indices_to_remove)} chunks for file {file_id}")
            
        except Exception as e:
            print(f"Error deleting document {file_id}: {str(e)}")
            raise e
    
    async def clear_all(self):
        """Clear all documents from vector store"""
        try:
            self.documents = []
            self.metadata = []
            self.vectors = None
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            await self.save_data()
            print("Simple vector store cleared")
            
        except Exception as e:
            print(f"Error clearing simple vector store: {str(e)}")
            raise e