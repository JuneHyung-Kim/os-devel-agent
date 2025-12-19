"""
Custom Ollama Embedding Function for ChromaDB
"""
import requests
from typing import List
from chromadb.api.types import Documents, Embeddings
import chromadb.utils.embedding_functions as embedding_functions


class OllamaEmbeddingFunction(embedding_functions.EmbeddingFunction[Documents]):
    """Custom embedding function for Ollama."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model_name: str = "mxbai-embed-large"):
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.embed_url = f"{self.base_url}/api/embeddings"
    
    def __call__(self, input: Documents) -> Embeddings:
        """Generate embeddings for the input documents."""
        embeddings = []
        
        for text in input:
            response = requests.post(
                self.embed_url,
                json={
                    "model": self.model_name,
                    "prompt": text
                }
            )
            
            if response.status_code != 200:
                raise RuntimeError(
                    f"Ollama embedding request failed: {response.status_code} - {response.text}"
                )
            
            result = response.json()
            embeddings.append(result["embedding"])
        
        return embeddings
