"""
Embedding Service using NVIDIA API.

This module handles text embedding using NVIDIA's embedding models
via OpenAI-compatible API.
"""

from typing import List, Optional
from openai import OpenAI

from .config import NVIDIA_API_KEY, NVIDIA_BASE_URL, EMBEDDING_MODEL


class EmbeddingService:
    """
    Service for generating text embeddings using NVIDIA API.
    
    Uses OpenAI-compatible client to interact with NVIDIA's embedding models.
    """
    
    def __init__(
        self,
        api_key: str = NVIDIA_API_KEY,
        base_url: str = NVIDIA_BASE_URL,
        model: str = EMBEDDING_MODEL
    ):
        """
        Initialize the embedding service.
        
        Args:
            api_key: NVIDIA API key
            base_url: NVIDIA API base URL
            model: Embedding model to use
        """
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )
        self.model = model
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        # Truncate if too long (most models have token limits)
        text = text[:8000] if len(text) > 8000 else text
        
        response = self.client.embeddings.create(
            input=[text],
            model=self.model,
            encoding_format="float",
            extra_body={"input_type": "passage", "truncate": "NONE"}
        )
        
        return response.data[0].embedding
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        # Process in batches to avoid rate limits
        batch_size = 10
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Truncate each text
            batch = [t[:8000] if len(t) > 8000 else t for t in batch]
            
            response = self.client.embeddings.create(
                input=batch,
                model=self.model,
                encoding_format="float",
                extra_body={"input_type": "passage", "truncate": "NONE"}
            )
            
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a query (for retrieval).
        
        Uses different input_type for queries vs passages.
        
        Args:
            query: Query text
            
        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(
            input=[query],
            model=self.model,
            encoding_format="float",
            extra_body={"input_type": "query", "truncate": "NONE"}
        )
        
        return response.data[0].embedding


def get_embedding_service() -> EmbeddingService:
    """Factory function to create embedding service."""
    return EmbeddingService()
