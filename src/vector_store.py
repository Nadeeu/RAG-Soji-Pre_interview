"""
Vector Store using ChromaDB.

This module handles:
1. Storing document embeddings in ChromaDB
2. Retrieving similar documents
3. Updating/deleting documents
"""

import os
from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings

from .config import CHROMA_PERSIST_DIR, COLLECTION_NAME
from .document_loader import Document
from .embeddings import EmbeddingService, get_embedding_service


class VectorStore:
    """
    Vector database using ChromaDB for storing and retrieving document embeddings.
    """
    
    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
        embedding_service: Optional[EmbeddingService] = None
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_dir: Directory to persist ChromaDB data
            collection_name: Name of the collection
            embedding_service: Service for generating embeddings
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(path=persist_dir)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        # Initialize embedding service
        self.embedding_service = embedding_service or get_embedding_service()
    
    def document_exists(self, doc_id: str) -> bool:
        """
        Check if a document already exists in the vector store.
        
        Args:
            doc_id: Document ID to check
            
        Returns:
            True if document exists, False otherwise
        """
        try:
            result = self.collection.get(ids=[doc_id])
            return len(result['ids']) > 0
        except Exception:
            return False
    
    def get_existing_ids(self, doc_ids: List[str]) -> set:
        """
        Get which document IDs already exist in the vector store.
        
        Args:
            doc_ids: List of document IDs to check
            
        Returns:
            Set of existing document IDs
        """
        if not doc_ids:
            return set()
        
        try:
            result = self.collection.get(ids=doc_ids)
            return set(result['ids'])
        except Exception:
            return set()
    
    def add_documents(self, documents: List[Document], skip_existing: bool = True) -> int:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of Document objects to add
            skip_existing: If True, skip documents that already exist
            
        Returns:
            Number of documents actually added
        """
        if not documents:
            return 0
        
        # Filter out existing documents if requested
        if skip_existing:
            doc_ids = [doc.doc_id for doc in documents]
            existing_ids = self.get_existing_ids(doc_ids)
            
            if existing_ids:
                print(f"Skipping {len(existing_ids)} existing documents")
                documents = [doc for doc in documents if doc.doc_id not in existing_ids]
            
            if not documents:
                print("All documents already exist in vector store")
                return 0
        
        print(f"Generating embeddings for {len(documents)} documents...")
        
        # Extract texts and metadata
        texts = [doc.content for doc in documents]
        ids = [doc.doc_id for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedding_service.embed_texts(texts)
        
        print(f"Adding documents to ChromaDB...")
        
        # Add to collection
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"Added {len(documents)} documents to vector store")
        return len(documents)
    
    def update_documents(self, documents: List[Document]) -> None:
        """
        Update existing documents in the vector store.
        
        If document doesn't exist, it will be added.
        
        Args:
            documents: List of Document objects to update
        """
        if not documents:
            return
        
        print(f"Updating {len(documents)} documents...")
        
        # Extract texts and metadata
        texts = [doc.content for doc in documents]
        ids = [doc.doc_id for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedding_service.embed_texts(texts)
        
        # Upsert to collection
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"Updated {len(documents)} documents")
    
    def delete_documents(self, doc_ids: List[str]) -> None:
        """
        Delete documents from the vector store.
        
        Args:
            doc_ids: List of document IDs to delete
        """
        if not doc_ids:
            return
        
        self.collection.delete(ids=doc_ids)
        print(f"Deleted {len(doc_ids)} documents")
    
    def delete_by_ad_id(self, ad_id: str) -> None:
        """
        Delete all documents associated with a specific AD ID.
        
        Args:
            ad_id: AD ID to delete documents for
        """
        # Query for documents with matching ad_id
        results = self.collection.get(
            where={"ad_id": ad_id}
        )
        
        if results and results['ids']:
            self.collection.delete(ids=results['ids'])
            print(f"Deleted {len(results['ids'])} documents for AD: {ad_id}")
        else:
            print(f"No documents found for AD: {ad_id}")
    
    def query(
        self, 
        query_text: str, 
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        """
        Query the vector store for similar documents.
        
        Args:
            query_text: Query text
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of matching documents with scores
        """
        # Generate query embedding
        query_embedding = self.embedding_service.embed_query(query_text)
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        if results and results['ids'] and results['ids'][0]:
            for i, doc_id in enumerate(results['ids'][0]):
                formatted_results.append({
                    "id": doc_id,
                    "content": results['documents'][0][i] if results['documents'] else "",
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else 0
                })
        
        return formatted_results
    
    def query_by_ad(
        self, 
        query_text: str, 
        ad_id: str,
        n_results: int = 5
    ) -> List[Dict]:
        """
        Query documents from a specific AD.
        
        Args:
            query_text: Query text
            ad_id: AD ID to filter by
            n_results: Number of results
            
        Returns:
            List of matching documents
        """
        return self.query(
            query_text=query_text,
            n_results=n_results,
            filter_metadata={"ad_id": ad_id}
        )
    
    def get_all_documents(self, ad_id: Optional[str] = None) -> List[Dict]:
        """
        Get all documents, optionally filtered by AD ID.
        
        Args:
            ad_id: Optional AD ID filter
            
        Returns:
            List of all documents
        """
        if ad_id:
            results = self.collection.get(
                where={"ad_id": ad_id},
                include=["documents", "metadatas"]
            )
        else:
            results = self.collection.get(include=["documents", "metadatas"])
        
        formatted_results = []
        if results and results['ids']:
            for i, doc_id in enumerate(results['ids']):
                formatted_results.append({
                    "id": doc_id,
                    "content": results['documents'][i] if results['documents'] else "",
                    "metadata": results['metadatas'][i] if results['metadatas'] else {}
                })
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        return {
            "collection_name": self.collection_name,
            "document_count": self.collection.count(),
            "persist_dir": self.persist_dir
        }
    
    def clear_collection(self) -> None:
        """Clear all documents from the collection."""
        # Delete and recreate collection
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Cleared collection: {self.collection_name}")


def get_vector_store() -> VectorStore:
    """Factory function to create vector store."""
    return VectorStore()
