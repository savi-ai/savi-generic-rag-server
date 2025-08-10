import os
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)

class VectorService:
    def __init__(self):
        self.client = None
        self.embedding_model = None
        self._init_chroma_client()
        self._init_embedding_model()
    
    def _init_chroma_client(self):
        """Initialize ChromaDB client"""
        try:
            os.makedirs(settings.chroma_persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=settings.chroma_persist_directory,
                settings=ChromaSettings(
                    anonymized_telemetry=False
                )
            )
            logger.info("ChromaDB client initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing ChromaDB client: {str(e)}")
            raise
    
    def _init_embedding_model(self):
        """Initialize sentence transformer model"""
        try:
            self.embedding_model = SentenceTransformer(settings.embedding_model)
            logger.info(f"Embedding model {settings.embedding_model} loaded successfully")
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            raise
    
    def _get_collection_name(self, usecase_id: str) -> str:
        """Get collection name for a use case"""
        return f"usecase_{usecase_id}"
    
    async def store_documents(
        self, 
        usecase_id: str, 
        documents: List[str], 
        metadata: Any  # Can be Dict[str, Any] or List[Dict[str, Any]]
    ) -> bool:
        """Store documents in vector database"""
        try:
            collection_name = self._get_collection_name(usecase_id)
            
            # Get or create collection
            try:
                collection = self.client.get_collection(name=collection_name)
            except:
                collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"usecase_id": usecase_id}
                )
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(documents).tolist()
            
            # Prepare documents for storage
            ids = [f"{usecase_id}_{i}" for i in range(len(documents))]
            
            # Handle metadata - can be single dict or list of dicts
            if isinstance(metadata, list):
                # If metadata is a list, it should match the number of documents
                if len(metadata) != len(documents):
                    raise ValueError(f"Metadata list length ({len(metadata)}) must match documents length ({len(documents)})")
                metadatas = metadata
            else:
                # If metadata is a single dict, use it for all documents
                metadatas = [metadata for _ in documents]
            
            # Add to collection
            collection.add(
                documents=documents,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )
            
            logger.info(f"Stored {len(documents)} documents for usecase_id: {usecase_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing documents: {str(e)}")
            raise
    
    async def similarity_search(
        self, 
        usecase_id: str, 
        query: str, 
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Perform similarity search"""
        try:
            collection_name = self._get_collection_name(usecase_id)
            
            # Get collection
            try:
                collection = self.client.get_collection(name=collection_name)
            except:
                logger.warning(f"No collection found for usecase_id: {usecase_id}")
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query]).tolist()
            
            # Search
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            search_results = []
            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    search_results.append({
                        "content": doc,
                        "metadata": metadata,
                        "score": 1 - distance,  # Convert distance to similarity score
                        "rank": i + 1
                    })
            
            logger.info(f"Found {len(search_results)} results for query in usecase_id: {usecase_id}")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            raise
    
    async def list_collections(self) -> List[str]:
        """List all available collections (use cases)"""
        try:
            collections = self.client.list_collections()
            usecases = []
            
            for collection in collections:
                if collection.name.startswith("usecase_"):
                    usecase_id = collection.name.replace("usecase_", "")
                    usecases.append(usecase_id)
            
            return usecases
            
        except Exception as e:
            logger.error(f"Error listing collections: {str(e)}")
            raise
    
    async def delete_collection(self, usecase_id: str) -> bool:
        """Delete a collection for a use case"""
        try:
            collection_name = self._get_collection_name(usecase_id)
            
            try:
                self.client.delete_collection(name=collection_name)
                logger.info(f"Deleted collection for usecase_id: {usecase_id}")
                return True
            except:
                logger.warning(f"Collection not found for usecase_id: {usecase_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error deleting collection {usecase_id}: {str(e)}")
            raise

    async def get_usecases(self) -> List[str]:
        """Get list of available use cases (alias for list_collections)"""
        return await self.list_collections()
    
    async def delete_usecase(self, usecase_id: str) -> bool:
        """Delete a use case and all associated data (alias for delete_collection)"""
        return await self.delete_collection(usecase_id)
    
    async def get_collection_stats(self, usecase_id: str) -> Dict[str, Any]:
        """Get statistics for a use case collection"""
        try:
            collection_name = self._get_collection_name(usecase_id)
            
            try:
                collection = self.client.get_collection(name=collection_name)
                count = collection.count()
                
                return {
                    "usecase_id": usecase_id,
                    "document_count": count,
                    "collection_name": collection_name
                }
            except:
                return {
                    "usecase_id": usecase_id,
                    "document_count": 0,
                    "collection_name": collection_name,
                    "error": "Collection not found"
                }
                
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            raise 