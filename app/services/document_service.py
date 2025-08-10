import os
import logging
import aiofiles
from typing import List, Dict, Any, Optional
import boto3
import pandas as pd
import json
from datetime import datetime

from app.core.config import settings
from app.services.vector_service import VectorService
from app.utils.file_processor import FileProcessor
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import shutil

logger = logging.getLogger(__name__)

class DocumentService:
    def __init__(self):
        self.vector_service = VectorService()
        self.file_processor = FileProcessor()
        self.upload_dir = settings.upload_dir
        os.makedirs(self.upload_dir, exist_ok=True)
        
        # Initialize S3 client if credentials are available
        self.s3_client = None
        if settings.aws_access_key_id and settings.aws_secret_access_key:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=settings.aws_access_key_id,
                aws_secret_access_key=settings.aws_secret_access_key,
                region_name=settings.aws_region
            )
    
    async def process_upload(
        self,
        files: List = None,
        usecase_id: str = None,
        is_vector: bool = False,
        chunk_size: int = 1000,
        overlap: int = 200,
        s3_config: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Process document upload from files or S3"""
        try:
            logger.info(f"Processing upload for usecase_id: {usecase_id}")
            
            processed_files = []
            total_chunks = 0
            
            if files:
                # Process uploaded files
                for file in files:
                    if file and file.filename:
                        result = await self._process_single_file(
                            file, usecase_id, is_vector, chunk_size, overlap
                        )
                        processed_files.append(result)
                        total_chunks += result.get("chunks_created", 0)
            
            elif s3_config:
                # Process S3 documents
                result = await self._process_s3_documents(
                    s3_config, usecase_id, is_vector, chunk_size, overlap
                )
                processed_files.extend(result.get("processed_files", []))
                total_chunks += result.get("total_chunks", 0)
            
            else:
                raise ValueError("Either files or S3 config must be provided")
            
            return {
                "usecase_id": usecase_id,
                "processed_files": processed_files,
                "total_chunks": total_chunks,
                "vectorization_enabled": is_vector
            }
            
        except Exception as e:
            logger.error(f"Error processing upload: {str(e)}")
            raise
    
    async def parse_evaluation_file(self, evaluation_file) -> List[Dict[str, Any]]:
        """Parse evaluation CSV file and return list of evaluation items"""
        try:
            logger.info(f"Parsing evaluation file: {evaluation_file.filename}")
            
            # Read file content
            content = await evaluation_file.read()
            
            # Save temporarily to parse with pandas
            temp_file_path = os.path.join(self.upload_dir, f"temp_eval_{evaluation_file.filename}")
            
            async with aiofiles.open(temp_file_path, 'wb') as f:
                await f.write(content)
            
            try:
                # Read CSV file
                df = pd.read_csv(temp_file_path)
                
                # Validate required columns
                required_columns = ["query", "expected_answer"]
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    raise ValueError(f"Missing required columns: {missing_columns}. Required: {required_columns}")
                
                # Convert to list of dictionaries
                evaluation_data = []
                for _, row in df.iterrows():
                    item = {
                        "query": str(row["query"]) if pd.notna(row["query"]) else "",
                        "expected_answer": str(row["expected_answer"]) if pd.notna(row["expected_answer"]) else ""
                    }
                    
                    # Skip empty rows
                    if item["query"].strip() and item["expected_answer"].strip():
                        evaluation_data.append(item)
                
                logger.info(f"Parsed {len(evaluation_data)} evaluation items from file")
                return evaluation_data
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    
        except Exception as e:
            logger.error(f"Error parsing evaluation file: {str(e)}")
            raise ValueError(f"Failed to parse evaluation file: {str(e)}")
    
    async def _process_single_file(
        self,
        file,
        usecase_id: str,
        is_vector: bool,
        chunk_size: int,
        overlap: int
    ) -> Dict[str, Any]:
        """Process a single uploaded file"""
        try:
            # Save file
            file_path = os.path.join(self.upload_dir, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Extract text
            text_content = await self.file_processor.extract_text(file_path)
            chunks_created = 0
            if is_vector:
                # Chunk and vectorize
                logger.info(f"Vectorizing...")

                loader = PyPDFLoader(file_path)
                document = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=overlap
                )
                chunked_documents = text_splitter.split_documents(document)
                chunks = []
                for chunk in chunked_documents:
                    chunks.append(chunk.page_content)
                logger.info(f"Vectorizing Completed")

                # Store in vector database
                logger.info(f"Storing to VectorDB...")
                await self.vector_service.store_documents(
                    usecase_id=usecase_id,
                    documents=chunks,
                    metadata={"filename": file.filename, "file_path": file_path}
                )
                logger.info(f"Storing to VectorDB Completed")

                chunks_created += len(chunks)
            else:
                # Store full document
                await self.vector_service.store_documents(
                    usecase_id=usecase_id,
                    documents=[text_content],
                    metadata={"filename": file.filename, "file_path": file_path}
                )
                chunks_created = 1
            return {
                "filename": file.filename,
                "text_length": len(text_content) if text_content else 0,
                "chunks_created": chunks_created,
                "status": "processed"
            }
            
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")
            return {
                "filename": file.filename,
                "error": str(e),
                "status": "failed"
            }
    
    async def _process_s3_documents(
        self,
        s3_config: Dict[str, Any],
        usecase_id: str,
        is_vector: bool,
        chunk_size: int,
        overlap: int
    ) -> Dict[str, Any]:
        """Process documents from S3"""
        try:
            if not self.s3_client:
                raise ValueError("S3 client not configured")
            
            bucket_name = s3_config.get("bucket_name")
            prefix = s3_config.get("prefix", "")
            
            # List objects in S3
            response = self.s3_client.list_objects_v2(
                Bucket=bucket_name,
                Prefix=prefix
            )
            
            processed_files = []
            total_chunks = 0
            
            for obj in response.get("Contents", []):
                key = obj["Key"]
                
                # Skip directories
                if key.endswith("/"):
                    continue
                
                # Download file
                local_path = os.path.join(self.upload_dir, f"{usecase_id}_{os.path.basename(key)}")
                self.s3_client.download_file(bucket_name, key, local_path)
                
                # Extract text
                text_content = await self.file_processor.extract_text(local_path)
                
                chunks_created = 0
                if is_vector and text_content:
                    # Chunk and vectorize
                    loader = PyPDFLoader(local_path)
                    document = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=overlap
                    )
                    chunked_documents = text_splitter.split_documents(document)
                    chunks = []
                    for chunk in chunked_documents:
                        chunks.append(chunk.page_content)
                    logger.info(f"Vectorizing Completed")

                    # Store in vector database
                    logger.info(f"Storing to VectorDB...")
                    await self.vector_service.store_documents(
                        usecase_id=usecase_id,
                        documents=chunks,
                        metadata={"filename": key, "file_path": local_path}
                    )
                    logger.info(f"Storing to VectorDB Completed")
                
                processed_files.append({
                    "s3_key": key,
                    "filename": os.path.basename(key),
                    "text_length": len(text_content) if text_content else 0,
                    "chunks_created": chunks_created,
                    "status": "processed"
                })
                
                total_chunks += chunks_created
                
                # Clean up local file
                os.remove(local_path)
            
            return {
                "processed_files": processed_files,
                "total_chunks": total_chunks
            }
            
        except Exception as e:
            logger.error(f"Error processing S3 documents: {str(e)}")
            raise
    
    async def list_usecases(self) -> List[str]:
        """List all available use cases"""
        try:
            return await self.vector_service.list_collections()
        except Exception as e:
            logger.error(f"Error listing use cases: {str(e)}")
            raise
    
    async def delete_usecase(self, usecase_id: str):
        """Delete a use case and all associated data"""
        try:
            logger.info(f"Deleting use case: {usecase_id}")
            
            # Delete from vector database
            await self.vector_service.delete_collection(usecase_id)
            
            # Clean up uploaded files for this use case
            for filename in os.listdir(self.upload_dir):
                if filename.startswith(f"{usecase_id}_"):
                    file_path = os.path.join(self.upload_dir, filename)
                    os.remove(file_path)
                    logger.info(f"Deleted file: {filename}")
            
            logger.info(f"Use case {usecase_id} deleted successfully")
            
        except Exception as e:
            logger.error(f"Error deleting use case {usecase_id}: {str(e)}")
            raise 