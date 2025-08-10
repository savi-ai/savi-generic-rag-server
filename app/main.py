from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional, Dict, Any
import json
import logging
from datetime import datetime
import os

from app.core.config import settings
from app.services.document_service import DocumentService
from app.services.vector_service import VectorService
from app.services.llm_service import LLMService
from app.services.agentic_service import AgenticService
from app.services.evaluation_service import EvaluationService
from app.services.guardrail_service import GuardrailService
from app.services.task_manager import task_manager
from app.models.schemas import (
    UploadRequest,
    TestRequest,
    EvaluationRequest,
    ResponseModel,
    ErrorResponse
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    debug=settings.debug
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
document_service = DocumentService()
vector_service = VectorService()
llm_service = LLMService()
agentic_service = AgenticService()
evaluation_service = EvaluationService()
guardrail_service = GuardrailService()

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return ResponseModel(
        success=True,
        message="Savi RAG Backend is running",
        data={"version": settings.api_version, "status": "healthy"},
        timestamp=datetime.now().isoformat()
    )

@app.post("/savi-rag-api/api/upload")
async def upload_documents(
    files: List[UploadFile] = File(None),
    usecase_id: str = Form(...),
    llm_parameters: str = Form(...),
    s3_config: Optional[str] = Form(None)
):
    """Upload and process documents for RAG"""
    try:
        llm_params = json.loads(llm_parameters)
        is_vector = llm_params.get("isVector", False)
        chunk_size = llm_params.get("chunkSize", settings.default_chunk_size)
        overlap = llm_params.get("overlap", settings.default_chunk_overlap)

        logger.info(f"Upload request for usecase_id: {usecase_id}")
        
        # Parse S3 config if provided
        s3_config_data = None
        if s3_config:
            try:
                s3_config_data = json.loads(s3_config)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid S3 config JSON")
        
        # Process upload
        result = await document_service.process_upload(
            files=files,
            usecase_id=usecase_id,
            is_vector=is_vector,
            chunk_size=chunk_size,
            overlap=overlap,
            s3_config=s3_config_data
        )
        
        return ResponseModel(
            success=True,
            message="Documents uploaded and processed successfully",
            data=result,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/savi-rag-api/api/test")
async def test_rag(
    usecase_id: str = Form(...),
    query: str = Form(...),
    system_prompt: Optional[str] = Form(""),
    llm_parameters: Optional[str] = Form("{}"),
    agentic_config: Optional[str] = Form("{}"),
    guardrail_config: Optional[str] = Form("{}"),
    tools: Optional[str] = Form("[]")  # For backward compatibility
):
    """Test RAG query with optional agentic mode and guardrails"""
    try:
        logger.info(f"Test request for usecase_id: {usecase_id}, query: {query[:100]}...")
        
        # Parse JSON parameters
        try:
            llm_params = json.loads(llm_parameters) if llm_parameters else {}
            agentic_params = json.loads(agentic_config) if agentic_config else {}
            guardrail_params = json.loads(guardrail_config) if guardrail_config else {}
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in parameters: {str(e)}")
        
        # Convert old UI parameters to new format
        if "enabled" in agentic_params:
            # Convert old format to new format
            agentic_params = {
                "isAgentic": agentic_params.get("enabled", False),
                "selectedAgenticType": agentic_params.get("type", "self_critic"),
                "useApiCall": agentic_params.get("useApiCall", False),
                "apiConfig": agentic_params.get("apiConfig", {})
            }
        
        # Convert old UI parameter names to new format
        if "topK" in llm_params:
            llm_params["top_k"] = llm_params.pop("topK")
        if "maxTokens" in llm_params:
            llm_params["max_tokens"] = llm_params.pop("maxTokens")
        if "useSelfCritic" in llm_params:
            # This was handled in agentic_params, remove from llm_params
            llm_params.pop("useSelfCritic", None)

        # Apply question guardrail if configured
        if guardrail_params.get("useGuardrail", False) and guardrail_params.get("questionGuardrails", "") != "":
            is_allowed, reason = await guardrail_service.apply_question_guardrail(
                query, guardrail_params, guardrail_params
            )
            if not is_allowed:
                return ResponseModel(
                    success=False,
                    message="Query blocked by guardrails",
                    data={"response": "Sorry could not generate response as guardrails blocked the question"},
                    timestamp=datetime.now().isoformat()
                )
        
        # Perform similarity search
        search_results = await vector_service.similarity_search(
            usecase_id=usecase_id,
            query=query,
            top_k=llm_params.get("top_k", 5)
        )
        
        # Generate response
        if agentic_params.get("enabled", False):
            # Use agentic mode
            response_data = await agentic_service.generate_response(
                query=query,
                context=search_results,
                system_prompt=system_prompt,
                llm_parameters=llm_params,
                agentic_config=agentic_params,
                usecase_id=usecase_id
            )
            response_text = response_data["response"]
        else:
            # Use standard LLM mode
            response_text = await llm_service.generate_response(
                query=query,
                context=search_results,
                system_prompt=system_prompt,
                llm_parameters=llm_params
            )
        
        # Apply answer guardrail if configured
        if guardrail_params.get("useGuardrails", False) and guardrail_params.get("answerGuardrails", "") != "":
            is_allowed, reason = await guardrail_service.apply_answer_guardrail(
                response_text, guardrail_params, guardrail_params
            )
            if not is_allowed:
                return ResponseModel(
                    success=False,
                    message="Response blocked by guardrails",
                    data={"response": "Sorry could not generate response as guardrails blocked the response"},
                    timestamp=datetime.now().isoformat()
                )

        return ResponseModel(
            success=True,
            message="Query processed successfully",
            data={
                "query": query,
                "response": response_text,
                "search_results_count": len(search_results),
                "usecase_id": usecase_id
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Test error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/savi-rag-api/api/evaluate")
async def start_evaluation(request: Request):
    """Start asynchronous evaluation and return task ID"""
    try:
        # Parse the form data manually
        form_data = await request.form()
        
        # Extract required fields
        usecase_id = form_data.get("usecase_id")
        if not usecase_id:
            raise HTTPException(status_code=400, detail="usecase_id is required")
        
        logger.info(f"Evaluation request for usecase_id: {usecase_id}")
        
        # Extract optional fields
        system_prompt = form_data.get("system_prompt", "")
        llm_parameters = form_data.get("llm_parameters", "{}")
        agentic_config = form_data.get("agentic_config", "{}")
        guardrail_config = form_data.get("guardrail_config", "{}")
        evaluation_data = form_data.get("evaluation_data")
        evaluation_file = form_data.get("evaluation_file")
        
        # Parse JSON parameters
        try:
            llm_params = json.loads(llm_parameters) if llm_parameters else {}
            agentic_params = json.loads(agentic_config) if agentic_config else {}
            guardrail_params = json.loads(guardrail_config) if guardrail_config else {}
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in parameters: {str(e)}")
        
        # Parse evaluation data - support both file upload and JSON data
        evaluation_items = []
        if evaluation_file and hasattr(evaluation_file, 'filename') and evaluation_file.filename:
            # New format: file upload
            evaluation_items = await document_service.parse_evaluation_file(evaluation_file)
        elif evaluation_data:
            # Old format: JSON string in form data
            try:
                evaluation_items = json.loads(evaluation_data)
                # Validate structure
                if not isinstance(evaluation_items, list):
                    raise ValueError("evaluation_data must be a list")
                
                # Ensure each item has required fields and convert old format if needed
                normalized_items = []
                for item in evaluation_items:
                    if isinstance(item, dict):
                        normalized_item = {
                            "query": item.get("query", ""),
                            "expected_answer": item.get("answer", item.get("expected_answer", ""))
                        }
                        if normalized_item["query"] and normalized_item["expected_answer"]:
                            normalized_items.append(normalized_item)
                
                evaluation_items = normalized_items
                
            except (json.JSONDecodeError, ValueError) as e:
                raise HTTPException(status_code=400, detail=f"Invalid evaluation_data format: {str(e)}")
        else:
            raise HTTPException(status_code=400, detail="Either evaluation_file or evaluation_data must be provided")
        
        if not evaluation_items:
            raise HTTPException(status_code=400, detail="No valid evaluation items found")
        
        # Convert old UI parameters to new format
        if "enabled" in agentic_params:
            # Convert old format to new format
            agentic_params = {
                "isAgentic": agentic_params.get("enabled", False),
                "selectedAgenticType": agentic_params.get("type", "self_critic"),
                "useApiCall": agentic_params.get("useApiCall", False),
                "apiConfig": agentic_params.get("apiConfig", {})
            }
        
        # Convert old UI parameter names to new format
        if "topK" in llm_params:
            llm_params["top_k"] = llm_params.pop("topK")
        if "maxTokens" in llm_params:
            llm_params["max_tokens"] = llm_params.pop("maxTokens")
        
        # Start async evaluation
        task_id = evaluation_service.start_async_evaluation(
            usecase_id=usecase_id,
            evaluation_data=evaluation_items,
            llm_parameters=llm_params,
            system_prompt=system_prompt,
            agentic_config=agentic_params,
            guardrail_config=guardrail_params
        )
        
        return ResponseModel(
            success=True,
            message="Evaluation started successfully",
            data={
                "task_id": task_id,
                "usecase_id": usecase_id,
                "total_queries": len(evaluation_items),
                "status_url": f"/savi-rag-api/api/evaluation/status/{task_id}",
                "download_url": f"/savi-rag-api/api/evaluation/download/{task_id}"
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/savi-rag-api/api/evaluation/status/{task_id}")
async def get_evaluation_status(task_id: str):
    """Get evaluation task status"""
    try:
        task_status = task_manager.get_task_status(task_id)
        
        if not task_status:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return ResponseModel(
            success=True,
            message="Task status retrieved successfully",
            data=task_status,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/savi-rag-api/api/evaluation/download/{task_id}")
async def get_evaluation_files(task_id: str):
    """Get list of downloadable files for an evaluation task"""
    try:
        result_files = task_manager.get_task_result_files(task_id)
        
        if not result_files:
            raise HTTPException(status_code=404, detail="No files found for this task")
        
        return ResponseModel(
            success=True,
            message="Result files retrieved successfully",
            data={
                "task_id": task_id,
                "files": [
                    {
                        "type": file["type"],
                        "download_url": f"/savi-rag-api/api/evaluation/download/{task_id}/{file['type']}",
                        "created_at": file["created_at"]
                    }
                    for file in result_files
                ]
            },
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File list error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/savi-rag-api/api/evaluation/download/{task_id}/{file_type}")
async def download_evaluation_file(task_id: str, file_type: str):
    """Download specific evaluation result file"""
    try:
        result_files = task_manager.get_task_result_files(task_id)
        
        # Find the requested file
        target_file = None
        for file in result_files:
            if file["type"] == file_type:
                target_file = file
                break
        
        if not target_file:
            raise HTTPException(status_code=404, detail=f"File type '{file_type}' not found for this task")
        
        file_path = target_file["path"]
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found on disk")
        
        # Determine media type
        media_type = "text/csv" if file_type == "csv" else "text/html"
        filename = f"evaluation_{task_id}.{file_type}"
        
        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File download error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/savi-rag-api/api/usecases")
async def list_usecases():
    """List all available use cases"""
    try:
        usecases = await document_service.list_usecases()
        
        return ResponseModel(
            success=True,
            message="Use cases retrieved successfully",
            data={"usecases": usecases},
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"List usecases error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/savi-rag-api/api/usecases/{usecase_id}")
async def delete_usecase(usecase_id: str):
    """Delete a use case and all associated data"""
    try:
        await document_service.delete_usecase(usecase_id)
        
        return ResponseModel(
            success=True,
            message=f"Use case '{usecase_id}' deleted successfully",
            data={"deleted_usecase_id": usecase_id},
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Delete usecase error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 