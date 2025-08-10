from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class ResponseModel(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: str

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    details: Optional[str] = None

class LLMParameters(BaseModel):
    is_vector: bool = False
    chunk_size: Optional[int] = 1000
    overlap: Optional[int] = 200
    top_k: Optional[int] = 5
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000
    use_self_critic: Optional[bool] = False
    provider: Optional[str] = "ollama"  # ollama, openai, anthropic
    model: Optional[str] = None  # Will use default model for the provider

class S3Config(BaseModel):
    bucket_name: str
    region: str = "us-east-1"
    prefix: Optional[str] = ""

class AgenticConfig(BaseModel):
    enabled: bool = False
    type: Optional[str] = None  # self_critic, react
    use_api_call: bool = False
    api_config: Optional[Dict[str, Any]] = None
    use_guardrail: bool = False
    guardrail_rules: Optional[str] = None

class UploadRequest(BaseModel):
    usecase_id: str
    llm_parameters: LLMParameters
    s3_config: Optional[S3Config] = None

class TestRequest(BaseModel):
    usecase_id: str
    query: str
    llm_parameters: LLMParameters
    system_prompt: str = ""
    tools: List[str] = []
    agentic_config: AgenticConfig = AgenticConfig()

class EvaluationData(BaseModel):
    query: str
    answer: str

class EvaluationRequest(BaseModel):
    usecase_id: str
    evaluation_data: List[EvaluationData]
    llm_parameters: LLMParameters
    system_prompt: str = ""
    tools: List[str] = []
    agentic_config: AgenticConfig = AgenticConfig()

class SearchResult(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float

class LLMResponse(BaseModel):
    response: str
    search_results: List[SearchResult]
    query: str
    usecase_id: str
    processing_time: float

class EvaluationResult(BaseModel):
    query: str
    expected_answer: str
    generated_answer: str
    similarity_score: float
    processing_time: float

class EvaluationSummary(BaseModel):
    total_queries: int
    average_similarity_score: float
    processing_time: float
    results: List[EvaluationResult] 