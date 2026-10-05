from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class DocumentMetadata(BaseModel):
    document_id: str
    document_name: str
    chunk_count: int
    upload_date: Optional[str] = None
    file_type: Optional[str] = None

class QuestionRequest(BaseModel):
    query: str = Field(..., description="User question or research prompt")
    selected_document_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of document_ids to filter internal search scope"
    )

class CitationInternal(BaseModel):
    document_name: str
    page_or_section: str
    document_id: str
    snippet: str
    score: Optional[float] = None

class CitationExternal(BaseModel):
    title: str
    url: str
    snippet: str
    score: Optional[float] = None

class QuestionResponse(BaseModel):
    query: str
    final_answer: str
    route_taken: str
    supervisor_reasoning: Optional[str] = None
    internal_sources: List[CitationInternal] = Field(default_factory=list)
    external_sources: List[CitationExternal] = Field(default_factory=list)
    circuit_breaker_status: Dict[str, Any] = Field(default_factory=dict)
    execution_time_seconds: float = 0.0
    guardrails_passed: bool = True
    guardrails_flags: List[str] = Field(default_factory=list)
    cached: bool = Field(default=False, description="Whether the response was served from Cache")

class DocumentUploadResponse(BaseModel):
    document_id: str
    document_name: str
    chunk_count: int
    status: str
    message: str
