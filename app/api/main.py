import time
import shutil
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, Query, status

from app.config import settings
from app.agents.models import (
    QuestionRequest,
    QuestionResponse,
    DocumentUploadResponse,
    DocumentMetadata,
    CitationInternal,
    CitationExternal
)
from app.guardrails.input_guard import validate_input_query
from app.guardrails.output_guard import validate_output_response
from app.graph.workflow import run_research_workflow
from app.rag.ingestion import (
    ingest_document,
    load_chunks_index,
    remove_document_from_index,
    SUPPORTED_EXTENSIONS
)
from app.rag.milvus_store import milvus_store
from app.reliability.circuit_breaker import (
    get_all_circuit_statuses,
    get_circuit_breaker,
    CircuitState
)
from app.services.pipeline_service import execute_scalable_research_pipeline
from app.cache.redis_cache import cache_service
from app.concurrency.limiter import concurrency_controller

logger = logging.getLogger(__name__)

router = APIRouter(prefix="", tags=["Research & Knowledge Assistant"])

@router.get("/health", summary="System & Reliability Health Check")
def health_check() -> Dict[str, Any]:
    """
    Returns service health status, Milvus vector store connectivity,
    Redis cache status, active concurrency, indexed chunk count, and live circuit breaker states.
    """
    milvus_cb = get_circuit_breaker("milvus")
    count = 0
    if milvus_cb.state != CircuitState.OPEN:
        try:
            count = milvus_store.count()
        except Exception:
            count = 0

    milvus_ok = (milvus_cb.state == CircuitState.CLOSED and count >= 0 and milvus_store._client is not None)

    chunks = load_chunks_index()
    unique_docs = len(set(c.get("document_id") for c in chunks if c.get("document_id")))

    return {
        "status": "healthy",
        "milvus_connected": milvus_ok,
        "redis_connected": cache_service.is_connected(),
        "active_concurrency": concurrency_controller.active_requests,
        "total_vector_records": count,
        "total_bm25_chunks": len(chunks),
        "total_documents": unique_docs,
        "circuit_breakers": get_all_circuit_statuses()
    }

@router.post(
    "/documents/upload",
    response_model=List[DocumentUploadResponse],
    summary="Upload & Index Internal Documents"
)
async def upload_documents(files: List[UploadFile] = File(...)) -> List[DocumentUploadResponse]:
    """
    Uploads one or more documents (PDF, DOCX, TXT, CSV, XLSX, PPTX, Images),
    extracts text with OCR fallback, chunks, generates embeddings, stores in Milvus,
    and appends to the BM25 chunk index.
    """
    responses: List[DocumentUploadResponse] = []

    for file in files:
        filename = file.filename or "uploaded_doc"
        file_ext = Path(filename).suffix.lower()

        if file_ext not in SUPPORTED_EXTENSIONS:
            responses.append(DocumentUploadResponse(
                document_id="",
                document_name=filename,
                chunk_count=0,
                status="rejected",
                message=f"Unsupported file type '{file_ext}'. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            ))
            continue

        doc_id = str(uuid.uuid4())[:8]
        safe_path = settings.DOCUMENTS_DIR / f"{doc_id}_{filename}"

        try:
            # Save file to disk safely
            with open(safe_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Ingest and index
            ingest_result = ingest_document(
                file_path=safe_path,
                original_filename=filename,
                document_id=doc_id
            )

            responses.append(DocumentUploadResponse(
                document_id=doc_id,
                document_name=filename,
                chunk_count=ingest_result.get("chunk_count", 0),
                status=ingest_result.get("status", "success"),
                message=ingest_result.get("message", "Ingestion completed.")
            ))
        except Exception as e:
            logger.error(f"Failed to process upload '{filename}': {e}", exc_info=True)
            responses.append(DocumentUploadResponse(
                document_id=doc_id,
                document_name=filename,
                chunk_count=0,
                status="error",
                message=f"Failed to ingest document: {str(e)}"
            ))

    return responses

@router.get(
    "/documents",
    response_model=List[DocumentMetadata],
    summary="List All Indexed Documents"
)
def list_documents() -> List[DocumentMetadata]:
    """
    Retrieves metadata of all currently indexed documents.
    """
    chunks = load_chunks_index()
    doc_map: Dict[str, Dict[str, Any]] = {}

    for c in chunks:
        doc_id = c.get("document_id")
        if not doc_id:
            continue
        if doc_id not in doc_map:
            doc_name = c.get("document_name", "Untitled")
            ext = Path(doc_name).suffix.lower()
            doc_map[doc_id] = {
                "document_id": doc_id,
                "document_name": doc_name,
                "chunk_count": 0,
                "file_type": ext,
                "upload_date": "Indexed"
            }
        doc_map[doc_id]["chunk_count"] += 1

    return [DocumentMetadata(**v) for v in doc_map.values()]

@router.delete("/documents/{document_id}", summary="Delete Document from System")
def delete_document(document_id: str) -> Dict[str, Any]:
    """
    Removes a document from both the Milvus vector store and the BM25 chunk index.
    """
    try:
        milvus_store.delete_document(document_id)
        remove_document_from_index(document_id)
        return {"status": "success", "message": f"Document '{document_id}' successfully removed."}
    except Exception as e:
        logger.error(f"Error deleting document {document_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/ask", response_model=QuestionResponse, summary="Ask Research Assistant")
def ask_question(request: QuestionRequest) -> QuestionResponse:
    """
    Orchestrates the complete research workflow:
    1. Input Guardrails validation
    2. LangGraph Multi-Agent reasoning (Supervisor -> RAG / Research -> Synthesis)
    3. Output Guardrails verification
    4. Formatted response with traceable citations and reliability status
    """
    start_time = time.time()

    # 1. Input Guardrail
    input_validation = validate_input_query(request.query)
    if not input_validation.is_valid:
        elapsed = time.time() - start_time
        return QuestionResponse(
            query=request.query,
            final_answer=f"⚠️ Guardrail Rejection: {input_validation.error_message}",
            route_taken="blocked_by_guardrails",
            supervisor_reasoning="Blocked at input validation stage.",
            internal_sources=[],
            external_sources=[],
            circuit_breaker_status=get_all_circuit_statuses(),
            execution_time_seconds=round(elapsed, 3),
            guardrails_passed=False,
            guardrails_flags=input_validation.flags
        )

    sanitized_query = input_validation.sanitized_query

    # 2. Scalable Execution: Concurrency Control -> Cache Check -> Worker Queue -> Batch Async I/O -> Save Cache
    is_cached = False
    try:
        workflow_output, is_cached = execute_scalable_research_pipeline(
            query=sanitized_query,
            selected_document_ids=request.selected_document_ids
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Workflow execution failure: {e}", exc_info=True)
        elapsed = time.time() - start_time
        return QuestionResponse(
            query=sanitized_query,
            final_answer=(
                "An unexpected internal error occurred while processing your query. "
                "Please check system connectivity or retry shortly."
            ),
            route_taken="error",
            supervisor_reasoning="Workflow encountered an unhandled exception.",
            internal_sources=[],
            external_sources=[],
            circuit_breaker_status=get_all_circuit_statuses(),
            execution_time_seconds=round(elapsed, 3),
            guardrails_passed=True,
            guardrails_flags=["WORKFLOW_EXCEPTION"],
            cached=False
        )

    raw_answer = workflow_output.get("final_answer", "")
    route = workflow_output.get("route", "rag")
    reasoning = workflow_output.get("supervisor_reasoning", "")
    sources = workflow_output.get("sources", {})
    raw_internal = sources.get("internal", [])
    raw_external = sources.get("external", [])

    # 3. Output Guardrail
    output_validation = validate_output_response(
        response=raw_answer,
        internal_sources_count=len(raw_internal),
        external_sources_count=len(raw_external),
        route=route
    )

    elapsed = time.time() - start_time

    # Format structured citations
    internal_citations = [
        CitationInternal(
            document_name=item.get("document_name", ""),
            page_or_section=item.get("page_or_section", ""),
            document_id=item.get("document_id", ""),
            snippet=item.get("snippet", ""),
            score=item.get("score")
        ) for item in raw_internal
    ]

    external_citations = [
        CitationExternal(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("snippet", ""),
            score=item.get("score")
        ) for item in raw_external
    ]

    combined_flags = list(set(input_validation.flags + output_validation.flags))

    return QuestionResponse(
        query=sanitized_query,
        final_answer=output_validation.sanitized_response,
        route_taken=route,
        supervisor_reasoning=reasoning,
        internal_sources=internal_citations,
        external_sources=external_citations,
        circuit_breaker_status=get_all_circuit_statuses(),
        execution_time_seconds=round(elapsed, 3),
        guardrails_passed=output_validation.is_valid,
        guardrails_flags=combined_flags,
        cached=is_cached
    )
