import logging
from typing import Dict, Any, List, Optional

from app.tasks.celery_app import celery_app
from app.graph.workflow import run_research_workflow

logger = logging.getLogger(__name__)

@celery_app.task(name="execute_research_workflow_task", bind=True)
def execute_research_workflow_task(
    self,
    query: str,
    selected_document_ids: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Background Worker Task:
    Executes the multi-agent research workflow (Supervisor -> RAG / Web -> Synthesis).
    """
    task_id = getattr(self.request, "id", "local")
    logger.info(f"Background Worker executing task [{task_id}] for query: '{query[:40]}'")
    output = run_research_workflow(
        query=query,
        selected_document_ids=selected_document_ids
    )
    logger.info(f"Background Worker completed task [{task_id}].")
    return output
