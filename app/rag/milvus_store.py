import re
import logging
from typing import List, Dict, Any, Optional
from pymilvus import MilvusClient
from app.config import settings
from app.reliability.resilient import resilient_call

logger = logging.getLogger(__name__)

def parse_page_number(page_val: Any) -> int:
    """Helper to convert page strings like 'Page 3' or 'Slide 2' into an integer."""
    if isinstance(page_val, int):
        return page_val
    if isinstance(page_val, str):
        match = re.search(r"\d+", page_val)
        if match:
            return int(match.group(0))
    return 1

class MilvusStore:
    """
    Manages vector storage, indexing, and filtered semantic search in Milvus.
    Includes circuit-breaker resilience and document scope filtering.
    """
    def __init__(self, uri: Optional[str] = None, collection_name: Optional[str] = None):
        self.uri = uri or settings.MILVUS_URI
        self.collection_name = collection_name or settings.MILVUS_COLLECTION
        self._client: Optional[MilvusClient] = None

    def get_client(self) -> MilvusClient:
        """Returns or initializes the MilvusClient."""
        if self._client is None:
            logger.info(f"Connecting to Milvus vector store at {self.uri}...")
            self._client = MilvusClient(uri=self.uri)
            self._ensure_collection()
        return self._client

    def _ensure_collection(self) -> None:
        """Verifies or creates the target Milvus collection."""
        try:
            if not self._client.has_collection(self.collection_name):
                logger.info(f"Collection '{self.collection_name}' not found. Creating schema...")
                self._client.create_collection(
                    collection_name=self.collection_name,
                    dimension=settings.EMBEDDING_DIM,
                    metric_type="COSINE",
                    auto_id=True,
                    enable_dynamic_field=True
                )
                logger.info(f"Milvus collection '{self.collection_name}' created successfully.")
            else:
                logger.info(f"Milvus collection '{self.collection_name}' already exists.")
        except Exception as e:
            logger.error(f"Failed to ensure Milvus collection: {e}")
            raise e

    def insert_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """
        Inserts vector chunks into Milvus collection with resilient error handling.
        Uses fields: chunk_id, document_id, document_name, page (int), text, embedding.
        """
        if not chunks:
            return 0

        def _insert():
            client = self.get_client()
            data = []
            for c in chunks:
                raw_page = c.get("page_or_section", c.get("page", 1))
                page_int = parse_page_number(raw_page)
                vec = c.get("embedding", c.get("vector"))

                data.append({
                    "chunk_id": str(c.get("chunk_id", "")),
                    "document_id": str(c.get("document_id", "")),
                    "document_name": str(c.get("document_name", "")),
                    "page": page_int,
                    "text": str(c.get("text", "")),
                    "embedding": vec
                })
            res = client.insert(collection_name=self.collection_name, data=data)
            logger.info(f"Inserted {len(data)} vectors into Milvus collection '{self.collection_name}'.")
            return len(data)

        return resilient_call("milvus", _insert, max_retries=2, backoff_factor=0.5)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        selected_document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic vector search with optional document scope filtering.
        """
        def _search():
            client = self.get_client()
            filter_expr = ""
            if selected_document_ids:
                escaped_ids = [f'"{doc_id}"' for doc_id in selected_document_ids if doc_id]
                if escaped_ids:
                    filter_expr = f"document_id in [{', '.join(escaped_ids)}]"

            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            results = client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                anns_field="embedding",
                limit=top_k,
                filter=filter_expr if filter_expr else "",
                output_fields=["chunk_id", "document_id", "document_name", "page", "text"]
            )

            hits: List[Dict[str, Any]] = []
            if results and len(results) > 0:
                for hit in results[0]:
                    entity = hit.get("entity", {})
                    page_num = entity.get("page", 1)
                    hits.append({
                        "chunk_id": entity.get("chunk_id", ""),
                        "document_id": entity.get("document_id", ""),
                        "document_name": entity.get("document_name", ""),
                        "page_or_section": f"Page {page_num}",
                        "text": entity.get("text", ""),
                        "score": float(hit.get("distance", 0.0))
                    })
            return hits

        def _search_fallback():
            logger.warning("Milvus search fallback triggered: returning empty semantic list.")
            return []

        return resilient_call("milvus", _search, fallback=_search_fallback)

    def delete_document(self, document_id: str) -> None:
        """Deletes all chunks belonging to a given document_id."""
        def _delete():
            client = self.get_client()
            filter_expr = f'document_id == "{document_id}"'
            client.delete(collection_name=self.collection_name, filter=filter_expr)
            logger.info(f"Deleted chunks for document_id '{document_id}' from Milvus.")

        resilient_call("milvus", _delete)

    def count(self) -> int:
        """Returns total vector count in the collection with circuit breaker protection."""
        from app.reliability.circuit_breaker import get_circuit_breaker, CircuitState
        cb = get_circuit_breaker("milvus")
        if cb.state == CircuitState.OPEN:
            return 0

        def _get_count() -> int:
            client = self.get_client()
            stats = client.get_collection_stats(self.collection_name)
            return stats.get("row_count", 0)

        try:
            return resilient_call("milvus", _get_count, max_retries=0)
        except Exception as e:
            logger.warning(f"Could not retrieve Milvus collection count: {e}")
            return 0

# Singleton MilvusStore instance
milvus_store = MilvusStore()
