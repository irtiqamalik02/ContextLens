import hashlib
import logging
from typing import List, Dict, Any

from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, MatchAny

import config

logger = logging.getLogger("contextlens")


def get_qdrant():
    import config as _cfg
    if _cfg.qdrant is None:
        from qdrant_client import QdrantClient
        _cfg.qdrant = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
    return _cfg.qdrant


def ensure_collection():
    client = get_qdrant()
    existing = [c.name for c in client.get_collections().collections]
    if config.QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION,
            vectors_config=VectorParams(size=config.EMBED_DIM, distance=Distance.COSINE),
        )


def load_docs_from_qdrant():
    client = get_qdrant()
    try:
        ensure_collection()
        scroll_result, _ = client.scroll(
            collection_name=config.QDRANT_COLLECTION,
            with_vectors=True,
            with_payload=True,
            limit=100_000,
        )
        config.DOCS.clear()
        for point in scroll_result:
            payload = point.payload
            payload["embedding"] = point.vector
            config.DOCS.append(payload)
    except Exception as e:
        logger.error(f"[qdrant] Failed to load docs on startup: {e}")


def deterministic_id(doc_id: str) -> int:
    return int(hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:15], 16)


def upsert_to_qdrant(docs: List[Dict[str, Any]]):
    if not docs:
        return
    client = get_qdrant()
    ensure_collection()
    points = [
        PointStruct(
            id=deterministic_id(doc["id"]),
            vector=doc["embedding"],
            payload={k: v for k, v in doc.items() if k != "embedding"},
        )
        for doc in docs
    ]
    client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)


def search_qdrant(
    query_vector: List[float], 
    top_k: int = 12,
    repo_names: List[str] = None,
    tags: List[str] = None
) -> List[Dict[str, Any]]:
    """
    Search Qdrant with optional filtering by repo_names and tags.
    
    Args:
        query_vector: The embedding vector to search with
        top_k: Number of results to return
        repo_names: Optional list of repo names to filter by (OR condition)
        tags: Optional list of tags to filter by (OR condition)
    
    Returns:
        List of documents with scores
    """
    client = get_qdrant()
    ensure_collection()
    
    # Build filter conditions
    query_filter = None
    filter_conditions = []
    
    if repo_names:
        # Filter by repo_name using OR (should) condition
        filter_conditions.append(
            FieldCondition(key="repo_name", match=MatchAny(any=repo_names))
        )
    
    if tags:
        # Filter by tag using OR (should) condition
        filter_conditions.append(
            FieldCondition(key="tag", match=MatchAny(any=tags))
        )
    
    # If we have filter conditions, combine them with AND logic
    if filter_conditions:
        query_filter = Filter(must=filter_conditions)
        logger.info(f"[search_qdrant] Filtering by repo_names={repo_names}, tags={tags}")
    
    response = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
        query_filter=query_filter,
        with_payload=True,
    )
    docs = []
    for point in response.points:
        doc = dict(point.payload)
        doc["score"] = point.score
        docs.append(doc)
    
    logger.info(f"[search_qdrant] Retrieved {len(docs)} results (top_k={top_k})")
    return docs


def clear_all_from_qdrant():
    client = get_qdrant()
    existing = [c.name for c in client.get_collections().collections]
    if config.QDRANT_COLLECTION in existing:
        client.delete_collection(collection_name=config.QDRANT_COLLECTION)
    ensure_collection()
    logger.info("[qdrant] Collection cleared and recreated")


def delete_repo_from_qdrant(repo_root: str):
    client = get_qdrant()
    ensure_collection()
    client.delete(
        collection_name=config.QDRANT_COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="repo_root", match=MatchValue(value=repo_root))]
        ),
    )


def delete_swagger_from_qdrant(source_name: str):
    client = get_qdrant()
    ensure_collection()
    client.delete(
        collection_name=config.QDRANT_COLLECTION,
        points_selector=Filter(
            must=[FieldCondition(key="swagger_source", match=MatchValue(value=source_name))]
        ),
    )
