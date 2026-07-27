import hashlib
import logging
from typing import List, Dict, Any

from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

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


def search_qdrant(query_vector: List[float], top_k: int = 12) -> List[Dict[str, Any]]:
    client = get_qdrant()
    ensure_collection()
    response = client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )
    docs = []
    for point in response.points:
        doc = dict(point.payload)
        doc["score"] = point.score
        docs.append(doc)
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
