import re
import logging
from typing import List, Dict, Any

from ollama_client import ollama_embed
from qdrant_store import search_qdrant

logger = logging.getLogger("contextlens")

QDRANT_CANDIDATE_MULTIPLIER = 3
MIN_SCORE_THRESHOLD = 0.25
SCORE_DROP_RATIO = 0.6


def normalize_tokens(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9_./-]+", text.lower())
    return [t for t in tokens if len(t) >= 3]


async def retrieve(question: str, top_k: int = 10) -> List[Dict[str, Any]]:
    q_emb = await ollama_embed(question)

    candidates = search_qdrant(q_emb, top_k=top_k * QDRANT_CANDIDATE_MULTIPLIER)

    if not candidates:
        return []

    q_tokens = set(normalize_tokens(question))

    for doc in candidates:
        semantic = doc["score"]

        doc_text_lower = doc["text"].lower()
        lexical_hits = sum(1 for tok in q_tokens if tok in doc_text_lower)
        lexical = lexical_hits / max(1, len(q_tokens))

        doc["score"] = (0.75 * semantic) + (0.25 * lexical)

    candidates.sort(key=lambda x: x["score"], reverse=True)

    best_score = candidates[0]["score"]
    score_floor = max(MIN_SCORE_THRESHOLD, best_score * SCORE_DROP_RATIO)

    filtered = []
    for doc in candidates[:top_k]:
        if doc["score"] < score_floor:
            break
        filtered.append(doc)

    logger.info(
        f"[retrieve] top_k={top_k}, candidates={len(candidates)}, "
        f"best={best_score:.4f}, floor={score_floor:.4f}, returned={len(filtered)}"
    )

    return filtered
