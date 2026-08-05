import asyncio
import logging
from typing import List, Dict, Any

import config

logger = logging.getLogger("contextlens")

MAX_EMBED_CHARS = 28000


async def ollama_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{config.OLLAMA_BASE_URL}{path}"
    resp = await config.http_client.post(url, json=payload)
    resp.raise_for_status()
    return resp.json()


async def ollama_embed(text: str) -> List[float]:
    truncated = text[:MAX_EMBED_CHARS] if len(text) > MAX_EMBED_CHARS else text
    data = await ollama_post("/api/embeddings", {
        "model": config.EMBED_MODEL,
        "prompt": truncated
    })
    return data["embedding"]


async def ollama_embed_batch(texts: List[str], concurrency: int = 10) -> List[List[float]]:
    semaphore = asyncio.Semaphore(concurrency)

    async def _embed(text: str) -> List[float]:
        async with semaphore:
            try:
                return await ollama_embed(text)
            except Exception as e:
                logger.warning(f"[embed] Failed to embed chunk ({len(text)} chars): {e}")
                return [0.0] * config.EMBED_DIM

    return await asyncio.gather(*[_embed(t) for t in texts])


async def ollama_chat(messages: List[Dict[str, str]]) -> str:
    data = await ollama_post("/api/chat", {
        "model": config.CHAT_MODEL,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_ctx": 16384,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
        },
    })
    return data["message"]["content"]
