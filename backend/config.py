import os
from typing import List, Dict, Any

import httpx
from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.1:latest")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "contextlens")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))  # nomic-embed-text default

qdrant: QdrantClient = None
http_client: httpx.AsyncClient = None

# In-memory index (populated from Qdrant on startup)
DOCS: List[Dict[str, Any]] = []

IGNORE_DIRS = {
    ".git", ".idea", ".vscode", "__pycache__", ".venv", "venv",
    "node_modules", "dist", "build", "target", ".next", ".mypy_cache",
    ".pytest_cache", ".cache"
}

ALLOWED_EXTS = {
    ".py", ".java", ".kt", ".js", ".ts", ".tsx", ".jsx",
    ".json", ".yml", ".yaml", ".md", ".txt", ".xml", ".html",
    ".sql", ".sh", ".properties", ".ini", ".cfg"
}

MAX_FILE_BYTES = 600_000
