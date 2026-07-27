import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

import yaml

import config
from ollama_client import ollama_embed_batch
from qdrant_store import upsert_to_qdrant, delete_repo_from_qdrant


def file_hash(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in config.ALLOWED_EXTS


def read_text_file(path: Path) -> str:
    try:
        if path.stat().st_size > config.MAX_FILE_BYTES:
            return ""
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def chunk_by_lines(text: str, lines_per_chunk: int = 40, overlap: int = 8) -> List[Tuple[int, int, str]]:
    lines = text.splitlines()
    if not lines:
        return []

    chunks = []
    start = 0
    total = len(lines)

    while start < total:
        end = min(total, start + lines_per_chunk)
        chunk_lines = lines[start:end]
        chunk_text = "\n".join(chunk_lines).strip()

        if chunk_text:
            chunks.append((start + 1, end, chunk_text))

        if end >= total:
            break

        start = max(0, end - overlap)

    return chunks


async def index_repo(repo_path: str, repo_name: str = "", tag: str = "") -> Dict[str, Any]:
    root = Path(repo_path).expanduser().resolve()
    if not root.exists():
        return {"ok": False, "message": f"Path does not exist: {root}"}

    if not root.is_dir():
        return {"ok": False, "message": f"Path is not a directory: {root}"}

    root_str = str(root)
    resolved_name = repo_name.strip() or root.name
    resolved_tag = tag.strip().lower() or ""
    config.DOCS[:] = [d for d in config.DOCS if d.get("repo_root") != root_str]
    delete_repo_from_qdrant(root_str)

    file_count = 0
    chunk_count = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if any(part in config.IGNORE_DIRS for part in path.parts):
            continue

        if not is_text_file(path):
            continue

        content = read_text_file(path)
        if not content.strip():
            continue

        file_count += 1
        chunks = chunk_by_lines(content)
        chunk_texts = [ct for _, _, ct in chunks]
        embeddings = await ollama_embed_batch(chunk_texts)
        new_docs = []

        for idx, ((start_line, end_line, chunk_text), emb) in enumerate(zip(chunks, embeddings)):
            doc = {
                "id": f"{file_hash(path)}-{idx}",
                "path": str(path.relative_to(root)),
                "full_path": str(path),
                "repo_root": root_str,
                "repo_name": resolved_name,
                "tag": resolved_tag,
                "start_line": start_line,
                "end_line": end_line,
                "chunk_index": idx,
                "text": chunk_text,
                "embedding": emb,
            }
            config.DOCS.append(doc)
            new_docs.append(doc)
            chunk_count += 1

        upsert_to_qdrant(new_docs)

    return {
        "ok": True,
        "root": str(root),
        "files_indexed": file_count,
        "chunks_indexed": chunk_count
    }


async def index_swagger_text(swagger_text: str, source_name: str = "swagger") -> Dict[str, Any]:
    try:
        parsed = yaml.safe_load(swagger_text)
        pretty_text = json.dumps(parsed, indent=2, ensure_ascii=False)
    except Exception:
        pretty_text = swagger_text

    chunks = chunk_by_lines(pretty_text)
    chunk_texts = [ct for _, _, ct in chunks]
    embeddings = await ollama_embed_batch(chunk_texts)

    new_docs = []
    for idx, ((start_line, end_line, chunk_text), emb) in enumerate(zip(chunks, embeddings)):
        doc = {
            "id": f"{source_name}-{idx}",
            "path": source_name,
            "full_path": source_name,
            "start_line": start_line,
            "end_line": end_line,
            "chunk_index": idx,
            "text": chunk_text,
            "embedding": emb,
        }
        config.DOCS.append(doc)
        new_docs.append(doc)

    upsert_to_qdrant(new_docs)
    return {"ok": True, "source": source_name, "chunks_added": len(new_docs)}
