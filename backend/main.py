import os
import re
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple

import httpx
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse

load_dotenv()

app = FastAPI(title="ContextLens")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.1:latest")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text:latest")

# In-memory index for MVP
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


class IndexRepoRequest(BaseModel):
    path: str


class ChatRequest(BaseModel):
    role: str
    question: str


def ollama_post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{OLLAMA_BASE_URL}{path}"
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()


def ollama_embed(text: str) -> List[float]:
    data = ollama_post("/api/embeddings", {
        "model": EMBED_MODEL,
        "prompt": text
    })
    return data["embedding"]


def ollama_chat(messages: List[Dict[str, str]]) -> str:
    data = ollama_post("/api/chat", {
        "model": CHAT_MODEL,
        "messages": messages,
        "stream": False,
    })
    return data["message"]["content"]


def file_hash(path: Path) -> str:
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]


def is_text_file(path: Path) -> bool:
    return path.suffix.lower() in ALLOWED_EXTS


def read_text_file(path: Path) -> str:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
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


def normalize_tokens(text: str) -> List[str]:
    tokens = re.findall(r"[A-Za-z0-9_./-]+", text.lower())
    return [t for t in tokens if len(t) >= 3]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def index_repo(repo_path: str) -> Dict[str, Any]:
    global DOCS
    DOCS = []

    root = Path(repo_path).expanduser().resolve()
    if not root.exists():
        return {"ok": False, "message": f"Path does not exist: {root}"}

    if not root.is_dir():
        return {"ok": False, "message": f"Path is not a directory: {root}"}

    file_count = 0
    chunk_count = 0

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if any(part in IGNORE_DIRS for part in path.parts):
            continue

        if not is_text_file(path):
            continue

        content = read_text_file(path)
        if not content.strip():
            continue

        file_count += 1
        chunks = chunk_by_lines(content)

        for idx, (start_line, end_line, chunk_text) in enumerate(chunks):
            emb = ollama_embed(chunk_text)

            DOCS.append({
                "id": f"{file_hash(path)}-{idx}",
                "path": str(path.relative_to(root)),
                "full_path": str(path),
                "start_line": start_line,
                "end_line": end_line,
                "chunk_index": idx,
                "text": chunk_text,
                "embedding": emb,
            })
            chunk_count += 1

    return {
        "ok": True,
        "root": str(root),
        "files_indexed": file_count,
        "chunks_indexed": chunk_count
    }


def index_swagger_text(swagger_text: str, source_name: str = "swagger") -> Dict[str, Any]:
    global DOCS

    try:
        parsed = yaml.safe_load(swagger_text)
        pretty_text = json.dumps(parsed, indent=2, ensure_ascii=False)
    except Exception:
        pretty_text = swagger_text

    chunks = chunk_by_lines(pretty_text)
    added = 0

    for idx, (start_line, end_line, chunk_text) in enumerate(chunks):
        emb = ollama_embed(chunk_text)
        DOCS.append({
            "id": f"{source_name}-{idx}",
            "path": source_name,
            "full_path": source_name,
            "start_line": start_line,
            "end_line": end_line,
            "chunk_index": idx,
            "text": chunk_text,
            "embedding": emb,
        })
        added += 1

    return {"ok": True, "source": source_name, "chunks_added": added}


def retrieve(question: str, top_k: int = 5) -> List[Dict[str, Any]]:
    if not DOCS:
        return []

    q_emb = ollama_embed(question)
    q_tokens = set(normalize_tokens(question))

    scored = []
    for doc in DOCS:
        semantic = cosine_similarity(q_emb, doc["embedding"])

        doc_text_lower = doc["text"].lower()
        lexical_hits = 0
        for tok in q_tokens:
            if tok in doc_text_lower:
                lexical_hits += 1

        lexical = lexical_hits / max(1, len(q_tokens))
        score = (0.75 * semantic) + (0.25 * lexical)

        scored.append({**doc, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def role_instruction(role: str) -> str:
    r = role.lower().strip()

    if r == "business":
        return (
            "Use plain business language. Avoid code-level detail unless necessary. "
            "Focus on feasibility, impact, risk, and whether the app already supports it."
        )

    if r in {"product manager", "pm"}:
        return (
            "Use a product manager style. Focus on scope, dependencies, user impact, effort, "
            "tradeoffs, and whether the request is feasible."
        )

    if r in {"developer", "dev", "software developer"}:
        return (
            "Use a developer style. Include technical flow, validation rules, APIs, classes, "
            "modules, data flow, and implementation impact."
        )

    if r in {"tester", "qa", "debugger"}:
        return (
            "Use a QA/testing style. Focus on validation, edge cases, negative cases, "
            "expected behavior, regression risk, and test scenarios."
        )

    return "Answer clearly and concisely."


def build_messages(role: str, question: str, sources: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    source_block_parts = []
    for i, src in enumerate(sources, start=1):
        source_block_parts.append(
            f"[Source {i}] {src['path']} lines {src['start_line']}-{src['end_line']}\n{src['text']}"
        )

    source_block = "\n\n".join(source_block_parts) if source_block_parts else "No sources found."

    system = (
        "You are ContextLens, an internal codebase assistant. "
        "Answer only from the provided sources. If the sources do not support the answer, say so. "
        "Be honest about uncertainty. Keep the answer useful and concise."
    )

    user = f"""
Role:
{role_instruction(role)}

Question:
{question}

Relevant sources:
{source_block}

Answer rules:
- Match the selected role.
- Use source paths/line ranges when referencing evidence.
- If the answer is not fully supported, say what is missing.
- Prefer concise, actionable output.
"""

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


@app.get("/", response_class=HTMLResponse)
def home():
    html_path = Path(__file__).with_name("index.html")
    return html_path.read_text(encoding="utf-8")


@app.post("/index/repo")
def api_index_repo(req: IndexRepoRequest):
    return index_repo(req.path)


@app.post("/index/swagger")
async def api_index_swagger(file: UploadFile = File(...)):
    content = (await file.read()).decode("utf-8", errors="ignore")
    source_name = file.filename or "swagger"
    return index_swagger_text(content, source_name=source_name)


@app.post("/chat")
def api_chat(req: ChatRequest):
    if not DOCS:
        return {
            "answer": "No repo or docs have been indexed yet.",
            "sources": []
        }

    sources = retrieve(req.question, top_k=5)
    messages = build_messages(req.role, req.question, sources)
    answer = ollama_chat(messages)

    return {
        "answer": answer,
        "sources": [
            {
                "path": s["path"],
                "lines": f"{s['start_line']}-{s['end_line']}",
                "score": round(float(s["score"]), 4)
            }
            for s in sources
        ]
    }