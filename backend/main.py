import logging
from pathlib import Path
from contextlib import asynccontextmanager

import httpx
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse

import config
from qdrant_store import load_docs_from_qdrant, get_qdrant, clear_all_from_qdrant
from ollama_client import ollama_chat
from indexer import index_repo, index_swagger_text
from retriever import retrieve
from prompts import build_messages, ContextWindowExceededError, CONTEXT_WINDOW

logger = logging.getLogger("contextlens")


@asynccontextmanager
async def lifespan(application: FastAPI):
    config.http_client = httpx.AsyncClient(timeout=120.0)
    try:
        load_docs_from_qdrant()
        logger.info(f"[qdrant] Loaded {len(config.DOCS)} chunks from collection '{config.QDRANT_COLLECTION}'")
    except Exception as e:
        logger.warning(f"[qdrant] Not available at startup, using empty in-memory store: {e}")
    yield
    await config.http_client.aclose()
    config.http_client = None


app = FastAPI(title="ContextLens", lifespan=lifespan)


class IndexRepoRequest(BaseModel):
    path: str
    repo_name: str = ""
    tag: str = ""


class HistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    role: str
    question: str
    history: list[HistoryItem] = []
    workspace_context: str = ""


@app.get("/", response_class=HTMLResponse)
def home():
    html_path = Path(__file__).with_name("index.html")
    return html_path.read_text(encoding="utf-8")


@app.get("/health")
async def health():
    status = {"ollama": "unknown", "qdrant": "unknown", "docs_loaded": len(config.DOCS)}
    try:
        resp = await config.http_client.get(f"{config.OLLAMA_BASE_URL}/api/tags")
        status["ollama"] = "ok" if resp.status_code == 200 else f"status {resp.status_code}"
    except Exception as e:
        status["ollama"] = f"error: {e}"
    try:
        get_qdrant().get_collections()
        status["qdrant"] = "ok"
    except Exception as e:
        status["qdrant"] = f"error: {e}"
    healthy = status["ollama"] == "ok" and status["qdrant"] == "ok"
    return JSONResponse(content=status, status_code=200 if healthy else 503)


@app.delete("/index/reset")
async def api_reset_index():
    try:
        config.DOCS.clear()
        clear_all_from_qdrant()
        return {"ok": True, "message": "All indexed data has been cleared."}
    except Exception as e:
        logger.exception("Error resetting index")
        return JSONResponse({"ok": False, "message": str(e)}, status_code=500)


@app.post("/index/repo")
async def api_index_repo(req: IndexRepoRequest):
    try:
        return await index_repo(req.path, repo_name=req.repo_name, tag=req.tag)
    except httpx.ConnectError:
        return JSONResponse({"ok": False, "message": "Cannot connect to Ollama. Is it running?"}, status_code=502)
    except Exception as e:
        logger.exception("Error indexing repo")
        return JSONResponse({"ok": False, "message": str(e)}, status_code=500)


@app.post("/index/swagger")
async def api_index_swagger(
    file: UploadFile = File(...),
    repo_name: str = Form(""),
    tag: str = Form(""),
):
    try:
        content = (await file.read()).decode("utf-8", errors="ignore")
        source_name = file.filename or "swagger"
        return await index_swagger_text(content, source_name=source_name, repo_name=repo_name, tag=tag)
    except httpx.ConnectError:
        return JSONResponse({"ok": False, "message": "Cannot connect to Ollama. Is it running?"}, status_code=502)
    except Exception as e:
        logger.exception("Error indexing swagger")
        return JSONResponse({"ok": False, "message": str(e)}, status_code=500)


@app.post("/chat")
async def api_chat(req: ChatRequest):
    if not config.DOCS:
        return {
            "answer": "No repo or docs have been indexed yet.",
            "sources": []
        }

    try:
        sources = await retrieve(req.question, top_k=6)
        history_dicts = [{"role": h.role, "content": h.content} for h in req.history]
        messages, token_estimate = build_messages(req.role, req.question, sources, history=history_dicts, workspace_context=req.workspace_context)
        answer = await ollama_chat(messages)

        return {
            "answer": answer,
            "sources": [
                {
                    "path": s["path"],
                    "lines": f"{s['start_line']}-{s['end_line']}",
                    "score": round(float(s["score"]), 4)
                }
                for s in sources
            ],
            "token_usage": {
                "prompt_tokens": token_estimate,
                "limit": CONTEXT_WINDOW,
            },
        }
    except ContextWindowExceededError as e:
        return JSONResponse(
            {
                "answer": str(e),
                "sources": [],
                "error_type": "context_window_exceeded",
            },
            status_code=413,
        )
    except httpx.ReadTimeout:
        return JSONResponse(
            {
                "answer": "Ollama response timed out. The model may be overloaded or the prompt is too large. Please clear your chat history and try again with a simpler question.",
                "sources": [],
                "error_type": "ollama timeout",
            },
            status_code=504,
        )
    except httpx.ConnectError:
        return JSONResponse({"answer": "Cannot connect to Ollama. Is it running?", "sources": []}, status_code=502)
    except Exception as e:
        logger.exception("Error in chat")
        return JSONResponse({"answer": f"Internal error: {e}", "sources": []}, status_code=500)