# api.py

from fastapi import FastAPI, File, UploadFile, Form, status, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from llm import langgraph_crag, generate_image_caption, update_memory, short_term_memory, _GEN_POOL


from utils import timing_decorator
from config import CORS_SETTINGS, ASSETS_DIR, HTML_DIR, MULTIMODAL_SETTINGS
import logging
import structlog
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import time
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import asyncio
from pathlib import Path
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address
import requests
import json
import re
def extract_output(result):
    # Try if result is a dict
    if isinstance(result, dict) and "output" in result:
        return result["output"]
    # Try if result is a string containing code block
    if isinstance(result, str) and "```" in result:
        # Extract JSON inside code block
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", result, re.IGNORECASE)
        if match:
            json_str = match.group(1).strip()
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and "output" in parsed:
                    return parsed["output"]
            except Exception:
                pass  # Fallback to next step
    # Try if result is a JSON string
    if isinstance(result, str) and result.strip().startswith("{") and "output" in result:
        try:
            parsed = json.loads(result)
            if isinstance(parsed, dict) and "output" in parsed:
                return parsed["output"]
        except Exception:
            pass  # Fallback to returning string
    # Otherwise, just return as is
    return str(result)

logger = structlog.get_logger(__name__)
IMAGES_DIR = Path("/Users/l/Desktop/LeapLead/RAG/assets/images")

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="LeapLead API",
    description="API for multimodal chat interactions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)
app.state.limiter = limiter

# Add rate limiter exception handler
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add rate limiter middleware using SlowAPIMiddleware
app.add_middleware(SlowAPIMiddleware)

# CORS middleware
app.add_middleware(CORSMiddleware, **CORS_SETTINGS)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    metadata: Optional[Dict[str, Any]] = None

class ImageMetadata(BaseModel):
    filename: str
    caption: str = ''
    url: str

class MultiModalResponse(ChatResponse):
    images: Optional[List[ImageMetadata]] = Field(
        default=None,
        description="List of relevant images with metadata"
    )
    timing: Optional[float] = Field(
        default=None,
        description="Processing time in seconds"
    )
def stream_langgraph_rag_with_images(message, image_caption=None, user_id=None, thread_id=None, on_stream_end=None):
    state = {"message": message}
    if image_caption:
        state["image_caption"] = image_caption
    if user_id:
        state["user_id"] = user_id
    if thread_id:
        state["thread_id"] = thread_id
    result = langgraph_crag.invoke(state, config={"configurable": {"thread_id": thread_id}})
    gen = _GEN_POOL.pop(result["answer_stream_id"])     # retrieve and remove
    for token in gen:                                   # unchanged UX
        yield token

    full_answer = "".join(answer_chunks)
    images = result.get('images_context', [])
    yield f"\n[[[IMAGES_JSON]]]{json.dumps(images)}[[[/IMAGES_JSON]]]\n"
    if on_stream_end:
        on_stream_end(full_answer)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": str(exc)},
    )

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc.detail)},
    )
def ensure_image(filename: str) -> Path:
    filepath = IMAGES_DIR / filename
    if filepath.exists():
        return filepath
    url = f"https://docs.blender.org/manual/en/latest/_images/{filename}"
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            filepath.write_bytes(resp.content)
            logger.info(
                "image_downloaded",
                filename=filename,
                url=url,
                saved_path=str(filepath),
                status_code=resp.status_code
            )
            return filepath
        else:
            logger.warning(
                "image_download_failed",
                filename=filename,
                url=url,
                status_code=resp.status_code
            )
    except Exception as e:
        logger.error(
            "image_download_exception",
            filename=filename,
            url=url,
            error=str(e)
        )
    return None
user_threads = {}  # {user_id: [thread_id, ...]}

SHORT_TERM_MAX_TURNS = 8

@app.get("/threads", response_model=List[str])
def list_threads(user_id: str = Query(...)):
    """Return all thread_ids for a user"""
    return user_threads.get(user_id, [])

@app.get("/threads/{thread_id}/messages", response_model=List[dict])
def get_thread_messages(user_id: str, thread_id: str):
    """Return conversation history for this thread"""
    return short_term_memory.get((user_id, thread_id), [])
@app.post("/threads")
def create_thread(user_id: str = Form(...)):
    thread_id = f"thread-{int(time.time() * 1000)}"
    user_threads.setdefault(user_id, []).append(thread_id)
    return {"thread_id": thread_id}

@app.post("/chat-multimodal", response_model=MultiModalResponse)
@timing_decorator
async def chat_multimodal(
    background_tasks: BackgroundTasks,
    message: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    user_id: str = Form(...),  
    thread_id: str = Form(...),
    
):
    image_caption = None
        # 1. Add thread to user's thread list if new
    if thread_id not in user_threads.get(user_id, []):
        user_threads.setdefault(user_id, []).append(thread_id)

    try:
        # Input validation
        if not message and not image:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either message or image must be provided"
            )
        
        if image:
            if not image.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail=f"Invalid image format: {image.content_type}"
                )
            try:
                image_bytes = await image.read()
                image_caption = generate_image_caption(image_bytes)
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to read image: {str(e)}"
                )
        answer_chunks: list[str] = []
        def streaming_gen():
            nonlocal answer_chunks
            state = {"message": message}
            if image_caption:
                state["image_caption"] = image_caption
            if user_id:
                state["user_id"] = user_id
            if thread_id:
                state["thread_id"] = thread_id
            result = langgraph_crag.invoke(state, config={"configurable": {"thread_id": thread_id}})
            gen = _GEN_POOL.pop(result["answer_stream_id"])     # retrieve and remove
            for token in gen:  
                answer_chunks.append(token)                                 # unchanged UX
                yield token

            images = result.get('images_context', [])
            full_answer = "".join(answer_chunks)
            # Update memory immediately before yielding images marker
            update_memory(user_id, thread_id, message, full_answer)
            yield f"\n[[[IMAGES_JSON]]]{json.dumps(images)}[[[/IMAGES_JSON]]]\n"


        return StreamingResponse(streaming_gen(), media_type="text/plain")

        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in chat-multimodal")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )

# Static files mounting
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
app.mount("/", StaticFiles(directory=str(HTML_DIR), html=True), name="static_html")
print("IMAGES_DIR resolved to:", str(IMAGES_DIR))
