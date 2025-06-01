from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os
import getpass
from langchain_openai import ChatOpenAI
import time
from transformers import CLIPProcessor, CLIPModel
import torch
import numpy as np
import pickle
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io
from typing import Optional
import torch.cuda
from pathlib import Path
from config import BASE_DIR, LLM_SETTINGS
import logging
from langgraph.graph import StateGraph, END
from langchain_core.messages.utils import trim_messages
from dotenv import load_dotenv
from typing import TypedDict
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langgraph.store.sqlite import SqliteStore
from langgraph.checkpoint.sqlite import SqliteSaver
from langmem import create_manage_memory_tool, create_search_memory_tool
import sqlite3
from uuid import uuid4

load_dotenv()
logger = logging.getLogger(__name__)

CHROMA_DB_DIR = BASE_DIR / "DB" / "text"
IMAGE_DB_DIR = BASE_DIR / "DB" / "image"


deepseek_api_key = os.environ.get("DEEPSEEK_API_KEY")
if not deepseek_api_key:
    deepseek_api_key = getpass.getpass("Enter DeepSeek API Key: ")
# LLM (DeepSeek via OpenAI-compatible interface)
llm = ChatOpenAI(
    api_key=deepseek_api_key,
    base_url="https://api.deepseek.com/v1",
    callbacks=[StreamingStdOutCallbackHandler()],
    **LLM_SETTINGS
)
state = {"user_id": ..., "messages": [...], "image_bytes": ...}


# Vector store and retriever
embeddings = OpenAIEmbeddings()  # Or HuggingFaceEmbeddings
vector_db = Chroma(persist_directory=str(CHROMA_DB_DIR), embedding_function=embeddings)
retriever = vector_db.as_retriever()
# one file = one user-memory corpus
LT_MEM_DB = BASE_DIR / "DB" / "long_term_mem.db"
store = SqliteStore(str(LT_MEM_DB))
# Run once on first start-up (or inside a small helper):


manage_memory = create_manage_memory_tool(
    namespace=("memories", "{user_id}"),
    store=store
)
search_memory = create_search_memory_tool(
    namespace=("memories", "{user_id}"),
    store=store
)

def long_term_search_node(state):
    user_id = state.get("user_id")          # None if absent
    if not user_id:                         # nothing to search
        state["long_term_facts"] = ""
        return state
    hits = search_memory.invoke(
        {"query": state["message"]},
        config={"configurable": {"user_id": state["user_id"]}}
    )
    state["long_term_facts"] = "\n".join(h["memory"]["text"] for h in hits)
    return state

def long_term_write_node(state):
    # Write every assistant reply verbatim; add a filter if you prefer
    manage_memory.invoke(
        {"text": state.get("final_answer", "")},
        config={"configurable": {"user_id": state["user_id"]}}
    )
    return state

USER_MEM_DB_DIR = BASE_DIR / "DB" / "user_memory"

# Use same embeddings for consistency
user_mem_vector_db = Chroma(
    persist_directory=str(USER_MEM_DB_DIR),
    embedding_function=embeddings
)


def add_user_fact(fact, user_id):
    # Avoid adding every chat turn—only what you want to persist across sessions!
    user_mem_vector_db.add_texts([fact], metadatas=[{"user_id": user_id}])

def retrieve_user_memory(query, user_id, top_k=3):
    # Use user_id to filter only relevant user's facts
    if not user_id:
        print("retrieve_user_memory called with user_id=None!")
        return ""
    results = user_mem_vector_db.similarity_search(query, k=top_k, filter={"user_id": user_id})
    return "\n".join([doc.page_content for doc in results])

def get_short_term_history_by_tokens(messages, max_tokens=512):
    return trim_messages(
        messages,
        strategy="last",
        max_tokens=max_tokens
    )

def load_models() -> tuple[BlipProcessor, BlipForConditionalGeneration]:
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)
        return blip_processor, blip_model
    except Exception as e:
        raise RuntimeError(f"Failed to load BLIP models: {str(e)}")

blip_processor, blip_model = load_models()

def generate_image_caption(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        out = blip_model.generate(**inputs)
        caption = blip_processor.decode(out[0], skip_special_tokens=True)
    return caption

# Load the model once at startup
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16", use_fast=True)
# Load your image Chroma DB (replace with your actual DB path and embedding function)
image_vector_db = Chroma(
    persist_directory=str(IMAGE_DB_DIR),
    embedding_function=None  # Set to None since we'll provide embeddings manually
)

def embed_text_with_clip(text):
    inputs = clip_processor(text=[text], return_tensors="pt")
    with torch.no_grad():
        text_emb = clip_model.get_text_features(**inputs)
        text_emb = text_emb / text_emb.norm(p=2)
    return text_emb.cpu().numpy().squeeze()

def retrieve_top_images(query, top_k=3, max_distance=0.89):
    """
    Retrieves top images relevant to the query, filtering by a minimum relevance score.
    """
    query_embedding = embed_text_with_clip(query)
    # Use the new LangChain function to get (Document, score) tuples
    results = image_vector_db.similarity_search_by_vector_with_relevance_scores(query_embedding, k=top_k)
    print(f"Query: {query}")
    filtered = []
    for doc, score in results:
        meta = doc.metadata
        filename = meta.get('filename')
        caption = meta.get('caption', '')
        print(f"  Image: {filename}, Score: {score}")
        if score < max_distance:
            filtered.append({
                "filename": filename,
                "caption": caption,
                "url": f"/assets/images/{filename}"
            })
    print(f"  Filtered: {len(filtered)} images returned\n")
    return filtered
def text_retrieval_node(state):
    docs = retriever.get_relevant_documents(state['message'])
    print("Retrieved docs:", [doc.page_content[:50] for doc in docs])
    state['text_context'] = "\n".join(doc.page_content for doc in docs)
    return state


def image_retrieval_by_answer_node(state):
    answer_text = state.get('final_answer', '')   # Or wherever you store the answer
    images = retrieve_top_images(answer_text)
    state['images_context'] = images
    return state
_GEN_POOL = {} 
def merge_and_llm_node(state):
    images_text = ''
    history = "\n".join([f"{m['role']}: {m['content']}" for m in state.get('short_term_history', [])])
    long_term = state.get('long_term_facts', '')
    prompt = (
        "You are an expert in Blender. Use the following context to answer the question. "
        "If the context does not contain the answer, answer using your own knowledge."
        f"User question: {state['message']}\n"
        f"User uploaded image caption: {state.get('image_caption','')}\n"
        f"Conversation history:\n{history}\n"
        f"**Known user facts:**\n{long_term}\n"
        f"Relevant text:\n{state['text_context']}\n"
    )
    # Streaming: Yield each token as soon as it's available
    
    def token_stream():
        answer_chunks = []
        try:
            for chunk in llm.stream(prompt):
                answer_chunks.append(chunk.content)
                yield chunk.content
        finally:
            # This will run even if there's an error or the generator is exhausted!
            state['final_answer'] = "".join(answer_chunks)

    gen_id = uuid4().hex
    _GEN_POOL[gen_id] = token_stream()   # keep the generator in RAM
    state["answer_stream_id"] = gen_id  
    return state

short_term_memory = {}  # For demo; use Redis/DB for production
SHORT_TERM_MAX_TURNS = 8
def memory_retrieval_node(state):
    user_id = state.get('user_id')
    thread_id = state.get('thread_id')
    print(f"memory_retrieval_node: user_id={user_id}, thread_id={thread_id}")
    if not user_id or not thread_id:
        print("memory_retrieval_node called without user_id or thread_id!")
        state['short_term_history'] = []
        state['long_term_facts'] = ""
        return state
    # Short-term
    prev_msgs = short_term_memory.get((user_id, thread_id), [])
    state['short_term_history'] = get_short_term_history_by_tokens(prev_msgs)
    # Long-term
    hits = search_memory.invoke(
    {"query": state["message"]},
    config={"configurable": {"user_id": user_id}},
    )
    state['long_term_facts'] = "\n".join(h["memory"]["text"] for h in hits)
    return state

def update_memory(user_id, thread_id, message, answer):
    user_history = short_term_memory.get((user_id, thread_id), [])
    user_history.append({"role": "user", "content": message})
    user_history.append({"role": "assistant", "content": answer})
    short_term_memory[(user_id, thread_id)] = user_history[-SHORT_TERM_MAX_TURNS*2:]

def update_memory_node(state):
    user_id = state.get('user_id')
    thread_id = state.get('thread_id')
    answer = state.get('final_answer', '')
    if not answer:
        print("Warning: 'final_answer' missing in update_memory_node. Memory may not be updated properly.")
    # Or wherever the final response is
    user_history = short_term_memory.get((user_id, thread_id), [])
    user_history.append({"role": "user", "content": state['message']})
    user_history.append({"role": "assistant", "content": answer})
    short_term_memory[(user_id, thread_id)] = user_history[-SHORT_TERM_MAX_TURNS*2:]
    return state


class RagState(TypedDict):
    message: str
    text_context: str
    images_context: list
    image_caption: str
    answer_stream: object
    answer_stream_id: str 
builder = StateGraph(state_schema=RagState)
builder.add_node("text_retrieval", text_retrieval_node)
builder.add_node("memory_retrieval", memory_retrieval_node)
builder.add_node("merge_and_llm", merge_and_llm_node)
builder.add_node("image_retrieval_by_answer", image_retrieval_by_answer_node)
builder.add_node("long_term_search", long_term_search_node)
builder.add_node("long_term_write",  long_term_write_node)



builder.set_entry_point("text_retrieval")
builder.add_edge("text_retrieval", "memory_retrieval")
builder.add_edge("memory_retrieval", "long_term_search")
builder.add_edge("long_term_search", "merge_and_llm") 
builder.add_edge("merge_and_llm","image_retrieval_by_answer")
builder.add_edge("image_retrieval_by_answer", END)


CHKPT_DB = BASE_DIR / "DB" / "graph_state.db"
# 1. open the connection – keep it global so it lives as long as the process
conn = sqlite3.connect(str(CHKPT_DB), check_same_thread=False)  # thread-safe for FastAPI
# 2. build the saver *without* a context-manager
saver = SqliteSaver(conn)
# 3. compile the graph with that saver
langgraph_crag = builder.compile(checkpointer=saver)


def run_langgraph_rag(message, user_id=None, thread_id=None):
    state = {"message": message}
    if user_id is not None:
        state['user_id'] = user_id
    if thread_id is not None:
        state['thread_id'] = thread_id
    result = langgraph_crag.invoke(
        state,
        config={"configurable": {"thread_id": thread_id}}
    )

    return result["final_answer"], result.get("images_context", [])
