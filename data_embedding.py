from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain.embeddings.base import Embeddings
from ebooklib.epub import read_epub, EpubHtml, EpubItem
from ebooklib import ITEM_IMAGE, ITEM_DOCUMENT
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
import getpass
import torch
import shutil
from pathlib import Path
import logging
from typing import List, Optional, Dict, Any
import chromadb
import uuid
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


# === Paths ===
# Get the current file's directory
BASE_DIR = Path(__file__).parent.resolve()

# Define paths relative to the base directory
CHROMA_TEXT_DB_DIR = BASE_DIR / "DB" / "text"
CHROMA_IMAGE_DB_DIR = BASE_DIR / "DB" / "image"
EPUB_PATH = BASE_DIR / "blender_manual_v440_en.epub"
IMAGE_OUTPUT_DIR = BASE_DIR / "images"

# Clear old DB
if os.path.exists(CHROMA_TEXT_DB_DIR):
    shutil.rmtree(CHROMA_TEXT_DB_DIR)

# === Initialize Embeddings ===
text_embedder = OpenAIEmbeddings()
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

class CLIPImageEmbeddings(Embeddings):
    def embed_documents(self, image_paths):
        embs = []
        for path in image_paths:
            try:
                image = Image.open(path).convert("RGB")      # RGB, not RGBA
                inputs = clip_processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    vec = clip_model.get_image_features(**inputs)[0]
                embs.append((vec / vec.norm(p=2)).cpu().tolist())  # list[float]
            except Exception as e:
                logger.error(f"{path}: {e}")
                embs.append([0.0] * 512)
        return embs

    def embed_query(self, image_path):
        return self.embed_documents([image_path])[0]



# === EPUB Parsing ===
def extract_images(epub_path: Path, output_folder: Path) -> Dict[str, str]:
    """Extract images from EPUB file and save them to output folder."""
    book = read_epub(str(epub_path))  # Convert Path to string for epub reader
    output_folder.mkdir(exist_ok=True)
    image_map = {}
    
    for item in book.get_items():
        if item.get_type() == ITEM_IMAGE:
            filename = Path(item.get_name()).name
            filepath = output_folder / filename
            with open(filepath, 'wb') as f:
                f.write(item.get_content())
            image_map[item.get_id()] = filename
    return image_map

def load_epub_with_images(epub_path: Path, image_map: Dict[str, str]) -> List[Document]:
    book = read_epub(str(epub_path))  # Convert Path to string
    docs = []
    
    for item in book.get_items():
        if item.get_type() == ITEM_DOCUMENT:
            logger.info(f"Processing document item: {item.get_name()}")
            
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            for img in soup.find_all('img'):
                src = img.get('src')
                if src:
                    img.insert_after(f"\n[Image: {Path(src).name}]")
            
            text = soup.get_text(separator="\n").strip()
            if text:
                logger.info(f"Extracted text from: {item.get_name()} ({len(text)} chars)")
                docs.append(Document(
                    page_content=text,
                    metadata={"source": item.get_name()}
                ))
    return docs


# === Text Chroma DB ===
def init_text_chroma():
    image_map = extract_images(EPUB_PATH, IMAGE_OUTPUT_DIR)
    docs = load_epub_with_images(EPUB_PATH, image_map)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )
    split_docs = text_splitter.split_documents(docs)

    CHROMA_TEXT_DB_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loaded {len(split_docs)} document chunks from EPUB")
    
    return Chroma.from_documents(
        documents=split_docs,
        embedding=text_embedder,
        persist_directory=str(CHROMA_TEXT_DB_DIR)  # Convert Path to string for Chroma
    )


def load_text_chroma():
    # Convert Path to string for Chroma
    return Chroma(
        persist_directory=str(CHROMA_TEXT_DB_DIR), 
        embedding_function=text_embedder
    )

# === Image Chroma DB ===
def init_image_chroma():
    if CHROMA_IMAGE_DB_DIR.exists():
        shutil.rmtree(CHROMA_IMAGE_DB_DIR)
    
    image_files = [
        f for f in IMAGE_OUTPUT_DIR.iterdir() 
        if f.suffix.lower() in ('.png', '.jpg', '.jpeg')
    ]

    docs = [
        Document(
            page_content=str(path.absolute()),
            metadata={
                "filename": path.name,
                "caption": "",
                "image_path": str(path.absolute())
            }
        )
        for path in image_files
    ]

    CHROMA_IMAGE_DB_DIR.mkdir(parents=True, exist_ok=True)
    
    # Build normalized image embeddings
    embedder = CLIPImageEmbeddings()
    image_embeddings = embedder.embed_documents([str(f) for f in image_files])

    # Use ChromaDB client to set cosine similarity
    client = chromadb.PersistentClient(path=str(CHROMA_IMAGE_DB_DIR))
    collection = client.get_or_create_collection(
        name="langchain",
        metadata={"hnsw:space": "cosine"}
    )
    # Add to collection
    collection.add(
        embeddings=[emb.tolist() for emb in image_embeddings],
        metadatas=[{
            "filename": path.name,
            "caption": "",
            "image_path": str(path.absolute())
        } for path in image_files],
        documents=[str(path.absolute()) for path in image_files],
        ids=[str(uuid.uuid4()) for _ in image_files]  
    )
    return collection


def load_image_chroma():
    embedder = CLIPImageEmbeddings()
    # Convert Path to string for Chroma
    return Chroma(
        persist_directory=str(CHROMA_IMAGE_DB_DIR), 
        embedding_function=embedder
    )

# === Image Search ===
def search_by_image(image_path, top_k=3):
    db = load_image_chroma()
    embedder = CLIPImageEmbeddings()
    embedding = embedder.embed_query(image_path)
    return db.similarity_search_by_vector(embedding, k=top_k)

def main():
    try:
        logger.info("Starting database rebuild process...")
        
        logger.info("Initializing text database...")
        init_text_chroma()
        
        logger.info("Initializing image database...")
        init_image_chroma()
        
        logger.info("Database rebuild completed successfully")
        return 0
    except Exception as e:
        logger.error(f"Failed to rebuild databases: {e}")
        return 1

if __name__ == "__main__":
    exit(main())
