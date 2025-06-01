from ebooklib import epub
from bs4 import BeautifulSoup
from langchain_core.documents import Document

def load_epub(epub_path):
    book = epub.read_epub(epub_path)
    docs = []

    for item in book.get_items():
        if item.get_type() == epub.EpubHtml:
            content = item.get_content()
            soup = BeautifulSoup(content, 'html.parser')

            # Clean and extract text
            text = soup.get_text(separator="\n").strip()
            if text:
                docs.append(Document(page_content=text, metadata={"source": item.get_name()}))
    
    return docs
