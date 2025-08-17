# tutor/loaders.py
from langchain_community.document_loaders import PyPDFLoader, YoutubeLoader, TextLoader
from pathlib import Path

class DocumentBundle:
    def __init__(self, docs, metadata=None):
        self.docs = docs
        self.metadata = metadata or {}

def load_pdf(path: str) -> DocumentBundle:
    loader = PyPDFLoader(path)
    docs = loader.load()
    for d in docs:
        d.metadata.setdefault("source", path)
        d.metadata.setdefault("kind", "pdf")
    return DocumentBundle(docs, {"kind": "pdf", "path": path})

def load_youtube(url: str) -> DocumentBundle:
    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
    docs = loader.load()
    for d in docs:
        d.metadata.setdefault("source", url)
        d.metadata.setdefault("kind", "youtube")
    return DocumentBundle(docs, {"kind": "youtube", "url": url})

def load_text(path: str) -> DocumentBundle:
    loader = TextLoader(path, autodetect_encoding=True)
    docs = loader.load()
    for d in docs:
        d.metadata.setdefault("source", path)
        d.metadata.setdefault("kind", "text")
    return DocumentBundle(docs, {"kind": "text", "path": path})