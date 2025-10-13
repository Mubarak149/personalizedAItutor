import re
import numpy as np
import tempfile
import requests
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
from rest_framework.exceptions import ValidationError
from pgvector.django import CosineDistance
from .models import DocumentChunk
# Optional imports
try:
    from youtube_transcript_api import YouTubeTranscriptApi
except Exception:
    YouTubeTranscriptApi = None


# 🧠 Sentence embedding model (used to turn text → numerical vector)
MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")


# ✅ Validate uploaded file extension and size
def validate_file(file_obj, allowed_ext=None, max_size=None):
    name = file_obj.name
    ext = (name.rsplit(".", 1)[-1] if "." in name else "").lower()
    dot_ext = "." + ext if ext else ""

    allowed_ext = allowed_ext or {".pdf", ".txt"}
    max_size = max_size or 10 * 1024 * 1024  # 10 MB

    if dot_ext not in allowed_ext:
        raise ValidationError(f"Unsupported file type: {dot_ext}. Allowed: {', '.join(allowed_ext)}")

    if file_obj.size > max_size:
        size_mb = round(file_obj.size / (1024 * 1024), 2)
        max_mb = round(max_size / (1024 * 1024), 2)
        raise ValidationError(f"File too large: {name} ({size_mb}MB). Max allowed is {max_mb}MB.")

    return True


# ✅ Extract text from PDF files
def parse_pdf_file(file_obj):
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp:
        for chunk in file_obj.chunks():
            tmp.write(chunk)
        tmp.flush()
        try:
            reader = PdfReader(tmp.name)
            texts = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(texts)
        except Exception:
            return ""


# ✅ Fetch YouTube transcript (if supported)
def fetch_youtube_transcript(video_id):
    if YouTubeTranscriptApi is None:
        raise RuntimeError("youtube-transcript-api not installed.")
    parts = YouTubeTranscriptApi.get_transcript(video_id)
    return " ".join([p.get("text", "") for p in parts])


# ✅ Fetch plain text from website (simple version)
def fetch_website_text(url, max_chars=30000):
    resp = requests.get(url, timeout=10, headers={"User-Agent": "ChatLearnerBot/1.0"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()
    texts = list(soup.stripped_strings)
    return " ".join(texts)[:max_chars]


# ✅ Split long text into smaller chunks
def chunk_content(text, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_text(text)


# ✅ Generate embeddings (vectors) for a list of text chunks
def embed_chunks(chunks):
    if not chunks:
        return np.array([])
    embeddings = MODEL.encode(chunks, convert_to_numpy=True)
    return embeddings


# Assume MODEL is your embedding model (like SentenceTransformer)
def search_similar_chunks(query: str, model=MODEL, top_k=5):
    # Step 1: Encode user query into vector
    query_embedding = model.encode([query], convert_to_numpy=True)[0].tolist()
    
    # Step 2: Search in database using cosine similarity
    similar_chunks = (
        DocumentChunk.objects
        .annotate(distance=CosineDistance('embedding', query_embedding))
        .order_by('distance')[:top_k]
    )
    
    # Step 3: Return results
    results = [
        {
            "text": chunk.text,
            "distance": float(chunk.distance),
            "document_id": chunk.document_id,
        }
        for chunk in similar_chunks
    ]
    return results
