import re
import numpy as np
import tempfile
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader

# ✅ LangChain imports
from langchain_community.document_loaders import YoutubeLoader, WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# ✅ Django and project imports
from rest_framework.exceptions import ValidationError
from pgvector.django import CosineDistance
from .models import DocumentChunk

# ✅ Fallback imports for direct transcript
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound, VideoUnavailable
except Exception:
    YouTubeTranscriptApi = None

# 🧠 LangChain embedding model (wrapped for compatibility)
EMBEDDING_MODEL = EMBEDDING_MODEL = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


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


# ✅ Extract text from PDF files (via LangChain loader)
def parse_pdf_file(file_obj):
    with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp:
        for chunk in file_obj.chunks():
            tmp.write(chunk)
        tmp.flush()
        try:
            loader = PyPDFLoader(tmp.name)
            docs = loader.load()
            return "\n".join([d.page_content for d in docs])
        except Exception:
            return ""


# ✅ Fetch YouTube transcript (LangChain style)
def fetch_youtube_transcript(video_id: str) -> str:
    """
    Fetch and combine transcript text for a given YouTube video ID.
    Uses LangChain YoutubeLoader first, fallback to direct API.
    """
    try:
        # ✅ Try LangChain YouTube loader first
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
        docs = loader.load()
        return "\n".join([d.page_content for d in docs])

    except Exception:
        # ✅ Fallback to youtube_transcript_api
        if YouTubeTranscriptApi is None:
            return "⚠️ Transcript API not available."
        try:
            ytt_api = YouTubeTranscriptApi()
            transcript_obj = ytt_api.fetch(video_id)
            return " ".join([snippet.text for snippet in transcript_obj.snippets])
        except TranscriptsDisabled:
            return "⚠️ Transcripts are disabled for this video."
        except NoTranscriptFound:
            return "⚠️ No transcript available for this video."
        except VideoUnavailable:
            return "⚠️ The requested video is unavailable."
        except Exception as e:
            return f"⚠️ Error fetching transcript: {str(e)}"


# ✅ Fetch plain text from website (LangChain style)
def fetch_website_text(url, max_chars=30000):
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        text = "\n".join([d.page_content for d in docs])
        return text[:max_chars]
    except Exception:
        # Fallback to manual BeautifulSoup method
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Linux Debian/1.0"})
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()
        texts = list(soup.stripped_strings)
        return " ".join(texts)[:max_chars]


# ✅ Split long text into smaller chunks (LangChain splitter)
def chunk_content(text, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len
    )
    return splitter.split_text(text)


# ✅ Generate embeddings using LangChain
def embed_chunks(chunks):
    if not chunks:
        return np.array([])
    return np.array(EMBEDDING_MODEL.embed_documents(chunks))


# ✅ Store and retrieve embeddings (pgvector search)
def search_similar_chunks(query: str, top_k=5):
    query_embedding = EMBEDDING_MODEL.embed_query(query)
    similar_chunks = (
        DocumentChunk.objects
        .annotate(distance=CosineDistance('embedding', query_embedding))
        .order_by('distance')[:top_k]
    )
    return [
        {
            "text": chunk.text,
            "distance": float(chunk.distance),
            "document_id": chunk.document_id,
        }
        for chunk in similar_chunks
    ]
