import os
import re
import logging
import json
import numpy as np
import tempfile
import traceback
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from django.conf import settings
from rest_framework.exceptions import ValidationError
from pgvector.django import CosineDistance
from pytubefix import YouTube

# ✅ Logging setup
logger = logging.getLogger("user_activity")

# ✅ LangChain imports
from langchain_community.document_loaders import YoutubeLoader, WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.embeddings import JinaEmbeddings

# ✅ Project imports
from .interface import QueryInterface

# --- Initialize helpers ---
query_interface = QueryInterface()

# --- Initialize embeddings and LLM ---
EMBEDDING_MODEL = JinaEmbeddings(
    jina_api_key=os.getenv("JINA_API_KEY"),
    model="jina-embeddings-v3",
    dimension=384
)

LLM = ChatGroq(
    model_name="qwen/qwen3-32b",
    temperature=0.4,
)

# --- AI-related functions ---
def generate_answer_from_chunks(question, chunks):
    context = "\n\n".join([c["text"] for c in chunks])
    prompt_template = ChatPromptTemplate.from_template("""
        You are an AI tutor that explains complex ideas in simple, easy-to-understand language.

        Use the following context to answer the student's question. Be clear, encouraging, and educational.
        If the answer cannot be found, politely say you don't know and suggest what to study next.

        Context:
        {context}

        Question:
        {question}

        Explain as if you are teaching a beginner student:
    """)
    chain = LLMChain(llm=LLM, prompt=prompt_template)
    response = chain.run({"context": context, "question": question})
    return response.strip()

# --- File validation and parsing ---
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

# --- Chunking and embeddings ---
def chunk_content(text, chunk_size=1000, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        length_function=len
    )
    return splitter.split_text(text)

def embed_chunks(chunks):
    if not chunks:
        return np.array([])
    return np.array(EMBEDDING_MODEL.embed_documents(chunks))

# --- Database interactions via QueryInterface ---
def save_document(session_key, title, source, content="", uploaded_file=None):
    """Save a document using QueryInterface."""
    data = {
        "session_key": session_key,
        "title": title,
        "source": source,
        "content": content,
    }

    if settings.ENVIRONMENT != "production":
        # Only store uploaded file locally
        from .models import Document
        doc = Document.objects.create(**data, uploaded_file=uploaded_file)
        return doc
    else:
        return query_interface.insert("Document", data)

def save_chunks(document_id, chunks, embeddings):
    """Save chunks depending on environment (local ORM vs Supabase)."""

    if not chunks:
        # Nothing to insert
        logger.warning(f"Document {document_id} has no chunks. Skipping insert.")
        return []

    records = [
        {
            "document_id": document_id,
            "text": chunk,
            "embedding": emb.tolist(),
            "chunk_index": idx,
        }
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))
    ]

    table_name = "document_chunk" if settings.ENVIRONMENT == "production" else "DocumentChunk"

    query_interface.bulk_insert(table_name, records)
    return records



def search_similar_chunks(query: str, docs=None, top_k=5):
    """Search similar chunks (vector similarity) using Supabase in prod, Django locally."""
    query_embedding = EMBEDDING_MODEL.embed_query(query)
    doc_ids = [d["id"] for d in docs] if docs else []

    if settings.ENVIRONMENT == "production":
        # If you plan to add Supabase vector search later
        results = query_interface.select("document_chunk", {"document_id": doc_ids})
        # (You could extend QueryInterface with vector filtering if needed)
        return results[:top_k]
    else:
        from .models import DocumentChunk
        chunks = (
            DocumentChunk.objects
            .filter(document_id__in=doc_ids)
            .annotate(distance=CosineDistance("embedding", query_embedding))
            .order_by("distance")[:top_k]
        )
        return [
            {
                "text": c.text,
                "distance": float(c.distance),
                "document_id": c.document_id,
            }
            for c in chunks
        ]

# --- External source utilities ---

def fetch_youtube_info(video_url: str) -> dict:
    """
    Fetches YouTube video info including title, description, thumbnail,
    channel, length, publish date, and transcript as plain text.
    """
    try:
        yt = YouTube(video_url)

        # Basic video info
        title = yt.title
        description = yt.description
        thumbnail = yt.thumbnail_url
        channel = yt.author
        length = yt.length
        publish_date = yt.publish_date

        # Transcript / captions
        transcript = ""
        if yt.captions:
            # Auto English captions 'a.en'
            caption = yt.captions.get('a.en') or yt.captions.get('en')
            if caption:
                srt_text = caption.generate_srt_captions()
                # Remove SRT numbers and timestamps
                transcript = re.sub(r"\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n", "", srt_text)
                # Remove extra newlines
                transcript = "\n".join([line.strip() for line in transcript.splitlines() if line.strip()])
            else:
                transcript = "No English subtitles available."
        else:
            transcript = "No subtitles found."

        return {
            "title": title,
            "description": description,
            "thumbnail": thumbnail,
            "channel": channel,
            "length_seconds": length,
            "publish_date": str(publish_date),
            "transcript": transcript
        }

    except Exception as e:
        return {"error": str(e)}


def fetch_website_text(url, max_chars=30000):
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        text = "\n".join([d.page_content for d in docs])
        return text[:max_chars]
    except Exception as e:
        return f"⚠️ Error fetching website: {str(e)}"

def extract_youtube_id(url):
    if not url:
        return None
    m = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{6,})", url)
    return m.group(1) if m else None



def log_user_action(session_key, action_type, details=None):
    """
    Logs user activity in a structured way.
    - session_key: identifies the user/session
    - action_type: e.g., 'upload_document', 'ask_question', 'process_youtube'
    - details: dict with extra info (like file names, question text, document ids)
    """
    entry = {
        "session_key": session_key,
        "action": action_type,
        "details": details or {},
    }
    logger.info(json.dumps(entry))



def get_traceback_str(exception):
    """Convert exception traceback to string for logging"""
    try:
        return ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))
    except:
        return str(exception)