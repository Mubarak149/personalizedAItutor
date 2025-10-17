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
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.embeddings import JinaEmbeddings


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
import os
from dotenv import load_dotenv

load_dotenv()


# Initialize Jina embeddings via LangChain
EMBEDDING_MODEL = JinaEmbeddings(
    jina_api_key=os.getenv("JINA_API_KEY"),
    model="jina-embeddings-v3",  # their latest model
    dimension=384
)
# Initialize Groq model
LLM = ChatGroq(
    model_name="qwen/qwen3-32b",  # You can change to "mixtral-8x7b" or "gemma-7b-it"
    temperature=0.4,
)

def generate_answer_from_chunks(question, chunks):
    # Combine text from the most relevant chunks
    context = "\n\n".join([c["text"] for c in chunks])

    # Prompt template for educational-style answers
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
def search_similar_chunks(query: str, docs, top_k=5):
    query_embedding = EMBEDDING_MODEL.embed_query(query)
    doc_ids = [d["id"] for d in docs]

    similar_chunks = (
        DocumentChunk.objects
        .filter(document_id__in=doc_ids)
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



# small helper used by ProcessLinkView
def extract_youtube_id(url):
    if not url:
        return None
    m = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{6,})", url)
    return m.group(1) if m else None

