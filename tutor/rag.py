# tutor/rag.py
import os
from pydantic_settings import BaseSettings
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
import json

class Settings(BaseSettings):
    OPENAI_API_KEY: str = ""
    VECTOR_BACKEND: str = "faiss"  # or "pgvector"
    INDEX_PATH: str = ".rag_index/faiss"

settings = Settings()

# --- Embeddings / LLM ---
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

# --- Text splitter ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", " ", ""]
)

# --- Vector store (FAISS to keep it simple) ---
_vs = None

def get_vs():
    global _vs
    if _vs is None:
        if os.path.exists(settings.INDEX_PATH):
            _vs = FAISS.load_local(settings.INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        else:
            os.makedirs(settings.INDEX_PATH, exist_ok=True)
            _vs = FAISS.from_texts(["Temporary placeholder"], embeddings, metadatas=[{"source":"bootstrap"}])
            _vs.save_local(settings.INDEX_PATH)
    return _vs


def add_documents(docs):
    vs = get_vs()
    texts = []
    metas = []
    for d in docs:
        # ensure string content
        content = d.page_content if hasattr(d, "page_content") else str(d)
        # split
        for chunk in text_splitter.split_documents([Document(page_content=content, metadata=d.metadata)]):
            texts.append(chunk.page_content)
            metas.append(chunk.metadata)
    vs.add_texts(texts, metas)
    vs.save_local(settings.INDEX_PATH)


def get_retriever(k=4):
    return get_vs().as_retriever(search_kwargs={"k": k})

# --- Chat (RAG) ---
CHAT_SYSTEM = """
You are a patient subject tutor. Answer using only the given context.
Cite sources inline like (Source: <filename or url>). Be concise, step-by-step when helpful.
If the answer isn't in context, say you don't know and suggest where to look next.
"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", CHAT_SYSTEM),
    ("human", "Question: {question}\n\nContext:\n{context}"),
])


def answer_with_context(question: str, retriever=None):
    retriever = retriever or get_retriever()
    # gather context
    docs = retriever.get_relevant_documents(question)
    ctx = "\n\n".join([d.page_content + f"\n(Source: {d.metadata.get('source','?')})" for d in docs])
    messages = RAG_PROMPT.format_messages(question=question, context=ctx)
    resp = llm.invoke(messages)
    return resp.content, docs

# --- Quiz generation ---
QUIZ_SYSTEM = """
You are a quiz generator. Given study context, produce a JSON with 5 questions.
Use this schema:
{
  "title": "<short quiz title>",
  "questions": [
    {"type": "mcq", "prompt": "...", "choices": ["A) ...","B) ...","C) ...","D) ..."], "answer": "B"},
    {"type": "open", "prompt": "...", "answer": "<short expected key points>"}
  ]
}
Make them unambiguous and based ONLY on the context.
"""
QUIZ_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QUIZ_SYSTEM),
    ("human", "Create a quiz from this context:\n{context}")
])


def generate_quiz_from_context(topic_query: str, retriever=None):
    retriever = retriever or get_retriever()
    docs = retriever.get_relevant_documents(topic_query)
    ctx = "\n\n".join([d.page_content for d in docs])
    msg = QUIZ_PROMPT.format_messages(context=ctx)
    resp = llm.invoke(msg)
    try:
        data = json.loads(resp.content)
    except Exception:
        # simple repair: try to extract JSON block
        s = resp.content
        start = s.find("{")
        end = s.rfind("}")
        data = json.loads(s[start:end+1])
    return data, docs