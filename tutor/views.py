import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from django.shortcuts import render
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, parsers, permissions
from django.db import transaction
from django.http import JsonResponse


from .models import Document, DocumentChunk
from .serializers import DocumentSerializer
from . import utils


# tutor/views.py
from django.shortcuts import render
from django.http import JsonResponse
from . import utils

def chatlearner_view(request):
    query = request.POST.get("question") or request.GET.get("question")

    if request.method == "POST":
        # ✅ Validate input
        if not query or not query.strip():
            return JsonResponse({"error": "Please enter a question"}, status=400)

        try:
            results = utils.search_similar_chunks(query)
            print(results)
            return JsonResponse({"results": results}, status=200)
        except Exception as e:
            print("❌ Error in search_similar_chunks:", e)
            return JsonResponse({"error": str(e)}, status=500)

    # Render UI (GET)
    return render(request, "chatlearner.html")


def semantic_search_view(request):
    query = request.GET.get("query")
    if not query:
        return JsonResponse({"error": "Missing 'query' parameter"}, status=400)
    
    results = utils.search_similar_chunks(query)
    print(results)
    return JsonResponse({"query": query, "results": results})

# 🧩 This view handles file uploads (PDFs or text files).
# It reads, validates, extracts text, chunks content, embeds it, and saves results.
class DocumentUploadView(APIView):
    parser_classes = [parsers.MultiPartParser, parsers.FormParser]
    permission_classes = [permissions.AllowAny]

    def post(self, request, *args, **kwargs):
        # 🔹 Ensure each visitor/session has a unique key (used to link uploaded documents)
        if not request.session.session_key:
            request.session.save()
        session_key = request.session.session_key

        # 🔹 Get uploaded files from request
        files = request.FILES.getlist("files")
        if not files:
            return Response({"error": "No files uploaded"}, status=status.HTTP_400_BAD_REQUEST)

        # 🔹 Limit and file type settings
        MAX_SIZE = getattr(settings, "MAX_UPLOAD_SIZE", 10 * 1024 * 1024)  # default 10MB
        allowed_ext = {".pdf", ".txt"}

        created = []  # to collect uploaded doc info

        # Use transaction so if one file fails, all roll back
        with transaction.atomic():
            for f in files:
                # ✅ Step 1: Validate file (type + size)
                validation_error = utils.validate_file(f, allowed_ext, MAX_SIZE)
                if not validation_error:
                    return validation_error

                # ✅ Step 2: Save document metadata
                name = f.name
                doc = Document.objects.create(
                    session_key=session_key,
                    title=name,
                    source=Document.UPLOAD,
                    uploaded_file=f
                )

                # ✅ Step 3: Read content (depends on file type)
                dot_ext = "." + f.name.split(".")[-1].lower()
                content = ""
                if dot_ext == ".pdf":
                    content = utils.parse_pdf_file(f)
                elif dot_ext == ".txt":
                    f.seek(0)
                    try:
                        content = f.read().decode("utf-8", errors="ignore")
                    except Exception:
                        content = ""

                # ✅ Step 4: Split text into smaller chunks
                chunks = utils.chunk_content(content)

                # ✅ Step 5: Generate embeddings for each chunk
                embeddings = utils.embed_chunks(chunks)

                # ✅ Step 6: Save chunks & embeddings in DB (optimized)
                chunk_objs = [
                    DocumentChunk(
                        document=doc,
                        text=chunk,
                        embedding=emb.tolist(),  # convert NumPy array → Python list
                        chunk_index=idx
                    )
                    for idx, (chunk, emb) in enumerate(zip(chunks, embeddings))
                ]

                DocumentChunk.objects.bulk_create(chunk_objs, batch_size=100)

                # ✅ Step 7: Save full content
                doc.content = content
                doc.save()

                # ✅ Step 8: Prepare response data
                serialized_doc = {
                    "id": doc.id,
                    "title": doc.title,
                    "chunks_count": len(chunks),
                }
                created.append(serialized_doc)

        # ✅ Step 9: Return summary of all uploaded documents
        return Response({"status": "ok", "documents": created}, status=status.HTTP_201_CREATED)
    
# POST JSON {type: "youtube"|"website", url: "...", videoId?: "..."}
class ProcessLinkView(APIView):
    permission_classes = [permissions.AllowAny]
    def post(self, request, *args, **kwargs):
        if not request.session.session_key:
            request.session.save()
        session_key = request.session.session_key

        data = request.data
        typ = data.get("type")
        url = data.get("url", "")
        if typ == "youtube":
            video_id = data.get("videoId") or _extract_youtube_id(url)
            if not video_id:
                return Response({"error": "No video id found"}, status=status.HTTP_400_BAD_REQUEST)
            try:
                text = utils.fetch_youtube_transcript(video_id)
            except Exception as exc:
                return Response({"error": f"Transcript fetch failed: {str(exc)}"}, status=status.HTTP_400_BAD_REQUEST)
            doc = Document.objects.create(
                session_key=session_key,
                title=f"YouTube:{video_id}",
                source=Document.YOUTUBE,
                content=text,
                meta={"url": url, "videoId": video_id}
            )
            return Response({"status": "ok", "document": DocumentSerializer(doc).data}, status=status.HTTP_201_CREATED)

        elif typ == "website":
            if not url:
                return Response({"error": "No URL provided"}, status=status.HTTP_400_BAD_REQUEST)
            try:
                text = utils.fetch_website_text(url)
            except Exception as exc:
                return Response({"error": f"Website fetch failed: {str(exc)}"}, status=status.HTTP_400_BAD_REQUEST)
            doc = Document.objects.create(
                session_key=session_key,
                title=f"Website:{url}",
                source=Document.WEBSITE,
                content=text,
                meta={"url": url}
            )
            return Response({"status": "ok", "document": DocumentSerializer(doc).data}, status=status.HTTP_201_CREATED)

        return Response({"error": "Invalid link type"}, status=status.HTTP_400_BAD_REQUEST)


# small helper used by ProcessLinkView
def _extract_youtube_id(url):
    if not url:
        return None
    m = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{6,})", url)
    return m.group(1) if m else None


# Chat endpoint: uses simple retrieval over session documents
class ChatAPIView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request, *args, **kwargs):
        if not request.session.session_key:
            request.session.save()
        session_key = request.session.session_key
        question = (request.data.get("question") or "").strip()
        docs_qs = Document.objects.filter(session_key=session_key)
        docs = [{"title": d.title, "content": d.content} for d in docs_qs]
        answer = utils.simple_retrieval_answer(question, docs, top_sentences=3)
        return Response({"answer": answer})
