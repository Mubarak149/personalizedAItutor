import re
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from django.shortcuts import render
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, parsers, permissions
from django.db import transaction
from django.http import JsonResponse
from .interface import QueryInterface
from . import utils
from .logging_helpers import log_user_action  # our logging helper


def chatlearner_view(request):
    if request.method == "POST":
        if not request.session.session_key:
            request.session.save()
        session_key = request.session.session_key

        try:
            data = json.loads(request.body.decode("utf-8"))
            query = data.get("question")
        except Exception:
            query = None

        if not query or not query.strip():
            return JsonResponse({"error": "Please enter a question"}, status=400)

        try:
            results = utils.search_similar_chunks(query)
            log_user_action(
                session_key,
                "ask_question_chatlearner",
                {"question": query, "results_count": len(results)}
            )
            return JsonResponse({"results": results}, status=200)
        except Exception as e:
            log_user_action(session_key, "error_chatlearner", {"error": str(e)})
            return JsonResponse({"error": str(e)}, status=500)

    return render(request, "chatlearner.html")


class DocumentUploadView(APIView):
    parser_classes = [parsers.MultiPartParser, parsers.FormParser]
    permission_classes = [permissions.AllowAny]

    def post(self, request, *args, **kwargs):
        if not request.session.session_key:
            request.session.save()
        session_key = request.session.session_key

        files = request.FILES.getlist("files")
        if not files:
            return Response({"error": "No files uploaded"}, status=status.HTTP_400_BAD_REQUEST)

        MAX_SIZE = getattr(settings, "MAX_UPLOAD_SIZE", 10 * 1024 * 1024)
        allowed_ext = {".pdf", ".txt"}
        created = []
        query_interface = QueryInterface()

        # Use transaction only for local development
        if settings.ENVIRONMENT != "production":
            transaction_context = transaction.atomic()
        else:
            from contextlib import contextmanager
            @contextmanager
            def no_transaction():
                yield
            transaction_context = no_transaction()

        with transaction_context:
            for f in files:
                try:
                    utils.validate_file(f, allowed_ext, MAX_SIZE)
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

                    doc_data = {
                        "session_key": session_key,
                        "title": f.name,
                        "source": "upload",
                        "content": content,
                    }

                    doc = query_interface.insert("document", doc_data)
                    doc_id = doc[0]["id"] if settings.ENVIRONMENT == "production" else doc.id

                    chunks = utils.chunk_content(content)
                    embeddings = utils.embed_chunks(chunks)
                    utils.save_chunks(doc_id, chunks, embeddings)

                    serialized_doc = {
                        "id": doc_id,
                        "title": f.name,
                        "chunks_count": len(chunks),
                    }
                    created.append(serialized_doc)

                    # Log the upload
                    log_user_action(
                        session_key,
                        "upload_document",
                        {"filename": f.name, "doc_id": doc_id, "chunks": len(chunks)}
                    )

                except Exception as e:
                    log_user_action(session_key, "error_upload_document", {"filename": f.name, "error": str(e)})
                    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"status": "ok", "documents": created}, status=status.HTTP_201_CREATED)


class ProcessLinkView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request, *args, **kwargs):
        if not request.session.session_key:
            request.session.save()
        session_key = request.session.session_key

        data = request.data
        typ = data.get("type")
        url = data.get("url", "")
        query_interface = QueryInterface()

        try:
            if typ == "youtube":
                video_id = data.get("videoId") or utils.extract_youtube_id(url)
                if not video_id:
                    return Response({"error": "No video ID found"}, status=status.HTTP_400_BAD_REQUEST)

                text = utils.fetch_youtube_transcript(video_id)
                doc_data = {
                    "session_key": session_key,
                    "title": f"YouTube:{video_id}",
                    "source": "youtube",
                    "content": text,
                }
                doc = query_interface.insert("document", doc_data)
                doc_id = doc["id"] if settings.ENVIRONMENT == "production" else doc.id

                chunks = utils.chunk_content(text, chunk_size=500, chunk_overlap=50)
                embeddings = utils.embed_chunks(chunks)
                utils.save_chunks(doc_id, chunks, embeddings)

                log_user_action(
                    session_key,
                    "process_youtube_link",
                    {"video_id": video_id, "doc_id": doc_id, "chunks": len(chunks)}
                )

                return Response({
                    "status": "ok",
                    "document": {"id": doc_id, "title": doc_data["title"], "source": doc_data["source"]}
                }, status=status.HTTP_201_CREATED)

            elif typ == "website":
                if not url:
                    return Response({"error": "No URL provided"}, status=status.HTTP_400_BAD_REQUEST)

                text = utils.fetch_website_text(url)
                doc_data = {
                    "session_key": session_key,
                    "title": f"Website:{url}",
                    "source": "website",
                    "content": text,
                }
                doc = query_interface.insert("document", doc_data)
                doc_id = doc["id"] if settings.ENVIRONMENT == "production" else doc.id

                chunks = utils.chunk_content(text, chunk_size=500, chunk_overlap=50)
                embeddings = utils.embed_chunks(chunks)
                utils.save_chunks(doc_id, chunks, embeddings)

                log_user_action(
                    session_key,
                    "process_website_link",
                    {"url": url, "doc_id": doc_id, "chunks": len(chunks)}
                )

                return Response({
                    "status": "ok",
                    "document": {"id": doc_id, "title": doc_data["title"], "source": doc_data["source"]}
                }, status=status.HTTP_201_CREATED)

        except Exception as e:
            log_user_action(session_key, "error_process_link", {"type": typ, "url": url, "error": str(e)})
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        return Response({"error": "Invalid link type"}, status=status.HTTP_400_BAD_REQUEST)


class ChatAPIView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request, *args, **kwargs):
        if not request.session.session_key:
            request.session.save()
        session_key = request.session.session_key

        question = (request.data.get("question") or "").strip()
        if not question:
            return Response({"error": "Please enter a question"}, status=400)

        try:
            query_interface = QueryInterface()
            docs = query_interface.select("document", {"session_key": session_key})

            retrieved_chunks = utils.search_similar_chunks(question, docs, top_k=3)
            answer = utils.generate_answer_from_chunks(question, retrieved_chunks)

            log_user_action(
                session_key,
                "ask_question_chat",
                {
                    "question": question,
                    "doc_ids": [c["document_id"] for c in retrieved_chunks],
                    "retrieved_chunks_count": len(retrieved_chunks)
                }
            )

            return Response({
                "answer": answer,
                "sources": [c["document_id"] for c in retrieved_chunks],
            })
        except Exception as e:
            log_user_action(session_key, "error_chat", {"question": question, "error": str(e)})
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
