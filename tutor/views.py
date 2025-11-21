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
from django.core.cache import cache
from django.conf import settings

from .interface import QueryInterface
from . import utils
from .logging_helpers import log_user_action  # our logging helper

def chatlearner_view(request):
    if not request.session.session_key:
        request.session.save()  
    session_key = request.session.session_key
    query = QueryInterface()
    datas = query.select("document", {"session_key": session_key})
    
    # Log page view
    log_user_action(
        session_key,
        "page_view",
        {
            "page": "chatlearner",
            "documents_count": len(datas),
            "document_ids": [doc.get('id') for doc in datas]
        }
    )
    
    return render(request, "chatlearner.html", {"datas": datas})


class DocumentUploadView(APIView):
    parser_classes = [parsers.MultiPartParser, parsers.FormParser]
    permission_classes = [permissions.AllowAny]

    def post(self, request, *args, **kwargs):
        if not request.session.session_key:
            request.session.save()
        session_key = request.session.session_key

        files = request.FILES.getlist("files")
        if not files:
            log_user_action(session_key, "upload_error", {"error": "No files uploaded"})
            return Response({"error": "No files uploaded"}, status=status.HTTP_400_BAD_REQUEST)

        MAX_SIZE = getattr(settings, "MAX_UPLOAD_SIZE", 10 * 1024 * 1024)
        allowed_ext = {".pdf", ".txt"}
        created = []
        query_interface = QueryInterface()

        # Log upload attempt
        log_user_action(
            session_key,
            "upload_attempt",
            {
                "file_count": len(files),
                "file_names": [f.name for f in files],
                "file_sizes": [f.size for f in files]
            }
        )

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

                    # Log successful upload with detailed info
                    log_user_action(
                        session_key,
                        "upload_success",
                        {
                            "filename": f.name,
                            "doc_id": doc_id,
                            "file_size": f.size,
                            "content_length": len(content),
                            "chunks_count": len(chunks),
                            "file_type": dot_ext
                        }
                    )

                except Exception as e:
                    # Log detailed upload error
                    log_user_action(
                        session_key, 
                        "upload_error",
                        {
                            "filename": f.name,
                            "file_size": f.size if f else 0,
                            "error_type": type(e).__name__,
                            "error_message": str(e),
                            "traceback": utils.get_traceback_str(e)  # Add this helper function
                        }
                    )
                    return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # Log overall upload completion
        log_user_action(
            session_key,
            "upload_complete",
            {
                "total_files_processed": len(created),
                "created_documents": created
            }
        )

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
        
        # Log link processing attempt
        log_user_action(
            session_key,
            "process_link_attempt",
            {
                "type": typ,
                "url": url,
                "request_data": data
            }
        )
        
        try:
            if typ == "youtube":
                video_id = data.get("videoId") or utils.extract_youtube_id(url)
                if not video_id:
                    log_user_action(session_key, "youtube_error", {"error": "No video ID found", "url": url})
                    return Response({"error": "No video ID found"}, status=status.HTTP_400_BAD_REQUEST)
                
                # Log YouTube processing start
                log_user_action(session_key, "youtube_processing_start", {"video_id": video_id, "url": url})
                
                yt_video_data = utils.fetch_youtube_info(url)
                text = yt_video_data["transcript"]
                doc_data = {
                    "session_key": session_key,
                    "title": yt_video_data['title'],
                    "source": "youtube",
                    "content": text,
                }
                doc = query_interface.insert("document", doc_data)
                doc_id = doc[0]["id"] if settings.ENVIRONMENT == "production" else doc.id

                chunks = utils.chunk_content(text, chunk_size=500, chunk_overlap=50)
                embeddings = utils.embed_chunks(chunks)
                
                utils.save_chunks(doc_id, chunks, embeddings)

                # Log successful YouTube processing
                log_user_action(
                    session_key,
                    "youtube_success",
                    {
                        "video_id": video_id,
                        "doc_id": doc_id,
                        "video_title": yt_video_data['title'],
                        "content_length": len(text),
                        "chunks_count": len(chunks),
                        "duration": yt_video_data.get('duration')
                    }
                )

                return Response({
                    "status": "ok",
                    "document": {"id": doc_id, "title": doc_data["title"], "source": doc_data["source"]}
                }, status=status.HTTP_201_CREATED)

            elif typ == "website":
                if not url:
                    log_user_action(session_key, "website_error", {"error": "No URL provided"})
                    return Response({"error": "No URL provided"}, status=status.HTTP_400_BAD_REQUEST)

                # Log website processing start
                log_user_action(session_key, "website_processing_start", {"url": url})
                
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

                # Log successful website processing
                log_user_action(
                    session_key,
                    "website_success",
                    {
                        "url": url,
                        "doc_id": doc_id,
                        "content_length": len(text),
                        "chunks_count": len(chunks)
                    }
                )

                return Response({
                    "status": "ok",
                    "document": {"id": doc_id, "title": doc_data["title"], "source": doc_data["source"]}
                }, status=status.HTTP_201_CREATED)

        except Exception as e:
            # Log detailed link processing error
            log_user_action(
                session_key, 
                "process_link_error",
                {
                    "type": typ,
                    "url": url,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": utils.get_traceback_str(e)
                }
            )
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        log_user_action(session_key, "invalid_link_type", {"type": typ})
        return Response({"error": "Invalid link type"}, status=status.HTTP_400_BAD_REQUEST)


class ChatAPIView(APIView):
    permission_classes = [permissions.AllowAny]

    def post(self, request, *args, **kwargs):
        # 1. Make sure session exists
        if not request.session.session_key:
            request.session.save()
        session_key = request.session.session_key

        # 2. Read question
        question = (request.data.get("question") or "").strip()
        if not question:
            log_user_action(session_key, "empty_question", {"request_data": request.data})
            return Response({"error": "Please enter a question"}, status=400)

        # 3. Read selected document IDs from request body
        selected_ids = request.data.get("selected_ids", [])
        print("POST DATA:", request.data)
        
        # Convert to list safely (if frontend sends comma-separated string)
        if isinstance(selected_ids, str):
            selected_ids = [i for i in selected_ids.split(",") if i.strip()]

        # If user didn't pick any doc → stop here
        if not selected_ids:
            log_user_action(
                session_key,
                "no_documents_selected",
                {
                    "question": question,
                    "selected_ids": selected_ids
                }
            )
            return Response({"error": "Please select at least one document"}, status=400)

        try:
            query_interface = QueryInterface()

            # 4. Fetch only the selected docs belonging to this session
            docs = query_interface.select(
                "document",
                {
                    "session_key": session_key,
                    "id": selected_ids
                }
            )

            # 5. If doc IDs don't exist for this session → error
            if not docs:
                log_user_action(
                    session_key,
                    "documents_not_found",
                    {
                        "question": question,
                        "selected_ids": selected_ids,
                        "found_documents": []
                    }
                )
                return Response({"error": "Selected documents not found"}, status=404)

            # Log chat question with context
            log_user_action(
                session_key,
                "chat_question",
                {
                    "question": question,
                    "selected_ids": selected_ids,
                    "available_documents": [doc.get('id') for doc in docs],
                    "question_length": len(question)
                }
            )

            # 6. Retrieve chunks from the selected documents only
            retrieved_chunks = utils.search_similar_chunks(
                question,
                docs,
                top_k=30
            )

            # 7. Generate final answer from chunks
            answer = utils.generate_answer_from_chunks(question, retrieved_chunks)

            # 8. Log comprehensive chat response
            log_user_action(
                session_key,
                "chat_response",
                {
                    "question": question,
                    "selected_ids": selected_ids,
                    "doc_ids": list(set([c["document_id"] for c in retrieved_chunks])),
                    "retrieved_chunks_count": len(retrieved_chunks),
                    "answer_length": len(answer),
                    "answer_preview": answer[:200] + "..." if len(answer) > 200 else answer,
                    "sources_used": [c["document_id"] for c in retrieved_chunks]
                }
            )

            # 9. Send back answer + source document IDs
            return Response({
                "answer": answer,
                "sources": list(set([c["document_id"] for c in retrieved_chunks])),
            })

        except Exception as e:
            # Log detailed chat error
            log_user_action(
                session_key,
                "chat_error",
                {
                    "question": question,
                    "selected_ids": selected_ids,
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": utils.get_traceback_str(e)
                }
            )
            return Response({"error": str(e)}, status=500)