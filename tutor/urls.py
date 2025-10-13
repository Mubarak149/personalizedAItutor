# app/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path("chatlearner/", views.chatlearner_view, name="chatlearner"),
    path("api/documents/upload/", views.DocumentUploadView.as_view(), name="api_document_upload"),
    path("api/documents/process-link/", views.ProcessLinkView.as_view(), name="api_process_link"),
    path("api/chat/", views.ChatAPIView.as_view(), name="api_chat"),
]
