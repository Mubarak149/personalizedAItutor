from django.db import models
from django.contrib.postgres.fields import ArrayField  # PostgreSQL array type (for embeddings)
from pgvector.django import VectorField

# 🧾 Represents each uploaded document (PDF, text, etc.)
class Document(models.Model):
    UPLOAD = "upload"
    YOUTUBE = "youtube"
    WEBSITE = "website"
    SOURCE_CHOICES = [
        (UPLOAD, "Upload"),
        (YOUTUBE, "YouTube"),
        (WEBSITE, "Website"),
    ]

    session_key = models.CharField(max_length=255, db_index=True)  # links uploads to a session
    title = models.CharField(max_length=255)
    source = models.CharField(max_length=20, choices=SOURCE_CHOICES)
    uploaded_file = models.FileField(upload_to="uploads/", null=True, blank=True)
    content = models.TextField(blank=True, default="")  # full extracted text
    date_uploaded = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title


# 🧠 Represents a small chunk of text + its embedding vector
# app/models.py

class DocumentChunk(models.Model):
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chunks")
    text = models.TextField()
    embedding = VectorField(dimensions=384)  # ✅ use pgvector type
    chunk_index = models.PositiveIntegerField(default=0)

    def __str__(self):
        return f"Chunk {self.chunk_index} of {self.document.title}"

