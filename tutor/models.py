# tutor/models.py
from django.db import models
from django.contrib.auth.models import User

class Resource(models.Model):
    TYPE_CHOICES = [
        ("pdf", "PDF"),
        ("youtube", "YouTube"),
        ("text", "Plain Text/Markdown"),
    ]
    title = models.CharField(max_length=255)
    kind = models.CharField(max_length=20, choices=TYPE_CHOICES)
    source_path = models.TextField(help_text="File path or URL")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.title} ({self.kind})"

class ChatSession(models.Model):
    user = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)
    topic = models.CharField(max_length=200, default="General")
    created_at = models.DateTimeField(auto_now_add=True)

class ChatMessage(models.Model):
    ROLE_CHOICES = [("user", "User"), ("ai", "AI")]
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=10, choices=ROLE_CHOICES)
    content = models.TextField()
    tokens = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

class Quiz(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.SET_NULL, null=True, blank=True, related_name="quizzes")
    title = models.CharField(max_length=200)
    created_at = models.DateTimeField(auto_now_add=True)
    raw_json = models.JSONField(default=dict)  # canonical quiz structure

class QuizQuestion(models.Model):
    quiz = models.ForeignKey(Quiz, on_delete=models.CASCADE, related_name="questions")
    prompt = models.TextField()
    choices = models.JSONField(default=list)   # ["A. ...", "B. ...", ...] or empty for open-ended
    answer = models.CharField(max_length=10, blank=True)  # e.g., "B" or free text

class QuizAttempt(models.Model):
    quiz = models.ForeignKey(Quiz, on_delete=models.CASCADE, related_name="attempts")
    user = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)
    responses = models.JSONField(default=dict)  # {q_id: "B", ...}
    score = models.FloatField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)