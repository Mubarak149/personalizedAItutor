# app/serializers.py
from rest_framework import serializers
from .models import Document


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ["id", "title", "source", "uploaded_file", "content", 'date_uploaded']
        read_only_fields = ["id", "content", "date_uploaded"]
