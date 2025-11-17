from django.conf import settings
from supabase import create_client, Client
import requests
import json


class QueryInterface:
    def __init__(self):
        self.env = getattr(settings, "ENVIRONMENT", "development")

        if self.env == "production":
            self.supabase: Client = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_KEY
            )
            self.rest_url = f"{settings.SUPABASE_URL}/rest/v1"
            self.headers = {
                "apikey": settings.SUPABASE_KEY,
                "Authorization": f"Bearer {settings.SUPABASE_KEY}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            }

        # 🔹 Insert one record
    def insert(self, model_name: str, data: dict):
        if self.env == "production":
            # Supabase table names are lowercase by default
            table_name = model_name.lower()
            response = self.supabase.table(table_name).insert(data).execute()
            return response.data
        else:
            model = self._get_model(model_name)
            return model.objects.create(**data)

    def bulk_insert(self, model_name: str, records: list[dict]):
        if self.env == "production":
            table_name = model_name.lower()
            response = self.supabase.table(table_name).insert(records).execute()
            return response.data
        else:
            model = self._get_model(model_name)
            objs = [model(**r) for r in records]
            model.objects.bulk_create(objs)
            return objs


    # 🔹 Select (fetch)
    def select(self, model_name: str, filters: dict = None):
        if self.env == "production":
            table_name = model_name.lower()
            query = self.supabase.table(table_name).select("*")
            if filters:
                for key, value in filters.items():
                    if isinstance(value, list):
                        # Use `in.` for lists
                        value_str = ",".join(map(str, value))
                        query = query.in_(key, value)
                    else:
                        query = query.eq(key, value)
            response = query.execute()
            return response.data
        else:
            model = self._get_model(model_name)
            qs = model.objects.filter(**filters) if filters else model.objects.all()
            return list(qs.values())


    # 🔹 Update
    def update(self, model_name: str, filters: dict, data: dict):
        if self.env == "production":
            query = self.supabase.table(model_name).update(data)
            for key, value in filters.items():
                query = query.eq(key, value)
            response = query.execute()
            return response.data
        else:
            model = self._get_model(model_name)
            return model.objects.filter(**filters).update(**data)

    # 🔹 Delete
    def delete(self, model_name: str, filters: dict):
        if self.env == "production":
            query = self.supabase.table(model_name).delete()
            for key, value in filters.items():
                query = query.eq(key, value)
            response = query.execute()
            return response.data
        else:
            model = self._get_model(model_name)
            return model.objects.filter(**filters).delete()

    # 🔹 Vector Search (REST API - pgvector)
    def vector_search(self, table: str = "DocumentChunk", embedding=None, document_ids=None, top_k=5):
        """
        Perform vector similarity search via Supabase REST API.
        Uses cosine similarity with pgvector column `embedding`.

        Args:
            table (str): Supabase table to query
            embedding (list[float]): Query embedding
            document_ids (list[int]): Optional filter by document IDs
            top_k (int): Number of results
        """
        if self.env == "production":
            query_params = {
                "select": "text,document_id,distance:embedding<->{}".format(json.dumps(embedding))
            }

            if document_ids:
                query_params["document_id"] = "in.({})".format(",".join(map(str, document_ids)))

            url = f"{self.rest_url}/{table}?order=distance.asc&limit={top_k}"
            response = requests.get(url, headers=self.headers, params=query_params)

            if response.status_code != 200:
                raise Exception(f"Supabase vector search failed: {response.text}")

            return response.json()
        else:
            # Local fallback
            from .models import DocumentChunk
            from pgvector.django import CosineDistance

            qs = DocumentChunk.objects.all()
            if document_ids:
                qs = qs.filter(document_id__in=document_ids)
            results = (
                qs.annotate(distance=CosineDistance("embedding", embedding))
                .order_by("distance")[:top_k]
            )
            return [
                {"text": r.text, "distance": float(r.distance), "document_id": r.document_id}
                for r in results
            ]

    # Internal helper for local ORM
    def _get_model(self, model_name: str):
        from django.apps import apps
        app_label = getattr(settings, "DEFAULT_APP_LABEL", "tutor")
        # Convert camel-case to snake_case for Supabase compatibility if needed
        normalized_name = model_name.lower()
        return apps.get_model(app_label, normalized_name)

