from django.conf import settings
from django.core.cache import cache
from supabase import create_client, Client
import requests
import json
import hashlib
import time

class QueryInterface:
    def __init__(self):
        self.env = getattr(settings, "ENVIRONMENT", "development")
        self.cache_enabled = getattr(settings, "CACHE_ENABLED", True)
        self.default_cache_timeout = getattr(settings, "CACHE_TIMEOUT", 3600)  # 1 hour default

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

    def select(self, model_name: str, filters: dict = None, cache_timeout: int = None):
        """
        Enhanced select with caching support and versioning
        """
        if cache_timeout is None:
            cache_timeout = self.default_cache_timeout
        
        # Generate cache key with model version
        cache_key = self._generate_cache_key("select", model_name, filters)
        
        # Try cache first if enabled
        if self.cache_enabled:
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Execute query
        if self.env == "production":
            table_name = model_name.lower()
            query = self.supabase.table(table_name).select("*")
            if filters:
                for key, value in filters.items():
                    if isinstance(value, list):
                        query = query.in_(key, value)
                    else:
                        query = query.eq(key, value)
            response = query.execute()
            result = response.data
        else:
            model = self._get_model(model_name)
            qs = model.objects.filter(**filters) if filters else model.objects.all()
            result = list(qs.values())
        
        # Store in cache if enabled
        if self.cache_enabled:
            cache.set(cache_key, result, timeout=cache_timeout)
        
        return result

    def insert(self, model_name: str, data: dict):
        """
        Insert with cache invalidation using versioning
        """
        # Invalidate cache BEFORE insert to ensure consistency
        if self.cache_enabled:
            self._invalidate_model_caches(model_name)
        
        result = self._execute_insert(model_name, data)
            
        return result

    def bulk_insert(self, model_name: str, records: list[dict]):
        """
        Bulk insert with cache invalidation
        """
        # Invalidate cache BEFORE bulk insert
        if self.cache_enabled:
            self._invalidate_model_caches(model_name)
            
        result = self._execute_bulk_insert(model_name, records)
            
        return result

    def update(self, model_name: str, filters: dict, data: dict):
        """
        Update with cache invalidation
        """
        # Invalidate cache BEFORE update
        if self.cache_enabled:
            self._invalidate_model_caches(model_name)
            self._invalidate_filter_caches(model_name, filters)
            
        result = self._execute_update(model_name, filters, data)
            
        return result

    def delete(self, model_name: str, filters: dict):
        """
        Delete with cache invalidation
        """
        # Invalidate cache BEFORE delete
        if self.cache_enabled:
            self._invalidate_model_caches(model_name)
            self._invalidate_filter_caches(model_name, filters)
            
        result = self._execute_delete(model_name, filters)
            
        return result

    # Private methods for actual execution (to avoid recursion)
    def _execute_insert(self, model_name: str, data: dict):
        if self.env == "production":
            table_name = model_name.lower()
            response = self.supabase.table(table_name).insert(data).execute()
            return response.data
        else:
            model = self._get_model(model_name)
            return model.objects.create(**data)

    def _execute_bulk_insert(self, model_name: str, records: list[dict]):
        if self.env == "production":
            table_name = model_name.lower()
            response = self.supabase.table(table_name).insert(records).execute()
            return response.data
        else:
            model = self._get_model(model_name)
            objs = [model(**r) for r in records]
            model.objects.bulk_create(objs)
            return objs

    def _execute_update(self, model_name: str, filters: dict, data: dict):
        if self.env == "production":
            query = self.supabase.table(model_name).update(data)
            for key, value in filters.items():
                query = query.eq(key, value)
            response = query.execute()
            return response.data
        else:
            model = self._get_model(model_name)
            return model.objects.filter(**filters).update(**data)

    def _execute_delete(self, model_name: str, filters: dict):
        if self.env == "production":
            query = self.supabase.table(model_name).delete()
            for key, value in filters.items():
                query = query.eq(key, value)
            response = query.execute()
            return response.data
        else:
            model = self._get_model(model_name)
            return model.objects.filter(**filters).delete()

    # Cache management methods with versioning
    def _generate_cache_key(self, operation: str, model_name: str, filters: dict = None) -> str:
        """
        Generate unique cache key that includes model version for automatic invalidation
        """
        key_parts = [operation, model_name.lower()]
        
        # Add model version to cache key - this ensures cache invalidation when model changes
        model_version = self._get_model_version(model_name)
        key_parts.append(f"v{model_version}")
        
        if filters:
            # Sort filters to ensure consistent keys
            sorted_filters = json.dumps(filters, sort_keys=True)
            # Create hash for long filter strings
            filter_hash = hashlib.md5(sorted_filters.encode()).hexdigest()[:8]
            key_parts.append(filter_hash)
        
        return "query_interface_" + "_".join(key_parts)

    def _get_model_version(self, model_name: str) -> int:
        """
        Get current version for a model. Version increments on every insert/update/delete.
        """
        version_key = f"model_version_{model_name.lower()}"
        current_version = cache.get(version_key, 1)
        return current_version

    def _invalidate_model_caches(self, model_name: str):
        """
        Invalidate all caches for a specific model by incrementing version
        """
        version_key = f"model_version_{model_name.lower()}"
        current_version = cache.get(version_key, 1)
        # Increment version - this automatically invalidates all existing cache keys
        cache.set(version_key, current_version + 1, timeout=None)  # Never expire

    def _invalidate_filter_caches(self, model_name: str, filters: dict):
        """
        Invalidate specific filter caches (additional to versioning)
        """
        if filters:
            # Also delete specific filter cache immediately
            cache_key = self._generate_cache_key("select", model_name, filters)
            cache.delete(cache_key)

    def clear_cache(self, model_name: str = None):
        """
        Clear cache manually - useful for admin actions
        """
        if model_name:
            # Reset model version to 1, invalidating all caches for this model
            version_key = f"model_version_{model_name.lower()}"
            cache.set(version_key, 1, timeout=None)
        else:
            # Clear entire cache
            cache.clear()

    # Vector Search (with caching option)
    def vector_search(self, table: str = "DocumentChunk", embedding=None, document_ids=None, top_k=5, use_cache: bool = True):
        """
        Vector search with optional caching
        """
        if use_cache and self.cache_enabled:
            cache_key = self._generate_cache_key(
                "vector_search", 
                table, 
                {
                    "embedding_hash": hashlib.md5(json.dumps(embedding).encode()).hexdigest()[:12], 
                    "document_ids": document_ids, 
                    "top_k": top_k
                }
            )
            
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result

        # Execute vector search
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

            result = response.json()
        else:
            from .models import DocumentChunk
            from pgvector.django import CosineDistance

            qs = DocumentChunk.objects.all()
            if document_ids:
                qs = qs.filter(document_id__in=document_ids)
            results = (
                qs.annotate(distance=CosineDistance("embedding", embedding))
                .order_by("distance")[:top_k]
            )
            result = [
                {"text": r.text, "distance": float(r.distance), "document_id": r.document_id}
                for r in results
            ]

        # Cache the result
        if use_cache and self.cache_enabled:
            # Shorter timeout for vector searches as they might change more frequently
            cache.set(cache_key, result, timeout=1800)  # 30 minutes

        return result

    # Internal helper for local ORM
    def _get_model(self, model_name: str):
        from django.apps import apps
        app_label = getattr(settings, "DEFAULT_APP_LABEL", "tutor")
        normalized_name = model_name.lower()
        return apps.get_model(app_label, normalized_name)