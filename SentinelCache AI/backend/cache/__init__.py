# backend/cache/__init__.py
"""Cache management for ML models and API responses"""
import cachetools
from datetime import datetime, timedelta
import json
import hashlib
from typing import Any, Optional

class ModelCache:
    """Efficient caching system for models and predictions"""
    
    def __init__(self, maxsize=100, ttl_seconds=300):
        self.prediction_cache = cachetools.TTLCache(maxsize=maxsize, ttl=ttl_seconds)
        self.model_cache = {}
        self.stats_cache = cachetools.TTLCache(maxsize=50, ttl=60)
        
    def get_prediction_cache_key(self, input_text: str, model_type: str) -> str:
        """Generate cache key for prediction"""
        content = f"{model_type}:{input_text}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_cached_prediction(self, cache_key: str) -> Optional[Any]:
        """Get cached prediction if exists"""
        return self.prediction_cache.get(cache_key)
    
    def cache_prediction(self, cache_key: str, result: Any):
        """Cache prediction result"""
        self.prediction_cache[cache_key] = result
    
    def get_cache_stats(self) -> dict:
        """Get cache statistics"""
        return {
            "prediction_cache_size": len(self.prediction_cache),
            "model_cache_size": len(self.model_cache),
            "stats_cache_size": len(self.stats_cache),
            "max_cache_size": self.prediction_cache.maxsize,
            "ttl_seconds": self.prediction_cache.ttl
        }
    
    def clear_all(self):
        """Clear all caches"""
        self.prediction_cache.clear()
        self.model_cache.clear()
        self.stats_cache.clear()

# Global cache instance
cache_manager = ModelCache()