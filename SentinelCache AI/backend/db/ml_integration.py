# backend/db/ml_integration.py
"""Integration layer between ML models and existing databases"""
import os
import json
import hashlib
import redis
import psycopg2
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import logging

load_dotenv()
logger = logging.getLogger(__name__)

# Database connections (reusing your existing setup)
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://postgres:password@localhost:5432/threat_db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")

class MLDatabaseIntegration:
    """Handles all database operations for ML predictions"""
    
    def __init__(self):
        self.postgres_conn = None
        self.redis_client = None
        self.mongo_client = None
        
    def get_postgres_connection(self):
        """Get PostgreSQL connection (reusing your schema)"""
        if self.postgres_conn is None or self.postgres_conn.closed:
            self.postgres_conn = psycopg2.connect(POSTGRES_URL)
        return self.postgres_conn
    
    def get_redis_client(self):
        """Get Redis client"""
        if self.redis_client is None:
            self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        return self.redis_client
    
    def get_mongo_client(self):
        """Get MongoDB client"""
        if self.mongo_client is None:
            self.mongo_client = MongoClient(MONGO_URL)
        return self.mongo_client
    
    def check_cache(self, input_value: str, input_type: str) -> Optional[Dict]:
        """Check L2 (Redis) and L3 (MongoDB) caches first"""
        
        input_hash = hashlib.sha256(input_value.encode()).hexdigest()
        
        # Check Redis first (L2 cache)
        redis_client = self.get_redis_client()
        cache_key = f"threat:v1:{input_hash}"
        cached = redis_client.get(cache_key)
        
        if cached:
            logger.info(f"Redis cache hit for {input_type}")
            return {
                "from_cache": "redis",
                "result": json.loads(cached),
                "input_hash": input_hash
            }
        
        # Check MongoDB (L3 cache)
        mongo_client = self.get_mongo_client()
        db = mongo_client["Cache_db"]
        collection = db["cache_results"]
        
        mongo_cached = collection.find_one({"input_hash": input_hash})
        if mongo_cached and mongo_cached.get("expires_at", datetime.utcnow()) > datetime.utcnow():
            logger.info(f"MongoDB cache hit for {input_type}")
            # Also populate Redis for next time
            redis_client.setex(cache_key, 3600, json.dumps(mongo_cached["result"]))
            return {
                "from_cache": "mongodb",
                "result": mongo_cached["result"],
                "input_hash": input_hash
            }
        
        return None
    
    def save_prediction(self, request_id: str, user_id: str, input_type: str, 
                        input_value: str, prediction: Dict, model_version: str,
                        inference_ms: float, severity: str = "medium", 
                        action_taken: str = "flagged"):
        """Save ML prediction to all three databases (matching your seed schema)"""
        
        input_hash = hashlib.sha256(input_value.encode()).hexdigest()
        conn = self.get_postgres_connection()
        cur = conn.cursor()
        
        try:
            # 1. Insert into scan_requests (matches your seed_db.py schema)
            cur.execute("""
                INSERT INTO scan_requests (id, user_id, input_type, input_value, input_hash, status, created_at)
                VALUES (%s, %s, %s, %s, %s, 'complete', %s)
                ON CONFLICT (input_hash) DO UPDATE SET 
                    status = 'complete',
                    updated_at = %s
                RETURNING id
            """, (
                request_id, user_id, input_type, input_value, input_hash, 
                datetime.utcnow(), datetime.utcnow()
            ))
            
            # Get or use existing request_id
            result = cur.fetchone()
            final_request_id = result[0] if result else request_id
            
            # 2. Insert into ai_predictions (matches your seed_db.py schema)
            prediction_id = f"pred_{input_hash[:16]}"
            cur.execute("""
                INSERT INTO ai_predictions
                    (id, request_id, prediction_label, threat_type, confidence_score,
                     explanation, indicators, model_version, inference_ms, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
                ON CONFLICT (request_id) DO UPDATE SET
                    prediction_label = EXCLUDED.prediction_label,
                    confidence_score = EXCLUDED.confidence_score,
                    updated_at = %s
            """, (
                prediction_id, final_request_id,
                prediction.get("label", "safe"),
                prediction.get("threat_type", "clean"),
                prediction.get("confidence", 0.5),
                prediction.get("explanation", ""),
                json.dumps(prediction.get("indicators", [])),
                model_version,
                inference_ms,
                datetime.utcnow(),
                datetime.utcnow()
            ))
            
            # 3. Insert into threat_logs (matches your seed_db.py schema)
            cur.execute("""
                INSERT INTO threat_logs (prediction_id, severity, action_taken, notes, created_at)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                prediction_id, severity, action_taken,
                f"ML Prediction: {prediction.get('explanation', 'No explanation')}",
                datetime.utcnow()
            ))
            
            conn.commit()
            
            # 4. Save to Redis cache (L2)
            redis_client = self.get_redis_client()
            cache_key = f"threat:v1:{input_hash}"
            redis_data = {
                "label": prediction.get("label"),
                "type": prediction.get("threat_type"),
                "score": prediction.get("confidence"),
                "indicators": prediction.get("indicators", []),
                "model": model_version
            }
            redis_client.setex(cache_key, 3600, json.dumps(redis_data))
            
            # 5. Save to MongoDB cache (L3)
            mongo_client = self.get_mongo_client()
            db = mongo_client["Cache_db"]
            collection = db["cache_results"]
            
            mongo_doc = {
                "cache_key": input_hash,
                "input_type": input_type,
                "input_hash": input_hash,
                "result": redis_data,
                "model_version": model_version,
                "hit_count": 0,
                "created_at": datetime.utcnow(),
                "expires_at": datetime.utcnow() + timedelta(hours=24),
                "request_id": final_request_id,
                "prediction_id": prediction_id
            }
            
            collection.update_one(
                {"input_hash": input_hash},
                {"$set": mongo_doc},
                upsert=True
            )
            
            logger.info(f"Prediction saved to all 3 databases for {input_type}")
            return {
                "request_id": final_request_id,
                "prediction_id": prediction_id,
                "from_cache": False
            }
            
        except Exception as e:
            conn.rollback()
            logger.error(f"Error saving prediction: {e}")
            raise
        finally:
            cur.close()
    
    def get_user_id(self, email_or_username: str) -> Optional[str]:
        """Get user_id from existing users table"""
        conn = self.get_postgres_connection()
        cur = conn.cursor()
        
        try:
            cur.execute("""
                SELECT id FROM users 
                WHERE email = %s OR username = %s
                LIMIT 1
            """, (email_or_username, email_or_username))
            
            result = cur.fetchone()
            return result[0] if result else None
        finally:
            cur.close()

# Global instance
ml_db = MLDatabaseIntegration()