# backend/routes/stats.py - UPDATED to work with your databases
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import json

from backend.db.ml_integration import ml_db

router = APIRouter(prefix="/stats", tags=["stats"])

@router.get("/history")
async def get_history(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
# Replace with this:
    scan_type: Optional[str] = Query(None, pattern="^(url|email)$"),    malicious_only: bool = False
):
    """Get scan history from PostgreSQL"""
    conn = ml_db.get_postgres_connection()
    cur = conn.cursor()
    
    query = """
        SELECT sr.id, sr.input_type, sr.input_value, sr.created_at,
               ap.prediction_label, ap.threat_type, ap.confidence_score, 
               ap.explanation, ap.model_version, ap.inference_ms
        FROM scan_requests sr
        LEFT JOIN ai_predictions ap ON sr.id = ap.request_id
        WHERE 1=1
    """
    params = []
    
    if scan_type:
        query += " AND sr.input_type = %s"
        params.append(scan_type)
    
    if malicious_only:
        query += " AND ap.prediction_label = 'malicious'"
    
    query += " ORDER BY sr.created_at DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])
    
    cur.execute(query, params)
    rows = cur.fetchall()
    cur.close()
    
    scans = []
    for row in rows:
        scans.append({
            "id": row[0],
            "type": row[1],
            "input": row[2][:200],
            "timestamp": row[3].isoformat() if row[3] else None,
            "prediction": row[4],
            "threat_type": row[5],
            "confidence": row[6],
            "explanation": row[7],
            "model": row[8],
            "inference_ms": row[9]
        })
    
    return {
        "total": len(scans),
        "limit": limit,
        "offset": offset,
        "scans": scans
    }

@router.get("/summary")
async def get_summary(hours: int = Query(24, ge=1, le=720)):
    """Get summary statistics from PostgreSQL"""
    conn = ml_db.get_postgres_connection()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT 
            COUNT(*) as total_scans,
            SUM(CASE WHEN input_type = 'url' THEN 1 ELSE 0 END) as url_scans,
            SUM(CASE WHEN input_type = 'email' THEN 1 ELSE 0 END) as email_scans,
            SUM(CASE WHEN ap.prediction_label = 'malicious' THEN 1 ELSE 0 END) as malicious_total,
            AVG(ap.confidence_score) as avg_confidence
        FROM scan_requests sr
        LEFT JOIN ai_predictions ap ON sr.id = ap.request_id
        WHERE sr.created_at >= NOW() - INTERVAL '%s hours'
    """, (hours,))
    
    row = cur.fetchone()
    cur.close()
    
    return {
        "period_hours": hours,
        "total_scans": row[0] or 0,
        "by_type": {
            "url": row[1] or 0,
            "email": row[2] or 0
        },
        "malicious_total": row[3] or 0,
        "avg_confidence": round((row[4] or 0) * 100, 2),
        "timestamp": datetime.now().isoformat()
    }

@router.get("/cache/status")
async def get_cache_status():
    """Get cache status from Redis and MongoDB"""
    redis_client = ml_db.get_redis_client()
    mongo_client = ml_db.get_mongo_client()
    
    redis_info = redis_client.info()
    mongo_db = mongo_client["Cache_db"]
    cache_collection = mongo_db["cache_results"]
    
    return {
        "redis": {
            "connected": True,
            "keys": redis_info.get("db0", {}).get("keys", 0),
            "used_memory_mb": round(redis_info.get("used_memory_rss", 0) / 1024 / 1024, 2),
            "cache_keys_pattern": "threat:v1:*"
        },
        "mongodb": {
            "connected": True,
            "cached_documents": cache_collection.count_documents({}),
            "database": "Cache_db",
            "collection": "cache_results"
        },
        "timestamp": datetime.now().isoformat()
    }