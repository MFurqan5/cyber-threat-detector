# backend/routes/stats.py
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import json
import logging

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from backend.db.ml_integration import ml_db

router = APIRouter(prefix="/stats", tags=["stats"])

@router.get("/history")
async def get_history(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    scan_type: Optional[str] = Query(None, pattern="^(url|email)$"),
    malicious_only: bool = False
):
    """Get scan history from PostgreSQL"""
    try:
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
                "input": row[2][:200] if row[2] else "",
                "timestamp": row[3].isoformat() if row[3] else None,
                "prediction": row[4],
                "threat_type": row[5],
                "confidence": row[6] if row[6] else 0,
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
    except Exception as e:
        logger.error(f"Error in get_history: {e}")
        return {"total": 0, "scans": [], "error": str(e)}

@router.get("/summary")
async def get_summary(hours: int = Query(24, ge=1, le=720)):
    """Get summary statistics from PostgreSQL"""
    try:
        conn = ml_db.get_postgres_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT 
                COUNT(*) as total_scans,
                SUM(CASE WHEN input_type = 'url' THEN 1 ELSE 0 END) as url_scans,
                SUM(CASE WHEN input_type = 'email' THEN 1 ELSE 0 END) as email_scans,
                SUM(CASE WHEN ap.prediction_label = 'malicious' THEN 1 ELSE 0 END) as malicious_total,
                AVG(ap.confidence_score) as avg_confidence,
                AVG(ap.inference_ms) as avg_time
            FROM scan_requests sr
            LEFT JOIN ai_predictions ap ON sr.id = ap.request_id
            WHERE sr.created_at >= NOW() - INTERVAL '%s hours'
        """, (hours,))
        
        row = cur.fetchone()
        cur.close()
        
        total = row[0] or 0
        malicious = row[3] or 0
        
        return {
            "period_hours": hours,
            "total_scans": total,
            "by_type": {
                "url": row[1] or 0,
                "email": row[2] or 0
            },
            "threats_detected": malicious,
            "safe_requests": total - malicious,
            "malicious_total": malicious,
            "avg_confidence": round((row[4] or 0) * 100, 2),
            "avg_time_ms": round(row[5] or 0, 2),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in get_summary: {e}")
        return {
            "period_hours": hours,
            "total_scans": 0,
            "by_type": {"url": 0, "email": 0},
            "threats_detected": 0,
            "safe_requests": 0,
            "avg_confidence": 0,
            "timestamp": datetime.now().isoformat()
        }

@router.get("/cache/status")
async def get_cache_status():
    """Get REAL cache status with statistics"""
    try:
        # Get real cache stats from ml_db
        stats = ml_db.get_cache_stats()
        
        return {
            "l1": {
                "hit_rate": stats["l1"]["hit_rate"],
                "hits": stats["l1"]["hits"],
                "misses": stats["l1"]["misses"]
            },
            "l2": {
                "hit_rate": stats["l2"]["hit_rate"],
                "hits": stats["l2"]["keys"],
                "misses": stats["l2"]["misses"]
            },
            "l3": {
                "hit_rate": stats["l3"]["hit_rate"],
                "hits": stats["l3"]["documents"],
                "misses": stats["l3"]["misses"]
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Cache status error: {e}")
        return {
            "l1": {"hit_rate": 0, "hits": 0, "misses": 0},
            "l2": {"hit_rate": 0, "hits": 0, "misses": 0},
            "l3": {"hit_rate": 0, "hits": 0, "misses": 0},
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        }