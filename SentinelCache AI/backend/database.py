# backend/db.py
"""SQLite database for persistent storage"""
import sqlite3
from datetime import datetime
from pathlib import Path
import json
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

DB_PATH = Path("backend/data/sentinelcache.db")

class Database:
    """Database manager for scan history and analytics"""
    
    def __init__(self):
        self.db_path = DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()
    
    def get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_db(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Scans table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_type TEXT NOT NULL,
                    input_hash TEXT NOT NULL,
                    input_preview TEXT,
                    is_malicious INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    prediction_time_ms REAL NOT NULL,
                    features_used TEXT,
                    session_id TEXT,
                    timestamp TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON scans(timestamp DESC)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_scan_type 
                ON scans(scan_type)
            """)
            
            # Models table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS models (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_name TEXT UNIQUE NOT NULL,
                    model_version TEXT NOT NULL,
                    file_path TEXT,
                    loaded_at TEXT,
                    is_active INTEGER DEFAULT 1
                )
            """)
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def save_scan(self, scan_data: dict):
        """Save scan result to database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO scans (
                    scan_type, input_hash, input_preview, is_malicious,
                    confidence, prediction_time_ms, features_used, 
                    session_id, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                scan_data['scan_type'],
                scan_data.get('input_hash', ''),
                scan_data.get('input_preview', '')[:200],
                1 if scan_data['is_malicious'] else 0,
                scan_data['confidence'],
                scan_data['prediction_time_ms'],
                json.dumps(scan_data.get('features_used', [])),
                scan_data.get('session_id'),
                scan_data.get('timestamp', datetime.now().isoformat())
            ))
            conn.commit()
            return cursor.lastrowid
    
    def get_scans(self, limit: int = 100, offset: int = 0, 
                  scan_type: Optional[str] = None, 
                  malicious_only: bool = False) -> List[dict]:
        """Get scans with filters"""
        query = "SELECT * FROM scans WHERE 1=1"
        params = []
        
        if scan_type:
            query += " AND scan_type = ?"
            params.append(scan_type)
        
        if malicious_only:
            query += " AND is_malicious = 1"
        
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    
    def get_stats(self, hours: int = 24) -> dict:
        """Get aggregated statistics"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Total scans in timeframe
            cursor.execute("""
                SELECT COUNT(*) as total,
                       SUM(CASE WHEN scan_type = 'url' THEN 1 ELSE 0 END) as url_count,
                       SUM(CASE WHEN scan_type = 'email' THEN 1 ELSE 0 END) as email_count,
                       SUM(CASE WHEN is_malicious = 1 THEN 1 ELSE 0 END) as malicious_total,
                       AVG(confidence) as avg_confidence,
                       AVG(prediction_time_ms) as avg_time
                FROM scans 
                WHERE julianday('now') - julianday(timestamp) <= ?
            """, (hours / 24.0,))
            
            stats = dict(cursor.fetchone())
            
            # Get additional metrics
            cursor.execute("""
                SELECT scan_type, 
                       COUNT(*) as count,
                       SUM(CASE WHEN is_malicious = 1 THEN 1 ELSE 0 END) as malicious_count
                FROM scans
                WHERE julianday('now') - julianday(timestamp) <= ?
                GROUP BY scan_type
            """, (hours / 24.0,))
            
            by_type = {}
            for row in cursor.fetchall():
                by_type[row['scan_type']] = {
                    "total": row['count'],
                    "malicious": row['malicious_count']
                }
            
            return {
                "period_hours": hours,
                "total_scans": stats.get('total', 0),
                "by_type": by_type,
                "malicious_total": stats.get('malicious_total', 0),
                "avg_confidence": round((stats.get('avg_confidence') or 0) * 100, 2),
                "avg_prediction_time_ms": round(stats.get('avg_time') or 0, 2)
            }
    
    def log_model_load(self, model_name: str, version: str = "1.0.0"):
        """Log model loading event"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO models 
                (model_name, model_version, loaded_at, is_active)
                VALUES (?, ?, ?, 1)
            """, (model_name, version, datetime.now().isoformat()))
            conn.commit()

# Global database instance
db = Database()