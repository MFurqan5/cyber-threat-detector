"""
seed_db.py  —  Populates ALL three databases with identical dummy data.

HOW TO RUN:
    cd backend
    python seed_db.py

WHAT IT DOES:
    1. Inserts 2 dummy users into PostgreSQL
    2. Inserts 5 dummy scan requests (3 URLs, 2 emails) into PostgreSQL
    3. Inserts matching AI predictions into PostgreSQL
    4. Inserts matching threat logs into PostgreSQL
    5. Mirrors the 5 results into MongoDB cache (with TTL)
    6. Mirrors the 5 results into Redis cache (with TTL)

WHY: Your local Docker databases start EMPTY. This gives every team
     member the same data so the frontend and dashboard work immediately.

TO ADD NEW TEST DATA:
    Add a new entry to SEED_DATA below and re-run this script.
"""

import os
import json
import hashlib
import redis
import psycopg2
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

# ── Connection strings from .env ─────────────────────────────
POSTGRES_URL = os.getenv("POSTGRES_URL")
REDIS_URL    = os.getenv("REDIS_URL")
MONGO_URL    = os.getenv("MONGO_URL")

# ── Seed data — add more entries here as needed ──────────────
SEED_DATA = [
    {
        "user_id":    "22222222-2222-2222-2222-222222222222",
        "input_type": "url",
        "input_value": "http://paypal-secure-login.xyz/verify",
        "prediction_label": "malicious",
        "threat_type": "phishing",
        "confidence_score": 0.97,
        "explanation": "Domain impersonates PayPal. Unregistered TLD. Login form detected.",
        "indicators": ["suspicious_tld", "login_form_detected", "brand_impersonation"],
        "model_version": "rf-url-v2.1",
        "inference_ms": 82,
        "severity": "high",
        "action_taken": "blocked",
    },
    {
        "user_id":    "22222222-2222-2222-2222-222222222222",
        "input_type": "url",
        "input_value": "http://amaz0n-free-prize.net/claim",
        "prediction_label": "malicious",
        "threat_type": "phishing",
        "confidence_score": 0.99,
        "explanation": "Character substitution (0 for o) in domain. Prize scam keywords detected.",
        "indicators": ["character_substitution", "prize_keywords", "suspicious_tld"],
        "model_version": "rf-url-v2.1",
        "inference_ms": 78,
        "severity": "critical",
        "action_taken": "blocked",
    },
    {
        "user_id":    "11111111-1111-1111-1111-111111111111",
        "input_type": "url",
        "input_value": "https://google.com",
        "prediction_label": "safe",
        "threat_type": "clean",
        "confidence_score": 0.99,
        "explanation": "Known safe domain. Valid HTTPS. No suspicious indicators.",
        "indicators": [],
        "model_version": "rf-url-v2.1",
        "inference_ms": 71,
        "severity": "low",
        "action_taken": "none",
    },
    {
        "user_id":    "22222222-2222-2222-2222-222222222222",
        "input_type": "email",
        "input_value": "CONGRATULATIONS! You have won $1,000,000. Click here NOW to claim your prize before it expires!",
        "prediction_label": "malicious",
        "threat_type": "spam",
        "confidence_score": 0.96,
        "explanation": "Prize-winning language. Urgency keywords. Emotional manipulation detected.",
        "indicators": ["prize_language", "urgency_keywords", "emotional_manipulation"],
        "model_version": "nb-email-v1.3",
        "inference_ms": 12,
        "severity": "medium",
        "action_taken": "flagged",
    },
    {
        "user_id":    "11111111-1111-1111-1111-111111111111",
        "input_type": "email",
        "input_value": "Hi team, the standup meeting is moved to 10am tomorrow. Please update your calendars.",
        "prediction_label": "safe",
        "threat_type": "clean",
        "confidence_score": 0.98,
        "explanation": "Normal professional communication. No suspicious patterns.",
        "indicators": [],
        "model_version": "nb-email-v1.3",
        "inference_ms": 9,
        "severity": "low",
        "action_taken": "none",
    },
]

DUMMY_USERS = [
    {
        "id": "11111111-1111-1111-1111-111111111111",
        "email": "analyst@securescan.com",
        "username": "admin_alice",
        "password_hash": "$2a$12$placeholder_hash_alice",
        "role": "analyst",
    },
    {
        "id": "22222222-2222-2222-2222-222222222222",
        "email": "user@company.com",
        "username": "bob_user",
        "password_hash": "$2a$12$placeholder_hash_bob",
        "role": "user",
    },
]

def make_hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()

# ── 1. Seed PostgreSQL ────────────────────────────────────────
def seed_postgres():
    print("\n[1/3] Seeding PostgreSQL...")
    conn = psycopg2.connect(POSTGRES_URL)
    cur  = conn.cursor()

    # Insert users
    for u in DUMMY_USERS:
        cur.execute("""
            INSERT INTO users (id, email, username, password_hash, role)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (email) DO NOTHING
        """, (u["id"], u["email"], u["username"], u["password_hash"], u["role"]))

    # Insert scan requests, predictions, and logs
    for i, item in enumerate(SEED_DATA):
        request_id    = f"aaaa{i:04d}-0000-0000-0000-000000000000"
        prediction_id = f"bbbb{i:04d}-0000-0000-0000-000000000000"
        input_hash    = make_hash(item["input_value"])

        cur.execute("""
            INSERT INTO scan_requests (id, user_id, input_type, input_value, input_hash, status)
            VALUES (%s, %s, %s, %s, %s, 'complete')
            ON CONFLICT (input_hash) DO NOTHING
        """, (request_id, item["user_id"], item["input_type"], item["input_value"], input_hash))

        cur.execute("""
            INSERT INTO ai_predictions
                (id, request_id, prediction_label, threat_type, confidence_score,
                 explanation, indicators, model_version, inference_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
            ON CONFLICT (request_id) DO NOTHING
        """, (
            prediction_id, request_id,
            item["prediction_label"], item["threat_type"], item["confidence_score"],
            item["explanation"], json.dumps(item["indicators"]),
            item["model_version"], item["inference_ms"]
        ))

        cur.execute("""
            INSERT INTO threat_logs (prediction_id, severity, action_taken, notes)
            VALUES (%s, %s, %s, %s)
        """, (prediction_id, item["severity"], item["action_taken"],
              f"Seeded by seed_db.py — {item['threat_type']}"))

    conn.commit()
    cur.close()
    conn.close()
    print(f"   Done — inserted {len(DUMMY_USERS)} users and {len(SEED_DATA)} scan records.")

# ── 2. Seed MongoDB ───────────────────────────────────────────
def seed_mongodb():
    print("\n[2/3] Seeding MongoDB (L3 Cache)...")
    client = MongoClient(MONGO_URL)
    db     = client["Cache_db"]
    col    = db["cache_results"]

    # Ensure indexes exist
    col.create_index("cache_key", unique=True)
    col.create_index("expires_at", expireAfterSeconds=0)
    col.create_index("input_hash")

    count = 0
    for item in SEED_DATA:
        input_hash = make_hash(item["input_value"])
        doc = {
            "cache_key":  input_hash,
            "input_type": item["input_type"],
            "input_hash": input_hash,
            "result": {
                "prediction_label": item["prediction_label"],
                "threat_type":      item["threat_type"],
                "confidence_score": item["confidence_score"],
                "explanation":      item["explanation"],
                "indicators":       item["indicators"],
            },
            "model_version": item["model_version"],
            "hit_count":     0,
            "created_at":    datetime.utcnow(),
            "expires_at":    datetime.utcnow() + timedelta(hours=24),
        }
        try:
            col.insert_one(doc)
            count += 1
        except Exception:
            pass  # Already exists — skip

    client.close()
    print(f"   Done — inserted {count} documents into MongoDB cache_results.")

# ── 3. Seed Redis ─────────────────────────────────────────────
def seed_redis():
    print("\n[3/3] Seeding Redis (L2 Cache)...")
    r = redis.from_url(REDIS_URL, decode_responses=True)

    count = 0
    for item in SEED_DATA:
        input_hash = make_hash(item["input_value"])
        cache_key  = f"threat:v1:{input_hash}"
        value = json.dumps({
            "label":      item["prediction_label"],
            "type":       item["threat_type"],
            "score":      item["confidence_score"],
            "indicators": item["indicators"],
        })
        r.setex(cache_key, 3600, value)  # Expires in 1 hour
        count += 1

    print(f"   Done — inserted {count} keys into Redis.")

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  Cyber Threat Detector — Database Seed Script")
    print("=" * 55)
    print("  Make sure Docker is running:  docker-compose up -d")
    print("  Then wait 10 seconds before running this script.")
    print("=" * 55)

    try:
        seed_postgres()
    except Exception as e:
        print(f"   ERROR (PostgreSQL): {e}")

    try:
        seed_mongodb()
    except Exception as e:
        print(f"   ERROR (MongoDB): {e}")

    try:
        seed_redis()
    except Exception as e:
        print(f"   ERROR (Redis): {e}")

    print("\n" + "=" * 55)
    print("  All done! Your databases are ready.")
    print("  Start the backend:  uvicorn main:app --reload")
    print("=" * 55)