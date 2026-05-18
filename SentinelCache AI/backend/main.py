# backend/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import time
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from backend.routes import scan, stats
from backend.db import db
from backend.cache import cache_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting SENTINELCACHE AI Backend...")
    
    # Ensure directories exist
    Path("backend/models").mkdir(parents=True, exist_ok=True)
    Path("backend/data").mkdir(parents=True, exist_ok=True)
    
    # Initialize database
    logger.info("📦 Initializing database...")
    db.init_db()
    
    logger.info("✅ SENTINELCACHE AI is ready!")
    logger.info(f"📊 Cache size: {cache_manager.get_cache_stats()['prediction_cache_size']}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down SENTINELCACHE AI...")

# Create FastAPI app
app = FastAPI(
    title="SENTINELCACHE AI - ML Security API",
    description="Advanced phishing detection API using Machine Learning",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
    response.headers["X-Cache-Status"] = "MISS"
    return response

# Exception handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Validation error",
            "errors": exc.errors(),
            "body": exc.body
        }
    )

# Include routers
app.include_router(scan.router)
app.include_router(stats.router)

# Root endpoint
@app.get("/")
async def root():
    return {
        "name": "SENTINELCACHE AI",
        "version": "1.0.0",
        "status": "operational",
        "description": "ML-powered phishing detection API",
        "endpoints": {
            "scan_url": "POST /scan/url",
            "scan_email": "POST /scan/email",
            "scan_app_name": "POST /scan/app-name",
            "scan_app_apk": "POST /scan/app",
            "history": "GET /stats/history",
            "summary": "GET /stats/summary",
            "models_info": "GET /stats/models/info",
            "cache_status": "GET /stats/cache/status",
            "performance": "GET /stats/performance",
            "docs": "/docs",
            "health": "/health"
        }
    }

# Health check
@app.get("/health")
async def health_check():
    from pathlib import Path
    
    models_path = Path("backend/models")
    url_model = (models_path / "url_model.pkl").exists()
    email_model = (models_path / "email_model.pkl").exists()
    app_model = (models_path / "app_model.pkl").exists()
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "database": "connected" if db.db_path.exists() else "initializing",
            "cache": "active",
            "models": {
                "url_model": url_model,
                "email_model": email_model,
                "app_model": app_model
            }
        },
        "cache_stats": cache_manager.get_cache_stats()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )