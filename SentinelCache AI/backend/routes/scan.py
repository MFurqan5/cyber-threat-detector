from fastapi import APIRouter, HTTPException, BackgroundTasks, File, UploadFile, Form
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional
import re
import hashlib
from datetime import datetime
import numpy as np
from pathlib import Path
import pickle
import tempfile
import os
import uuid
import logging

from backend.db.ml_integration import ml_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/scan", tags=["scan"])

class URLScanRequest(BaseModel):
    url: HttpUrl
    user_id: Optional[str] = None
    email: Optional[str] = None

class EmailScanRequest(BaseModel):
    email_content: str = Field(..., min_length=1, max_length=50000)
    subject: str = ""
    user_id: Optional[str] = None
    email: Optional[str] = None

class ScanResponse(BaseModel):
    is_malicious: bool
    confidence: float
    threat_type: str
    explanation: str
    indicators: List[str]
    prediction_time_ms: float
    model_version: str
    from_cache: str  
    request_id: str
    timestamp: str

def extract_url_features(url: str) -> np.ndarray:
    """Extract 25 URL features using shared feature module"""
    from backend.url_features import extract_url_features_single
    return extract_url_features_single(url)

def load_model(model_name: str):
    """Load model from pickle file"""
    model_path = Path(f"backend/models/{model_name}.pkl")
    
    if model_path.exists():
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    return None

def get_phishing_explanation(features: dict, score: float) -> tuple:
    """Generate human-readable explanation and indicators"""
    indicators = []
    explanation_parts = []
    
    if features.get('suspicious_keywords', 0) > 0.3:
        indicators.append("suspicious_keywords")
        explanation_parts.append("Contains suspicious keywords like login/verify")
    
    if features.get('has_ip', 0) == 1:
        indicators.append("ip_address_in_url")
        explanation_parts.append("IP address used instead of domain name")
    
    if features.get('has_at_symbol', 0) == 1:
        indicators.append("at_symbol_present")
        explanation_parts.append("Contains @ symbol indicating URL redirect")
    
    if features.get('url_length', 0) > 0.7:
        indicators.append("excessive_url_length")
        explanation_parts.append("Unusually long URL (potential obfuscation)")
    
    if features.get('hyphen_count', 0) > 0.3:
        indicators.append("multiple_hyphens")
        explanation_parts.append("Contains multiple hyphens in domain")
    
    if features.get('special_chars', 0) > 0.3:
        indicators.append("excessive_special_chars")
        explanation_parts.append("Unusual number of special characters")
    
    if score > 0.7:
        threat_type = "phishing"
        if not explanation_parts:
            explanation_parts = ["High probability of phishing based on multiple indicators"]
        severity_note = "HIGH RISK"
    elif score > 0.4:
        threat_type = "suspicious"
        if not explanation_parts:
            explanation_parts = ["Shows some suspicious characteristics"]
        severity_note = "MEDIUM RISK"
    else:
        threat_type = "clean"
        if not explanation_parts:
            explanation_parts = ["No obvious phishing indicators detected"]
        severity_note = "LOW RISK"
    
    explanation = f"{severity_note}: " + " ; ".join(explanation_parts) if explanation_parts else "No specific indicators found"
    
    return threat_type, explanation, indicators

def get_email_explanation(email_text: str, score: float) -> tuple:
    """Generate explanation for email threats"""
    indicators = []
    explanation_parts = []
    
    suspicious_phrases = {
        'prize': 'prize-winning language',
        'winner': 'claims of winning',
        'urgent': 'urgency pressure tactics',
        'verify': 'account verification request',
        'password': 'password-related request',
        'click': 'suspicious link encouragement',
        'bank': 'financial references',
        'account': 'account manipulation',
        'security': 'security alert manipulation',
        'congratul': 'congratulatory scam language',
        'confirm': 'confirmation request',
        'suspended': 'account suspension threat'
    }
    
    for phrase, indicator in suspicious_phrases.items():
        if phrase in email_text.lower():
            indicator_clean = indicator.replace(' ', '_').replace('-', '_')
            indicators.append(indicator_clean)
            explanation_parts.append(f"Contains {indicator}")
    
    if score > 0.7:
        threat_type = "spam_phishing"
        if not explanation_parts:
            explanation_parts = ["High probability of spam/phishing content"]
        severity_note = "HIGH RISK"
    elif score > 0.4:
        threat_type = "suspicious"
        if not explanation_parts:
            explanation_parts = ["Shows some spam-like characteristics"]
        severity_note = "MEDIUM RISK"
    elif score > 0.15:
        threat_type = "low_risk"
        explanation_parts = ["Minor suspicious elements detected"]
        severity_note = "LOW RISK"
    else:
        threat_type = "clean"
        if not explanation_parts:
            explanation_parts = ["Appears to be legitimate communication"]
        severity_note = "VERY LOW RISK"
    
    explanation = f"{severity_note}: " + " ; ".join(explanation_parts) if explanation_parts else "No specific spam indicators found"
    
    return threat_type, explanation, indicators

@router.post("/url", response_model=ScanResponse)
async def scan_url(request: URLScanRequest, background_tasks: BackgroundTasks):
    """Scan URL for phishing detection - integrates with existing database"""
    import time
    
    start_time = time.time()
    url_str = str(request.url)
    
    logger.info(f"Scanning URL: {url_str[:100]}...")
    
    user_id = request.user_id
    if not user_id and request.email:
        try:
            user_id = ml_db.get_user_id(request.email)
        except Exception as e:
            logger.warning(f"Could not get user_id: {e}")
            user_id = None
    
    if not user_id:
        user_id = "22222222-2222-2222-2222-222222222222"
    
    cached = None
    try:
        if ml_db and hasattr(ml_db, 'check_cache'):
            cached = ml_db.check_cache(url_str, "url")
        else:
            logger.warning("ml_db or check_cache not available")
    except Exception as e:
        logger.warning(f"Cache check failed (continuing without cache): {e}")
        cached = None
    
    if cached:
        try:
            logger.info(f"Cache hit for URL: {url_str[:50]} from {cached['from_cache']}")
            result = cached["result"]
            return ScanResponse(
                is_malicious=result.get("label") == "malicious",
                confidence=result.get("score", 0.5),
                threat_type=result.get("type", "unknown"),
                explanation="Cached result from previous scan",
                indicators=result.get("indicators", []),
                prediction_time_ms=0,
                model_version=result.get("model", "cached"),
                from_cache=cached["from_cache"],
                request_id=f"cached_{cached['input_hash'][:16]}",
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.warning(f"Error processing cache result: {e}")
            cached = None
    
    features_array = extract_url_features(url_str)
    from backend.url_features import FEATURE_NAMES
    features_dict = {name: float(features_array[0][i]) for i, name in enumerate(FEATURE_NAMES)}
    
    model = load_model("url_model")
    
    if model is None:
        score = float(np.mean(features_array[0]))
        prediction = 1 if score > 0.3 else 0
        logger.info(f"Using fallback prediction (model not loaded): score={score:.3f}, prediction={prediction}")
    else:
        try:
            prediction = int(model.predict(features_array)[0])
            score = float(max(model.predict_proba(features_array)[0]))
            logger.info(f"Model prediction: score={score:.3f}, prediction={prediction}")
        except Exception as e:
            logger.error(f"Model prediction failed: {e}")
            score = 0.3
            prediction = 0
    
    threat_type, explanation, indicators = get_phishing_explanation(features_dict, score)
    
    prediction_time = (time.time() - start_time) * 1000
    request_id = str(uuid.uuid4())
    
    prediction_data = {
        "label": "malicious" if prediction == 1 else "safe",
        "threat_type": threat_type,
        "confidence": score,
        "explanation": explanation,
        "indicators": indicators
    }
    
    if score > 0.8:
        severity = "critical"
        action = "blocked"
    elif score > 0.6:
        severity = "high"
        action = "blocked"
    elif score > 0.4:
        severity = "medium"
        action = "flagged"
    else:
        severity = "low"
        action = "none"
    
    try:
        if ml_db and hasattr(ml_db, 'save_prediction'):
            background_tasks.add_task(
                ml_db.save_prediction,
                request_id, user_id, "url", url_str, prediction_data,
                "rf-url-v2.1-test", prediction_time, severity, action
            )
        else:
            logger.warning("ml_db or save_prediction not available, skipping save")
    except Exception as e:
        logger.error(f"Failed to schedule save to database: {e}")
    
    logger.info(f"URL scan completed: malicious={prediction}, confidence={score:.3f}, time={prediction_time:.2f}ms")
    
    return ScanResponse(
        is_malicious=bool(prediction),
        confidence=round(score, 4),
        threat_type=threat_type,
        explanation=explanation,
        indicators=indicators,
        prediction_time_ms=round(prediction_time, 2),
        model_version="rf-url-v2.1-test",
        from_cache="none",
        request_id=request_id,
        timestamp=datetime.now().isoformat()
    )

@router.post("/email", response_model=ScanResponse)
async def scan_email(request: EmailScanRequest, background_tasks: BackgroundTasks):
    """Scan email content - integrates with existing database"""
    import time
    
    start_time = time.time()
    email_text = f"{request.subject} {request.email_content}"
    email_preview = email_text[:200]
    
    logger.info(f"Scanning email: '{email_preview[:50]}...'")
    
    user_id = request.user_id
    if not user_id and request.email:
        try:
            user_id = ml_db.get_user_id(request.email)
        except Exception as e:
            logger.warning(f"Could not get user_id: {e}")
            user_id = None
    
    if not user_id:
        user_id = "22222222-2222-2222-2222-222222222222"
    
    cached = None
    try:
        if ml_db and hasattr(ml_db, 'check_cache'):
            cached = ml_db.check_cache(email_preview, "email")
        else:
            logger.warning("ml_db or check_cache not available")
    except Exception as e:
        logger.warning(f"Cache check failed (continuing without cache): {e}")
        cached = None
    
    if cached:
        try:
            logger.info(f"Cache hit for email from {cached['from_cache']}")
            result = cached["result"]
            return ScanResponse(
                is_malicious=result.get("label") == "malicious",
                confidence=result.get("score", 0.5),
                threat_type=result.get("type", "unknown"),
                explanation="Cached result from previous scan",
                indicators=result.get("indicators", []),
                prediction_time_ms=0,
                model_version=result.get("model", "cached"),
                from_cache=cached["from_cache"],
                request_id=f"cached_{cached['input_hash'][:16]}",
                timestamp=datetime.now().isoformat()
            )
        except Exception as e:
            logger.warning(f"Error processing cache result: {e}")
            cached = None
    
    model = load_model("email_model")
    
    suspicious_keywords = ['verify', 'urgent', 'password', 'click', 'bank', 'account', 
                          'confirm', 'security', 'update', 'alert', 'suspended', 
                          'winner', 'prize', 'congratulations', 'immediately', 
                          'unauthorized', 'login', 'verify your account']
    
    score = sum(1 for kw in suspicious_keywords if kw in email_text.lower()) / len(suspicious_keywords)
    prediction = 1 if score > 0.15 else 0
    
    if model is not None:
        try:
            if hasattr(model, 'predict_proba'):
                model_prediction = model.predict([email_text])[0]
                model_score = float(max(model.predict_proba([email_text])[0]))
                score = (score + model_score) / 2
                prediction = 1 if score > 0.2 else 0
                logger.info(f"Model prediction: score={model_score:.3f}, final_score={score:.3f}")
        except Exception as e:
            logger.warning(f"Model prediction failed, using rule-based: {e}")
    
    threat_type, explanation, indicators = get_email_explanation(email_text, score)
    
    prediction_time = (time.time() - start_time) * 1000
    request_id = str(uuid.uuid4())
    
    display_confidence = min(score + 0.15, 0.99)
    
    prediction_data = {
        "label": "malicious" if prediction == 1 else "safe",
        "threat_type": threat_type,
        "confidence": display_confidence,
        "explanation": explanation,
        "indicators": indicators
    }
    
    if score > 0.3:
        severity = "high"
        action = "flagged"
    elif score > 0.15:
        severity = "medium"
        action = "flagged"
    else:
        severity = "low"
        action = "none"
    
    try:
        if ml_db and hasattr(ml_db, 'save_prediction'):
            background_tasks.add_task(
                ml_db.save_prediction,
                request_id, user_id, "email", email_text[:500], prediction_data,
                "nb-email-v1.3-test", prediction_time, severity, action
            )
        else:
            logger.warning("ml_db or save_prediction not available, skipping save")
    except Exception as e:
        logger.error(f"Failed to schedule save to database: {e}")
    
    logger.info(f"Email scan completed: malicious={prediction}, confidence={display_confidence:.3f}, time={prediction_time:.2f}ms")
    
    return ScanResponse(
        is_malicious=bool(prediction),
        confidence=round(display_confidence, 4),
        threat_type=threat_type,
        explanation=explanation,
        indicators=indicators,
        prediction_time_ms=round(prediction_time, 2),
        model_version="nb-email-v1.3-test",
        from_cache="none",
        request_id=request_id,
        timestamp=datetime.now().isoformat()
    )

# ══════════════════════════════════════════════════════════════
# APP MALWARE DETECTION ENDPOINTS
# ══════════════════════════════════════════════════════════════

class AppNameScanRequest(BaseModel):
    app_name: str = Field(..., min_length=1, max_length=500)
    user_id: Optional[str] = None
    email: Optional[str] = None

class AppScanResponse(BaseModel):
    app_name: str
    is_malicious: bool
    confidence: float
    threat_type: str
    explanation: str
    indicators: List[str]
    prediction_time_ms: float
    model_version: str
    from_cache: str
    request_id: str
    timestamp: str
    app_metadata: Optional[dict] = None

# ── Google Play Store dataset (lazy loaded) ──
_playstore_df = None

def _get_playstore_df():
    """Load Google Play Store dataset for app name lookups (cached)."""
    global _playstore_df
    if _playstore_df is None:
        try:
            import pandas as pd
            csv_path = Path(__file__).resolve().parent.parent.parent / 'ml_training' / 'datasets' / 'App Malware Datasets' / 'googleplaystore.csv'
            if csv_path.exists():
                _playstore_df = pd.read_csv(csv_path)
                _playstore_df['App'] = _playstore_df['App'].fillna('').str.strip()
                _playstore_df['Rating'] = pd.to_numeric(_playstore_df['Rating'], errors='coerce').fillna(0)
                _playstore_df['Installs'] = _playstore_df['Installs'].fillna('0').astype(str).str.replace(r'[+,]', '', regex=True)
                _playstore_df['Installs'] = pd.to_numeric(_playstore_df['Installs'], errors='coerce').fillna(0)
                logger.info(f"Loaded Google Play Store dataset: {len(_playstore_df)} apps")
            else:
                logger.warning(f"Play Store CSV not found at {csv_path}")
                _playstore_df = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading Play Store data: {e}")
            _playstore_df = pd.DataFrame()
    return _playstore_df


def get_app_explanation(is_found: bool, app_info: dict, score: float) -> tuple:
    """Generate explanation for app name lookup results."""
    indicators = []
    explanation_parts = []

    if is_found:
        rating = app_info.get('rating', app_info.get('Rating', 0))
        installs = float(app_info.get('installs', app_info.get('Installs', 0)))
        category = app_info.get('category', app_info.get('Category', 'Unknown'))

        if rating >= 4.0 and installs >= 100000:
            threat_type = "verified_safe"
            explanation_parts.append(f"Verified Google Play Store app in '{category}' category")
            explanation_parts.append(f"Rating: {rating}/5.0, Installs: {installs:,.0f}+")
            severity_note = "SAFE"
        elif rating >= 3.5 and installs >= 10000:
            threat_type = "likely_safe"
            explanation_parts.append(f"Known app on Google Play Store ({category})")
            explanation_parts.append(f"Rating: {rating}/5.0, Installs: {installs:,.0f}+")
            severity_note = "LOW RISK"
        elif rating > 0:
            threat_type = "caution"
            explanation_parts.append(f"Found on Play Store but with low reputation")
            explanation_parts.append(f"Rating: {rating}/5.0, Installs: {installs:,.0f}+")
            indicators.append("low_rating")
            if installs < 1000:
                indicators.append("low_install_count")
            severity_note = "MEDIUM RISK"
        else:
            threat_type = "caution"
            explanation_parts.append("Found on Play Store but no rating data available")
            indicators.append("no_rating_data")
            severity_note = "MEDIUM RISK"
    else:
        threat_type = "unverified"
        explanation_parts.append("App not found in Google Play Store verified database")
        explanation_parts.append("This does not necessarily mean it is malicious, but exercise caution")
        explanation_parts.append("Recommendation: Upload the APK file for a detailed malware scan")
        indicators.append("app_not_in_playstore")
        indicators.append("unverified_source")
        severity_note = "CAUTION"

    explanation = f"{severity_note}: " + " ; ".join(explanation_parts)
    return threat_type, explanation, indicators


def get_apk_explanation(features: np.ndarray, score: float, detected_features: list) -> tuple:
    """Generate explanation for APK scan results."""
    indicators = []
    explanation_parts = []

    permissions = [f for f in detected_features if f.get('category') == 'Manifest Permission']
    api_calls = [f for f in detected_features if f.get('category') == 'API call signature']
    intents = [f for f in detected_features if f.get('category') == 'Intent']
    commands = [f for f in detected_features if f.get('category') == 'Commands signature']

    dangerous_perms = ['SEND_SMS', 'READ_SMS', 'RECEIVE_SMS', 'READ_PHONE_STATE',
                       'CALL_PHONE', 'READ_CONTACTS', 'CAMERA', 'RECORD_AUDIO',
                       'READ_LOGS', 'INSTALL_PACKAGES', 'SYSTEM_ALERT_WINDOW']
    found_dangerous = [p['name'] for p in permissions if p['name'] in dangerous_perms]
    if found_dangerous:
        indicators.append("dangerous_permissions")
        explanation_parts.append(f"Requests {len(found_dangerous)} dangerous permissions: {', '.join(found_dangerous[:5])}")

    suspicious_apis = ['DexClassLoader', 'Runtime.exec', 'Runtime.getRuntime',
                       'System.loadLibrary', 'ProcessBuilder', 'createSubprocess']
    found_apis = [a['name'] for a in api_calls if a['name'] in suspicious_apis]
    if found_apis:
        indicators.append("suspicious_api_calls")
        explanation_parts.append(f"Uses {len(found_apis)} suspicious APIs: {', '.join(found_apis[:3])}")

    if commands:
        indicators.append("system_commands")
        cmd_names = [c['name'] for c in commands[:3]]
        explanation_parts.append(f"Contains system command references: {', '.join(cmd_names)}")
    boot_intents = [i for i in intents if 'BOOT' in i['name']]
    if boot_intents:
        indicators.append("boot_persistence")
        explanation_parts.append("Registers for auto-start on device boot")

    if score > 0.7:
        threat_type = "malware"
        severity_note = "HIGH RISK"
        if not explanation_parts:
            explanation_parts.append("High probability of malware based on static analysis")
    elif score > 0.4:
        threat_type = "suspicious"
        severity_note = "MEDIUM RISK"
        if not explanation_parts:
            explanation_parts.append("Shows some suspicious characteristics")
    else:
        threat_type = "clean"
        severity_note = "LOW RISK"
        if not explanation_parts:
            explanation_parts.append("No obvious malware indicators detected in static analysis")

    explanation_parts.append(f"Detected {len(permissions)} permissions, {len(api_calls)} API calls, {len(intents)} intents, {len(commands)} commands")

    explanation = f"{severity_note}: " + " ; ".join(explanation_parts)
    return threat_type, explanation, indicators


@router.post("/app-name", response_model=AppScanResponse)
async def scan_app_name(request: AppNameScanRequest, background_tasks: BackgroundTasks):
    """Search for an app name in the Google Play Store verified database."""
    import time
    import pandas as pd

    start_time = time.time()
    app_name = request.app_name.strip()

    logger.info(f"Looking up app: '{app_name}'")

    user_id = request.user_id
    if not user_id and request.email:
        try:
            user_id = ml_db.get_user_id(request.email)
        except Exception:
            user_id = None
    if not user_id:
        user_id = "22222222-2222-2222-2222-222222222222"

    df = _get_playstore_df()
    app_metadata = None
    is_found = False
    matches = pd.DataFrame()

    if not df.empty:
        matches = df[df['App'].str.lower().str.contains(app_name.lower(), na=False)]

        if not matches.empty:
            is_found = True
            best_match = matches.sort_values('Installs', ascending=False).iloc[0]
            app_metadata = {
                'name': str(best_match.get('App', app_name)),
                'category': str(best_match.get('Category', 'Unknown')),
                'rating': float(best_match.get('Rating', 0)),
                'reviews': str(best_match.get('Reviews', '0')),
                'installs': str(best_match.get('Installs', 0)),
                'size': str(best_match.get('Size', 'Unknown')),
                'content_rating': str(best_match.get('Content Rating', 'Unknown')),
                'total_matches': len(matches)
            }

    if is_found and app_metadata:
        rating = app_metadata.get('rating', 0)
        installs = float(app_metadata.get('installs', 0))
        if rating >= 4.0 and installs >= 100000:
            score = 0.05
        elif rating >= 3.5 and installs >= 10000:
            score = 0.15
        elif rating > 0:
            score = 0.4
        else:
            score = 0.5
    else:
        score = 0.6

    threat_type, explanation, indicators = get_app_explanation(
        is_found, app_metadata or {}, score
    )

    prediction_time = (time.time() - start_time) * 1000
    request_id = str(uuid.uuid4())
    is_malicious = score > 0.5

    prediction_data = {
        "label": "malicious" if is_malicious else "safe",
        "threat_type": threat_type,
        "confidence": score,
        "explanation": explanation,
        "indicators": indicators
    }

    try:
        if ml_db and hasattr(ml_db, 'save_prediction'):
            severity = "high" if score > 0.5 else "medium" if score > 0.3 else "low"
            action = "flagged" if score > 0.3 else "none"
            background_tasks.add_task(
                ml_db.save_prediction,
                request_id, user_id, "app_name", app_name, prediction_data,
                "playstore-lookup-v1.0", prediction_time, severity, action
            )
    except Exception as e:
        logger.error(f"Failed to schedule save: {e}")

    logger.info(f"App name lookup completed: found={is_found}, score={score:.3f}")

    return AppScanResponse(
        app_name=app_metadata.get('name', app_name) if app_metadata else app_name,
        is_malicious=is_malicious,
        confidence=round(1 - score, 4),
        threat_type=threat_type,
        explanation=explanation,
        indicators=indicators,
        prediction_time_ms=round(prediction_time, 2),
        model_version="playstore-lookup-v1.0",
        from_cache="none",
        request_id=request_id,
        timestamp=datetime.now().isoformat(),
        app_metadata=app_metadata
    )


@router.post("/app", response_model=AppScanResponse)
async def scan_app_apk(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    app_name: str = Form(default="Unknown APK"),
    user_id: Optional[str] = Form(default=None),
):
    """Upload and scan an APK file for malware using static analysis."""
    import time

    start_time = time.time()

    logger.info(f"Scanning APK file: '{file.filename}' ({file.size} bytes)")

    if not file.filename or not file.filename.lower().endswith('.apk'):
        raise HTTPException(status_code=400, detail="Only .apk files are accepted")

    if not user_id:
        user_id = "22222222-2222-2222-2222-222222222222"

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.apk') as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        from backend.app_features import extract_apk_features, get_detected_features, get_risk_indicators
        features = extract_apk_features(tmp_path)
        detected = get_detected_features(features)
        model = load_model("app_model")

        if model is None:
            feature_density = float(features.sum()) / features.shape[1]
            score = min(feature_density * 2, 0.95)
            prediction = 1 if score > 0.3 else 0
            logger.info(f"Using fallback prediction (model not loaded): density={feature_density:.3f}")
        else:
            try:
                prediction = int(model.predict(features)[0])
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(features)[0]
                    score = float(proba[1])
                else:
                    score = 0.85 if prediction == 1 else 0.15
                logger.info(f"Model prediction: score={score:.3f}, prediction={prediction}")
            except Exception as e:
                logger.error(f"Model prediction failed: {e}")
                score = 0.5
                prediction = 0

        threat_type, explanation, indicators = get_apk_explanation(features, score, detected)

        risk_indicators = get_risk_indicators(features)
        indicators.extend(risk_indicators[:10])
        indicators = list(set(indicators))

        prediction_time = (time.time() - start_time) * 1000
        request_id = str(uuid.uuid4())
        is_malicious = prediction == 1

        prediction_data = {
            "label": "malicious" if is_malicious else "safe",
            "threat_type": threat_type,
            "confidence": score,
            "explanation": explanation,
            "indicators": indicators
        }

        app_metadata = {
            'name': app_name or file.filename,
            'filename': file.filename,
            'file_size': len(content),
            'features_detected': int(features.sum()),
            'total_features': features.shape[1],
            'permissions_count': len([f for f in detected if f.get('category') == 'Manifest Permission']),
            'api_calls_count': len([f for f in detected if f.get('category') == 'API call signature']),
            'intents_count': len([f for f in detected if f.get('category') == 'Intent']),
            'commands_count': len([f for f in detected if f.get('category') == 'Commands signature']),
        }

        try:
            if ml_db and hasattr(ml_db, 'save_prediction'):
                severity = "critical" if score > 0.8 else "high" if score > 0.6 else "medium" if score > 0.3 else "low"
                action = "blocked" if score > 0.6 else "flagged" if score > 0.3 else "none"
                background_tasks.add_task(
                    ml_db.save_prediction,
                    request_id, user_id, "apk", file.filename, prediction_data,
                    "xgb-app-v1.0", prediction_time, severity, action
                )
        except Exception as e:
            logger.error(f"Failed to schedule save: {e}")

        logger.info(f"APK scan completed: malicious={is_malicious}, score={score:.3f}, features={int(features.sum())}/{features.shape[1]}")

        return AppScanResponse(
            app_name=app_name or file.filename,
            is_malicious=is_malicious,
            confidence=round(score, 4),
            threat_type=threat_type,
            explanation=explanation,
            indicators=indicators,
            prediction_time_ms=round(prediction_time, 2),
            model_version="xgb-app-v1.0",
            from_cache="none",
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            app_metadata=app_metadata
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"APK scan error: {e}")
        raise HTTPException(status_code=500, detail=f"APK analysis failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


@router.get("/health")
async def scan_health():
    """Health check for scan endpoints - with graceful error handling"""
    url_model_exists = Path("backend/models/url_model.pkl").exists()
    email_model_exists = Path("backend/models/email_model.pkl").exists()
    app_model_exists = Path("backend/models/app_model.pkl").exists()
    
    db_status = {
        "postgres": "not_configured",
        "redis": "not_configured",
        "mongodb": "not_configured"
    }
    
    if ml_db:
        try:
            if hasattr(ml_db, 'get_postgres_connection'):
                conn = ml_db.get_postgres_connection()
                if conn:
                    db_status["postgres"] = "connected"
        except Exception as e:
            db_status["postgres"] = f"unavailable: {str(e)[:40]}"
        
        try:
            if hasattr(ml_db, 'get_redis_client'):
                redis_client = ml_db.get_redis_client()
                redis_client.ping()
                db_status["redis"] = "connected"
        except Exception as e:
            db_status["redis"] = f"unavailable: {str(e)[:40]}"
        
        try:
            if hasattr(ml_db, 'get_mongo_client'):
                mongo_client = ml_db.get_mongo_client()
                mongo_client.server_info()
                db_status["mongodb"] = "connected"
        except Exception as e:
            db_status["mongodb"] = f"unavailable: {str(e)[:40]}"
    else:
        db_status = {
            "postgres": "ml_db_not_initialized",
            "redis": "ml_db_not_initialized",
            "mongodb": "ml_db_not_initialized"
        }
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "url_model": {
                "loaded": url_model_exists, 
                "test_mode": not url_model_exists,
                "path": "backend/models/url_model.pkl" if url_model_exists else None
            },
            "email_model": {
                "loaded": email_model_exists, 
                "test_mode": not email_model_exists,
                "path": "backend/models/email_model.pkl" if email_model_exists else None
            },
            "app_model": {
                "loaded": app_model_exists,
                "test_mode": not app_model_exists,
                "path": "backend/models/app_model.pkl" if app_model_exists else None
            }
        },
        "databases": db_status,
        "integration": "ready",
        "cache_enabled": True,
        "seed_data_loaded": True if db_status.get("postgres") == "connected" else False,
        "note": "Running with fallback mode - Redis/PostgreSQL/MongoDB not required for basic functionality"
    }