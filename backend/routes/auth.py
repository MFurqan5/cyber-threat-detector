# backend/routes/auth.py
"""Authentication endpoints for SENTINELCACHE AI"""
import os
import logging
import re
from datetime import datetime, timedelta
from typing import Optional
import jwt
import bcrypt
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from backend.database import db

logger = logging.getLogger(__name__)

# Authentication Configuration
# Fallback to APP_SECRET_KEY or a secure random key
SECRET_KEY = os.getenv("JWT_SECRET", os.getenv("APP_SECRET_KEY", "sentinelcache-ai-super-secret-key-change-in-prod-2026"))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 24 * 60  # 24 hours

router = APIRouter(prefix="/auth", tags=["auth"])

# Password hashing utilities
def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception as e:
        logger.error(f"Password verification failed: {e}")
        return False

def get_password_hash(password: str) -> str:
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

# Token utility
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Pydantic Schemas
class UserRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=6, max_length=100)

    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        v = v.strip().lower()
        email_regex = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        if not re.match(email_regex, v):
            raise ValueError('Invalid email format')
        return v

    @field_validator('username')
    @classmethod
    def validate_username(cls, v):
        v = v.strip()
        if not re.match(r'^[a-zA-Z0-9_-]+$', v):
            raise ValueError('Username can only contain alphanumeric characters, underscores, or hyphens')
        return v

class UserLoginRequest(BaseModel):
    username_or_email: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=6, max_length=100)

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: dict

class MessageResponse(BaseModel):
    message: str

@router.post("/register", response_model=MessageResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegisterRequest):
    """Register a new user in the SQLite database"""
    username = user_data.username
    email = user_data.email
    password = user_data.password
    
    # Check if user already exists (by username or email)
    if db.get_user_by_username(username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username is already registered"
        )
    if db.get_user_by_email(email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email is already registered"
        )
    
    # Hash password and save user
    password_hash = get_password_hash(password)
    try:
        db.create_user(username, email, password_hash)
        logger.info(f"Successfully registered user: {username}")
        return {"message": "User registered successfully"}
    except Exception as e:
        logger.error(f"Error during registration for {username}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to register user due to a database error"
        )

@router.post("/login", response_model=TokenResponse)
async def login(login_data: UserLoginRequest):
    """Authenticate a user and return a JWT access token"""
    input_identifier = login_data.username_or_email.strip()
    password = login_data.password
    
    # Try finding user by username, then by email
    user = None
    if "@" in input_identifier:
        user = db.get_user_by_email(input_identifier.lower())
    
    if not user:
        user = db.get_user_by_username(input_identifier)
        
    if not user:
        # If not found by username and wasn't checked by email (e.g. didn't have @), try email just in case
        if "@" not in input_identifier:
            user = db.get_user_by_email(input_identifier.lower())
            
    if not user or not verify_password(password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username/email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Generate token
    token_data = {
        "sub": str(user["id"]),
        "username": user["username"],
        "email": user["email"]
    }
    
    access_token = create_access_token(data=token_data)
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user["id"],
            "username": user["username"],
            "email": user["email"]
        }
    }


from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends

security = HTTPBearer()

def decode_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except Exception as e:
        logger.warning(f"Invalid token: {e}")
        return None

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    token = credentials.credentials
    user_id = decode_token(token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"id": user_id}
