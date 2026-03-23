"""
PMIS Platform — Authentication
User registration, login, JWT token management.
"""

import bcrypt
import jwt
import uuid
from datetime import datetime, timezone, timedelta
from db import get_users_db, _now, _uuid, get_user_memory_path

SECRET_KEY = "pmis-platform-secret-change-in-production"
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 72


def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(12)).decode()


def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())


def create_jwt(user_id, username, role="member"):
    payload = {
        "user_id": user_id,
        "username": username,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_jwt(token):
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def register_user(username, password, display_name=None):
    """Register a new user. Creates their memory DB directory."""
    conn = get_users_db()

    # Check if username exists
    existing = conn.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
    if existing:
        conn.close()
        return {"error": f"Username '{username}' already exists"}

    user_id = _uuid()
    pw_hash = hash_password(password)

    conn.execute(
        "INSERT INTO users (id, username, display_name, password_hash, created_at, last_active, role) VALUES (?,?,?,?,?,?,?)",
        (user_id, username, display_name or username, pw_hash, _now(), _now(), "member")
    )
    conn.commit()
    conn.close()

    # Create user's memory directory
    mem_path = get_user_memory_path(username)
    mem_path.parent.mkdir(parents=True, exist_ok=True)

    token = create_jwt(user_id, username)

    return {
        "user_id": user_id,
        "username": username,
        "display_name": display_name or username,
        "token": token,
    }


def login_user(username, password):
    """Authenticate user and return JWT."""
    conn = get_users_db()
    user = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()

    if not user:
        conn.close()
        return {"error": "Invalid username or password"}

    if not verify_password(password, user["password_hash"]):
        conn.close()
        return {"error": "Invalid username or password"}

    # Update last active
    conn.execute("UPDATE users SET last_active=? WHERE id=?", (_now(), user["id"]))
    conn.commit()
    conn.close()

    token = create_jwt(user["id"], user["username"], user["role"])

    return {
        "user_id": user["id"],
        "username": user["username"],
        "display_name": user["display_name"],
        "role": user["role"],
        "token": token,
    }


def get_user_from_token(token):
    """Validate token and return user info."""
    payload = decode_jwt(token)
    if not payload:
        return None

    conn = get_users_db()
    user = conn.execute("SELECT * FROM users WHERE id=?", (payload["user_id"],)).fetchone()
    conn.close()

    if not user:
        return None

    return {
        "user_id": user["id"],
        "username": user["username"],
        "display_name": user["display_name"],
        "role": user["role"],
    }


def list_users():
    """List all registered users (admin function)."""
    conn = get_users_db()
    users = conn.execute(
        "SELECT id, username, display_name, role, created_at, last_active FROM users ORDER BY created_at"
    ).fetchall()
    conn.close()
    return [dict(u) for u in users]
