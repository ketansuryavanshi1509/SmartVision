"""
Firebase client module for SmartVision application.
Replaces Supabase with Firebase Admin SDK + Cloud Firestore.
"""
import os
from typing import Any, Dict, Optional

import firebase_admin
from firebase_admin import auth as firebase_auth
from firebase_admin import credentials, firestore
from google.cloud.firestore_v1.base_query import FieldFilter
import requests as _requests

from app.config import config
from app.core.logger import logger

# Import optimizer for performance enhancements
try:
    from app.database.firebase_optimizer import FirebaseSchemaOptimizer, get_schema_optimizer
    OPTIMIZER_AVAILABLE = True
except ImportError:
    OPTIMIZER_AVAILABLE = False
    FirebaseSchemaOptimizer = None
    get_schema_optimizer = None

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_firebase_app = None
_firestore_db = None


class SessionExpiredError(Exception):
    """Raised when a token is expired and cannot be refreshed."""


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def init_firebase():
    """Initialize Firebase Admin SDK and Firestore client."""
    global _firebase_app, _firestore_db

    if _firebase_app is not None:
        logger.info("Firebase already initialized")
        return

    cred_path = config.FIREBASE_CREDENTIALS
    if not os.path.isabs(cred_path):
        # Resolve relative to project root
        cred_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), cred_path)

    if not os.path.exists(cred_path):
        logger.error(f"Firebase credentials file not found: {cred_path}")
        raise FileNotFoundError(
            f"Firebase credentials file not found: {cred_path}. "
            "Set FIREBASE_CREDENTIALS env var to the correct path."
        )

    cred = credentials.Certificate(cred_path)
    _firebase_app = firebase_admin.initialize_app(cred)
    _firestore_db = firestore.client()
    logger.info("Firebase initialized successfully (project: %s)", _firebase_app.project_id)


def get_db():
    """Return the Firestore client.  Initializes Firebase if needed."""
    global _firestore_db
    if _firestore_db is None:
        init_firebase()
    return _firestore_db


def get_optimizer() -> Optional[FirebaseSchemaOptimizer]:
    """Get Firebase schema optimizer for performance enhancements."""
    if not OPTIMIZER_AVAILABLE:
        logger.warning("Firebase optimizer not available")
        return None
    
    db = get_db()
    if db is None:
        return None
    
    return get_schema_optimizer(db)


# ---------------------------------------------------------------------------
# Firebase Auth — token verification (server-side)
# ---------------------------------------------------------------------------

def verify_firebase_token(id_token: str) -> Dict[str, Any]:
    """Verify a Firebase Auth ID token using the Admin SDK.
    """
    if not id_token:
        raise ValueError("ID token is required")

    try:
        decoded = firebase_auth.verify_id_token(id_token, check_revoked=True)
        return {
            "id": decoded["uid"],
            "email": decoded.get("email", ""),
            "role": "authenticated",
        }
    except firebase_auth.ExpiredIdTokenError:
        raise SessionExpiredError("Firebase ID token expired")
    except firebase_auth.RevokedIdTokenError:
        raise SessionExpiredError("Firebase ID token has been revoked")
    except firebase_auth.InvalidIdTokenError as exc:
        raise ValueError(f"Invalid Firebase ID token: {exc}") from exc
    except Exception as exc:
        logger.warning("Token verification failed: %s", exc)
        raise


# ---------------------------------------------------------------------------
# Firebase Auth REST API — sign-in / sign-up (password-based)
# ---------------------------------------------------------------------------

_FIREBASE_API_KEY: Optional[str] = None


def _get_firebase_api_key() -> str:
    global _FIREBASE_API_KEY
    if _FIREBASE_API_KEY is None:
        _FIREBASE_API_KEY = config.FIREBASE_API_KEY
    if not _FIREBASE_API_KEY:
        raise ValueError(
            "FIREBASE_API_KEY is not set in config or environment."
        )
    return _FIREBASE_API_KEY


def sign_in_with_email_password(email: str, password: str) -> Dict[str, Any]:
    """Sign in a user via Firebase Auth REST API.

    Returns dict with keys: idToken, refreshToken, localId (uid), email, …
    Raises on failure.
    """
    api_key = _get_firebase_api_key()
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    resp = _requests.post(url, json=payload, timeout=10)
    data = resp.json()
    if resp.status_code != 200:
        error_msg = data.get("error", {}).get("message", "Login failed")
        raise ValueError(error_msg)
    return data


def sign_up_with_email_password(email: str, password: str) -> Dict[str, Any]:
    """Create a user via Firebase Auth REST API.

    Returns dict with keys: idToken, refreshToken, localId (uid), email, …
    Raises on failure.
    """
    api_key = _get_firebase_api_key()
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signUp?key={api_key}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    resp = _requests.post(url, json=payload, timeout=10)
    data = resp.json()
    if resp.status_code != 200:
        error_msg = data.get("error", {}).get("message", "Signup failed")
        raise ValueError(error_msg)
    return data


# ---------------------------------------------------------------------------
# Helpers — offline / degraded mode
# ---------------------------------------------------------------------------

def _is_timeout(exc: Exception) -> bool:
    """Return True if the exception looks like a network timeout."""
    msg = str(exc).lower()
    return "timed out" in msg or "timeout" in msg or "connect" in msg


