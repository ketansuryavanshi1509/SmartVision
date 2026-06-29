import time
from functools import wraps
from flask import jsonify, request, g, session
from app.database.firebase_client import verify_firebase_token, SessionExpiredError

def _get_bearer_token() -> str:
    header = request.headers.get("Authorization", "")
    if not header:
        return ""
    parts = header.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return ""

def require_auth(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        token = _get_bearer_token()
        if not token:
            return jsonify({"success": False, "message": "Missing Bearer token"}), 401

        now = time.time()
        # Simple caching in session to avoid redundant Firebase verification
        cached_user_id = session.get("user_id")
        cached_token = session.get("access_token")
        verified_at = session.get("token_verified_at", 0)
        
        if cached_user_id and cached_token == token and (now - verified_at) < 300:
            g.user_id = cached_user_id
            g.access_token = token
            return fn(*args, **kwargs)

        try:
            user = verify_firebase_token(token)
            user_id = user.get("id")
            if not user_id:
                raise ValueError("User ID missing in token")
            
            g.user_id = user_id
            g.access_token = token
            
            # Update cache
            session["user_id"] = user_id
            session["access_token"] = token
            session["token_verified_at"] = now
            
            # Record as last active user for background services
            try:
                from app.api.web_server import update_last_active_user
                update_last_active_user(user_id)
            except ImportError:
                pass
            
            return fn(*args, **kwargs)
        except SessionExpiredError:
            return jsonify({"success": False, "error": "session_expired", "message": "Session expired"}), 401
        except Exception as e:
            return jsonify({"success": False, "message": f"Authentication failed: {str(e)}"}), 401

    return wrapper
