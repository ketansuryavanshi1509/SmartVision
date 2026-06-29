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


def register_auth_handlers(app):
    @app.after_request
    def _attach_refreshed_tokens(response):
        access = getattr(g, "refreshed_access_token", None)
        refresh = getattr(g, "refreshed_refresh_token", None)
        if access:
            response.headers["X-Access-Token"] = access
        if refresh:
            response.headers["X-Refresh-Token"] = refresh
        return response


def require_auth(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        token = _get_bearer_token()
        if not token:
            return jsonify({"success": False, "message": "Missing Bearer token"}), 401

        now = time.time()
        cached_user_id = session.get("user_id")
        cached_token = session.get("access_token")
        verified_at = session.get("token_verified_at", 0)
        if cached_user_id and cached_token == token and (now - verified_at) < 300:
            g.user_id = cached_user_id
            g.user = {"id": cached_user_id}
            g.access_token = token
            return fn(*args, **kwargs)

        try:
            user = verify_firebase_token(token)
        except SessionExpiredError:
            return jsonify({"error": "session_expired"}), 401
        except Exception:
            return jsonify({"success": False, "message": "Invalid or expired token"}), 401

        user_id = user.get("id") if isinstance(user, dict) else None
        if not user_id:
            return jsonify({"success": False, "message": "User ID missing in token"}), 401

        g.user_id = user_id
        g.user = user
        g.access_token = token

        session["user_id"] = user_id
        session["access_token"] = token
        session["token_verified_at"] = now

        return fn(*args, **kwargs)

    return wrapper
