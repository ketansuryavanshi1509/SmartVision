"""
Flask web server for mobile-responsive Vision Assistant.
Replaces desktop Tkinter GUI with web interface.
"""

from flask import Flask, render_template, jsonify, request, g, current_app, session
from flask_cors import CORS
import threading
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import os
import time
import uuid
import logging

from app.config import config
from app.core.logger import logger
from app.auth.firebase_auth import require_auth
from app.database.firebase_client import (
    init_firebase,
    get_db,
    sign_in_with_email_password,
    sign_up_with_email_password,
    verify_firebase_token,
    _is_timeout,
)
from firebase_admin import auth as firebase_auth
from google.cloud.firestore_v1.base_query import FieldFilter
from app.services.navigation.location_service import LocationTrackerManager, reverse_geocode
from app.services.emergency.emergency_system import normalize_phone_number
from app.services.speech.speech_manager import create_speech_api

app = Flask(__name__)
app.secret_key = config.FLASK_SECRET_KEY

# Add speech API routes
create_speech_api(app)
CORS(app)  # Enable CORS for mobile access

# Suppress noisy access logs for frequent polling endpoints without changing poll frequency.
class _SuppressPollsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not (
            "GET /api/status" in msg
            or "GET /api/speech/status" in msg
        )

_werkzeug_logger = logging.getLogger("werkzeug")
_werkzeug_logger.addFilter(_SuppressPollsFilter())

# Global reference to vision assistant
vision_assistant = None
latest_location = None
location_timestamp = None  # Track when location was last updated
navigation_sessions = {}
_last_process_frame_ts = {}
_process_frame_min_interval_s = 0.1
_recent_command_ids = {}
_last_command = None
last_active_user_id = None

def update_last_active_user(user_id):
    """Track the most recently active user ID."""
    global last_active_user_id
    if user_id and user_id != last_active_user_id:
        last_active_user_id = user_id
        logger.info(f"Updated last active user ID: {user_id}")

# Firestore database reference
db = None
location_tracker = LocationTrackerManager()

def store_navigation_session(session_id, polyline):
    if not session_id or not polyline:
        return
    navigation_sessions[session_id] = {
        'polyline': polyline
    }

def record_command(command: str, source: str = "backend", command_id: str = None):
    if not command:
        return None
    now = time.time()
    with _command_lock:
        # Drop old command ids (10s TTL)
        stale = [cid for cid, ts in _recent_command_ids.items() if (now - ts) > 10.0]
        for cid in stale:
            _recent_command_ids.pop(cid, None)
        if command_id:
            if command_id in _recent_command_ids:
                return None
            _recent_command_ids[command_id] = now
        else:
            command_id = str(uuid.uuid4())
            _recent_command_ids[command_id] = now

        global _last_command
        _last_command = {
            "id": command_id,
            "command": command,
            "source": source or "backend",
            "ts": now,
        }
        return _last_command

def get_session(session_id):
    if not session_id:
        return None
    return navigation_sessions.get(session_id)


def _normalize_session_id(session_id):
    if session_id is None:
        return None
    normalized = str(session_id).strip()
    if not normalized or normalized in ("null", "undefined"):
        return None
    return normalized


def _parse_match_threshold(raw_value, minimum=0.80):
    if raw_value in (None, "", "null", "undefined"):
        return None
    try:
        threshold = float(raw_value)
    except (TypeError, ValueError):
        logger.warning("Invalid match_threshold %r received; using automatic default", raw_value)
        return None
    return max(minimum, min(1.0, threshold))


def init_db():
    """Initialize Firebase backend."""
    global db
    try:
        init_firebase()
        db = get_db()
        logger.info("Firebase initialized successfully")
    except Exception as e:
        logger.warning("Firebase initialization failed: %s – running in degraded mode", e)
        db = None

def init_vision_assistant(va_instance):
    """Initialize the vision assistant instance."""
    global vision_assistant
    vision_assistant = va_instance


def _get_obj_attr(obj, key, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _json_success(message=None, speak=None, status=200, **payload):
    body = {"success": True}
    if message is not None:
        body["message"] = message
    if speak:
        body["speak"] = speak
    body.update(payload)
    return jsonify(body), status


@app.route('/')
def index():
    """Serve the main mobile-responsive web interface."""
    return render_template(
        'index.html', 
        google_maps_api_key=config.GOOGLE_MAPS_API_KEY,
        firebase_config={
            "apiKey": config.FIREBASE_API_KEY,
            "authDomain": config.FIREBASE_AUTH_DOMAIN,
            "projectId": config.FIREBASE_PROJECT_ID,
            "storageBucket": config.FIREBASE_STORAGE_BUCKET
        }
    )


@app.route('/login')
def login():
    """Serve the login page."""
    return render_template(
        'login.html',
        firebase_config={
            "apiKey": config.FIREBASE_API_KEY,
            "authDomain": config.FIREBASE_AUTH_DOMAIN,
            "projectId": config.FIREBASE_PROJECT_ID,
            "storageBucket": config.FIREBASE_STORAGE_BUCKET
        }
    )


@app.route('/signup')
def signup():
    """Serve the signup page."""
    return render_template(
        'signup.html',
        firebase_config={
            "apiKey": config.FIREBASE_API_KEY,
            "authDomain": config.FIREBASE_AUTH_DOMAIN,
            "projectId": config.FIREBASE_PROJECT_ID,
            "storageBucket": config.FIREBASE_STORAGE_BUCKET
        }
    )


@app.route('/favicon.ico')
def favicon():
    """Avoid noisy 404s when browsers request a favicon."""
    return ("", 204)


@app.route('/api/login', methods=['POST'])
def api_login():
    """Handle login API using Firebase Auth."""
    try:
        data = request.json or {}
        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({'success': False, 'message': 'Email and password are required'}), 400

        try:
            result = sign_in_with_email_password(email, password)
        except Exception as e:
            error_msg = str(e)
            if _is_timeout(e):
                logger.warning("Firebase timeout during login for %s", email)
                return jsonify({
                    'success': False,
                    'message': 'Login is temporarily unavailable because Firebase could not be reached. Please try again.',
                }), 503
            return jsonify({'success': False, 'message': f'Login error: {error_msg}'}), 401

        id_token = result.get('idToken')
        refresh_token = result.get('refreshToken')
        user_id = result.get('localId')

        if not id_token or not user_id:
            return jsonify({'success': False, 'message': 'Login failed'}), 401

        return jsonify({
            'success': True,
            'access_token': id_token,
            'refresh_token': refresh_token,
            'user_id': user_id,
        })
    except Exception as e:
        error_msg = str(e)
        logger.error("Login error: %s", e)
        return jsonify({'success': False, 'message': f'Login error: {error_msg}'}), 500


@app.route('/api/signup', methods=['POST'])
def api_signup():
    """Handle signup API using Firebase Auth."""
    try:
        data = request.json or {}
        email = data.get('email')
        password = data.get('password')
        full_name = data.get('full_name')

        if not email or not password:
            return jsonify({'success': False, 'message': 'Email and password are required'}), 400

        try:
            result = sign_up_with_email_password(email, password)
        except Exception as e:
            error_msg = str(e)
            logger.error("Signup error: %s", e)
            if "EMAIL_EXISTS" in error_msg.upper():
                return jsonify({'success': False, 'message': 'An account with this email already exists'}), 400
            if "TOO_MANY_ATTEMPTS" in error_msg.upper() or "QUOTA" in error_msg.upper():
                return jsonify({'success': False, 'message': 'Please wait before requesting another signup'}), 429
            return jsonify({'success': False, 'message': f'Signup error: {error_msg}'}), 500

        id_token = result.get('idToken')
        refresh_token = result.get('refreshToken')
        user_id = result.get('localId')

        if not user_id:
            return jsonify({'success': False, 'message': 'Signup failed'}), 400

        # Create user profile in Firestore
        try:
            firestore_db = get_db()
            if firestore_db:
                profile_data = {"full_name": full_name or "", "created_at": time.time()}

                # Handle face registration during signup if image is provided
                image_data = data.get('image')
                if image_data:
                    try:
                        import base64
                        import cv2
                        import numpy as np
                        from deepface import DeepFace

                        # Decode base64 image
                        img_data = image_data.split(',')[1] if ',' in image_data else image_data
                        image_bytes = base64.b64decode(img_data)
                        nparr = np.frombuffer(image_bytes, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                        # Generate embedding with the same baseline model used by face auth
                        results = DeepFace.represent(img, model_name="Facenet", enforce_detection=True)
                        if results:
                            profile_data["face_encoding"] = results[0]["embedding"]
                            profile_data["face_embedding_dim"] = len(results[0]["embedding"])
                    except Exception as face_err:
                        logger.warning("Face registration during signup failed: %s", face_err)

                firestore_db.collection("profiles").document(user_id).set(profile_data)
        except Exception as exc:
            logger.warning("Profile creation failed: %s", exc)

        return jsonify({
            'success': True,
            'user_id': user_id,
            'access_token': id_token,
            'refresh_token': refresh_token,
            'email_confirmed': True,
            'requires_email_confirm': False,
        })
    except Exception as e:
        error_msg = str(e)
        logger.error("Signup error: %s", e)
        return jsonify({'success': False, 'message': f'Signup error: {error_msg}'}), 500





@app.route('/api/logout', methods=['POST'])
def api_logout():
    """Handle logout API."""
    return jsonify({'success': True, 'message': 'Logged out successfully'})


@app.route('/api/signup_with_face', methods=['POST'])
def api_signup_with_face():
    """Handle signup with email/password + face enrollment."""
    try:
        data = request.json or {}
        email = data.get('email')
        password = data.get('password')
        full_name = data.get('full_name')
        face_images = data.get('face_images', [])  # Array of base64 images

        header = request.headers.get("Authorization", "")
        parts = header.split()
        bearer_token = parts[1].strip() if len(parts) == 2 and parts[0].lower() == "bearer" else ""

        user_id = None
        id_token = bearer_token or None
        refresh_token = data.get('refresh_token')

        if bearer_token:
            try:
                verified_user = verify_firebase_token(bearer_token)
            except Exception as exc:
                logger.error("Signup-with-face token verification failed: %s", exc)
                return jsonify({'success': False, 'message': 'Invalid or expired token'}), 401

            user_id = verified_user.get('id')
            email = email or verified_user.get('email')

            if not user_id:
                return jsonify({'success': False, 'message': 'Authenticated user not found'}), 401
        else:
            if not email or not password:
                return jsonify({'success': False, 'message': 'Email and password are required'}), 400

            try:
                result = sign_up_with_email_password(email, password)
            except Exception as e:
                error_msg = str(e)
                logger.error("Signup error: %s", e)
                if "EMAIL_EXISTS" in error_msg.upper():
                    return jsonify({'success': False, 'message': 'An account with this email already exists'}), 400
                if "TOO_MANY_ATTEMPTS" in error_msg.upper() or "QUOTA" in error_msg.upper():
                    return jsonify({'success': False, 'message': 'Please wait before requesting another signup'}), 429
                return jsonify({'success': False, 'message': f'Signup error: {error_msg}'}), 500

            id_token = result.get('idToken')
            refresh_token = result.get('refreshToken')
            user_id = result.get('localId')

            if not user_id:
                return jsonify({'success': False, 'message': 'Signup failed'}), 400

        # Create user profile in Firestore
        firestore_db = get_db()
        if firestore_db:
            profile_data = {
                "full_name": full_name or "",
                "email": email,
                "created_at": time.time()
            }
            firestore_db.collection("users").document(user_id).set(profile_data, merge=True)

        # Enroll face if images provided
        face_enrolled = False
        face_enrollment_message = None
        if face_images and len(face_images) >= 5:
            try:
                from app.auth.face_auth import get_face_auth_manager
                import cv2
                import numpy as np
                
                logger.info(f"[FACE SIGNUP] Starting face enrollment for user {user_id} with {len(face_images)} images")
                
                face_auth = get_face_auth_manager()
                logger.info("[FACE SIGNUP] Face auth manager initialized successfully")
                
                # Decode all face images
                frames = []
                for i, img_data in enumerate(face_images):
                    try:
                        img_bytes = base64.b64decode(img_data.split(',')[1] if ',' in img_data else img_data)
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if img is not None:
                            frames.append(img)
                            logger.debug(f"[FACE SIGNUP] Frame {i+1} decoded: {img.shape}")
                        else:
                            logger.warning(f"[FACE SIGNUP] Frame {i+1} failed to decode")
                    except Exception as img_err:
                        logger.error(f"[FACE SIGNUP] Error decoding frame {i+1}: {img_err}")
                
                logger.info(f"[FACE SIGNUP] Successfully decoded {len(frames)} frames out of {len(face_images)}")
                
                if len(frames) >= 5:
                    logger.info(f"[FACE SIGNUP] Calling enroll_face with {len(frames)} frames")
                    success, message = face_auth.enroll_face(frames, user_id)
                    face_enrolled = success
                    face_enrollment_message = message
                    logger.info(f"[FACE SIGNUP] Face enrollment result: success={success}, message={message}")
                    if success:
                        logger.info(f"Face enrolled during signup for user {user_id}")
                    else:
                        logger.warning(f"Face enrollment failed: {message}")
                else:
                    face_enrollment_message = f"Only {len(frames)} valid frames received (need at least 5)"
                    logger.error(f"[FACE SIGNUP] Insufficient frames: {len(frames)}")
            except Exception as face_err:
                logger.error(f"Face enrollment during signup failed: {face_err}")
                import traceback
                traceback.print_exc()
                face_enrollment_message = str(face_err)
        else:
            logger.warning(f"[FACE SIGNUP] No face images or insufficient count: {len(face_images) if face_images else 0}")

        response_message = 'Account created successfully'
        if face_enrolled:
            response_message += ' with face recognition'
        elif face_images:
            detail = face_enrollment_message or 'Face enrollment was not completed.'
            response_message += f'. Face login was not enabled: {detail}'

        response = {
            'success': True,
            'user_id': user_id,
            'access_token': id_token,
            'refresh_token': refresh_token,
            'face_enrolled': face_enrolled,
            'face_enrollment_message': face_enrollment_message,
            'message': response_message,
        }
        
        return jsonify(response)
        
    except Exception as e:
        error_msg = str(e)
        logger.error("Signup with face error: %s", e)
        return jsonify({'success': False, 'message': f'Signup error: {error_msg}'}), 500


@app.route('/api/register_face', methods=['POST'])
@require_auth
def api_register_face():
    """Register / update face encoding for an already logged-in user.

    Called from the dashboard \"Register Face for Login\" button.
    Accepts a single base64 JPEG image and enrolls a face embedding for the
    current user without requiring a full signup flow.
    """
    try:
        data = request.json or {}
        image_data = data.get('image')
        user_id = g.user_id

        if not image_data:
            return jsonify({'success': False, 'message': 'No image provided'}), 400

        if not user_id:
            return jsonify({'success': False, 'message': 'User not authenticated'}), 401

        # Decode the image
        try:
            img_bytes = base64.b64decode(image_data.split(',')[1] if ',' in image_data else image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return jsonify({'success': False, 'message': 'Could not decode image'}), 400
        except Exception as decode_err:
            return jsonify({'success': False, 'message': f'Image decode error: {decode_err}'}), 400

        from app.auth.face_auth import get_face_auth_manager
        face_auth = get_face_auth_manager()

        # Use the single frame repeated to meet minimum enrollment requirement
        frames = [img] * face_auth.min_enrollment_frames
        success, message = face_auth.enroll_face(frames, user_id)

        if success:
            logger.info("Face registered from dashboard for user %s", user_id)
            return jsonify({'success': True, 'message': 'Face registered successfully! You can now login with your face.'})
        else:
            logger.warning("Dashboard face registration failed for user %s: %s", user_id, message)
            return jsonify({'success': False, 'message': message}), 400

    except Exception as e:
        logger.error("Register face error: %s", e)
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/test_face_modules', methods=['GET'])
@require_auth
def test_face_modules():
    """Test endpoint to verify face authentication modules are working."""
    try:
        from app.auth.face_auth import get_face_auth_manager
        
        result = {
            'face_auth_initialized': False,
            'embedding_generator': None,
            'liveness_detector': None,
            'errors': []
        }
        
        try:
            face_auth = get_face_auth_manager()
            result['face_auth_initialized'] = True
            
            if face_auth.face_embedding_gen:
                result['embedding_generator'] = {
                    'initialized': True,
                    'model_name': getattr(face_auth.face_embedding_gen, 'model_name', 'unknown'),
                    'embedding_dim': getattr(face_auth.face_embedding_gen, 'embedding_dim', None)
                }
            else:
                result['errors'].append('Face embedding generator not initialized')
            
            if face_auth.liveness_detector:
                result['liveness_detector'] = {
                    'initialized': True,
                    'type': type(face_auth.liveness_detector).__name__
                }
            else:
                result['errors'].append('Liveness detector not initialized')
                
        except Exception as e:
            result['errors'].append(str(e))
            import traceback
            traceback.print_exc()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Test face modules error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/face_login', methods=['POST'])
def api_face_login():
    """Login using face recognition (no password needed)."""
    try:
        data = request.json or {}
        email = data.get('email')
        face_images = data.get('images') or []
        face_image = data.get('image')
        if not face_images and face_image:
            face_images = [face_image]

        if not email:
            return jsonify({'success': False, 'message': 'Email is required'}), 400
        
        if len(face_images) < 3:
            return jsonify({
                'success': False,
                'message': 'At least 3 face frames are required for liveness verification.'
            }), 400

        # Find user by email
        try:
            user_record = firebase_auth.get_user_by_email(email)
            user_id = user_record.uid
        except Exception:
            return jsonify({'success': False, 'message': 'User not found'}), 404

        # Check if user has registered face
        from app.auth.face_auth import get_face_auth_manager
        face_auth = get_face_auth_manager()
        
        if not face_auth.has_registered_face(user_id):
            return jsonify({
                'success': False,
                'message': 'No face registered for this user. Please use password login.',
                'requires_password': True
            }), 400

        frames = []
        for encoded_image in face_images:
            img_data = encoded_image.split(',')[1] if ',' in encoded_image else encoded_image
            image_bytes = base64.b64decode(img_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)

        if len(frames) < 3:
            return jsonify({
                'success': False,
                'message': 'Could not decode enough valid face frames for verification.'
            }), 400

        # Authenticate with liveness-verified face sequence.
        authenticated, message, similarity = face_auth.authenticate_face(
            frames[-1],
            user_id,
            liveness_frames=frames,
        )

        if authenticated:
            # Generate Firebase custom token
            custom_token = firebase_auth.create_custom_token(user_id)
            if isinstance(custom_token, bytes):
                custom_token = custom_token.decode("utf-8")

            logger.info(f"Face login successful for user {user_id} (similarity: {similarity:.4f})")
            
            return jsonify({
                'success': True,
                'custom_token': custom_token,
                'user_id': user_id,
                'similarity': similarity,
                'message': 'Face login successful'
            })
        else:
            logger.warning(f"Face login failed for user {user_id}: {message}")
            return jsonify({
                'success': False,
                'message': message,
                'similarity': similarity
            }), 401
            
    except Exception as e:
        logger.error("Face login error: %s", e)
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/liveness_check', methods=['POST'])
def api_liveness_check():
    """Verify liveness from sequence of face images."""
    try:
        data = request.json or {}
        face_images = data.get('images', [])

        if not face_images or len(face_images) < 3:
            return jsonify({
                'success': False,
                'message': 'Need at least 3 consecutive frames for liveness check'
            }), 400

        from app.auth.face_auth import get_face_auth_manager
        import cv2
        import numpy as np
        
        face_auth = get_face_auth_manager()
        
        # Decode all images
        frames = []
        for img_data in face_images:
            img_bytes = base64.b64decode(img_data.split(',')[1] if ',' in img_data else img_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is not None:
                frames.append(img)
        
        if len(frames) < 3:
            return jsonify({
                'success': False,
                'message': 'Could not decode enough valid frames'
            }), 400

        # Check liveness
        is_live, metrics = face_auth.verify_liveness(frames)

        if is_live:
            return _json_success(
                'Liveness verified',
                blink_detected=metrics.get('blink_detected'),
                blink_count=metrics.get('blink_count'),
                head_movement_detected=metrics.get('head_movement_detected'),
                head_movements=metrics.get('head_movements')
            )
        else:
            return jsonify({
                'success': False,
                'message': 'Liveness check failed. Please look at camera naturally and blink or move your head.',
                'metrics': metrics
            }), 401
            
    except Exception as e:
        logger.error("Liveness check error: %s", e)
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/start_camera', methods=['POST'])
@require_auth
def start_camera():
    """Start camera capture."""
    try:
        if vision_assistant and vision_assistant.vision_engine:
            success = vision_assistant.vision_engine.start_camera()
            if success:
                return _json_success("Camera started", speak="Camera started")
            return jsonify({'success': False, 'message': 'Failed to start camera'})
        return jsonify({'success': False, 'message': 'Vision engine not available'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/stop_camera', methods=['POST'])
@require_auth
def stop_camera():
    """Stop camera capture."""
    try:
        if vision_assistant and vision_assistant.vision_engine:
            vision_assistant.vision_engine.stop_camera()
            return _json_success("Camera stopped", speak="Camera stopped")
        return jsonify({'success': False, 'message': 'Vision engine not available'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/process_frame', methods=['POST'])
@require_auth
def process_frame():
    """Process a frame sent from mobile device."""
    try:
        now = time.monotonic()
        throttle_key = getattr(g, "user_id", None) or request.remote_addr or "global"
        last_ts = _last_process_frame_ts.get(throttle_key, 0.0)
        if (now - last_ts) < _process_frame_min_interval_s:
            return jsonify({
                'success': False,
                'message': 'Too many requests. Please slow down.'
            }), 429
        _last_process_frame_ts[throttle_key] = now

        data = request.json
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Decode base64 image
        image_data = data['image'].split(',')[1] if ',' in data['image'] else data['image']
        image_bytes = base64.b64decode(image_data)
        image = Image.open(BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if vision_assistant and vision_assistant.vision_engine:
            # Keep latest frame available for other features (e.g., personal object storage)
            vision_assistant.vision_engine.current_frame = frame
            # Detect objects
            detections = vision_assistant.vision_engine.detect_objects(frame)

            # Personalized object recognition via embeddings
            recognition = None
            match_threshold = _parse_match_threshold(data.get("match_threshold"))
            try:
                results = vision_assistant.vision_engine.search_similar_objects(
                    frame=frame,
                    access_token=g.access_token,
                    match_count=1,
                    match_threshold=match_threshold,
                    user_id=g.user_id,
                )
                if results:
                    best = results[0]
                    similarity = float(best.get("similarity", 0.0))
                    object_name = best.get("object_name", "object")
                    blip_caption = best.get("blip_caption")
                    applied_threshold = float(best.get("threshold_used", match_threshold or 0.80))
                    logger.info("Similarity score: %.4f", similarity)
                    if similarity >= applied_threshold:
                        ok_to_speak, reason = vision_assistant.vision_engine._should_speak_personal_match(object_name)
                        if ok_to_speak:
                            logger.info("Match accepted above threshold %.2f", applied_threshold)
                            message = vision_assistant.vision_engine.format_personal_match(object_name)
                            recognition = {
                                "type": "recognized_personal_object",
                                "object_name": object_name,
                                "blip_caption": blip_caption,
                                "message": message,
                                "similarity": similarity,
                                "threshold_used": applied_threshold,
                            }
                        else:
                            if reason == "cooldown":
                                logger.info("Match suppressed due to cooldown for %s", object_name)
                            else:
                                logger.info("Match suppressed due to smoothing for %s", object_name)
                    else:
                        logger.info("Match ignored due to threshold %.2f", applied_threshold)
                        vision_assistant.vision_engine.reset_personal_match_state()
                else:
                    vision_assistant.vision_engine.reset_personal_match_state()
            except Exception as e:
                if _is_timeout(e):
                    logger.debug("Personal object match skipped: Firebase unreachable")
                else:
                    logger.exception("Personal object match failed: %s", e)

            # Get scene description or BLIP fallback with cooldown
            description = None
            get_description = bool(data.get("get_description", False))
            navigation_active = bool(getattr(vision_assistant, "navigation_active", False))
            voice_state = getattr(getattr(vision_assistant, "voice_engine", None), "conversation_state", "idle")
            if recognition is not None:
                description = recognition["message"]
            elif get_description and not navigation_active and voice_state == "idle":
                if vision_assistant.vision_engine.should_generate_scene_caption():
                    description = vision_assistant.vision_engine.generate_blip_caption(frame)
                    if not description:
                        vision_assistant.vision_engine.current_frame = frame
                        description = vision_assistant.vision_engine.describe_scene()
                    if description:
                        vision_assistant.vision_engine.record_scene_caption()

            # Format detections for JSON
            detections_json = []
            for det in detections:
                detections_json.append({
                    'class_name': det['class_name'],
                    'confidence': float(det['score']),
                    'bbox': [float(x) for x in det['bbox']]
                })

            detection_names = [det["class_name"] for det in detections_json]
            unique_names = list(dict.fromkeys(detection_names))
            message = None
            speak_text = None
            if recognition is not None:
                message = recognition["message"]
                speak_text = recognition["message"]
            elif description:
                message = description
                speak_text = description
            elif unique_names:
                joined = ", ".join(unique_names[:3])
                message = f"Object detected: {joined}"
            else:
                message = "No objects detected"

            return jsonify({
                'success': True,
                'message': message,
                'speak': speak_text,
                'detections': detections_json,
                'description': description,
                'recognition': recognition,
                'navigation_mode': getattr(vision_assistant.vision_engine, 'navigation_mode', False),
            })
        return jsonify({'success': False, 'message': 'Vision engine not available'})
    except Exception as e:
        logger.exception("process_frame failed")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/describe_scene', methods=['POST'])
@require_auth
def describe_scene():
    """Get scene description."""
    try:
        if vision_assistant:
            description = vision_assistant.describe_scene()
            return _json_success(description, speak=description, description=description)
        return jsonify({'success': False, 'message': 'Vision assistant not available'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/navigate', methods=['POST'])
@require_auth
def navigate():
    """Start navigation."""
    try:
        data = request.json
        destination = data.get('destination')
        mode = data.get('mode', 'walking')
        session_id = data.get('session_id') or str(uuid.uuid4())
        
        if vision_assistant:
            if vision_assistant.navigation_active:
                return jsonify({'success': False, 'message': 'Navigation already active'})
            
            # Start navigation in background
            threading.Thread(
                target=vision_assistant.navigation_flow,
                args=(destination, mode, session_id),
                daemon=True
            ).start()

            try:
                record_command(f"NAVIGATE:{destination}:{mode}", source="web", command_id=None)
            except Exception:
                pass

            speak_text = f"Calculating route to {destination}."
            return _json_success('Navigation started', speak=speak_text, session_id=session_id)
        return jsonify({'success': False, 'message': 'Vision assistant not available'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stop_navigation', methods=['POST'])
@require_auth
def stop_navigation():
    """Stop navigation."""
    try:
        if vision_assistant:
            vision_assistant.stop_navigation()
            try:
                user_id = g.user_id
                location_tracker.end_session(user_id, access_token=g.access_token)
            except Exception:
                pass
            try:
                record_command("STOP_NAVIGATION", source="web", command_id=None)
            except Exception:
                pass
            return _json_success('Navigation stopped', speak='Navigation stopped')
        return jsonify({'success': False, 'message': 'Vision assistant not available'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/navigation_status', methods=['GET'])
@require_auth
def navigation_status():
    """Return the latest live navigation snapshot for the active route."""
    try:
        if not vision_assistant or not getattr(vision_assistant, "navigation_manager", None):
            return jsonify({'success': False, 'message': 'Navigation assistant not available'}), 503
        manager = vision_assistant.navigation_manager
        status = manager.get_navigation_status()
        return jsonify({'success': True, 'navigation': status})
    except Exception as e:
        logger.exception("navigation_status failed")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/location', methods=['POST'])
@require_auth
def set_location():
    """Set location from device GPS or IP fallback."""
    global latest_location, location_timestamp
    try:
        import time
        data = request.json or {}
        lat = data.get('lat')
        lng = data.get('lng')
        use_ip_fallback = bool(data.get("use_ip_fallback"))
        speak_location = bool(data.get("speak", False)) # New flag for active speech

        if lat is None or lng is None:
            return jsonify({'success': False, 'message': 'GPS coordinates required (lat and lng)'}), 400

        # Validate coordinates
        if not (-90 <= lat <= 90 and -180 <= lng <= 180):
            logger.warning("Invalid GPS coordinates: %s, %s", lat, lng)
            return jsonify({'success': False, 'message': 'Invalid coordinates'}), 400

        latest_location = (lat, lng)
        location_timestamp = time.time()
        
        # Human-readable reverse geocoding
        location_name = reverse_geocode(lat, lng)
        logger.info("Location updated: %s", location_name)
        try:
            if vision_assistant:
                vision_assistant.current_location = (lat, lng)
                nav_mgr = getattr(vision_assistant, "navigation_manager", None)
                if nav_mgr is not None:
                    nav_mgr.current_location = (lat, lng)
        except Exception:
            pass
        try:
            user_id = g.user_id
            location_tracker.ingest_point(user_id, lat, lng, ts=location_timestamp, access_token=g.access_token)
        except Exception as e:
            logger.warning("Location tracking error: %s", e)

        speak_text = f"You are currently near {location_name}" if speak_location else None
        return _json_success(
            f"Location found: {location_name}",
            speak=speak_text,
            lat=lat,
            lng=lng,
            location_name=location_name,
        )
    except Exception as e:
        logger.error("Error setting location: %s", e)
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/voice_command', methods=['POST'])
@require_auth
def voice_command():
    """Handle voice command from mobile device."""
    try:
        data = request.json
        command = data.get('command')
        command_id = data.get('command_id')
        source = data.get('source', 'web')
        
        if vision_assistant and command:
            user_id = g.user_id
            voice_engine = getattr(vision_assistant, "voice_engine", None)
            backend_voice_active = bool(getattr(voice_engine, "listening", False))

            # Prevent duplicate/conflicting command streams when backend STT is active.
            if source == "web" and backend_voice_active:
                logger.info("Ignoring web voice command because backend STT is active")
                return jsonify({
                    'success': True,
                    'ignored': True,
                    'reason': 'backend_stt_active',
                    'message': 'Web voice command ignored while backend STT is active',
                    'command_id': command_id
                })

            recorded = record_command(command, source=source, command_id=command_id)
            if recorded is None and command_id:
                return jsonify({'success': True, 'message': 'Duplicate command ignored', 'command_id': command_id})
            vision_assistant.handle_voice_command(command, user_id=user_id)
            return jsonify({'success': True, 'message': 'Command processed', 'command_id': command_id})
        return jsonify({'success': False, 'message': 'No command provided'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/emergency_contact', methods=['GET', 'POST'])
@require_auth
def emergency_contact():
    """Get or update user's emergency contact phone."""
    try:
        user_id = g.user_id
        if not user_id:
            return jsonify({'success': False, 'message': 'User not authenticated'}), 401

        firestore_db = get_db()
        if not firestore_db:
            return jsonify({'success': True, 'emergency_phone': None, '_offline': True})

        profile_ref = firestore_db.collection('profiles').document(user_id)

        if request.method == 'GET':
            doc = profile_ref.get()
            phone = None
            if doc.exists:
                phone = doc.to_dict().get('emergency_phone')
            return jsonify({'success': True, 'emergency_phone': phone})

        data = request.json or {}
        emergency_phone = data.get('emergency_phone', '')
        # Update user profile
        default_country = os.getenv('DEFAULT_COUNTRY_CODE', '')
        normalized = normalize_phone_number(emergency_phone, default_country=default_country)
        if not normalized:
            return jsonify({'success': False, 'message': 'Invalid phone number format'}), 400
            
        db.collection("users").document(user_id).set({"emergency_phone": normalized}, merge=True)
        return _json_success(
            'Emergency contact updated',
            speak='Emergency contact updated',
            emergency_phone=normalized,
        )
    except Exception as e:
        if _is_timeout(e):
            logger.warning("emergency_contact skipped: Firebase unreachable")
            return jsonify({'success': True, 'emergency_phone': None, '_offline': True})
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/emergency_trigger', methods=['POST'])
@require_auth
def emergency_trigger():
    """Trigger emergency alert."""
    try:
        user_id = g.user_id
        data = request.json or {}
        trigger_type = data.get('trigger_type', 'voice')
        if not vision_assistant:
            return jsonify({'success': False, 'message': 'Vision assistant not available'}), 500
        ok, msg = vision_assistant.handle_emergency(user_id=user_id, trigger_type=trigger_type, access_token=g.access_token)
        status = 200 if ok else 400
        speak_text = "Emergency alert sent successfully. Help is on the way." if ok else msg
        return jsonify({'success': ok, 'message': msg, 'speak': speak_text}), status
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/feedback', methods=['POST'])
@require_auth
def feedback():
    """Submit user feedback."""
    try:
        data = request.json
        feedback_text = data.get('feedback')
        
        if feedback_text:
            with open("user_feedback.txt", "a", encoding="utf-8") as f:
                f.write(feedback_text + "\n")
            return _json_success('Feedback recorded', speak='Thank you for your feedback')
        return jsonify({'success': False, 'message': 'No feedback provided'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
@require_auth
def status():
    """Get application status."""
    try:
        status_data = {
            'status': 'running',
            'camera_running': False,
            'navigation_active': False,
            'navigation_mode': False,
            'voice_active': False
        }
        
        # Safely access vision assistant attributes
        if vision_assistant:
            try:
                if hasattr(vision_assistant, 'vision_engine') and vision_assistant.vision_engine:
                    status_data['camera_running'] = getattr(vision_assistant.vision_engine, 'is_running', False)
                    status_data['navigation_mode'] = getattr(vision_assistant.vision_engine, 'navigation_mode', False)
                status_data['navigation_active'] = getattr(vision_assistant, 'navigation_active', False)
                session_id = getattr(vision_assistant, 'navigation_session_id', None)
                if not session_id:
                    nav_mgr = getattr(vision_assistant, 'navigation_manager', None)
                    session_id = getattr(nav_mgr, '_session_id', None) if nav_mgr else None
                status_data['navigation_session_id'] = session_id
            except AttributeError:
                # Handle any attribute errors gracefully
                pass
            
            # Check if voice engine is active
            try:
                if hasattr(vision_assistant, 'voice_engine') and vision_assistant.voice_engine:
                    status_data['voice_active'] = getattr(vision_assistant.voice_engine, 'listening', False)
            except AttributeError:
                pass

        try:
            status_data['last_command'] = _last_command
        except Exception:
            status_data['last_command'] = None
        
        return jsonify({'success': True, 'status': status_data})
    except Exception as e:
        # Log the error but return a valid response
        logger.error("Status endpoint error: %s", e)
        return jsonify({
            'success': True,
            'status': {
                'status': 'running',
                'camera_running': False,
                'navigation_active': False,
                'navigation_mode': False,
                'voice_active': False
            },
            'error': 'Internal error occurred but system is running'
        })


@app.route('/api/location_sessions', methods=['GET'])
@require_auth
def location_sessions():
    """List recent location sessions for the current user."""
    try:
        user_id = g.user_id
        limit = int(request.args.get('limit', 10))
        
        db = get_db()
        if not db:
            return jsonify({'success': True, 'sessions': [], '_offline': True})

        docs = db.collection("users").document(user_id).collection("location_sessions")\
            .order_by("started_at", direction="DESCENDING").limit(limit).get()

        sessions = []
        for doc in docs:
            data = doc.to_dict()
            data['id'] = doc.id
            sessions.append(data)

        return jsonify({'success': True, 'sessions': sessions})
    except Exception as e:
        if _is_timeout(e):
            logger.warning("location_sessions skipped: Firebase unreachable")
            return jsonify({'success': True, 'sessions': [], '_offline': True})
        logger.exception("location_sessions failed")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/location_points', methods=['GET'])
@require_auth
def location_points():
    """Get location points for a session."""
    try:
        user_id = g.user_id
        session_id = request.args.get('session_id')
        if not session_id:
            return jsonify({'success': False, 'message': 'session_id is required'}), 400

        db = get_db()
        if not db:
            return jsonify({'success': True, 'points': [], '_offline': True})

        docs = db.collection("users").document(user_id).collection("location_points")\
            .where(filter=FieldFilter("session_id", "==", session_id))\
            .order_by("recorded_at").get()

        points = []
        for doc in docs:
            d = doc.to_dict()
            points.append({
                'latitude': d.get('latitude'),
                'longitude': d.get('longitude'),
                'recorded_at': d.get('recorded_at'),
            })

        return jsonify({'success': True, 'points': points})
    except Exception as e:
        if _is_timeout(e):
            logger.warning("location_points skipped: Firebase unreachable")
            return jsonify({'success': True, 'points': [], '_offline': True})
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/location_summary', methods=['GET'])
@require_auth
def location_summary():
    """Get summary for a session (or active session if not provided)."""
    try:
        user_id = g.user_id
        session_id = request.args.get('session_id') or location_tracker.get_active_session_id(user_id)
        if not session_id:
            return jsonify({'success': False, 'message': 'No active session'}), 400

        db = get_db()
        if not db:
            return jsonify({'success': True, 'session': None, '_offline': True})

        doc = db.collection("users").document(user_id).collection("location_sessions").document(session_id).get()
        if doc.exists:
            data = doc.to_dict()
            data['id'] = doc.id
            return jsonify({'success': True, 'session': data})
        return jsonify({'success': True, 'session': None})
    except Exception as e:
        if _is_timeout(e):
            logger.warning("location_summary skipped: Firebase unreachable")
            return jsonify({'success': True, 'session': None, '_offline': True})
        return jsonify({'success': False, 'error': str(e)}), 500


def _encode_polyline(coords):
    """Encode a list of (lat, lng) pairs into a polyline string."""
    result = []
    prev_lat = 0
    prev_lng = 0
    for lat, lng in coords:
        lat_e5 = int(round(lat * 1e5))
        lng_e5 = int(round(lng * 1e5))
        dlat = lat_e5 - prev_lat
        dlng = lng_e5 - prev_lng
        for value in (dlat, dlng):
            value = ~(value << 1) if value < 0 else (value << 1)
            while value >= 0x20:
                result.append(chr((0x20 | (value & 0x1f)) + 63))
                value >>= 5
            result.append(chr(value + 63))
        prev_lat = lat_e5
        prev_lng = lng_e5
    return ''.join(result)


@app.route('/api/location_polyline', methods=['GET'])
@require_auth
def location_polyline():
    """Return encoded polyline for a session."""
    session_id = _normalize_session_id(request.args.get('session_id'))
    if not session_id:
        return jsonify({"polyline": ""})

    cached_session = get_session(session_id)
    if cached_session and 'polyline' in cached_session:
        return jsonify({"polyline": cached_session['polyline']})

    try:
        user_id = g.user_id
        session_id = _normalize_session_id(request.args.get('session_id'))
        
        if not session_id:
            return jsonify({"polyline": ""})
            
        db = get_db()
        if not db:
            return jsonify({"polyline": ""})

        try:
            docs = db.collection("users").document(user_id).collection("location_points")\
                .where(filter=FieldFilter("session_id", "==", session_id))\
                .order_by("recorded_at").get()
            points = [doc.to_dict() for doc in docs]
        except Exception as e:
            if "requires an index" in str(e).lower():
                logger.warning("Polyline query requires an index, falling back to in-memory sorting")
                docs = db.collection("users").document(user_id).collection("location_points")\
                    .where(filter=FieldFilter("session_id", "==", session_id)).get()
                points = [doc.to_dict() for doc in docs]
                points = sorted(points, key=lambda d: d.get("recorded_at", 0))
            else:
                raise
        if not points:
            return jsonify({"polyline": ""})
        coords = [(p["latitude"], p["longitude"]) for p in points]
        return jsonify({"polyline": _encode_polyline(coords)})
    except Exception as e:
        if _is_timeout(e):
            logger.warning("location_polyline skipped: Firebase unreachable")
            return jsonify({"polyline": ""})
        logger.exception("location_polyline failed")
        return jsonify({"error": str(e)}), 500


@app.route("/api/add_personal_object", methods=["POST"])
@require_auth
def add_personal_object():
    """Add a personal object for recognition."""
    try:
        if not vision_assistant or not vision_assistant.vision_engine:
            return jsonify({
                "success": False,
                "message": "Vision engine not available. Please start the main application first."
            }), 500

        data = request.get_json() or {}
        object_name = data.get("object_name")
        image_url = data.get("image_url")

        if not object_name or not image_url:
            return jsonify({
                "success": False,
                "message": "Missing object_name or image_url"
            }), 400

        try:
            firestore_db = get_db()
            result = vision_assistant.vision_engine.store_personal_object(
                supabase=firestore_db,
                user_id=g.user_id,
                object_name=object_name,
                image_url=image_url,
            )
        except Exception as exc:
            if _is_timeout(exc):
                return jsonify({"success": False, "message": "Cannot save object while offline"}), 503
            raise

        return jsonify({
            "success": True,
            "message": f"Saved personal object {object_name}",
            "speak": f"Saved personal object {object_name}",
            "data": result
        }), 200

    except Exception as e:
        current_app.logger.exception("Add personal object failed")
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@app.route('/api/search_personal_object', methods=['POST'])
@require_auth
def search_personal_object():
    """Search for a personal object in the current frame."""
    try:
        if not vision_assistant or not vision_assistant.vision_engine:
            return jsonify({'success': False, 'message': 'Vision engine not available'}), 500
        
        data = request.json
        object_name = data.get('object_name')
        
        if not object_name:
            return jsonify({'success': False, 'message': 'Object name is required'}), 400
        
        # Search for the object in the current frame
        result = vision_assistant.vision_engine.search_for_personal_object(object_name, user_id=g.user_id)

        return jsonify({'success': True, 'message': result, 'speak': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/similarity_search', methods=['POST'])
@require_auth
def similarity_search():
    """Find similar stored objects."""
    try:
        if not vision_assistant or not vision_assistant.vision_engine:
            return jsonify({'success': False, 'message': 'Vision engine not available'}), 500

        data = request.json or {}
        image_data = data.get('image')
        match_count = int(data.get('match_count', 5))
        match_threshold = _parse_match_threshold(data.get('match_threshold'))

        frame = None
        if image_data:
            try:
                encoded = image_data.split(',')[1] if ',' in image_data else image_data
                image_bytes = base64.b64decode(encoded)
                image = Image.open(BytesIO(image_bytes))
                frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            except Exception as e:
                return jsonify({'success': False, 'message': f'Invalid image data: {e}'}), 400

        try:
            results = vision_assistant.vision_engine.search_similar_objects(
                frame=frame,
                access_token=g.access_token,
                match_count=match_count,
                match_threshold=match_threshold,
                user_id=g.user_id,
            )
        except Exception as exc:
            if _is_timeout(exc):
                logger.warning("similarity_search skipped: Firebase unreachable")
                return jsonify({'success': True, 'results': [], '_offline': True})
            raise
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/personal_objects', methods=['GET'])
@require_auth
def get_personal_objects():
    """Get list of stored personal objects."""
    try:
        if not vision_assistant or not vision_assistant.vision_engine:
            return jsonify({'success': False, 'message': 'Vision engine not available'}), 500
        
        objects_list = vision_assistant.vision_engine.get_personal_objects_list(user_id=g.user_id)
        
        return jsonify({'success': True, 'objects': objects_list})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stop_all_speech', methods=['POST'])
@require_auth
def api_stop_all_speech():
    """Emergency stop all speech and clear queue"""
    try:
        if vision_assistant and hasattr(vision_assistant, 'speech_manager'):
            vision_assistant.speech_manager.cancel_all_speech()
            return jsonify({"success": True, "message": "All speech stopped"}), 200
        else:
            return jsonify({"success": False, "error": "Speech manager not available"}), 500
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/emergency_stop', methods=['POST'])
@require_auth
def api_emergency_stop():
    """Emergency stop all systems including speech, navigation, and camera"""
    try:
        # Stop all speech
        if vision_assistant and hasattr(vision_assistant, 'speech_manager'):
            vision_assistant.speech_manager.cancel_all_speech()
        
        # Stop navigation
        if vision_assistant and hasattr(vision_assistant, 'stop_navigation'):
            vision_assistant.stop_navigation()
        
        # Stop camera
        if vision_assistant and hasattr(vision_assistant, 'vision_engine'):
            vision_assistant.vision_engine.stop_camera()
        
        return jsonify({"success": True, "message": "Emergency stop activated"}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500



def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the Flask server."""
    # Initialize the database
    init_db()
    ssl_context = None
    if os.getenv("ENABLE_HTTPS", "").strip().lower() in {"1", "true", "yes"}:
        ssl_context = "adhoc"
        logger.info("HTTPS enabled with ad-hoc certificate")
    else:
        print("⚠ Access via http://192.168.10.35:5000 (NOT https)")
    app.run(host=host, port=port, debug=debug, threaded=True, ssl_context=ssl_context)


if __name__ == '__main__':
    run_server(debug=True)
