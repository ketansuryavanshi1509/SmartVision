import sys
import traceback
import re
import threading
import time
import os
import math
import uuid
import requests
from typing import Optional
import logging

from app.config import config
from app.core.logger import logger
from app.services.speech.speech_manager import SpeechPriority, speech_manager
from app.utils.helpers import clean_html_instruction as clean_instruction

from app.services.navigation.location_service import LocationManager, reverse_geocode
from app.services.navigation.navigation_service import NavigationManager

def global_exception_handler(exc_type, exc_value, exc_traceback):
    logger.error("FATAL ERROR CAUGHT", exc_info=(exc_type, exc_value, exc_traceback))

sys.excepthook = global_exception_handler

def calculate_route(origin_lat, origin_lng, destination, mode, api_key):
    logger.info("=== CALLING GOOGLE DIRECTIONS API ===")
    logger.info(f"[NAVIGATION] Origin: ({origin_lat}, {origin_lng})")
    logger.info(f"[NAVIGATION] Destination: {destination}")
    logger.info(f"[NAVIGATION] Mode: {mode}")
    url = 'https://maps.googleapis.com/maps/api/directions/json'
    params = {
        'origin': f"{origin_lat},{origin_lng}",
        'destination': destination,
        'mode': mode,
        'alternatives': 'false',
        'units': 'metric',
        'key': api_key,
    }
    response = requests.get(url, params=params, timeout=10)
    data = response.json()
    logger.info(f"[DIRECTIONS STATUS]: {data.get('status')}")
    if data.get("status") != "OK":
        logger.error(f"[DIRECTIONS ERROR]: {data}")
        return None
    route = data["routes"][0]
    leg = route["legs"][0]
    distance = leg["distance"]["text"]
    duration = leg["duration"]["text"]
    steps = []
    for step in leg["steps"]:
        steps.append({
            "instruction": clean_instruction(step.get("html_instructions", "")),
            "distance_text": step.get("distance", {}).get("text"),
            "distance_value": step.get("distance", {}).get("value"),
            "end_location": step.get("end_location", {}),
        })
    logger.info(f"[NAVIGATION] Extracted {len(steps)} steps")
    overview_polyline = route["overview_polyline"]["points"]
    return {
        "distance": distance,
        "duration": duration,
        "steps": steps,
        "polyline": overview_polyline,
    }

# Google Maps is optional at startup. If the package is missing, we disable
# navigation geocoding instead of mutating the environment or blocking on pip.
try:
    import googlemaps
except ImportError:
    googlemaps = None
    logger.warning("[CONFIG] googlemaps package not installed; navigation geocoding will be disabled.")

# Google Maps API
GOOGLE_MAPS_API_KEY = config.GOOGLE_MAPS_API_KEY
if GOOGLE_MAPS_API_KEY:
    logger.info(f"[CONFIG] Using Google Maps API key: {GOOGLE_MAPS_API_KEY[:20]}...")
else:
    logger.warning("[CONFIG] Google Maps API key not set; geocoding will be disabled.")

# Initialize Google Maps client
try:
    if GOOGLE_MAPS_API_KEY and googlemaps is not None:
        gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)
        # Do not call the network during module import.
        logger.info("[CONFIG] Google Maps client initialized")
        logger.info("[CONFIG] ✓ Google Maps API key is valid and working")
    elif GOOGLE_MAPS_API_KEY:
        gmaps = None
        logger.warning("[CONFIG] Google Maps API key is set but googlemaps package is missing")
    else:
        gmaps = None
        logger.info("[CONFIG] Google Maps client not initialized due to missing API key")
except Exception as e:
    logger.error("[CONFIG] Google Maps initialization error: %s", e)
    gmaps = None

from app.services.vision.vision_engine import VisionEngine
from app.services.speech.voice_engine import VoiceEngine
from app.api.web_server import init_vision_assistant, run_server
from app.services.emergency.emergency_system import EmergencyAlertManager, contains_emergency_keyword
from app.services.speech.speech_manager import SpeechPriority, speech_manager, speech_lock


class VisionAssistant:
    def __init__(self, web_mode=True):
        self.vision_engine: Optional[VisionEngine] = None
        self.voice_engine: Optional[VoiceEngine] = None
        self.navigation_active = False
        self.is_navigating = False
        self.web_mode = web_mode
        self.navigation_stop_event = threading.Event()
        self.current_route_instructions = []
        self.navigation_steps = []
        self.current_destination = None
        self.current_transport = None
        self.route_step_coordinates = []  # Store coordinates for each step
        self.navigation_session_id = None
        self.navigation_setup_active = False  # Flag to prevent multiple navigation setups
        self.navigation_state = {
            "active": False,
            "awaiting_destination": False,
            "awaiting_mode": False,
            "destination": None,
            "mode": None
        }
        self._mobile_gps = None
        self._mobile_gps_ts = 0.0
        self.current_location = None
        self.last_obstacle_time = 0.0
        self.last_obstacle_key = None
        self.obstacle_cooldown = 10.0
        self.interaction_mode = False  # Flag for exclusive interaction mode
        self.speech_manager = speech_manager
        self.emergency_manager = EmergencyAlertManager(
            cooldown_seconds=int(os.getenv("EMERGENCY_COOLDOWN_SECONDS", "60"))
        )
        self.location_manager = LocationManager(gmaps_client=gmaps, web_mode=self.web_mode, mobile_max_age_s=120.0)
        self.nearby_service = None   # wired after NavigationManager init

        try:
            self.vision_engine = VisionEngine()
            
            # Voice engine is optional in web mode (uses Web Speech API)
            # Initialize voice engine for BOTH web & desktop
            try:
                self.voice_engine = VoiceEngine()
                self.vision_engine.set_voice_engine(self.voice_engine)
                self.voice_engine.set_speech_manager(self.speech_manager)
                self.voice_engine.set_places_api_key(GOOGLE_MAPS_API_KEY)
                self.voice_engine.set_location_provider(self.get_current_location)
                self.speech_manager.set_speech_callbacks(
                    self.voice_engine.pause_listening_for_tts,
                    self.voice_engine.resume_listening_after_tts,
                )

                # ALWAYS start streaming listening if microphone exists
                if self.voice_engine.microphone:
                    self.voice_engine.command_callback = self.handle_voice_command
                    self.voice_engine.start_listening()
                    print("Voice input enabled (streaming listening active)")
                else:
                    print("Voice input unavailable: no microphone detected")

            except Exception as e:
                print(f"Warning: Voice engine not available: {e}")
                self.voice_engine = None


            # Set default emergency classes (can be changed)
            self.vision_engine.set_emergency_classes(['person', 'car', 'motorcycle', 'truck', 'bus', 'bicycle'])

            # Initialize web server if in web mode
            if web_mode:
                init_vision_assistant(self)
                print("Web server mode initialized")

            self.navigation_manager = NavigationManager(
                api_key=GOOGLE_MAPS_API_KEY,
                location_manager=self.location_manager,
                speech_manager=self.speech_manager,
                vision_engine=self.vision_engine,
                voice_engine=self.voice_engine,
                on_route_calculated=self._store_navigation_polyline,
                on_start=self._on_navigation_start,
                on_stop=self._on_navigation_stop,
            )
            logger.info("All engines initialized successfully!")
        except Exception as e:
            logger.error(f"Error initializing engines: {e}")
            raise

        # ── NearbyPlacesService ────────────────────────────────────────────────────
        from app.services.navigation.nearby_places_service import NearbyPlacesService
        self.nearby_service = NearbyPlacesService(
            api_key=GOOGLE_MAPS_API_KEY,
            location_provider=self.get_current_location,
            speak_fn=lambda text: self._safe_speak(text, priority=SpeechPriority.SYSTEM),
            start_navigation_fn=self._start_navigation_coords,
        )
        logger.info("[INIT] NearbyPlacesService ready")

    def get_current_location(self):
        """
        Get current location using the best available method.
        Priority: 1) Mobile GPS, 2) IP fallback.
        Returns (lat, lng, city) or None.
        """
        if not getattr(self, "location_manager", None):
            return None
        return self.location_manager.get_best_available_location()

    def get_current_gps_location(self):
        """Get current location using only mobile GPS (no IP fallback)."""
        if not getattr(self, "location_manager", None):
            return self.get_current_location()
        return self.location_manager.get_mobile_gps_location()

    def get_current_location_name(self, loc=None) -> str:

        if loc is None:
            loc = self.get_current_location()
        if loc and len(loc) >= 3 and loc[2]:
            return loc[2]
        if loc and len(loc) >= 2:
            return f"Location at {loc[0]:.4f}, {loc[1]:.4f}"
        return "an unknown location"

    def _store_navigation_polyline(self, session_id, polyline):
        if not session_id or not polyline:
            return
        try:
            from app.api import web_server
            web_server.store_navigation_session(session_id, polyline)
        except Exception:
            pass

    def _on_navigation_start(self):
        self.navigation_active = True
        self.is_navigating = True
        try:
            if self.speech_manager:
                self.speech_manager.set_navigation_suppression(True)
        except Exception:
            pass
        if self.voice_engine:
            try:
                self.voice_engine.set_state("navigating")
                self.voice_engine.pause_listening_for_navigation()
            except Exception:
                pass
        if self.vision_engine:
            try:
                self.vision_engine.set_navigation_mode(True)
            except Exception:
                pass

    def _on_navigation_stop(self):
        self.navigation_active = False
        self.is_navigating = False
        try:
            if self.speech_manager:
                self.speech_manager.set_navigation_suppression(False)
        except Exception:
            pass
        if self.voice_engine:
            try:
                self.voice_engine.resume_listening_after_navigation()
            except Exception:
                pass
        if self.vision_engine:
            try:
                self.vision_engine.set_navigation_mode(False)
            except Exception:
                pass
        if self.voice_engine:
            try:
                self.voice_engine.set_state("idle")
                self.voice_engine.resume_listening_after_navigation()
            except Exception:
                pass

    def handle_gps(self):
        """Handle GPS/location request."""
        loc = self.get_current_location()
        if loc:
            lat, lng = loc[0], loc[1]
            place = reverse_geocode(lat, lng)
            self._safe_speak(f"You are currently near {place}")
        else:
            self._safe_speak("Unable to determine your location.")
    
    def _get_city_from_coords(self, lat, lng):
        """Extract city name from coordinates using simplified geocoding service."""
        return reverse_geocode(lat, lng)



    # ---------------- Helper methods ----------------
    def _safe_speak(
        self,
        text: str,
        blocking: bool = False,
        priority: SpeechPriority = SpeechPriority.REGULAR,
        source: str = "main_app",
    ):
        """Safely speak text using centralized speech manager with state tracking."""
        if not text or not text.strip():
            return

        if hasattr(self, "speech_manager") and self.speech_manager:
            try:
                if priority == SpeechPriority.NAVIGATION and source == "main_app":
                    source = "navigation"
                if blocking:
                    done_event = threading.Event()

                    def _on_done():
                        done_event.set()

                    ok = self.speech_manager.speak(text, priority, source, _on_done)
                    if ok:
                        done_event.wait()
                else:
                    self.speech_manager.speak(text, priority, source)
                return
            except Exception as e:
                print(f"[SPEECH] Error speaking: {e}")
                return

    def _speak_blocking(self, text: str, priority: SpeechPriority = SpeechPriority.REGULAR):
        if hasattr(self, 'speech_manager') and self.speech_manager:
            try:
                self.speech_manager.cancel_all_speech()
            except Exception:
                pass
        self._safe_speak(text, blocking=True, priority=priority)
    
    # ---------------- Destination extraction ----------------
    def extract_destination(self, command: str) -> str:
        """Extract clean destination name from conversational speech. Always returns string."""
        if not command:
            return ""
        if self.voice_engine and hasattr(self.voice_engine, "extract_destination"):
            dest = self.voice_engine.extract_destination(command)
            return dest or ""
        return command.strip()
    
    # ---------------- Voice command handling ----------------
    def handle_voice_command(self, command: str, user_id: str = None):
        if not command:
            return
        print(f"Voice command received: {command}")
        try:
            from app.api import web_server
            web_server.record_command(command, source="backend_stt", command_id=None)
        except Exception:
            pass
        
        # ── Nearby places: handle ongoing selection ──────────────────────────────
        nearby = getattr(self, "nearby_service", None)
        if nearby and nearby.is_awaiting_selection:
            if nearby.handle_selection(command, mode="walking"):
                return

        # ── Classify new place intent ────────────────────────────────────────────
        if nearby and command:
            from app.services.navigation.nearby_places_service import classify_place_intent
            _place_intent = classify_place_intent(command)
            if _place_intent == "nearby":
                threading.Thread(
                    target=nearby.handle_nearby_query,
                    args=(command,),
                    kwargs={"mode": "walking"},
                    daemon=True,
                ).start()
                return
            if _place_intent == "specific":
                threading.Thread(
                    target=nearby.handle_specific_query,
                    args=(command,),
                    kwargs={"mode": "walking"},
                    daemon=True,
                ).start()
                return
        
        # Check if this is a navigation signal from voice engine
        if command.startswith("NAVIGATE:"):
            parts = command.split(":")
            if len(parts) == 3:
                destination = parts[1]
                mode = parts[2]
                print(f"[MAIN] Navigation request: {destination} by {mode}")
                self._start_navigation_from_voice(destination, mode)
            return

        token = command.strip().upper()
        if token in {"DESCRIBE", "STOP", "STOP_NAVIGATION", "EMERGENCY", "GPS"}:
            self._handle_action_token(token, user_id=user_id)
            return

        if contains_emergency_keyword(command):
            self._handle_action_token("EMERGENCY", user_id=user_id)
            return

        if self.voice_engine:
            try:
                action = self.voice_engine.handle_voice_command(command, emit_callback=False)
                if action:
                    self.handle_voice_command(action, user_id=user_id)
                return
            except Exception as e:
                print(f"[NAV ERROR] {e}")
                traceback.print_exc()
                self.current_destination = None
                self.current_transport = None
                self._safe_speak("Navigation failed. Please try again.", priority=SpeechPriority.SYSTEM)
                return

        # Fallback: simple keyword routing
        lowered = command.lower().strip()
        if lowered == "navigate" or lowered.startswith("navigate"):
            self._handle_action_token("NAVIGATE", user_id=user_id)
        elif "describe" in lowered:
            self._handle_action_token("DESCRIBE", user_id=user_id)
        elif "stop" in lowered:
            self._handle_action_token("STOP", user_id=user_id)
        elif "emergency" in lowered:
            self._handle_action_token("EMERGENCY", user_id=user_id)
        elif "location" in lowered or "gps" in lowered:
            self._handle_action_token("GPS", user_id=user_id)
        else:
            self._safe_speak("Sorry, I did not understand that command.", priority=SpeechPriority.SYSTEM)

    def _handle_action_token(self, token: str, user_id: str = None):
        if token == "EMERGENCY":
            self._safe_speak("Emergency detected. Processing immediately.", priority=SpeechPriority.EMERGENCY)
            ok, msg = self.handle_emergency(user_id=user_id, trigger_type="voice")
            if not ok:
                self._safe_speak(msg, priority=SpeechPriority.SYSTEM)
            return
        if token == "DESCRIBE":
            self.describe_scene()
            return
        if token == "GPS":
            self.handle_gps()
            return
        if token in {"STOP", "STOP_NAVIGATION"}:
            if self.navigation_active:
                self.stop_navigation()
                self._safe_speak("Navigation stopped.", priority=SpeechPriority.SYSTEM)
            else:
                self._safe_speak("No navigation is active.", priority=SpeechPriority.SYSTEM)
            return
        if token == "NAVIGATE":
            if self.voice_engine:
                self.voice_engine.start_navigation_flow()
            else:
                self._safe_speak("Voice engine not available.", priority=SpeechPriority.SYSTEM)

    def _start_navigation_from_voice(self, destination: str, mode: str):
        """Start navigation from voice engine signal."""
        print(f"[MAIN] Starting navigation: {destination} by {mode}")
        if hasattr(self, "speech_manager") and self.speech_manager:
            try:
                self.speech_manager.cancel_all_speech()
            except Exception:
                pass

        # Set navigation flags
        self.navigation_setup_active = True

        try:
            if not destination or not isinstance(destination, str):
                print("[NAV] Invalid destination input")
                self._safe_speak("I didn't catch the destination. Please say it again.", priority=SpeechPriority.SYSTEM)
                if self.voice_engine:
                    self.voice_engine.conversation_state = "awaiting_destination"
                self.navigation_setup_active = False
                return

            # Voice engine already confirmed; avoid duplicate speech here.
            self.navigation_setup_active = False

            if not self.navigation_manager:
                self._safe_speak("Navigation system not available.", priority=SpeechPriority.SYSTEM)
                return

            session_id = str(uuid.uuid4())
            self.navigation_session_id = session_id
            started = self.navigation_manager.start_navigation(destination, mode, session_id)
            if not started:
                self._safe_speak("Navigation already active or failed to start.", priority=SpeechPriority.SYSTEM)
            
        except Exception as e:
            print(f"[MAIN] Error starting navigation: {e}")
            self._safe_speak("Navigation setup failed.", priority=SpeechPriority.SYSTEM)
            self.navigation_setup_active = False

    def _start_navigation_coords(self, lat: float, lng: float, mode: str, name: str):
        """Navigate to exact GPS coordinates from nearby places selection."""
        import uuid
        if not self.navigation_manager:
            self._safe_speak("Navigation system not available.", priority=SpeechPriority.SYSTEM)
            return
        session_id = str(uuid.uuid4())
        self.navigation_session_id = session_id
        started = self.navigation_manager.start_navigation((lat, lng), mode, session_id)
        if not started:
            self._safe_speak(
                "Navigation already active or failed to start.",
                priority=SpeechPriority.SYSTEM,
            )

    def handle_emergency(self, user_id: str, trigger_type: str, access_token: str = ""):
        """Central emergency handler."""
        try:
            # Fallback to last active user from web server if user_id is None
            if not user_id:
                from app.api import web_server
                user_id = getattr(web_server, "last_active_user_id", None)
            
            if not user_id:
                logger.error("User ID not set for emergency and no active web user found")
                return False, "User ID not set. Please log in."

            return self.emergency_manager.trigger_emergency(
                user_id=user_id,
                trigger_type=trigger_type,
                location_provider=self.get_current_location,
                voice_engine=self.voice_engine,
                access_token=access_token
            )
        except Exception as e:
            return False, f"Emergency handler failed: {e}"

    # ---------------- Describe flow ----------------
    def describe_scene(self):
        if not self.vision_engine:
            error_msg = "Vision engine not available"
            self._safe_speak(error_msg, priority=SpeechPriority.SYSTEM)
            return None
        
        try:
            # Provide immediate feedback that processing is starting
            self._safe_speak("Analyzing scene...", priority=SpeechPriority.SYSTEM)
            
            try:
                description = self.vision_engine.describe_scene()
            except Exception:
                dets = getattr(self.vision_engine, "detections", [])
                if dets:
                    names = [d["class_name"] for d in dets]
                    unique = sorted(set(names))
                    description = "I see " + ", ".join(unique)
                else:
                    description = "I do not detect any objects around you."
            
            # Speak the description with appropriate priority
            self._safe_speak(description, priority=SpeechPriority.REGULAR)
            
            print(f"Scene description: {description}")
            return description
        except Exception as e:
            print(f"Error in describe_scene: {e}")
            error_msg = "Sorry, I cannot describe the scene right now"
            self._safe_speak(error_msg, priority=SpeechPriority.SYSTEM)
            return error_msg

    # ---------------- Navigation flow ----------------
    def navigation_flow(self, destination: str = None, mode: str = None, session_id: str = None):
        """Strict navigation flow with proper state control."""

        # If already navigating, ask user if they want to restart
        if self.navigation_active:
            self._safe_speak("Navigation is already active. Say 'stop navigation' to cancel current route first.", priority=SpeechPriority.SYSTEM)
            return

        if hasattr(self, "speech_manager") and self.speech_manager:
            try:
                self.speech_manager.cancel_all_speech()
            except Exception:
                pass
        
        # Prevent multiple navigation setups from happening simultaneously
        if self.navigation_setup_active:
            print("[NAVIGATION] Navigation setup already in progress, ignoring new request")
            return
        self.navigation_setup_active = True

        try:
            # If no destination provided, defer to continuous listener
            if not destination:
                if self.voice_engine:
                    self.voice_engine.start_navigation_flow()
                    self.navigation_setup_active = False
                    return
                self._safe_speak("Voice engine not available. Cannot get destination.", priority=SpeechPriority.SYSTEM)
                self.navigation_setup_active = False
                return

            # Ask for mode if not provided - defer to continuous listener
            if not mode:
                if self.voice_engine:
                    self.voice_engine.set_destination_and_prompt_transport(destination)
                    self.navigation_setup_active = False
                    return
                mode = "walking"

            if not destination or not isinstance(destination, str):
                print("[NAV] Invalid destination input")
                self._safe_speak("I didn't catch the destination. Please say it again.", priority=SpeechPriority.SYSTEM)
                if self.voice_engine:
                    self.voice_engine.conversation_state = "awaiting_destination"
                self.navigation_setup_active = False  # Reset state on failure
                return



            # Navigation using Google Maps API
            self._safe_speak(f"Starting navigation to {destination} by {mode}", priority=SpeechPriority.SYSTEM)

            # Reset the navigation setup flag now that navigation is starting
            self.navigation_setup_active = False

            if not self.navigation_manager:
                self._safe_speak("Navigation system not available.", priority=SpeechPriority.SYSTEM)
                return

            if not session_id:
                session_id = str(uuid.uuid4())
            self.navigation_session_id = session_id
            started = self.navigation_manager.start_navigation(destination, mode, session_id)
            if not started:
                self._safe_speak("Navigation already active or failed to start.", priority=SpeechPriority.SYSTEM)

            
        except Exception as e:
            print(f"[NAVIGATION] Error in navigation flow: {e}")
            self._safe_speak("Navigation setup failed. Please try again.", priority=SpeechPriority.SYSTEM)
            self.navigation_setup_active = False  # Reset state on failure



    def _calculate_distance(self, lat1, lng1, lat2, lng2):
        r = 6371000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lng2 - lng1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return r * c

    def start_live_navigation(self):
        trigger_index = 0

        try:
            while self.is_navigating:
                time.sleep(3)

                if self.navigation_stop_event.is_set():
                    break

                if self.speech_manager and self.speech_manager.is_speaking:
                    continue

                if trigger_index >= len(self.navigation_steps):
                    self._speak_blocking('You have reached your destination.', priority=SpeechPriority.NAVIGATION)
                    break

                current_location = self.current_location
                if not current_location:
                    if self.web_mode:
                        current_location = self.get_current_gps_location()
                    else:
                        current_location = self.get_current_location()
                    self.current_location = current_location

                if not current_location:
                    continue

                user_lat, user_lng = current_location[0], current_location[1]
                trigger_step = self.navigation_steps[trigger_index]
                end_loc = trigger_step.get('end_location', {}) or {}
                step_lat = end_loc.get('lat')
                step_lng = end_loc.get('lng')
                if step_lat is None or step_lng is None:
                    trigger_index += 1
                    continue

                distance_to_turn = self._calculate_distance(user_lat, user_lng, step_lat, step_lng)
                if distance_to_turn < 40:
                    instruction = trigger_step.get('instruction', '').strip()
                    distance_text = trigger_step.get('distance_text') or 'an unknown distance'
                    if instruction:
                        self._speak_blocking(
                            f"{instruction} For {distance_text}.",
                            priority=SpeechPriority.NAVIGATION,
                        )
                    trigger_index += 1
        except Exception as e:
            print(f"[NAV_LIVE] ??? Error: {e}")
            import traceback
            traceback.print_exc()
            self._safe_speak('Navigation interrupted due to an error.', priority=SpeechPriority.SYSTEM)
        finally:
            print('[NAV_LIVE] Cleaning up...')
            self.navigation_active = False
            self.is_navigating = False
            try:
                self.vision_engine.set_navigation_mode(False)
            except Exception:
                pass
            self.current_route_instructions = []
            self.navigation_steps = []
            self.route_step_coordinates = []
            self.navigation_stop_event.clear()
            self.navigation_session_id = None

            # Reset voice engine state
            if self.voice_engine:
                try:
                    self.voice_engine.set_state('idle')
                    self.voice_engine.resume_listening_after_navigation()
                    print('[NAVIGATION] Voice engine state reset to idle')
                except Exception as e:
                    print(f"[NAVIGATION] Warning: Could not reset voice engine state: {e}")



    def stop_navigation(self):
        """Stop any running navigation."""
        if self.navigation_manager:
            self.navigation_manager.stop_navigation()
        else:
            self.navigation_active = False
            self.is_navigating = False
            self.navigation_setup_active = False


    # ---------------- Quit and lifecycle ----------------
    def quit_application(self):
        print("Quitting Vision Assistant...")
        try:
            self._safe_speak("Shutting down. Goodbye.", blocking=True)
        except Exception:
            pass
        self.cleanup()
        sys.exit(0)



    def run(self):
        if not self.vision_engine:
            print("Error: Vision engine not initialized")
            return
        
        if self.web_mode:
            # Run web server
            print("=" * 50)
            print("Vision Assistant - Web Server Mode")
            print("=" * 50)
            use_https = os.getenv("ENABLE_HTTPS", "").strip().lower() in {"1", "true", "yes"}
            protocol = "https" if use_https else "http"
            print(f"Starting web server on {protocol}://0.0.0.0:5000")
            print("Access the app from your mobile device at:")
            print(f"  - {protocol}://localhost:5000 (same device)")
            print(f"  - {protocol}://<your-ip>:5000 (mobile device on same network)")
            if use_https:
                print("  ⚠ Accept the self-signed certificate warning in your browser")
            print("=" * 50)
            try:
                run_server(host='0.0.0.0', port=5000, debug=False)
            except KeyboardInterrupt:
                print("\nKeyboardInterrupt")
                self.cleanup()


    def cleanup(self):
        print("Cleaning up resources...")
        try:
            # Emergency speech cancellation
            if hasattr(self, 'speech_manager') and self.speech_manager:
                self.speech_manager.cancel_all_speech()
                print("Speech cancelled and queue cleared")
            
            if self.navigation_active:
                self.stop_navigation()
            if self.vision_engine:
                self.vision_engine.cleanup()
            if self.voice_engine:
                self.voice_engine.cleanup()
        except Exception as e:
            print(f"Cleanup error: {e}")
        print("Cleanup completed")


def check_dependencies(web_mode: bool = True):
    required_modules = ['cv2', 'torch', 'ultralytics', 'PIL', 'numpy']
    if not web_mode:
        required_modules.extend(['google.cloud.speech', 'tkinter'])
    missing = []
    for m in required_modules:
        try:
            __import__(m)
        except ImportError:
            missing.append(m)
    if missing:
        print("Missing required modules:", missing)
        return False
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Vision Assistant')
    parser.add_argument('--desktop', action='store_true', help='Run in desktop GUI mode (legacy)')
    args = parser.parse_args()
    web_mode = not args.desktop
    
    print("=" * 50)
    print("Vision Assistant - Mobile Web App" if web_mode else "Vision Assistant - Desktop App")
    print("=" * 50)
    if not check_dependencies(web_mode=web_mode):
        sys.exit(1)
    
    app = VisionAssistant(web_mode=web_mode)
    app.run()


if __name__ == "__main__":
    main()
