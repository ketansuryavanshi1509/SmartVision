import time
import logging
import traceback
import subprocess
import threading
import queue
import requests
import os
import re
import math
from difflib import SequenceMatcher

import sounddevice as sd
from app.config import config
from app.core.logger import logger
from app.services.speech.groq_whisper import GroqWhisperSTT
from app.services.speech.speech_manager import SpeechPriority, speech_manager


def clean_text(text):
    text = (text or "").lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return text


FILLER_WORDS = {
    "yeah",
    "yes",
    "ok",
    "okay",
    "very good",
    "hmm",
    "huh",
}

NON_DESTINATION_PHRASES = {
    "good morning",
    "good afternoon",
    "good evening",
    "good night",
    "hello",
    "hi",
    "hey",
    "hey there",
    "how are you",
    "thank you",
    "thanks",
    "thanks a lot",
    "there we go",
    "oh no",
    "never mind",
    "not now",
}

NAVIGATION_COMMAND_WORDS = {
    "navigate",
    "navigation",
    "direction",
    "directions",
    "route",
    "routing",
    "map",
    "maps",
    "start",
    "begin",
    "destination",
    "location",
    "go",
    "travel",
}

TRANSPORT_WORDS = {
    "walk",
    "walking",
    "drive",
    "driving",
    "car",
    "bike",
    "bicycle",
    "motorbike",
    "motorcycle",
    "scooter",
    "two",
    "wheeler",
}

GENERIC_DESTINATION_TOKENS = {
    "station", "road", "street", "place", "market", "mall", "hospital", "school",
    "college", "temple", "mandir", "masjid", "mosque", "church", "bridge",
}

NEARBY_QUERY_MARKERS = ("nearest", "nearby", "closest")

PLACE_QUERY_NORMALIZATIONS = {
    "wine shop": "liquor store",
    "wine store": "liquor store",
    "liquor shop": "liquor store",
    "medical shop": "pharmacy",
    "chemist shop": "pharmacy",
    "chemist": "pharmacy",
    "medicine shop": "pharmacy",
    "petrol pump": "gas station",
    "petrol bunk": "gas station",
}

PLACE_TYPE_HINTS = {
    "liquor store": "liquor_store",
    "pharmacy": "pharmacy",
    "hospital": "hospital",
    "atm": "atm",
    "restaurant": "restaurant",
    "train station": "train_station",
    "railway station": "train_station",
    "bus station": "bus_station",
    "bus stop": "bus_station",
    "gas station": "gas_station",
    "airport": "airport",
    "mall": "shopping_mall",
}

MODE_PROMPT = "How would you like to travel? Walking, driving, two-wheeler, or public transport?"
MODE_REPROMPT = "Please say walking, driving, two-wheeler, or public transport."
PLACE_SELECTION_PROMPT = "Say the first one, second one, third one, or say the place name you want."
FRAGMENT_LEADERS = {
    "tell", "show", "find", "take", "go", "where", "what",
    "how", "navigate", "help", "describe", "nearest", "closest"
}

INDIAN_PLACE_ALIASES = {
    "allahabad": "prayagraj",
    "banaras": "varanasi",
    "benares": "varanasi",
    "bombay": "mumbai",
    "calcutta": "kolkata",
    "madras": "chennai",
    "bangalore": "bengaluru",
    "poona": "pune",
    "baroda": "vadodara",
}

SYSTEM_BOOT_PHRASE = clean_text("Vision Assistant I am ready start camera and voice recognition")

VALID_STATES = {"idle", "awaiting_destination", "awaiting_place_selection", "awaiting_mode", "navigating"}
VALID_STATE_TRANSITIONS = {
    "idle": {"awaiting_destination", "awaiting_place_selection", "navigating"},
    "awaiting_destination": {"idle", "awaiting_place_selection", "awaiting_mode"},
    "awaiting_place_selection": {"idle", "awaiting_destination", "awaiting_mode"},
    "awaiting_mode": {"idle", "awaiting_destination", "awaiting_place_selection", "navigating"},
    "navigating": {"idle"},
}


def is_valid_transition(old_state: str, new_state: str) -> bool:
    if new_state not in VALID_STATES:
        return False
    if old_state not in VALID_STATES:
        return True
    return new_state == old_state or new_state in VALID_STATE_TRANSITIONS.get(old_state, set())


def is_system_echo(transcript: str) -> bool:
    cleaned = clean_text(transcript)
    if not cleaned:
        return True
    if "vision assistant" in cleaned:
        return True
    if "camera and voice recognition" in cleaned:
        return True
    last_tts = getattr(speech_manager, "last_spoken_text", None)
    if last_tts:
        # Only consider recent TTS (within 8s) to avoid blocking valid commands
        last_ended_ts = getattr(speech_manager, "_last_speech_ended_ts", 0.0)
        if last_ended_ts and (time.time() - last_ended_ts) > 8.0:
            return False
        last_clean = clean_text(last_tts)
        if cleaned == last_clean:
            return True
        try:
            # Substring check: only flag as echo when the transcript is a
            # substantial portion of the TTS text (not a short response to a
            # long prompt like "first one" from "Say the first one, second...").
            if cleaned in last_clean and len(cleaned) >= len(last_clean) * 0.4:
                return True
            if last_clean in cleaned:
                return True
            # Word-overlap check: require high overlap in BOTH directions so
            # short user replies (e.g. "first one") that share words with a
            # long TTS prompt are not mistakenly suppressed.
            transcript_words = set(cleaned.split())
            tts_words = set(last_clean.split())
            if len(transcript_words) >= 3 and len(tts_words) >= 3:
                fwd = len(transcript_words & tts_words) / len(transcript_words)
                rev = len(transcript_words & tts_words) / len(tts_words)
                if fwd >= 0.60 and rev >= 0.40:
                    return True
            if SequenceMatcher(None, cleaned, last_clean).ratio() >= 0.55:
                return True
        except Exception:
            pass
    return False


class VoiceEngine:
    """
    Voice processing engine with deterministic state machine navigation.
    States: idle, awaiting_destination, awaiting_mode, navigating
    """

    def __init__(self):
        self.tts_available = True
        self.system_boot_completed = False
        self.use_external_navigation = True

        # State machine
        self._state = "idle"
        self.is_navigating = False
        self.failed_geocode_attempts = 0
        self.pending_destination = None
        self.destination = None
        self.transport_mode = None
        self.current_location = None
        self.navigation_steps = []
        self.current_step_index = 0
        self.navigation_polyline = None
        self.pending_destination_coords = None
        self.command_callback = None
        self._emergency_override_until = 0.0
        self.min_confidence = 0.5
        self.place_match_threshold = config.PLACE_MATCH_THRESHOLD
        self.places_api_key = None
        self.location_provider = None
        self.pending_place_suggestion = None
        self.pending_place_suggestions = []
        self.pending_nearby_results = []
        self.pending_nearby_query = None
        self._nav_last_valid_input_at = time.time()
        self._pending_fragment_text = ""
        self._pending_fragment_ts = 0.0
        self._fragment_merge_window_s = max(1.0, float(config.STT_FRAGMENT_MERGE_SECONDS))

        # Continuous listener
        self.listening = False
        self._listen_enabled = threading.Event()
        self._listen_enabled.set()
        self._pause_lock = threading.Lock()
        self.stt = GroqWhisperSTT(
            sample_rate_hz=config.MIC_SAMPLE_RATE,
            language_code=(config.STT_LANGUAGE or "en"),
        )
        self._transcript_queue = queue.Queue()
        self._transcript_thread = threading.Thread(target=self._transcript_worker, daemon=True)
        self._transcript_thread.start()

        # Speech queue
        self._speech_queue = queue.Queue()
        self._speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self._speech_thread.start()
        self._current_speech = None
        self._current_speech_text = ""
        self._speech_lock = threading.Lock()
        self._last_queued_text = ""

        self.microphone = None
        self._initialize_microphone()

        logger.info("Voice engine initialized with Groq Whisper STT")

    def _initialize_microphone(self):
        """Initialize microphone with error handling."""
        try:
            info = sd.query_devices(kind="input")
            self.microphone = info
            print("Microphone initialized")
        except Exception as e:
            self.microphone = None
            print(f"Microphone initialization failed: {e}")

    def set_speech_manager(self, speech_manager):
        """Set reference to speech manager."""
        self.speech_manager = speech_manager

    def set_places_api_key(self, api_key: str):
        """Set API key for Google Places Autocomplete."""
        self.places_api_key = api_key

    def set_location_provider(self, provider):
        """Set a callable that returns (lat, lng, city) or None."""
        self.location_provider = provider

    def get_current_location(self):
        if self.location_provider and callable(self.location_provider):
            try:
                loc = self.location_provider()
                if loc and isinstance(loc, (list, tuple)) and len(loc) >= 2:
                    return {"lat": loc[0], "lng": loc[1]}
            except Exception:
                pass
        return None

    # ---------------- TTS ----------------
    def speak(self, text: str, blocking: bool = False, priority="NORMAL"):
        """Convert text to speech."""
        if not text:
            return

        priority_value = self._normalize_priority(priority)

        if isinstance(priority, str) and priority.upper() == "EMERGENCY":
            self.stop_speaking_immediately()

        # We no longer pause listening here to allow barge-in (STOP/EMERGENCY).
        # Echo suppression in is_system_echo handles the feedback.

        if hasattr(self, "speech_manager") and self.speech_manager:
            done_event = threading.Event()

            def _on_done():
                done_event.set()
                self._resume_listening()

            try:
                priority_enum = self._priority_to_enum(priority_value)
                ok = self.speech_manager.speak(text, priority_enum, "voice_engine", _on_done)
                if not ok:
                    self._resume_listening()
                    return
            except Exception as e:
                print(f"Speech manager failed: {e}")
                self._resume_listening()
                return

            if blocking:
                done_event.wait()
            return

        with self._speech_lock:
            if text == self._last_queued_text or text == self._current_speech_text:
                return

        done_event = threading.Event()
        self._last_queued_text = text
        self._speech_queue.put((text, done_event))
        if blocking:
            done_event.wait()

    def stop_speaking_immediately(self):
        """Interrupt all speech immediately (emergency override)."""
        if hasattr(self, "speech_manager") and self.speech_manager:
            try:
                self.speech_manager.cancel_all_speech()
            except Exception:
                pass
        try:
            requests.post("http://localhost:5000/api/speech/cancel", timeout=1)
        except Exception:
            pass
        self._clear_speech_queue()

    def _normalize_priority(self, priority):
        if isinstance(priority, int):
            return priority
        if isinstance(priority, str):
            p = priority.upper().strip()
            if p == "EMERGENCY":
                return 100
            if p == "SYSTEM":
                return 60
            if p == "NAVIGATION":
                return 80
            if p == "BACKGROUND":
                return 20
            return 40
        try:
            return int(priority)
        except Exception:
            return 40

    def _priority_to_enum(self, priority_value: int) -> SpeechPriority:
        for p in SpeechPriority:
            if p.value == priority_value:
                return p
        if priority_value >= 80:
            return SpeechPriority.HIGH
        if priority_value >= 40:
            return SpeechPriority.REGULAR
        return SpeechPriority.BACKGROUND

    def _speak_sync(self, text: str):
        try:
            print(f"Speaking: {text}")
            safe_text = text.replace("'", "''")
            cmd = (
                f"powershell -Command \"Add-Type -AssemblyName System.Speech; "
                f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{safe_text}');\""
            )
            subprocess.run(cmd, shell=True, capture_output=True)
        except Exception as e:
            print(f"Error speaking: {e}")

    def _speech_worker(self):
        while True:
            text, done_event = self._speech_queue.get()
            try:
                with self._speech_lock:
                    if self._current_speech is not None:
                        self._current_speech.wait()
                    self._current_speech = done_event
                    self._current_speech_text = text
                    self._speak_sync(text)
            finally:
                with self._speech_lock:
                    self._current_speech = None
                    self._current_speech_text = ""
                done_event.set()
                self._resume_listening()

    def _clear_speech_queue(self):
        cleared = 0
        while True:
            try:
                _, done_event = self._speech_queue.get_nowait()
                done_event.set()
                cleared += 1
            except queue.Empty:
                break
        if cleared:
            print(f"[SPEECH] Cleared {cleared} queued items")

    def _pause_listening(self):
        with self._pause_lock:
            self._listen_enabled.clear()
        try:
            self.stt.pause()
        except Exception:
            pass

    def _resume_listening(self):
        with self._pause_lock:
            self._listen_enabled.set()
        try:
            self.stt.resume()
        except Exception:
            pass

    def pause_listening_for_tts(self):
        # We no longer pause listening during TTS to allow "STOP" or "EMERGENCY" barge-in.
        # Echo suppression logic in is_system_echo handles the feedback.
        pass

    def resume_listening_after_tts(self):
        # No-op since we didn't pause.
        pass

    def pause_listening_for_navigation(self):
        self.is_navigating = True
        # Keep listening active during navigation so we can hear "STOP" or "EMERGENCY"
        # We no longer call self._pause_listening() here.
        pass

    def resume_listening_after_navigation(self):
        self.is_navigating = False
        # self._resume_listening() is not needed since we didn't pause.
        self.set_state("idle")
        self.start_listening()

    # ---------------- STATE MACHINE ----------------
    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, new_state: str):
        self.set_state(new_state)

    @property
    def conversation_state(self):
        return self._state

    @conversation_state.setter
    def conversation_state(self, new_state: str):
        self.set_state(new_state)

    def set_state(self, new_state: str):
        voice_logger = logging.getLogger("smartvision.voice")
        if not new_state:
            return
        old_state = self._state
        if new_state == old_state:
            return
        if not is_valid_transition(old_state, new_state):
            voice_logger.warning("[STATE] blocked transition %s -> %s", old_state, new_state)
            return
        voice_logger.info(f"[STATE] {old_state} -> {new_state}")
        self._state = new_state
        if new_state in {"awaiting_destination", "awaiting_place_selection", "awaiting_mode"}:
            self._nav_last_valid_input_at = time.time()

    def _reset_to_idle(self):
        if self.state != "idle":
            self.set_state("idle")
        self.destination = None
        self.transport_mode = None
        self.pending_destination = None
        self.pending_destination_coords = None
        self._emergency_override_until = 0.0
        self.failed_geocode_attempts = 0
        self.pending_place_suggestion = None
        self.pending_place_suggestions = []
        self.pending_nearby_results = []
        self.pending_nearby_query = None
        self._nav_last_valid_input_at = time.time()

    # ---------------- INTENT + ENTITY LAYERS ----------------
    def detect_intent(self, text: str) -> str:
        normalized_text = self._normalize_indian_voice_text(text)
        if not normalized_text:
            return "unknown"
        if "stop navigation" in normalized_text or "cancel" in normalized_text or "stop it" in normalized_text or normalized_text.strip() == "stop":
            return "stop_navigation"
        if normalized_text == "navigate" or normalized_text.startswith("navigate"):
            return "navigate"
        
        # "I want to visit X" and similar → navigate intent
        if re.search(r'\bi\s+want\s+to\s+(visit|go\s+to|go)\b', normalized_text):
            return "navigate"
        if re.search(r'\bwanna\s+(visit|go\s+to|go)\b', normalized_text):
            return "navigate"
        
        if (
            normalized_text.startswith("go to")
            or normalized_text.startswith("directions")
            or normalized_text.startswith("take me")
            or normalized_text.startswith("find")
            or normalized_text.startswith("search")
            or normalized_text.startswith("show me")
            or normalized_text.startswith("want to go")
            or normalized_text.startswith("i want to go")
            or normalized_text.startswith("would like to go")
            or normalized_text.startswith("like to go")
            or normalized_text.startswith("me to")
            or any(marker in normalized_text for marker in NEARBY_QUERY_MARKERS)
            or "near me" in normalized_text
            or "around me" in normalized_text
            or "guide me" in normalized_text
            or "route" in normalized_text
            or "rasta" in normalized_text
            or "jana hai" in normalized_text
            or "le chalo" in normalized_text
        ):
            return "navigate"
        if "describe" in normalized_text or "what do you see" in normalized_text:
            return "describe_scene"
        if (
            "location" in normalized_text
            or "where am i" in normalized_text
            or "where i am" in normalized_text
            or normalized_text in {"gps", "my location", "tell me my location", "tell my location", "me my location"}
        ):
            return "get_location"
        return "unknown"

    def extract_destination(self, text: str):
        text = (text or "").lower().strip()
        if not text:
            return None

        filler_phrases = [
            "i would like to go to",
            "i want to go to",
            "take me to",
            "navigate to",
            "go to",
            "after translation",
            "oh sorry",
            "sorry",
            "please",
            "can you",
        ]

        cleaned = text
        for phrase in filler_phrases:
            if phrase in cleaned:
                cleaned = cleaned.replace(phrase, " ")

        cleaned = re.sub(r"[^\w\s]", " ", cleaned)
        filler_words = {"i", "would", "like", "to", "the", "a"}
        tokens = [t for t in cleaned.split() if t not in filler_words]
        cleaned = " ".join(tokens).strip()
        cleaned = " ".join(cleaned.split())

        if len(cleaned) < 3:
            return None
        if len(cleaned.split()) < 1:
            return None
        if self._contains_profanity(cleaned):
            return None
        if any(verb in cleaned.split() for verb in ("go", "determine", "make", "do")):
            return None

        print(f"[ENTITY CLEANED]: {cleaned}")
        return cleaned

    def correct_destination(self, text: str):
        cleaned = (text or "").lower().strip()
        if not cleaned:
            return None
        corrections = {
            "turn a railway station": "torna railway station",
            "town railway station": "torna railway station",
            "tane": "thane",
        }
        for wrong, right in corrections.items():
            if wrong in cleaned:
                cleaned = cleaned.replace(wrong, right)
        return cleaned

    def _normalize_place_text(self, text: str) -> str:
        return " ".join(self._normalize_indian_voice_text(text).split())

    def _current_location_tuple(self):
        if self.location_provider and callable(self.location_provider):
            try:
                loc = self.location_provider()
                if loc and isinstance(loc, (list, tuple)) and len(loc) >= 2:
                    return (float(loc[0]), float(loc[1]), loc[2] if len(loc) >= 3 else None)
            except Exception:
                pass
        return None

    def _clear_pending_fragment(self):
        self._pending_fragment_text = ""
        self._pending_fragment_ts = 0.0

    def _merge_pending_fragment(self, transcript: str) -> str:
        pending_text = getattr(self, "_pending_fragment_text", "") or ""
        pending_ts = float(getattr(self, "_pending_fragment_ts", 0.0) or 0.0)
        if not pending_text or not pending_ts:
            return transcript
        if (time.time() - pending_ts) > float(getattr(self, "_fragment_merge_window_s", 4.0) or 4.0):
            self._clear_pending_fragment()
            return transcript
        merged = f"{pending_text} {transcript}".strip()
        self._clear_pending_fragment()
        return merged

    def _should_buffer_unknown_fragment(self, transcript: str) -> bool:
        if self.state not in {"idle", "awaiting_destination"}:
            return False
        normalized = clean_text(transcript)
        if not normalized:
            return False
        words = normalized.split()
        if len(words) > 4:
            return False
        if any(marker in normalized for marker in NEARBY_QUERY_MARKERS):
            return False
        if "location" in normalized or "where am i" in normalized:
            return False
        if self.extract_destination(normalized):
            return False
        return len(words) <= 2 or words[0] in FRAGMENT_LEADERS

    def _extract_selection_index(self, text: str):
        normalized = self._normalize_indian_voice_text(text)
        ordinal_tokens = [
            ("third", 3),
            ("second", 2),
            ("first", 1),
            ("3", 3),
            ("2", 2),
            ("1", 1),
            ("three", 3),
            ("two", 2),
            ("one", 1),
        ]
        for token, idx in ordinal_tokens:
            if re.search(rf"\b{re.escape(token)}\b", normalized):
                return idx
        return None

    def _local_interpret_utterance(self, transcript: str):
        normalized_text = self._normalize_indian_voice_text(transcript)
        nearby_query = self._extract_nearby_search_query(normalized_text)
        selection_index = self._extract_selection_index(normalized_text)
        candidate_intent = self.detect_intent(normalized_text)
        if candidate_intent == "unknown" and self.state == "awaiting_place_selection" and (selection_index or normalized_text):
            candidate_intent = "select_option"

        destination = ""
        if candidate_intent == "navigate":
            destination = transcript
        elif self.state == "awaiting_destination":
            destination = transcript

        if nearby_query:
            candidate_intent = "navigate_nearby"

        if self._is_yes(transcript):
            candidate_intent = "yes"
        elif self._is_no(transcript):
            candidate_intent = "no"
        elif self.state == "awaiting_place_selection" and selection_index is not None:
            candidate_intent = "select_option"

        return {
            "cleaned_text": normalized_text or transcript,
            "intent": candidate_intent,
            "destination": destination,
            "nearby_query": nearby_query,
        }

    def _get_location_bias(self):
        if self.location_provider and callable(self.location_provider):
            try:
                loc = self.location_provider()
                if loc and isinstance(loc, (list, tuple)) and len(loc) >= 2:
                    return (loc[0], loc[1])
            except Exception:
                pass
        return None

    def _places_autocomplete(self, text: str):
        if not text:
            return []
        api_key = self.places_api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        if not api_key:
            return []

        params = {
            "input": text,
            "key": api_key,
            "components": "country:in",
            "radius": 50000,
        }
        loc = self._get_location_bias()
        if loc:
            params["location"] = f"{loc[0]},{loc[1]}"

        try:
            resp = requests.get(
                "https://maps.googleapis.com/maps/api/place/autocomplete/json",
                params=params,
                timeout=6,
            )
            if resp.status_code != 200:
                return []
            payload = resp.json()
            predictions = payload.get("predictions") or []
            if not predictions:
                return []
            suggestions = []
            for prediction in predictions[:3]:
                description = prediction.get("description")
                if description:
                    suggestions.append(description)
            return suggestions
        except Exception:
            return []

    def geocode_destination(self, place_name: str):
        api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY not set")

        url = "https://maps.googleapis.com/maps/api/geocode/json"
        params = {
            "address": place_name,
            "key": api_key,
            "region": "in",
        }

        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        if data.get("status") == "OK" and data.get("results"):
            result = data["results"][0]
            location = result["geometry"]["location"]
            return {
                "lat": location["lat"],
                "lng": location["lng"],
                "formatted_address": result.get("formatted_address") or place_name,
            }
        else:
            logger = logging.getLogger("smartvision.voice")
            logger.warning("Google forward geocode failed: status=%s", data.get("status"))

        return None

    def start_navigation(self, dest_lat, dest_lng, mode="walking"):
        if self.conversation_state != "navigating":
            return
        worker = threading.Thread(
            target=self._start_navigation_worker,
            args=(dest_lat, dest_lng, mode),
            daemon=True,
        )
        worker.start()

    def _start_navigation_from_confirmation(self):
        if not self.destination:
            return
        if not self.transport_mode:
            self.transport_mode = "walking"
        if not self.pending_destination_coords:
            try:
                result = self.geocode_destination(self.destination)
                if result:
                    self.pending_destination_coords = (result["lat"], result["lng"])
            except Exception:
                pass
        if self.pending_destination_coords:
            self.start_navigation(
                self.pending_destination_coords[0],
                self.pending_destination_coords[1],
                self.transport_mode,
            )

    def _start_navigation_worker(self, dest_lat, dest_lng, mode):
        logger = logging.getLogger("smartvision.voice")
        api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not api_key:
            self.speak("Unable to calculate route.")
            return

        if not self.current_location:
            self.current_location = self.get_current_location()

        if not self.current_location:
            self.speak("Unable to calculate route.")
            return

        origin = f"{self.current_location['lat']},{self.current_location['lng']}"
        destination = f"{dest_lat},{dest_lng}"

        url = "https://maps.googleapis.com/maps/api/directions/json"
        api_mode = mode or "walking"
        if api_mode == "two_wheeler":
            api_mode = "driving"

        print(f"[NAVIGATION] Origin: {origin}")
        print(f"[NAVIGATION] Destination: {destination}")
        print(f"[NAVIGATION] Mode: {api_mode}")

        params = {
            "origin": origin,
            "destination": destination,
            "mode": api_mode,
            "key": api_key,
        }

        response = requests.get(url, params=params, timeout=5)
        data = response.json()

        print("[DIRECTIONS STATUS]:", data.get("status"))
        if data.get("status") != "OK":
            print("[DIRECTIONS ERROR]:", data)
            self.speak("Unable to calculate route.")
            return

        route = data["routes"][0]
        leg = route["legs"][0]
        distance = leg["distance"]["text"]
        duration = leg["duration"]["text"]

        steps = []
        for step in leg["steps"]:
            instruction = step["html_instructions"]
            instruction = clean_html_instruction(instruction)
            lat = step["end_location"]["lat"]
            lng = step["end_location"]["lng"]
            steps.append({
                "instruction": instruction,
                "lat": lat,
                "lng": lng,
            })

        print(f"[NAVIGATION] Extracted {len(steps)} steps")
        self.navigation_polyline = route["overview_polyline"]["points"]
        self.navigation_steps = steps
        self.current_step_index = 0

        logger.info("Navigation started")
        if steps:
            self.speak(f"Route found. Distance {distance}. Estimated time {duration}.", priority="NAVIGATION")
            self.speak(f"First instruction: {steps[0]['instruction']}", priority="NAVIGATION")
        else:
            self.speak("Route found but no instructions available.", priority="NAVIGATION")

    def speak_next_step(self):
        logger = logging.getLogger("smartvision.voice")
        if self.conversation_state != "navigating":
            return
        if self.current_step_index >= len(self.navigation_steps):
            logger.info("Navigation completed")
            self.speak("You have arrived at your destination.")
            self.set_state("idle")
            return

        step = self.navigation_steps[self.current_step_index]
        instruction = step["html_instructions"]

        clean_instruction = clean_html_instruction(instruction)

        logger.info("Speaking step %d", self.current_step_index + 1)
        self.speak(clean_instruction, priority="NAVIGATION")

    def _place_match_score(self, raw_text: str, suggestion: str) -> float:
        raw_norm = self._normalize_place_text(raw_text)
        sug_norm = self._normalize_place_text(suggestion)
        if not raw_norm or not sug_norm:
            return 0.0
        ratio = SequenceMatcher(None, raw_norm, sug_norm).ratio()
        raw_tokens = set(raw_norm.split())
        sug_tokens = set(sug_norm.split())
        if not raw_tokens:
            return ratio
        overlap = len(raw_tokens & sug_tokens) / max(1, len(raw_tokens))
        return 0.7 * ratio + 0.3 * overlap

    def is_valid_destination(self, dest: str) -> bool:
        cleaned = self._normalize_indian_voice_text(dest)
        if not cleaned:
            return False
        if cleaned in FILLER_WORDS:
            return False
        if self._is_small_talk_or_noise(cleaned):
            return False
        if len(cleaned) < 3:
            return False
        if self._contains_profanity(cleaned):
            return False
        tokens = cleaned.split()
        if len(tokens) > 6:
            return False
        if len(tokens) == 1 and tokens[0] in {"town", "station", "place", "road"}:
            return False
        return True

    def _is_small_talk_or_noise(self, cleaned: str) -> bool:
        cleaned = " ".join((cleaned or "").lower().split())
        if not cleaned:
            return True
        if cleaned in NON_DESTINATION_PHRASES:
            return True
        tokens = cleaned.split()
        noise_words = {
            "good", "morning", "afternoon", "evening", "night",
            "hello", "hi", "hey", "thanks", "thank", "you",
            "there", "we", "go", "oh", "no", "okay", "ok",
        }
        if tokens and all(token in noise_words for token in tokens):
            return True
        return False

    def _contains_profanity(self, text: str) -> bool:
        bad_words = {
            "shit",
            "shitty",
            "fuck",
            "fucking",
            "bitch",
            "asshole",
            "bastard",
            "damn",
        }
        words = set((text or "").lower().split())
        return any(word in bad_words for word in words)

    def _summarize_place_label(self, label: str) -> str:
        parts = [part.strip() for part in str(label or "").split(",") if part.strip()]
        if not parts:
            return ""
        return ", ".join(parts[:2])

    def _format_spoken_distance(self, meters: float) -> str:
        if meters is None:
            return ""
        meters_value = float(meters)
        if meters_value < 1000:
            rounded = int(10 * round(meters_value / 10.0))
            return f"{max(rounded, 10)} meters"
        return f"{meters_value / 1000.0:.1f} kilometers"

    def _normalize_place_query(self, query: str) -> str:
        normalized = self._normalize_indian_voice_text(query)
        for source, target in PLACE_QUERY_NORMALIZATIONS.items():
            normalized = re.sub(rf"\b{re.escape(source)}\b", target, normalized)
        return " ".join(normalized.split())

    def _extract_nearby_search_query(self, cleaned: str) -> str:
        normalized = self._normalize_place_query(cleaned)
        if not normalized:
            return ""
        
        # Handle "I want to visit X" / "take me to X" — extract destination as search query
        specific_match = re.search(
            r'\bi\s+want\s+to\s+(?:visit|go\s+to|go)\s+(.+)', normalized
        )
        if not specific_match:
            specific_match = re.search(
                r'\b(?:take\s+me\s+to|wanna\s+(?:go\s+to|visit)|visit)\s+(.+)', normalized
            )
        if specific_match:
            dest = specific_match.group(1).strip()
            dest = re.sub(r'\b(near me|around me|close by)\b', '', dest).strip()
            return dest
        
        for marker in NEARBY_QUERY_MARKERS:
            if marker in normalized:
                query = normalized.split(marker, 1)[1].strip()
                query = re.sub(r"^(to|the|a|an|my|me)\s+", "", query)
                query = re.sub(r"\b(near me|nearby|around me|around here|close by|close to me)\b", " ", query)
                return " ".join(query.split())
        near_me_match = re.search(
            r"(?:find|search|show me|take me to|guide me to|me to)?\s*(?:the\s+)?(.+?)\s+(?:near me|around me|around here|close by|close to me)$",
            normalized,
        )
        if near_me_match:
            query = near_me_match.group(1).strip()
            query = re.sub(r"^(to|the|a|an|my|me)\s+", "", query)
            return " ".join(query.split())
        return ""

    def _search_nearby_place(self, query: str):
        api_key = self.places_api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        loc = self._current_location_tuple()
        if not api_key or not loc or not query:
            return None

        normalized_query = self._normalize_place_query(query)
        params = {
            "location": f"{loc[0]},{loc[1]}",
            "keyword": normalized_query,
            "rankby": "distance",
            "key": api_key,
        }
        place_type = PLACE_TYPE_HINTS.get(normalized_query)
        if place_type:
            params["type"] = place_type

        try:
            response = requests.get(
                "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
                params=params,
                timeout=6,
            )
            if response.status_code != 200:
                return None
            payload = response.json()
            results = payload.get("results") or []
            if not results:
                return None
            top_result = results[0]
            geometry = (top_result.get("geometry") or {}).get("location") or {}
            if "lat" not in geometry or "lng" not in geometry:
                return None
            distance_m = self._haversine_distance_m(loc[0], loc[1], float(geometry["lat"]), float(geometry["lng"]))
            vicinity = top_result.get("vicinity") or top_result.get("formatted_address") or ""
            name = top_result.get("name") or normalized_query.title()
            formatted_address = f"{name}, {vicinity}".strip(", ")
            return {
                "lat": float(geometry["lat"]),
                "lng": float(geometry["lng"]),
                "formatted_address": formatted_address,
                "name": name,
                "distance_m": distance_m,
                "search_query": normalized_query,
            }
        except Exception:
            return None

    def _search_nearby_places(self, query: str, max_results: int = 2):
        """Search nearby places: NearbySearch first, TextSearch as fallback. Returns max 2 results."""
        api_key = self.places_api_key or os.getenv("GOOGLE_MAPS_API_KEY")
        loc = self._current_location_tuple()
        if not api_key or not loc or not query:
            return []

        normalized_query = self._normalize_place_query(query)
        cap = max(1, int(max_results))

        def _parse(raw_items):
            out = []
            for item in raw_items:
                geo = (item.get("geometry") or {}).get("location") or {}
                if "lat" not in geo or "lng" not in geo:
                    continue
                p_lat, p_lng = float(geo["lat"]), float(geo["lng"])
                dist_m = self._haversine_distance_m(loc[0], loc[1], p_lat, p_lng)
                vicinity = item.get("vicinity") or item.get("formatted_address") or ""
                name = item.get("name") or normalized_query.title()
                out.append({
                    "lat": p_lat, "lng": p_lng,
                    "formatted_address": f"{name}, {vicinity}".strip(", "),
                    "name": name, "vicinity": vicinity,
                    "distance_m": dist_m,
                    "rating": item.get("rating"),
                    "place_id": item.get("place_id", ""),
                    "search_query": normalized_query,
                })
            out.sort(key=lambda x: x["distance_m"])
            return out[:cap]

        # 1. NearbySearch
        try:
            params = {
                "location": f"{loc[0]},{loc[1]}",
                "keyword": normalized_query,
                "rankby": "distance",
                "key": api_key,
            }
            place_type = PLACE_TYPE_HINTS.get(normalized_query)
            if place_type:
                params["type"] = place_type
            resp = requests.get(
                "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
                params=params, timeout=7,
            )
            if resp.status_code == 200:
                payload = resp.json()
                if payload.get("status") == "OK":
                    results = _parse(payload.get("results") or [])
                    if results:
                        return results
        except Exception:
            pass

        # 2. TextSearch fallback
        try:
            resp = requests.get(
                "https://maps.googleapis.com/maps/api/place/textsearch/json",
                params={
                    "query": normalized_query,
                    "location": f"{loc[0]},{loc[1]}",
                    "radius": 20000,
                    "key": api_key,
                    "region": "in",
                },
                timeout=7,
            )
            if resp.status_code == 200:
                payload = resp.json()
                if payload.get("status") == "OK":
                    results = _parse(payload.get("results") or [])
                    if results:
                        return results
        except Exception:
            pass

        return []

    def _present_nearby_places(self, results, query: str, priority: str = "SYSTEM") -> bool:
        """Store results, set state to awaiting_place_selection, speak 2 clear options."""
        if not results:
            return False
        results = results[:2]
        self.pending_nearby_results = list(results)
        self.pending_nearby_query = query
        self.pending_place_suggestion = None
        self.pending_place_suggestions = []
        self.pending_destination = None
        self.pending_destination_coords = None
        self.destination = None
        self.set_state("awaiting_place_selection")

        parts = []
        for idx, r in enumerate(results, 1):
            name = r.get("name") or self._summarize_place_label(r.get("formatted_address", ""))
            vicinity = r.get("vicinity", "")
            dist_text = self._format_spoken_distance(r.get("distance_m"))
            rating = r.get("rating")
            desc = f"Option {idx}: {name}"
            if vicinity:
                desc += f", at {vicinity}"
            if dist_text:
                desc += f", about {dist_text} away"
            if rating:
                desc += f", rated {rating} out of 5"
            parts.append(desc)

        self.speak(
            f"I found {len(results)} nearby {query}. {' '.join(parts)}. "
            f"Say first or second to choose, or say the name of the place.",
            priority=priority,
        )
        return True

    def _match_nearby_selection(self, user_input: str):
        options = list(self.pending_nearby_results or [])
        if not options:
            return None

        index_value = None
        if index_value is None:
            normalized = self._normalize_indian_voice_text(user_input)
            ordinal_tokens = [
                ("third", 3),
                ("second", 2),
                ("first", 1),
                ("3", 3),
                ("2", 2),
                ("1", 1),
                ("three", 3),
                ("two", 2),
                ("one", 1),
            ]
            for token, idx in ordinal_tokens:
                if re.search(rf"\b{re.escape(token)}\b", normalized):
                    index_value = idx
                    break
        if index_value is not None and 1 <= int(index_value) <= len(options):
            return options[int(index_value) - 1]

        candidate_names = [user_input]

        best_match = None
        best_score = 0.0
        for candidate in candidate_names:
            candidate_norm = self._normalize_place_text(candidate)
            if not candidate_norm:
                continue
            for result in options:
                label = result.get("name") or result.get("formatted_address") or ""
                label_norm = self._normalize_place_text(label)
                if not label_norm:
                    continue
                score = SequenceMatcher(None, candidate_norm, label_norm).ratio()
                if candidate_norm in label_norm:
                    score = max(score, 0.92)
                if score > best_score:
                    best_score = score
                    best_match = result
        if best_score >= 0.55:
            return best_match
        return None

    def _select_nearby_destination(self, selected_result: dict, priority: str = "SYSTEM") -> bool:
        if not selected_result:
            return False
        self.pending_nearby_results = []
        self.pending_nearby_query = None
        self.pending_destination = selected_result
        self.destination = selected_result.get("formatted_address")
        self.pending_destination_coords = (selected_result["lat"], selected_result["lng"])
        self.set_state("awaiting_mode")
        name = selected_result.get("name") or self._summarize_place_label(self.destination)
        self.speak(f"Okay. {name} selected. {MODE_PROMPT}", priority=priority)
        return True

    def _is_street_level_place(self, label: str) -> bool:
        normalized = self._normalize_indian_voice_text(label)
        if not normalized:
            return False
        if re.match(r"^\d+", normalized):
            return True
        street_markers = {
            "road", "rd", "street", "st", "lane", "ln", "marg", "gali", "nagar",
            "sector", "phase", "opp", "opposite", "market area",
        }
        return any(marker in normalized for marker in street_markers)

    def _haversine_distance_m(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        earth_radius_m = 6371000.0
        d_lat = math.radians(lat2 - lat1)
        d_lng = math.radians(lng2 - lng1)
        a = (
            math.sin(d_lat / 2.0) ** 2
            + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lng / 2.0) ** 2
        )
        return 2.0 * earth_radius_m * math.asin(math.sqrt(a))

    def _is_ambiguous_far_result(self, cleaned_input: str, result: dict) -> bool:
        tokens = [token for token in (cleaned_input or "").split() if token]
        if len(tokens) != 1:
            return False
        token = tokens[0]
        if token in GENERIC_DESTINATION_TOKENS:
            return True

        formatted_address = result.get("formatted_address") or ""
        if not self._is_street_level_place(formatted_address):
            return False

        current_loc = self._current_location_tuple()
        if not current_loc:
            return False

        try:
            distance_m = self._haversine_distance_m(
                current_loc[0],
                current_loc[1],
                float(result["lat"]),
                float(result["lng"]),
            )
        except Exception:
            return False

        return distance_m >= 50000.0

    def normalize_transport(self, text: str):
        normalized = self._normalize_indian_voice_text(text)
        if not normalized:
            return None

        normalized = re.sub(r"\btu\b", "to", normalized)
        normalized = re.sub(r"\btoo\b", "to", normalized)

        phrase_mapping = {
            "public transportation": "transit",
            "public transport": "transit",
            "by public transport": "transit",
            "by public transportation": "transit",
            "auto rickshaw": "transit",
            "auto rikshaw": "transit",
            "auto": "transit",
            "rickshaw": "transit",
            "bus stand": "transit",
            "bus": "transit",
            "train": "transit",
            "metro": "transit",
            "subway": "transit",
            "tram": "transit",
            "two wheeler": "two_wheeler",
            "two wheel": "two_wheeler",
            "tu wheeler": "two_wheeler",
            "2 wheeler": "two_wheeler",
            "two wheeled": "two_wheeler",
            "motor bike": "two_wheeler",
            "motorbike": "two_wheeler",
            "motor cycle": "two_wheeler",
            "motorcycle": "two_wheeler",
            "scooter": "two_wheeler",
            "scooty": "two_wheeler",
            "activa": "two_wheeler",
            "rapido": "two_wheeler",
            "bike": "two_wheeler",
            "on foot": "walking",
            "by foot": "walking",
            "paidal": "walking",
            "paidal hi": "walking",
            "walk": "walking",
            "walking": "walking",
            "by car": "driving",
            "car": "driving",
            "cab": "driving",
            "taxi": "driving",
            "uber": "driving",
            "ola": "driving",
            "gadi": "driving",
            "drive": "driving",
            "driving": "driving",
            "gaadi": "driving",
            "bicycle": "bicycling",
            "cycle": "bicycling",
            "cycling": "bicycling",
            "bicycling": "bicycling",
        }

        for phrase, mode in sorted(phrase_mapping.items(), key=lambda item: len(item[0]), reverse=True):
            if re.search(rf"\b{re.escape(phrase)}\b", normalized):
                return mode

        stripped = re.sub(
            r"\b(i|im|i am|want|would|like|to|go|travel|move|take|me|by|with|on|in|using|via|through|please|can|you|need|now)\b",
            " ",
            normalized,
        )
        stripped = " ".join(stripped.split())
        if stripped and stripped != normalized:
            return self.normalize_transport(stripped)
        return None

    def _is_yes(self, text: str) -> bool:
        normalized = self._normalize_confirmation_text(text)
        if normalized in {
            "yes", "yeah", "yas", "chal", "ha", "haan", "han", "haan ji", "han ji",
            "bilkul", "sahi", "correct", "yep", "yup", "sure", "thats right", "that is right",
        }:
            return True
        tokens = normalized.split()
        if not tokens:
            return False
        yes_heads = {"yes", "yeah", "yep", "yup", "sure", "correct", "okay", "ok", "haan", "han", "bilkul", "sahi"}
        yes_tail_fillers = {
            "i", "want", "to", "go", "ahead", "please", "right", "thats", "that", "is",
            "it", "now", "do", "can", "you", "navigation", "navigate", "ji",
            "there", "lets", "start", "the", "place", "one", "this",
        }
        return tokens[0] in yes_heads and all(token in yes_tail_fillers for token in tokens[1:])

    def _is_no(self, text: str) -> bool:
        normalized = self._normalize_confirmation_text(text)
        if normalized in {"no", "nope", "nah", "incorrect", "wrong", "not that", "nahi", "nahi ji", "na"}:
            return True
        tokens = normalized.split()
        if not tokens:
            return False
        no_heads = {"no", "nope", "nah", "incorrect", "wrong", "nahi", "na"}
        no_tail_fillers = {
            "i", "do", "dont", "not", "want", "that", "this", "one", "please", "thanks", "ji"
        }
        return tokens[0] in no_heads and all(token in no_tail_fillers for token in tokens[1:])

    def _normalize_confirmation_text(self, text: str) -> str:
        clean = self._normalize_indian_voice_text(text)
        clean = re.sub(r"\btu\b", "to", clean)
        clean = re.sub(r"\btoo\b", "to", clean)
        clean = " ".join(clean.split())
        return clean

    def _clean_destination_input(self, destination: str) -> str:
        raw = self._normalize_indian_voice_text(destination)
        # Common STT substitution in Indian accents: "tu" instead of "to".
        raw = re.sub(r"\btu\b", "to", raw)
        raw = re.sub(
            r"\b(i want to go to|i want to|want to go to|want to|"
            r"start navigation|begin navigation|go to|navigate to|navigate|navigation|take me to|me to|"
            r"direction to|directions to|guide me to|route to|after translation|oh sorry|sorry|please|can you)\b",
            " ",
            raw,
        )
        raw = re.sub(r"\bto\b", " ", raw)
        return " ".join(raw.split())

    def _normalize_indian_voice_text(self, text: str) -> str:
        normalized = clean_text(text)
        if not normalized:
            return ""

        replacements = {
            "gadi": "gaadi",
            "haanji": "haan ji",
            "hanji": "han ji",
            "nahi ji": "nahi ji",
            "railway stn": "railway station",
            "stn": "station",
            "metro stn": "metro station",
            "busstop": "bus stop",
            "two wheeler": "two wheeler",
        }
        for source, target in replacements.items():
            normalized = normalized.replace(source, target)

        for source, target in INDIAN_PLACE_ALIASES.items():
            normalized = re.sub(rf"\b{re.escape(source)}\b", target, normalized)

        return " ".join(normalized.split())

    def _should_accept_short_utterance(self, text: str) -> bool:
        if self.state not in {"awaiting_destination", "awaiting_place_selection", "awaiting_mode"}:
            return False
        if self._is_yes(text) or self._is_no(text):
            return True
        if self.state == "awaiting_place_selection":
            return True
        if self.state == "awaiting_mode":
            return bool(self.normalize_transport(text))
        cleaned_destination = self._clean_destination_input(text)
        if not cleaned_destination or len(cleaned_destination) < 3:
            return False
        if self._is_small_talk_or_noise(cleaned_destination):
            return False
        if self._is_command_like_destination(cleaned_destination):
            return False
        return True

    def _dispatch_command_token(self, command: str, emit_callback: bool) -> str:
        if emit_callback and self.command_callback:
            self.command_callback(command)
            return ""
        return command

    def _handle_non_navigation_intent(self, intent: str, emit_callback: bool = True) -> str:
        if intent == "describe_scene":
            return self._dispatch_command_token("DESCRIBE", emit_callback)
        if intent == "get_location":
            return self._dispatch_command_token("GPS", emit_callback)
        return ""

    def _is_command_like_destination(self, cleaned: str) -> bool:
        tokens = [token for token in (cleaned or "").split() if token]
        if not tokens:
            return True

        # Single-word command/transit phrases should never be treated as destination.
        if len(tokens) == 1 and (tokens[0] in NAVIGATION_COMMAND_WORDS or tokens[0] in TRANSPORT_WORDS):
            return True

        # Two-word command phrases like "start navigation", "go route", etc.
        if len(tokens) <= 2 and all(t in NAVIGATION_COMMAND_WORDS for t in tokens):
            return True

        # Mode-only phrases while awaiting destination (e.g., "walking", "driving")
        if all(t in TRANSPORT_WORDS for t in tokens):
            return True

        return False

    def _resolve_destination(self, raw_input: str, priority: str = "SYSTEM") -> bool:
        logger = logging.getLogger("smartvision.voice")
        cleaned = self._clean_destination_input(raw_input)
        original_cleaned = cleaned
        logger.info(f"[NAV] Destination raw: {raw_input}")
        logger.info(f"[NAV] Cleaned destination: {cleaned}")
        if not cleaned:
            self.set_state("awaiting_destination")
            self.speak("Please say the destination clearly.", priority=priority)
            return False
        if cleaned in FILLER_WORDS or len(cleaned) < 3:
            self.set_state("awaiting_destination")
            self.speak("Please tell me your destination clearly.", priority=priority)
            return False
        if self._is_small_talk_or_noise(cleaned):
            self.set_state("awaiting_destination")
            self.speak("Please say only the destination name, for example Pune Railway Station.", priority=priority)
            return False
        if self._is_command_like_destination(cleaned):
            self.set_state("awaiting_destination")
            self.speak("Please say the place name, for example Pune Railway Station.", priority=priority)
            return False

        if not self.is_valid_destination(cleaned):
            self.set_state("awaiting_destination")
            self.speak("I couldn't catch that place. Please say it again.", priority=priority)
            return False

        nearby_query = self._extract_nearby_search_query(cleaned)
        if nearby_query:
            nearby_result = self._search_nearby_place(nearby_query)
            if nearby_result:
                self.pending_place_suggestions = []
                self.pending_place_suggestion = None
                self.pending_destination = nearby_result
                self.destination = nearby_result.get("formatted_address")
                self.pending_destination_coords = (nearby_result["lat"], nearby_result["lng"])
                self.set_state("awaiting_mode")
                name = nearby_result.get("name") or self._summarize_place_label(self.destination)
                distance_text = self._format_spoken_distance(nearby_result.get("distance_m"))
                intro = f"I found {name} near you."
                if distance_text:
                    intro = f"{intro} It is about {distance_text} away."
                self.speak(f"{intro} {MODE_PROMPT}", priority=priority)
                self._nav_last_valid_input_at = time.time()
                return True

        suggestions = self._places_autocomplete(cleaned)
        candidate_query = suggestions[0] if suggestions else cleaned

        try:
            result = self.geocode_destination(candidate_query)
        except Exception as exc:
            logger.warning("[NAV] Geocode failed: %s", exc)
            result = None

        if result is None:
            logger.info(f"[NAV] Suggestions: {suggestions}")
            self.pending_nearby_results = []
            self.pending_nearby_query = None
            if suggestions:
                self.pending_place_suggestions = suggestions
                self.pending_place_suggestion = suggestions[0]
                self.pending_destination = None
                self.pending_destination_coords = None
                self.set_state("awaiting_destination")
                self.speak(f"Did you mean {suggestions[0]}?", priority=priority)
            else:
                self.set_state("awaiting_destination")
                self.speak("I couldn't find that location. Please say it again clearly.", priority=priority)
            return False

        resolved_label = result.get("formatted_address") or candidate_query
        if self._is_ambiguous_far_result(original_cleaned, result):
            self.pending_place_suggestions = []
            self.pending_place_suggestion = None
            self.pending_destination = None
            self.pending_destination_coords = None
            self.set_state("awaiting_destination")
            place_summary = self._summarize_place_label(resolved_label) or resolved_label
            self.speak(
                f"I heard {original_cleaned}, but that matches {place_summary}, which seems far away. "
                "Please say the full station, landmark, or area name.",
                priority=priority,
            )
            return False
        similarity = self._place_match_score(original_cleaned, resolved_label)
        min_accept_score = max(0.52, min(float(self.place_match_threshold), 0.75))
        if similarity < min_accept_score:
            self.pending_destination = result
            self.pending_destination_coords = (result["lat"], result["lng"])
            self.pending_place_suggestion = resolved_label
            self.pending_place_suggestions = suggestions or [resolved_label]
            self.set_state("awaiting_destination")
            self.speak(f"Did you mean {resolved_label}?", priority=priority)
            return False

        self.pending_place_suggestions = []
        self.pending_place_suggestion = None
        self.pending_nearby_results = []
        self.pending_nearby_query = None
        self.pending_destination = result
        self.destination = result.get("formatted_address") or resolved_label
        self.pending_destination_coords = (result["lat"], result["lng"])
        self.set_state("awaiting_mode")
        self.speak(MODE_PROMPT, priority=priority)
        self._nav_last_valid_input_at = time.time()
        return True

    def _confirm_pending_destination(self, priority: str = "SYSTEM") -> bool:
        if self.pending_place_suggestion and not self.pending_destination_coords:
            result = self.geocode_destination(self.pending_place_suggestion)
            if not result:
                self.set_state("awaiting_destination")
                self.speak("I couldn't find that location. Please say it again clearly.", priority=priority)
                return False
            self.pending_destination = result
            self.destination = result.get("formatted_address") or self.pending_place_suggestion
            self.pending_destination_coords = (result["lat"], result["lng"])

        if not self.pending_destination:
            self.set_state("awaiting_destination")
            self.speak("Please say the destination again.", priority=priority)
            return False

        if not self.destination:
            self.destination = self.pending_destination.get("formatted_address") or self.pending_place_suggestion

        self.pending_place_suggestion = None
        self.pending_place_suggestions = []
        self.set_state("awaiting_mode")
        self.speak(MODE_PROMPT, priority=priority)
        self._nav_last_valid_input_at = time.time()
        return True

    # ---------------- STREAMING LISTENER ----------------
    def start_listening(self):
        voice_logger = logging.getLogger("smartvision.voice")

        def on_transcript(text):
            voice_logger.info("[VOICE RAW] %s", text)
            try:
                self._transcript_queue.put_nowait(text)
            except queue.Full:
                pass

        if self.listening:
            return True
        if hasattr(self, "speech_manager") and self.speech_manager:
            if self.speech_manager.is_speaking:
                return False
        self.listening = True
        def _run():
            try:
                self.stt.listen(on_transcript)
            finally:
                self.listening = False

        threading.Thread(
            target=_run,
            daemon=True,
        ).start()
        return True

    def _transcript_worker(self):
        while True:
            text = self._transcript_queue.get()
            try:
                if not text:
                    continue
                # We no longer drop transcripts while speaking to allow barge-in.
                # Echo suppression (is_system_echo) handles filtering out the assistant's own voice.
                if not self._listen_enabled.is_set():
                    continue
                cleaned = clean_text(text)
                if not self.system_boot_completed:
                    if cleaned == SYSTEM_BOOT_PHRASE:
                        self.system_boot_completed = True
                        continue
                    self.system_boot_completed = True
                allow_short_input = self._should_accept_short_utterance(text)
                if (len(cleaned) <= 3 or cleaned in FILLER_WORDS) and not allow_short_input:
                    continue
                if not is_system_echo(text):
                    self.handle_command(text)
            except Exception:
                traceback.print_exc()

    def stop_listening(self):
        self.listening = False
        try:
            self.stt.stop()
        except Exception:
            pass

    def start_navigation_flow(self):
        """Non-blocking navigation setup."""
        if self.conversation_state != "idle":
            print(f"[NAV] Cannot start navigation - state is {self.conversation_state}")
            return False
        self.set_state("awaiting_destination")
        self.destination = None
        self.transport_mode = None
        self.speak("Where would you like to go?", priority="SYSTEM")
        return True

    def set_destination_and_prompt_transport(self, destination: str):
        if self.state == "idle":
            self.set_state("awaiting_destination")
        return self._resolve_destination(destination, priority="SYSTEM")

    def handle_command(self, text: str):
        return self.handle_voice_command(text, emit_callback=True)

    def handle_voice_command(self, text: str, emit_callback: bool = True):
        try:
            user_input = self._merge_pending_fragment(text)
            if not user_input or not user_input.strip():
                return ""

            parsed_input = self._local_interpret_utterance(user_input)
            cleaned_user_input = parsed_input.get("cleaned_text") or user_input
            text = cleaned_user_input.lower().strip()
            normalized = clean_text(user_input)
            allow_short_input = self._should_accept_short_utterance(user_input)
            if (len(normalized) <= 3 or normalized in FILLER_WORDS) and not allow_short_input:
                if self.state == "awaiting_destination":
                    self.speak("Please tell me your destination clearly.", priority="SYSTEM")
                return ""

            if time.time() < self._emergency_override_until:
                return ""

            # Emergency override always wins
            emergency_keywords = (
                "emergency", "help", "save me", "danger", "medical emergency", 
                "trouble", "call police", "i fell", "hurt", "injured"
            )
            if any(keyword in text for keyword in emergency_keywords):
                self._activate_emergency_override()
                command = "EMERGENCY"
                if emit_callback and self.command_callback:
                    self.command_callback(command)
                    return ""
                return command

            intent = parsed_input.get("intent") or self.detect_intent(text)
            if intent not in {
                "navigate", "navigate_destination", "navigate_nearby", "describe_scene",
                "get_location", "stop_navigation", "select_option", "yes", "no", "unknown",
            }:
                intent = self.detect_intent(text)
            if intent == "navigate_destination":
                intent = "navigate"
            if intent in {"yes", "no"}:
                # Let the state-specific confirmation branches handle these naturally.
                intent = "unknown"

            if intent == "unknown" and self._should_buffer_unknown_fragment(user_input):
                self._pending_fragment_text = user_input.strip()
                self._pending_fragment_ts = time.time()
                return ""
            self._clear_pending_fragment()

            if intent == "stop_navigation" and self.state in {"awaiting_destination", "awaiting_place_selection", "awaiting_mode", "navigating"}:
                self._reset_to_idle()
                self.speak("Navigation cancelled.", priority="SYSTEM")
                if emit_callback and self.command_callback:
                    self.command_callback("STOP_NAVIGATION")
                return ""

            if self.state == "idle":
                if intent in {"navigate", "navigate_nearby"}:
                    nearby_query = parsed_input.get("nearby_query") or self._extract_nearby_search_query(text)
                    if nearby_query:
                        nearby_results = self._search_nearby_places(nearby_query)
                        if self._present_nearby_places(nearby_results, nearby_query, priority="SYSTEM"):
                            return ""
                        self.speak(f"I couldn't find any nearby {nearby_query}. Please try another place type.", priority="SYSTEM")
                        return ""

                    destination_candidate = parsed_input.get("destination") or cleaned_user_input
                    cleaned_destination = self._clean_destination_input(destination_candidate)
                    if cleaned_destination and not self._is_command_like_destination(cleaned_destination):
                        self.set_state("awaiting_destination")
                        self._resolve_destination(destination_candidate, priority="SYSTEM")
                    else:
                        self.set_state("awaiting_destination")
                        self.speak("Where would you like to go?", priority="SYSTEM")
                    return ""
                if intent in {"describe_scene", "get_location"}:
                    return self._handle_non_navigation_intent(intent, emit_callback=emit_callback)
                if intent == "stop_navigation":
                    self.speak("No navigation is active.")
                    return ""
                return ""

            if self.state == "awaiting_destination":
                if intent in {"describe_scene", "get_location"}:
                    return self._handle_non_navigation_intent(intent, emit_callback=emit_callback)

                nearby_query = parsed_input.get("nearby_query") or self._extract_nearby_search_query(text)
                if nearby_query:
                    nearby_results = self._search_nearby_places(nearby_query)
                    if self._present_nearby_places(nearby_results, nearby_query, priority="SYSTEM"):
                        return ""
                    self.speak(f"I couldn't find any nearby {nearby_query}. Please try another place type.", priority="SYSTEM")
                    return ""

                # If user repeats "navigate" while already awaiting, just re-prompt
                if intent == "navigate" and not self.extract_destination(text):
                    self.speak("Where would you like to go?", priority="SYSTEM")
                    return ""

                if self.pending_place_suggestion:
                    if self._is_yes(user_input):
                        self._confirm_pending_destination(priority="SYSTEM")
                        return ""
                    if self._is_no(user_input):
                        self.pending_place_suggestion = None
                        self.pending_place_suggestions = []
                        self.pending_destination = None
                        self.pending_destination_coords = None
                        self.speak("Please say the destination again.", priority="SYSTEM")
                        return ""
                    # User may provide a corrected destination instead of yes/no.
                    retry_cleaned = self._clean_destination_input(user_input)
                    if retry_cleaned and retry_cleaned not in FILLER_WORDS and not self._is_command_like_destination(retry_cleaned):
                        self.pending_place_suggestion = None
                        self.pending_place_suggestions = []
                        self.pending_destination = None
                        self.pending_destination_coords = None
                        self._resolve_destination(parsed_input.get("destination") or user_input, priority="SYSTEM")
                        return ""
                    self.speak("Please say yes or no, or say a different destination.", priority="SYSTEM")
                    return ""

                self._resolve_destination(parsed_input.get("destination") or user_input, priority="SYSTEM")
                return ""

            if self.state == "awaiting_place_selection":
                if intent in {"describe_scene", "get_location"}:
                    return self._handle_non_navigation_intent(intent, emit_callback=emit_callback)
                if self._is_no(user_input):
                    self.pending_nearby_results = []
                    self.pending_nearby_query = None
                    self.set_state("awaiting_destination")
                    self.speak("Okay. Tell me what nearby place you want to find.", priority="SYSTEM")
                    return ""

                selected_result = self._match_nearby_selection(user_input)
                if selected_result:
                    self._select_nearby_destination(selected_result, priority="SYSTEM")
                    return ""

                if self.pending_nearby_results:
                    self.speak(PLACE_SELECTION_PROMPT, priority="SYSTEM")
                    return ""
                return ""

            if self.state == "awaiting_mode":
                if intent in {"describe_scene", "get_location"}:
                    return self._handle_non_navigation_intent(intent, emit_callback=emit_callback)

                if self._is_no(user_input):
                    self.set_state("awaiting_destination")
                    self.speak("Please say the destination again.", priority="SYSTEM")
                    return ""

                mode = self.normalize_transport(user_input)
                if not mode:
                    self.speak(MODE_REPROMPT, priority="SYSTEM")
                    return ""

                self.transport_mode = mode
                self.set_state("navigating")
                if hasattr(self, "speech_manager") and self.speech_manager:
                    try:
                        self.speech_manager.cancel_background()
                        self.speech_manager.clear_queue()
                    except Exception:
                        pass
                self.speak("Starting navigation.", priority="NAVIGATION")
                if not self.use_external_navigation:
                    self._start_navigation_from_confirmation()
                command = f"NAVIGATE:{self.destination}:{self.transport_mode}"
                if emit_callback and self.command_callback:
                    self.command_callback(command)
                    return ""
                return command

            if self.state == "navigating":
                if intent == "stop_navigation":
                    self.set_state("idle")
                    command = "STOP_NAVIGATION"
                    if emit_callback and self.command_callback:
                        self.command_callback(command)
                        return ""
                    return command
                
                if intent in {"describe_scene", "get_location"}:
                    return self._handle_non_navigation_intent(intent, emit_callback=emit_callback)
                
                return ""

            print("[VOICE] Unknown command")
            return ""
        except Exception as e:
            print(f"[NAV ERROR] {e}")
            traceback.print_exc()
            self.speak("Navigation failed. Please try again.")
            return ""

    def _activate_emergency_override(self):
        self._emergency_override_until = time.time() + 2.5
        try:
            self.stop_speaking_immediately()
        except Exception:
            pass

    def emergency_override(self, alert_text: str):
        """Emergency override to interrupt speech and announce alert."""
        self._activate_emergency_override()
        if alert_text:
            self.speak(alert_text, priority="EMERGENCY")

    def cleanup(self):
        """Cleanup resources."""
        self.stop_listening()
        print("Voice engine cleaned up")

    def test_tts(self):
        """Simple TTS test hook for GUI."""
        self.speak("Testing voice output.", blocking=False)
