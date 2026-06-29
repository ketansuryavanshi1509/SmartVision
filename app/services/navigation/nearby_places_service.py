"""
NearbyPlacesService
===================
Gemini-like nearby place discovery for SmartVision.

Two search flows:
  GENERIC  — "find the nearest medical shop near me"
             Uses Google Places NearbySearch API (ranked by distance from GPS).

  SPECIFIC — "I want to visit Mamledar Misal in Bhandup"
             Uses Google Places TextSearch API (handles name + area queries).

Both flows:
  1. Fetch top-2 results sorted by distance from user's GPS
  2. Read them aloud: name, address, distance, rating
  3. Wait for voice selection ("first", "second", or place name)
  4. Call start_navigation_fn(lat, lng, mode, name) → NavigationManager

No circular imports. All Google API calls run in the calling thread
(caller must use threading.Thread if needed).
"""

import math
import re
import threading
import requests
from difflib import SequenceMatcher
from typing import Optional, Tuple, List, Dict, Any, Callable

from app.core.logger import logger


# ── Distance helpers ───────────────────────────────────────────────────────

def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Return distance in metres between two GPS coordinates."""
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lat2 - lat1)
    dp = math.radians(lng2 - lng1)
    a = math.sin(dl / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dp / 2) ** 2
    return 2 * R * math.asin(math.sqrt(max(0.0, a)))


def _fmt_dist(metres: float) -> str:
    """Format distance for natural speech."""
    if metres < 1000:
        return f"{int(round(metres / 10) * 10)} meters"
    return f"{metres / 1000:.1f} kilometres"


# ── Intent helpers (also imported by main.py and voice_command_parser.py) ─

def classify_place_intent(text: str) -> str:
    """
    Classify whether a voice utterance is a place-search command.

    Returns 'nearby', 'specific', or 'none'.
    """
    t = text.lower()
    if re.search(r'\b(nearest|nearby|closest|near\s+me)\b', t):
        return 'nearby'
    if re.search(r'\bfind\s+(a\s+|the\s+|some\s+|any\s+)?(nearest|nearby|closest)\b', t):
        return 'nearby'
    if re.search(r'\bwhere\s+(is|are)\s+the\s+(nearest|closest)\b', t):
        return 'nearby'
    if re.search(r'\bi\s+want\s+to\s+(visit|go\s+to|go)\b', t):
        return 'specific'
    if re.search(r'\bwanna\s+(visit|go\s+to|go)\b', t):
        return 'specific'
    if re.search(r'\b(take\s+me\s+to|navigate\s+to|go\s+to|directions?\s+to)\b', t):
        return 'specific'
    return 'none'


def extract_nearby_query(text: str) -> str:
    """Strip intent words; return the search target. E.g. 'nearest medical shop near me' → 'medical shop'."""
    t = text.lower()
    t = re.sub(r'^(find|show|tell|get|give)\s+(me\s+)?', '', t)
    t = re.sub(r'\b(the\s+|a\s+|an\s+)?(nearest|nearby|closest)\s+', '', t)
    t = re.sub(r'\bnear\s+(me|my\s+(current\s+)?location)\b', '', t)
    t = re.sub(r'\baround\s+me\b', '', t)
    return t.strip(' .,')


def extract_specific_query(text: str) -> str:
    """Strip intent prefix; return destination phrase. E.g. 'I want to visit X in Y' → 'X in Y'."""
    patterns = [
        r'\bi\s+want\s+to\s+(?:visit|go\s+to|go)\s+(.+)',
        r'\bwanna\s+(?:visit|go\s+to|go)\s+(.+)',
        r'\b(?:take\s+me\s+to|navigate\s+to|go\s+to|directions?\s+to)\s+(.+)',
        r'\bvisit\s+(.+)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip(' ,.')
    return text.strip()


# ── Main service ───────────────────────────────────────────────────────────

class NearbyPlacesService:
    """
    Stateful service for Gemini-like place discovery.

    Lifecycle:
      1. handle_nearby_query(text) or handle_specific_query(text)
         → searches Google Places → speaks 2 options → is_awaiting_selection = True
      2. handle_selection(text)
         → matches input to option → starts navigation → resets state

    Wire once in VisionAssistant.__init__ after NavigationManager init.
    """

    MAX_RESULTS = 2
    SEARCH_RADIUS_M = 20_000

    def __init__(
        self,
        api_key: str,
        location_provider: Callable[[], Optional[Tuple]],
        speak_fn: Callable[[str], None],
        start_navigation_fn: Callable[[float, float, str, str], None],
    ):
        self.api_key = api_key
        self._get_location = location_provider
        self._speak = speak_fn
        self._start_navigation = start_navigation_fn
        self._pending_results: List[Dict[str, Any]] = []
        self._pending_query: str = ""
        self._awaiting_selection: bool = False
        self._lock = threading.Lock()

    def handle_nearby_query(self, raw_text: str, mode: str = "walking") -> bool:
        """Handle a generic nearby search utterance e.g. 'nearest medical shop'."""
        query = extract_nearby_query(raw_text)
        if not query:
            self._speak("Please tell me what you are looking for.")
            return False
        logger.info("[NEARBY] Generic search: %s", query)
        return self._search_and_present(query, search_type="nearby", mode=mode)

    def handle_specific_query(self, raw_text: str, mode: str = "walking") -> bool:
        """Handle a named destination utterance e.g. 'I want to visit Mamledar Misal in Bhandup'."""
        query = extract_specific_query(raw_text)
        if not query:
            self._speak("Please tell me where you want to go.")
            return False
        logger.info("[NEARBY] Specific search: %s", query)
        return self._search_and_present(query, search_type="specific", mode=mode)

    def handle_selection(self, user_input: str, mode: str = "walking") -> bool:
        """Match user's spoken choice to an option and start navigation."""
        with self._lock:
            if not self._awaiting_selection or not self._pending_results:
                return False
            results = list(self._pending_results)

        selected = self._match_selection(user_input, results)
        if not selected:
            self._speak(
                "I didn't catch that. Please say first or second, "
                "or the name of the place."
            )
            return False

        with self._lock:
            self._awaiting_selection = False
            self._pending_results = []
            self._pending_query = ""

        name = selected.get("name", "your destination")
        dist = _fmt_dist(selected.get("distance_m", 0))
        self._speak(f"Starting navigation to {name}, {dist} away.")
        self._start_navigation(selected["lat"], selected["lng"], mode, name)
        return True

    @property
    def is_awaiting_selection(self) -> bool:
        """True when options have been spoken and we are waiting for the user's choice."""
        return self._awaiting_selection

    def cancel(self):
        """Reset state — call when user says 'cancel' or 'stop'."""
        with self._lock:
            self._awaiting_selection = False
            self._pending_results = []
            self._pending_query = ""

    # ── Search orchestration ───────────────────────────────────────────────

    def _search_and_present(self, query: str, search_type: str, mode: str) -> bool:
        """Run search, update state, speak results."""
        loc = self._get_location()
        if not loc:
            self._speak(
                "I cannot determine your current location. "
                "Please enable GPS and try again."
            )
            return False

        lat, lng = float(loc[0]), float(loc[1])
        self._speak(f"Searching for {query} near you. Please wait.")

        results = (
            self._nearby_search(query, lat, lng)
            if search_type == "nearby"
            else self._text_search(query, lat, lng)
        )

        # Fallback to the other API if primary returns nothing
        if not results:
            logger.info("[NEARBY] Primary empty, trying fallback API")
            results = (
                self._text_search(query, lat, lng)
                if search_type == "nearby"
                else self._nearby_search(query, lat, lng)
            )

        if not results:
            self._speak(
                f"Sorry, I could not find any {query} near you. "
                "Please try a different search."
            )
            return False

        capped = results[: self.MAX_RESULTS]
        with self._lock:
            self._pending_results = capped
            self._pending_query = query
            self._awaiting_selection = True

        self._speak_results(capped, query)
        return True

    # ── Google Places API calls ────────────────────────────────────────────

    def _nearby_search(self, query: str, lat: float, lng: float) -> List[Dict]:
        """NearbySearch ranked by distance — best for generic place types."""
        try:
            resp = requests.get(
                "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
                params={
                    "location": f"{lat},{lng}",
                    "keyword": query,
                    "rankby": "distance",
                    "key": self.api_key,
                },
                timeout=8,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "OK":
                    return self._parse_results(data.get("results", []), lat, lng)
                logger.warning("[NEARBY] NearbySearch status: %s", data.get("status"))
        except Exception:
            logger.exception("[NEARBY] NearbySearch failed")
        return []

    def _text_search(self, query: str, lat: float, lng: float) -> List[Dict]:
        """TextSearch — best for named destinations with area context."""
        try:
            resp = requests.get(
                "https://maps.googleapis.com/maps/api/place/textsearch/json",
                params={
                    "query": query,
                    "location": f"{lat},{lng}",
                    "radius": self.SEARCH_RADIUS_M,
                    "key": self.api_key,
                    "region": "in",
                },
                timeout=8,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("status") == "OK":
                    return self._parse_results(data.get("results", []), lat, lng)
                logger.warning("[NEARBY] TextSearch status: %s", data.get("status"))
        except Exception:
            logger.exception("[NEARBY] TextSearch failed")
        return []

    def _parse_results(
        self, raw: list, user_lat: float, user_lng: float
    ) -> List[Dict]:
        """Parse raw Places API items into clean dicts sorted by distance."""
        out = []
        for item in raw:
            geo = (item.get("geometry") or {}).get("location") or {}
            if "lat" not in geo or "lng" not in geo:
                continue
            p_lat, p_lng = float(geo["lat"]), float(geo["lng"])
            dist_m = _haversine_m(user_lat, user_lng, p_lat, p_lng)
            vicinity = item.get("vicinity") or item.get("formatted_address") or ""
            name = item.get("name") or "Unknown place"
            out.append({
                "name": name,
                "vicinity": vicinity,
                "formatted_address": f"{name}, {vicinity}".strip(", "),
                "lat": p_lat,
                "lng": p_lng,
                "distance_m": dist_m,
                "rating": item.get("rating"),
                "place_id": item.get("place_id", ""),
            })
        out.sort(key=lambda x: x["distance_m"])
        return out[: self.MAX_RESULTS]

    # ── TTS output ─────────────────────────────────────────────────────────

    def _speak_results(self, results: List[Dict], query: str):
        """Build and speak the options prompt."""
        parts = []
        for i, r in enumerate(results, 1):
            desc = f"Option {i}: {r['name']}"
            if r.get("vicinity"):
                desc += f", at {r['vicinity']}"
            desc += f", about {_fmt_dist(r['distance_m'])} away"
            if r.get("rating"):
                desc += f", rated {r['rating']} out of 5"
            parts.append(desc)

        self._speak(
            f"I found {len(results)} nearby {query}. "
            f"{'. '.join(parts)}. "
            f"Say first or second to choose, or say the name of the place."
        )

    # ── Selection matching ─────────────────────────────────────────────────

    def _match_selection(
        self, user_input: str, results: List[Dict]
    ) -> Optional[Dict]:
        """Match voice input to a result via ordinals or fuzzy name matching."""
        text = user_input.lower().strip()

        # Check multi-word phrases FIRST (before single words like "one", "two")
        multi_word_ordinals = {
            "first one": 1, "option one": 1, "number one": 1,
            "second one": 2, "option two": 2, "number two": 2,
        }
        for phrase, idx in multi_word_ordinals.items():
            if phrase in text:
                if 1 <= idx <= len(results):
                    return results[idx - 1]
        
        # Then check single-word ordinals and numbers
        ordinals = {
            "first": 1, "one": 1, "1": 1,
            "option 1": 1, "option 2": 2,
            "second": 2, "two": 2, "2": 2,
        }
        for phrase, idx in ordinals.items():
            if phrase in text:
                if 1 <= idx <= len(results):
                    return results[idx - 1]

        best, best_score = None, 0.0
        for r in results:
            name_lower = r["name"].lower()
            score = SequenceMatcher(None, text, name_lower).ratio()
            for word in name_lower.split():
                if len(word) > 3 and word in text:
                    score = max(score, 0.75)
            if score > best_score:
                best_score, best = score, r

        return best if best_score >= 0.50 else None
