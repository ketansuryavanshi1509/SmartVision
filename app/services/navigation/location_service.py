"""
Merged Location Services
Combines location_manager (GPS polling) and location_tracker (Firebase tracking)
"""

# ==========================================
# FROM: location_manager.py
# ==========================================
import time
import requests

# Google Maps only — no OSM/geocoder fallback

from app.config import config
# Simple in-memory cache for location lookups
# Key: (rounded_lat, rounded_lng), Value: location_name
_location_cache = {}

def reverse_geocode(lat, lng):
    """
    Convert coordinates to a human-readable location name (City, State).
    Rounds coordinates to 4 decimal places for street-level cache efficiency (~11m accuracy).
    """
    if lat is None or lng is None:
        return "Unknown Location"
        
    lat_r = round(float(lat), 4)
    lng_r = round(float(lng), 4)
    cache_key = (lat_r, lng_r)
    
    if cache_key in _location_cache:
        return _location_cache[cache_key]
        
    api_key = config.GOOGLE_MAPS_API_KEY
    logger.debug("GMAPS key present: %s", bool(api_key))

    if not api_key:
        return f"{lat_r}, {lng_r}"
        
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "latlng": f"{lat},{lng}",
        "key": api_key
    }
    
    try:
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        logger.debug("Geocode response status: %s", data.get("status"))

        if data.get("status") != "OK":
            logger.warning("Google reverse geocode failed: status=%s, error=%s",
                           data.get("status"), data.get("error_message", ""))
            return f"{lat_r}, {lng_r}"

        if data.get("results"):
            components = data["results"][0].get("address_components", [])
            
            route = None
            sublocality2 = None
            sublocality = None
            neighborhood = None
            city = None
            state = None
            
            for c in components:
                types = c.get("types", [])
                if "route" in types:
                    route = c.get("long_name")
                elif "sublocality_level_2" in types:
                    sublocality2 = c.get("long_name")
                elif "sublocality_level_1" in types:
                    sublocality = c.get("long_name")
                elif "neighborhood" in types:
                    neighborhood = c.get("long_name")
                elif "locality" in types:
                    city = c.get("long_name")
                elif "administrative_area_level_1" in types:
                    state = c.get("long_name")
            
            # Build a detailed address: street, area, city
            parts = []
            if route:
                parts.append(route)
            if sublocality2:
                parts.append(sublocality2)
            if sublocality:
                parts.append(sublocality)
            elif neighborhood:
                parts.append(neighborhood)
            if city:
                parts.append(city)
            
            if parts:
                location_name = ", ".join(parts)
            elif state:
                location_name = state
            else:
                location_name = data["results"][0].get("formatted_address", f"{lat_r}, {lng_r}")
                # Simplify formatted address if too long
                if len(location_name) > 60:
                    location_name = ", ".join(location_name.split(",")[:3])
            
            _location_cache[cache_key] = location_name
            return location_name
        else:
            return f"{lat_r}, {lng_r}"
            
    except Exception as e:
        logger.warning("reverse_geocode exception: %s", e)
        return f"{lat_r}, {lng_r}"

from app.core.logger import logger


class LocationManager:
    def __init__(self, gmaps_client=None, web_mode=True, cache_ttl_s=5.0, mobile_max_age_s=20.0):
        self.gmaps = gmaps_client
        self.web_mode = bool(web_mode)
        self.cache_ttl_s = float(cache_ttl_s)
        self.mobile_max_age_s = float(mobile_max_age_s)

        self._mobile_gps = None
        self._mobile_gps_ts = 0.0
        self._browser_gps = None
        self._browser_gps_ts = 0.0
        self._last_location = None
        self._last_location_ts = 0.0

    def _cache(self, loc):
        self._last_location = loc
        self._last_location_ts = time.time()

    def get_best_available_location(self, use_cache: bool = True):
        now = time.time()
        if use_cache and self._last_location and (now - self._last_location_ts) < self.cache_ttl_s:
            return self._last_location

        # 1) Mobile GPS (/api/location)
        loc = self.get_mobile_gps_location()
        if loc:
            self._cache(loc)
            return loc

        # Last known
        if self._last_location:
            return self._last_location

        return None

    def get_mobile_gps_location(self):
        if not self.web_mode:
            return None
        now = time.time()
        try:
            from app.api.web_server import latest_location, location_timestamp
        except Exception:
            return None

        if latest_location and latest_location[0] is not None and latest_location[1] is not None:
            lat, lng = latest_location
            if -90 <= lat <= 90 and -180 <= lng <= 180:
                self._mobile_gps = (lat, lng)
                self._mobile_gps_ts = location_timestamp or now

        if self._mobile_gps and self._mobile_gps_ts:
            age = now - self._mobile_gps_ts
            if age <= self.mobile_max_age_s:
                lat, lng = self._mobile_gps
                city_name = self._get_city_from_coords(lat, lng)
                return (lat, lng, city_name)
        return None

    def get_browser_gps_location(self):
        return None

    def get_ip_location(self):
        """Deprecated – GPS only. Returns None."""
        return None


    def _get_city_from_coords(self, lat, lng):
        return reverse_geocode(lat, lng)


# ==========================================
# FROM: location_tracker.py
# ==========================================
import math
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field

from app.database.firebase_client import get_db, _is_timeout


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = p2 - p1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def classify_movement(speed_mps):
    if speed_mps is None:
        return "stationary"
    if speed_mps < 0.5:
        return "stationary"
    if speed_mps < 3.0:
        return "walking"
    return "vehicle"


def _now_iso(ts=None):
    dt = datetime.fromtimestamp(ts or time.time(), tz=timezone.utc)
    return dt.isoformat()


@dataclass
class _SessionState:
    session_id: str = None
    last_point: tuple = None
    last_ts: float = None
    total_distance: float = 0.0
    total_time: float = 0.0
    stops_count: int = 0
    start_lat: float = None
    start_lng: float = None
    last_movement: str = None
    buffer: list = field(default_factory=list)


class LocationTrackerManager:
    def __init__(self, sample_interval_s=8, min_distance_m=5, batch_size=10, inactivity_timeout_s=300):
        self.sample_interval_s = max(5, int(sample_interval_s))
        self.min_distance_m = max(2, float(min_distance_m))
        self.batch_size = max(5, int(batch_size))
        self.inactivity_timeout_s = max(60, int(inactivity_timeout_s))
        self._sessions = {}

    def _get_state(self, user_id):
        if user_id not in self._sessions:
            self._sessions[user_id] = _SessionState()
        return self._sessions[user_id]

    def _create_session(self, user_id, access_token):
        db = get_db()
        if not db:
            raise RuntimeError("Firestore not available")
        payload = {"created_at": _now_iso(), "started_at": _now_iso()}
        _, doc_ref = db.collection("users").document(user_id).collection("location_sessions").add(payload)
        return doc_ref.id

    def _update_session(self, user_id, state: _SessionState, access_token, end_lat=None, end_lng=None, ended=False):
        db = get_db()
        if not db or not state.session_id:
            return
        avg_speed = state.total_distance / state.total_time if state.total_time > 0 else 0
        payload = {
            "total_distance_m": state.total_distance,
            "total_time_s": state.total_time,
            "avg_speed_mps": avg_speed,
            "stops_count": state.stops_count,
            "start_lat": state.start_lat,
            "start_lng": state.start_lng,
            "end_lat": end_lat,
            "end_lng": end_lng,
        }
        if ended:
            payload["ended_at"] = _now_iso()
        db.collection("users").document(user_id).collection("location_sessions").document(state.session_id).update(payload)

    def _flush_points(self, state: _SessionState, access_token):
        if not state.buffer:
            return
        db = get_db()
        if not db:
            return
        batch = db.batch()
        user_ref = db.collection("users").document(state.buffer[0]["user_id"])
        for point in state.buffer:
            # Remove user_id from point payload as it's now in the path
            point_data = {k: v for k, v in point.items() if k != "user_id"}
            doc_ref = user_ref.collection("location_points").document()
            batch.set(doc_ref, point_data)
        batch.commit()
        state.buffer = []

    def ingest_point(self, user_id, lat, lng, ts=None, access_token=None):
        if not user_id or not access_token:
            return
        now_ts = ts or time.time()
        state = self._get_state(user_id)

        # Close stale session if inactive
        if state.last_ts and (now_ts - state.last_ts) > self.inactivity_timeout_s:
            self.end_session(user_id, access_token=access_token)
            state = self._get_state(user_id)

        if state.last_ts and (now_ts - state.last_ts) < self.sample_interval_s:
            return

        if state.session_id is None:
            try:
                state.session_id = self._create_session(user_id, access_token)
            except Exception:
                return

        distance = 0.0
        speed = 0.0
        if state.last_point:
            distance = haversine_m(state.last_point[0], state.last_point[1], lat, lng)
            dt = max(1.0, now_ts - state.last_ts)
            speed = distance / dt

        movement = classify_movement(speed)
        if distance < self.min_distance_m and state.last_point:
            movement = "stationary"

        if state.last_movement and state.last_movement != "stationary" and movement == "stationary":
            state.stops_count += 1

        if movement != "stationary":
            if state.last_ts:
                state.total_time += max(1.0, now_ts - state.last_ts)
            state.total_distance += distance

        if state.start_lat is None:
            state.start_lat, state.start_lng = lat, lng

        state.buffer.append({
            "session_id": state.session_id,
            "user_id": user_id,
            "latitude": lat,
            "longitude": lng,
            "recorded_at": _now_iso(now_ts),
            "movement_type": movement,
            "speed_mps": speed,
            "distance_increment_m": distance
        })

        state.last_point = (lat, lng)
        state.last_ts = now_ts
        state.last_movement = movement

        if len(state.buffer) >= self.batch_size:
            try:
                self._flush_points(state, access_token)
                self._update_session(user_id, state, access_token, end_lat=lat, end_lng=lng)
            except Exception:
                pass

    def end_session(self, user_id, access_token=None):
        if not access_token:
            return
        state = self._get_state(user_id)
        if not state.session_id:
            return
        try:
            self._flush_points(state, access_token)
        except Exception:
            pass
        end_lat, end_lng = None, None
        if state.last_point:
            end_lat, end_lng = state.last_point
        try:
            self._update_session(user_id, state, access_token, end_lat=end_lat, end_lng=end_lng, ended=True)
        except Exception:
            pass
        self._sessions[user_id] = _SessionState()

    def get_active_session_id(self, user_id):
        state = self._get_state(user_id)
        return state.session_id

