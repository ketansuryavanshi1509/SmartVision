"""
Merged Navigation Services
Combines navigation_manager (routing) and navigation_live (turn-by-turn navigation state)
"""

# ==========================================
# FROM: navigation_manager.py
# ==========================================
import html
import math
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from app.core.logger import logger
from app.services.speech.speech_manager import SpeechPriority, speech_manager as default_speech_manager

Coordinate = Tuple[float, float]
StepDict = Dict[str, Any]


def clean_instruction(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("<div", ". <div").replace("<wbr/>", " ")
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    earth_radius_m = 6371000.0
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(d_lng / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return earth_radius_m * c


class NavigationManager:
    def __init__(
        self,
        api_key: str,
        location_manager,
        speech_manager=default_speech_manager,
        vision_engine=None,
        voice_engine=None,
        on_route_calculated: Optional[Callable[[str, str], None]] = None,
        on_start: Optional[Callable[[], None]] = None,
        on_stop: Optional[Callable[[], None]] = None,
    ):
        self.api_key = api_key
        self.location_manager = location_manager
        self.speech_manager = speech_manager
        self.vision_engine = vision_engine
        self.voice_engine = voice_engine
        self.on_route_calculated = on_route_calculated
        self.on_start = on_start
        self.on_stop = on_stop

        self.is_navigating: bool = False
        self.navigation_active: bool = False
        self.steps: List[StepDict] = []
        self.current_step_index: int = 0
        self.destination: Optional[Coordinate] = None
        self.origin: Optional[Coordinate] = None
        self.navigation_thread: Optional[threading.Thread] = None
        self.last_spoken_stage: Optional[str] = None
        self.last_remaining_announcement: Optional[int] = None
        self.polyline_points: List[Coordinate] = []

        self.current_location: Optional[Coordinate] = None
        self.navigation_state: Optional[Dict[str, Any]] = None

        self._thread: Optional[threading.Thread] = None
        self._destination_raw: Any = None
        self._mode: str = "driving"
        self._session_id: Optional[str] = None
        self._encoded_polyline: Optional[str] = None
        self._total_distance_m: int = 0
        self._total_duration_s: int = 0
        self._last_reroute_at: float = 0.0
        self._reroute_cooldown_s: float = 10.0
        self._loop_interval_s: float = 1.0
        self._route_lock = threading.RLock()
        self._stop_event = threading.Event()
        self._on_start_emitted: bool = False
        self._last_progress_bucket: Optional[Tuple[str, int]] = None

    def start_navigation(self, *args, **kwargs) -> bool:
        destination, mode, session_id, origin = self._parse_start_arguments(*args, **kwargs)
        if destination is None:
            return False
        if isinstance(destination, str) and not destination.strip():
            return False

        mode = self._normalize_mode(mode)
        origin_coord = self._normalize_coordinate(origin)

        with self._route_lock:
            if self.navigation_thread and self.navigation_thread.is_alive():
                return False

            self._stop_event.clear()
            self.is_navigating = True
            self.navigation_active = True
            self.steps = []
            self.current_step_index = 0
            self.last_spoken_stage = None
            self.last_remaining_announcement = None
            self.polyline_points = []
            self.navigation_state = None

            self._destination_raw = destination
            self._mode = mode
            self._session_id = session_id
            self.origin = origin_coord
            self.destination = None
            self._encoded_polyline = None
            self._total_distance_m = 0
            self._total_duration_s = 0
            self._last_reroute_at = 0.0
            self._on_start_emitted = False
            self._last_progress_bucket = None

            self.navigation_thread = threading.Thread(
                target=self._navigation_worker,
                name="smartvision-navigation-loop",
                daemon=True,
            )
            self._thread = self.navigation_thread
            self.navigation_thread.start()
        return True

    def stop_navigation(self):
        self._stop_event.set()
        with self._route_lock:
            self.is_navigating = False
            self.navigation_active = False
            active_thread = self.navigation_thread
        if not active_thread or not active_thread.is_alive():
            self._finalize_navigation(stopped=True)

    def recalculate_route(self, lat: float = None, lng: float = None) -> bool:
        with self._route_lock:
            if not self.is_navigating or self._destination_raw is None:
                return False
            now = time.time()
            if (now - self._last_reroute_at) < self._reroute_cooldown_s:
                return False
            self._last_reroute_at = now
            destination_input = self._destination_raw
            mode = self._mode

        if lat is not None and lng is not None:
            origin = (float(lat), float(lng))
        else:
            origin = self._get_live_location()
            if not origin:
                return False

        directions = self._request_directions(origin, destination_input, mode)
        if not directions:
            return False

        parsed = self._parse_directions(directions)
        if not parsed:
            return False

        self._apply_parsed_route(parsed, reset_progress=True)
        logger.info("[NAV] Route calculated")
        self._enqueue_speech("Route recalculated.", SpeechPriority.NAVIGATION)
        if parsed["steps"]:
            guidance = self._lane_guidance_text(parsed["steps"][0])
            self._enqueue_speech(f"Next, {guidance}", SpeechPriority.NAVIGATION)
        return True

    def _navigation_worker(self):
        try:
            if not self.api_key:
                self._enqueue_speech("Navigation is unavailable. Missing Google API key.", SpeechPriority.SYSTEM)
                return

            route_origin = self.origin or self._get_live_location()
            if not route_origin:
                self._enqueue_speech("I could not determine your current location.", SpeechPriority.SYSTEM)
                return

            directions = self._request_directions(route_origin, self._destination_raw, self._mode)
            if not directions:
                self._enqueue_speech("I could not calculate a route right now.", SpeechPriority.SYSTEM)
                return

            parsed = self._parse_directions(directions)
            if not parsed or not parsed.get("steps"):
                self._enqueue_speech("Route found, but turn-by-turn steps are unavailable.", SpeechPriority.SYSTEM)
                return

            self._apply_parsed_route(parsed, reset_progress=True)
            self._emit_on_start_once()
            self._announce_route_summary()
            logger.info("[NAV] Route calculated")
            self._navigation_loop()
        except Exception:
            logger.exception("[NAV] Navigation worker failed")
            self._enqueue_speech("Navigation interrupted due to an internal error.", SpeechPriority.SYSTEM)
        finally:
            self._finalize_navigation(stopped=self._stop_event.is_set())

    def _navigation_loop(self):
        while not self._stop_event.is_set():
            with self._route_lock:
                if not self.is_navigating:
                    break
                has_steps = bool(self.steps)
                destination = self.destination
                polyline_points = list(self.polyline_points)

            if not has_steps:
                self._stop_event.wait(self._loop_interval_s)
                continue

            current_location = self._get_live_location()
            if not current_location:
                self._stop_event.wait(self._loop_interval_s)
                continue

            self.current_location = current_location

            if destination:
                destination_distance = haversine_distance(
                    current_location[0],
                    current_location[1],
                    destination[0],
                    destination[1],
                )
                if destination_distance < 30.0:
                    logger.info("[NAV] Arrival detected")
                    self._enqueue_speech("You have arrived at your destination.", SpeechPriority.NAVIGATION)
                    self._stop_event.set()
                    break

            if polyline_points:
                off_route_m = self._distance_from_polyline(current_location, polyline_points)
                if off_route_m > 50.0:
                    logger.info("[NAV] Off route detected")
                    self._enqueue_speech("You are off route. Recalculating.", SpeechPriority.NAVIGATION)
                    self.recalculate_route(current_location[0], current_location[1])
                    self._stop_event.wait(self._loop_interval_s)
                    continue

            with self._route_lock:
                if self.current_step_index >= len(self.steps):
                    self._stop_event.wait(self._loop_interval_s)
                    continue
                step = self.steps[self.current_step_index]
                step_index = self.current_step_index
                total_steps = len(self.steps)

            distance_to_step = self._distance_to_step(current_location, step)
            if distance_to_step is None:
                self._stop_event.wait(self._loop_interval_s)
                continue

            logger.info(
                "[NAV LOOP] distance_to_step=%.2f step_index=%d total_steps=%d",
                distance_to_step,
                step_index,
                total_steps,
            )

            off_route_m = None
            if polyline_points:
                off_route_m = self._distance_from_polyline(current_location, polyline_points)
            self._maybe_announce_remaining_distance(current_location)
            self._update_navigation_state(current_location, step, distance_to_step, off_route_m=off_route_m)
            self._process_step_guidance(step, distance_to_step)

            self._stop_event.wait(self._loop_interval_s)

    def _distance_to_step(self, current_location: Coordinate, step: StepDict = None) -> Optional[float]:
        if not current_location:
            return None
        if step is None:
            with self._route_lock:
                if self.current_step_index >= len(self.steps):
                    return None
                step = self.steps[self.current_step_index]
        end_location = step.get("end_location")
        if not end_location:
            return None
        return haversine_distance(
            current_location[0],
            current_location[1],
            end_location[0],
            end_location[1],
        )

    def _distance_from_polyline(
        self,
        current_location: Coordinate,
        polyline_points: List[Coordinate] = None,
    ) -> float:
        if not current_location:
            return float("inf")
        points = polyline_points if polyline_points is not None else self.polyline_points
        if not points:
            return float("inf")
        if len(points) == 1:
            return haversine_distance(
                current_location[0],
                current_location[1],
                points[0][0],
                points[0][1],
            )

        ref_lat = current_location[0]
        px, py = self._project_to_meters(current_location[0], current_location[1], ref_lat)
        min_distance = float("inf")

        for i in range(len(points) - 1):
            ax, ay = self._project_to_meters(points[i][0], points[i][1], ref_lat)
            bx, by = self._project_to_meters(points[i + 1][0], points[i + 1][1], ref_lat)

            abx = bx - ax
            aby = by - ay
            apx = px - ax
            apy = py - ay
            denom = (abx * abx) + (aby * aby)

            if denom <= 0.0:
                cx, cy = ax, ay
            else:
                t = max(0.0, min(1.0, ((apx * abx) + (apy * aby)) / denom))
                cx = ax + (abx * t)
                cy = ay + (aby * t)

            distance_m = math.hypot(px - cx, py - cy)
            if distance_m < min_distance:
                min_distance = distance_m

        return min_distance

    def distance_from_polyline(
        self,
        current_location: Coordinate,
        polyline_points: List[Coordinate],
    ) -> float:
        return self._distance_from_polyline(current_location, polyline_points)

    def _clean_html(self, text: str) -> str:
        return clean_instruction(text)

    def _parse_directions(self, directions_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(directions_payload, dict):
            return None

        routes = directions_payload.get("routes") or []
        if not routes:
            return None

        first_route = routes[0]
        legs = first_route.get("legs") or []
        if not legs:
            return None

        leg = legs[0]
        raw_steps = leg.get("steps") or []
        if not raw_steps:
            return None

        parsed_steps: List[StepDict] = []
        for raw_step in raw_steps:
            start = raw_step.get("start_location") or {}
            end = raw_step.get("end_location") or {}
            if "lat" not in start or "lng" not in start or "lat" not in end or "lng" not in end:
                continue

            instruction = self._clean_html(raw_step.get("html_instructions", ""))
            maneuver = str(raw_step.get("maneuver") or "straight").strip().lower()
            distance_value = int((raw_step.get("distance") or {}).get("value") or 0)
            duration_value = int((raw_step.get("duration") or {}).get("value") or 0)
            distance_text = (raw_step.get("distance") or {}).get("text") or ""
            duration_text = (raw_step.get("duration") or {}).get("text") or ""

            parsed_steps.append(
                {
                    "instruction": instruction,
                    "distance": distance_value,
                    "duration": duration_value,
                    "start_location": (float(start["lat"]), float(start["lng"])),
                    "end_location": (float(end["lat"]), float(end["lng"])),
                    "maneuver": maneuver,
                    "distance_text": distance_text,
                    "duration_text": duration_text,
                }
            )

        if not parsed_steps:
            return None

        encoded_polyline = (first_route.get("overview_polyline") or {}).get("points")
        polyline_points = self._decode_polyline(encoded_polyline) if encoded_polyline else []

        leg_end = leg.get("end_location") or {}
        destination = None
        if "lat" in leg_end and "lng" in leg_end:
            destination = (float(leg_end["lat"]), float(leg_end["lng"]))

        return {
            "steps": parsed_steps,
            "polyline": encoded_polyline,
            "polyline_points": polyline_points,
            "total_distance_m": int((leg.get("distance") or {}).get("value") or 0),
            "total_duration_s": int((leg.get("duration") or {}).get("value") or 0),
            "total_distance_text": (leg.get("distance") or {}).get("text") or "",
            "total_duration_text": (leg.get("duration") or {}).get("text") or "",
            "destination": destination,
            "origin": parsed_steps[0]["start_location"],
        }

    def _request_directions(
        self,
        origin: Coordinate,
        destination: Any,
        mode: str,
    ) -> Optional[Dict[str, Any]]:
        destination_param = self._destination_to_api_value(destination)
        if not destination_param:
            return None

        params = {
            "origin": f"{origin[0]},{origin[1]}",
            "destination": destination_param,
            "mode": self._normalize_mode(mode),
            "units": "metric",
            "alternatives": "false",
            "key": self.api_key,
        }
        try:
            response = requests.get(
                "https://maps.googleapis.com/maps/api/directions/json",
                params=params,
                timeout=12,
            )
            payload = response.json()
        except Exception:
            logger.exception("[NAV] Directions API failure")
            return None

        if payload.get("status") != "OK":
            logger.warning("[NAV] Directions API returned status=%s", payload.get("status"))
            return None
        return payload

    def _apply_parsed_route(self, parsed_route: Dict[str, Any], reset_progress: bool) -> None:
        callback_session_id = None
        callback_polyline = None

        with self._route_lock:
            self.steps = parsed_route["steps"]
            self.polyline_points = parsed_route.get("polyline_points") or []
            self._encoded_polyline = parsed_route.get("polyline")
            self._total_distance_m = int(parsed_route.get("total_distance_m") or 0)
            self._total_duration_s = int(parsed_route.get("total_duration_s") or 0)

            if parsed_route.get("origin"):
                self.origin = parsed_route["origin"]
            if parsed_route.get("destination"):
                self.destination = parsed_route["destination"]

            if reset_progress:
                self.current_step_index = 0
            self.last_spoken_stage = None

            if self._total_distance_m > 0:
                self.last_remaining_announcement = int(self._total_distance_m // 1000) + 1
            else:
                self.last_remaining_announcement = None

            self.navigation_state = {
                "active": True,
                "steps": self.steps,
                "current_step_index": self.current_step_index,
                "polyline": self._encoded_polyline,
                "total_distance": parsed_route.get("total_distance_text"),
                "total_duration": parsed_route.get("total_duration_text"),
                "mode": self._mode,
            }

            callback_session_id = self._session_id
            callback_polyline = self._encoded_polyline

        if callback_polyline and self.on_route_calculated:
            try:
                self.on_route_calculated(callback_session_id, callback_polyline)
            except Exception:
                logger.exception("[NAV] on_route_calculated callback failed")

    def _announce_route_summary(self):
        with self._route_lock:
            if not self.steps:
                return
            first_step = self.steps[0]
            total_distance_m = self._total_distance_m
            total_duration_s = self._total_duration_s

        first_guidance = self._format_guidance_with_distance(first_step, first_step.get("distance"))
        if total_distance_m > 0 and total_duration_s > 0:
            summary = (
                f"Route calculated. {self._format_distance(total_distance_m)} to destination, "
                f"about {self._format_duration(total_duration_s)}. {first_guidance}"
            )
        else:
            summary = f"Route calculated. {first_guidance}"
        self._enqueue_speech(summary, SpeechPriority.NAVIGATION)

    def _process_step_guidance(self, step: StepDict, distance_to_step: float):
        say_text = None
        advance_step = False

        with self._route_lock:
            stage = self.last_spoken_stage
            enabled_stages = self._enabled_guidance_stages(step, distance_to_step)
            turn_threshold = self._immediate_turn_threshold(step)

            if distance_to_step <= turn_threshold:
                if stage != "turn":
                    say_text = self._immediate_turn_text(step)
                self.last_spoken_stage = "turn"
                advance_step = True
            elif "50" in enabled_stages and distance_to_step <= 50.0:
                if stage not in {"50", "turn"}:
                    say_text = self._format_guidance_with_distance(step, 50)
                    self.last_spoken_stage = "50"
            elif "100" in enabled_stages and distance_to_step <= 100.0:
                if stage not in {"100", "turn"}:
                    say_text = self._format_guidance_with_distance(step, 100)
                    self.last_spoken_stage = "100"
            elif "200" in enabled_stages and distance_to_step <= 200.0:
                if stage not in {"200", "100", "turn"}:
                    say_text = self._format_guidance_with_distance(step, 200)
                    self.last_spoken_stage = "200"
            elif "500" in enabled_stages and distance_to_step <= 500.0:
                if stage not in {"500", "200", "100", "50", "turn"}:
                    say_text = self._format_guidance_with_distance(step, 500)
                    self.last_spoken_stage = "500"

        if say_text:
            self._enqueue_speech(say_text, SpeechPriority.NAVIGATION)

        if advance_step:
            self._advance_to_next_step()

    def _advance_to_next_step(self):
        next_step = None
        with self._route_lock:
            if self.current_step_index < len(self.steps):
                self.current_step_index += 1
            self.last_spoken_stage = None
            if self.navigation_state is not None:
                self.navigation_state["current_step_index"] = self.current_step_index
            if self.current_step_index < len(self.steps):
                next_step = self.steps[self.current_step_index]
            logger.info("[NAV] Step advanced")

        if next_step:
            self._enqueue_speech(self._next_step_preview_text(next_step), SpeechPriority.NAVIGATION)

    def _remaining_distance_m(self, current_location: Coordinate) -> Optional[float]:
        with self._route_lock:
            if not self.steps:
                return None
            step_index = self.current_step_index
            steps_copy = list(self.steps)
            destination = self.destination

        if step_index >= len(steps_copy):
            if destination:
                return haversine_distance(
                    current_location[0],
                    current_location[1],
                    destination[0],
                    destination[1],
                )
            return 0.0

        distance_m = self._distance_to_step(current_location, steps_copy[step_index]) or 0.0
        for future_step in steps_copy[step_index + 1 :]:
            distance_m += float(future_step.get("distance") or 0.0)
        return distance_m

    def _maybe_announce_remaining_distance(self, current_location: Coordinate):
        remaining_m = self._remaining_distance_m(current_location)
        if remaining_m is None:
            return

        bucket = self._remaining_progress_bucket(remaining_m)
        if bucket is None:
            return

        should_announce = False
        with self._route_lock:
            if self._last_progress_bucket != bucket:
                self._last_progress_bucket = bucket
                should_announce = True

        if should_announce:
            _, bucket_value = bucket
            self._enqueue_speech(
                f"{self._format_distance(bucket_value)} remaining.",
                SpeechPriority.NAVIGATION,
            )

    def _update_navigation_state(
        self,
        current_location: Coordinate,
        current_step: Optional[StepDict],
        distance_to_step: Optional[float],
        off_route_m: Optional[float] = None,
    ) -> None:
        remaining_m = self._remaining_distance_m(current_location)
        remaining_duration_s = self._estimate_remaining_duration(remaining_m)
        next_instruction = self._lane_guidance_text(current_step) if current_step else None

        with self._route_lock:
            if self.navigation_state is None:
                self.navigation_state = {"active": True}

            self.navigation_state.update(
                {
                    "active": bool(self.is_navigating),
                    "mode": self._mode,
                    "current_step_index": self.current_step_index,
                    "total_steps": len(self.steps),
                    "current_location": {
                        "lat": current_location[0],
                        "lng": current_location[1],
                    } if current_location else None,
                    "next_instruction": next_instruction,
                    "distance_to_next_turn_m": round(float(distance_to_step), 1) if distance_to_step is not None else None,
                    "distance_to_next_turn_text": self._format_distance(distance_to_step) if distance_to_step is not None else None,
                    "remaining_distance_m": round(float(remaining_m), 1) if remaining_m is not None else None,
                    "remaining_distance_text": self._format_distance(remaining_m) if remaining_m is not None else None,
                    "remaining_duration_s": int(remaining_duration_s) if remaining_duration_s is not None else None,
                    "remaining_duration_text": self._format_duration(remaining_duration_s) if remaining_duration_s is not None else None,
                    "off_route_distance_m": round(float(off_route_m), 1) if off_route_m is not None else None,
                }
            )

    def _estimate_remaining_duration(self, remaining_m: Optional[float]) -> Optional[int]:
        if remaining_m is None:
            return None
        with self._route_lock:
            total_distance_m = self._total_distance_m
            total_duration_s = self._total_duration_s
        if remaining_m <= 0:
            return 0
        if total_distance_m > 0 and total_duration_s > 0:
            ratio = max(0.0, min(1.0, float(remaining_m) / float(total_distance_m)))
            return int(round(total_duration_s * ratio))
        avg_speed_mps = 1.4 if self._mode == "walking" else 6.0
        if self._mode == "bicycling":
            avg_speed_mps = 4.2
        if self._mode == "transit":
            avg_speed_mps = 7.0
        return int(round(float(remaining_m) / avg_speed_mps))

    def _enabled_guidance_stages(self, step: StepDict, distance_to_step: float) -> set:
        step_distance = float(step.get("distance") or 0.0)
        effective_distance = max(step_distance, float(distance_to_step or 0.0))
        enabled = set()
        if effective_distance >= 550.0:
            enabled.add("500")
        if effective_distance >= 230.0:
            enabled.add("200")
        if effective_distance >= 120.0:
            enabled.add("100")
        if effective_distance >= 60.0:
            enabled.add("50")
        return enabled

    def _immediate_turn_threshold(self, step: StepDict) -> float:
        maneuver = str((step or {}).get("maneuver") or "").lower()
        if maneuver in {"merge", "roundabout", "roundabout-left", "roundabout-right"}:
            return 20.0
        if maneuver in {"straight", "continue", ""}:
            return 12.0
        return 18.0

    def _remaining_progress_bucket(self, remaining_m: float) -> Optional[Tuple[str, int]]:
        if remaining_m is None or remaining_m < 150.0:
            return None
        if remaining_m >= 1000.0:
            bucket_value = int(remaining_m // 1000) * 1000
            return ("km", max(bucket_value, 1000))
        for threshold in (750, 500, 250):
            if remaining_m >= threshold:
                return ("m", threshold)
        return None

    def _lane_guidance_text(self, step: StepDict) -> str:
        maneuver = str(step.get("maneuver") or "").lower()
        instruction = (step.get("instruction") or "Continue ahead.").strip()

        if instruction:
            return instruction.rstrip(".")
        if maneuver in {"turn-left", "uturn-left", "ramp-left", "fork-left"}:
            return "Turn left"
        if maneuver in {"turn-right", "uturn-right", "ramp-right", "fork-right"}:
            return "Turn right"
        if maneuver == "merge":
            return "Merge ahead"
        if maneuver == "roundabout-left":
            return "Take the left-side exit at the roundabout"
        if maneuver == "roundabout-right":
            return "Take the right-side exit at the roundabout"
        if maneuver == "roundabout":
            return "Enter the roundabout"
        if maneuver == "keep-left":
            return "Keep left"
        if maneuver == "keep-right":
            return "Keep right"
        return "Continue ahead"

    def _immediate_turn_text(self, step: StepDict) -> str:
        maneuver = str(step.get("maneuver") or "").lower()
        instruction = (step.get("instruction") or "").strip()

        if instruction:
            return instruction.rstrip(".")
        if maneuver in {"turn-left", "uturn-left", "ramp-left", "fork-left"}:
            return "Turn left."
        if maneuver in {"turn-right", "uturn-right", "ramp-right", "fork-right"}:
            return "Turn right."
        if maneuver == "merge":
            return "Merge now. Stay in your lane."
        if maneuver in {"roundabout", "roundabout-left", "roundabout-right"}:
            return "Enter the roundabout now."
        if instruction:
            return instruction
        return "Continue."

    def _next_step_preview_text(self, step: StepDict) -> str:
        step_distance = float(step.get("distance") or 0.0)
        if step_distance >= 40.0:
            return f"Then, {self._format_guidance_with_distance(step, step_distance)}"
        return f"Then, {self._lane_guidance_text(step)}."

    def _format_guidance_with_distance(self, step: StepDict, distance_m: Optional[float]) -> str:
        guidance = self._lane_guidance_text(step).rstrip(".")
        if distance_m is None or distance_m <= 0:
            return f"{guidance}."
        return f"In {self._format_distance(distance_m)}, {guidance}."

    def _enqueue_speech(
        self,
        text: str,
        priority: SpeechPriority = SpeechPriority.NAVIGATION,
        source: str = "navigation",
    ):
        if not text or not str(text).strip():
            return
        try:
            if hasattr(self.speech_manager, "enqueue"):
                self.speech_manager.enqueue(text=text, priority=priority, source=source)
            else:
                self.speech_manager.speak(str(text), priority, source=source)
        except TypeError:
            try:
                if hasattr(self.speech_manager, "enqueue"):
                    self.speech_manager.enqueue(str(text))
                else:
                    self.speech_manager.speak(str(text), priority, source=source)
            except Exception:
                logger.exception("[NAV] Speech enqueue failed")
        except Exception:
            logger.exception("[NAV] Speech enqueue failed")

    def _get_live_location(self) -> Optional[Coordinate]:
        location = None

        if self.current_location and self._normalize_coordinate(self.current_location):
            location = self._normalize_coordinate(self.current_location)

        if self.location_manager is not None:
            try:
                if getattr(self.location_manager, "web_mode", False) and hasattr(
                    self.location_manager,
                    "get_mobile_gps_location",
                ):
                    location = self._normalize_coordinate(self.location_manager.get_mobile_gps_location()) or location
            except Exception:
                logger.exception("[NAV] Failed to fetch mobile GPS location")

            if location is None:
                try:
                    if hasattr(self.location_manager, "get_best_available_location"):
                        try:
                            raw = self.location_manager.get_best_available_location(use_cache=False)
                        except TypeError:
                            raw = self.location_manager.get_best_available_location()
                        location = self._normalize_coordinate(raw)
                except Exception:
                    logger.exception("[NAV] Failed to fetch best available location")

        if location:
            self.current_location = location
        return location

    def get_navigation_status(self) -> Dict[str, Any]:
        with self._route_lock:
            status = dict(self.navigation_state or {})
            status.setdefault("active", bool(self.is_navigating))
            status.setdefault("current_step_index", self.current_step_index)
            status.setdefault("total_steps", len(self.steps))
            status.setdefault("mode", self._mode)
        return status

    def _emit_on_start_once(self):
        should_emit = False
        with self._route_lock:
            if not self._on_start_emitted:
                self._on_start_emitted = True
                should_emit = True
        if should_emit and self.on_start:
            try:
                self.on_start()
            except Exception:
                logger.exception("[NAV] on_start callback failed")

    def _finalize_navigation(self, stopped: bool):
        with self._route_lock:
            self._on_start_emitted = False

            self.is_navigating = False
            self.navigation_active = False
            self.steps = []
            self.current_step_index = 0
            self.last_spoken_stage = None
            self.last_remaining_announcement = None
            self._last_progress_bucket = None
            self.polyline_points = []
            self.navigation_state = None
            self.destination = None
            self.origin = None

            self._destination_raw = None
            self._encoded_polyline = None
            self._total_distance_m = 0
            self._total_duration_s = 0
            self._mode = "driving"
            self._session_id = None

            self.navigation_thread = None
            self._thread = None
            self._stop_event.clear()

        # Always call on_stop so voice engine state is reset even when
        # navigation failed before the route was calculated (e.g. ZERO_RESULTS).
        if self.on_stop:
            try:
                self.on_stop()
            except Exception:
                logger.exception("[NAV] on_stop callback failed")

    def _parse_start_arguments(self, *args, **kwargs) -> Tuple[Any, str, Optional[str], Optional[Coordinate]]:
        destination = kwargs.get("destination")
        mode = kwargs.get("mode", "driving")
        session_id = kwargs.get("session_id")
        origin = kwargs.get("origin")

        if args:
            if len(args) >= 3 and self._normalize_coordinate(args[0]) is not None:
                origin = args[0]
                destination = args[1]
                mode = args[2]
                if len(args) >= 4 and session_id is None:
                    session_id = args[3]
            else:
                destination = args[0]
                if len(args) >= 2:
                    mode = args[1]
                if len(args) >= 3 and session_id is None:
                    session_id = args[2]
                if len(args) >= 4 and origin is None:
                    origin = args[3]

        if isinstance(destination, str):
            destination = destination.strip()
        origin_coord = self._normalize_coordinate(origin)
        return destination, mode, session_id, origin_coord

    def _normalize_coordinate(self, value: Any) -> Optional[Coordinate]:
        if value is None:
            return None
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return (float(value[0]), float(value[1]))
            except Exception:
                return None
        if isinstance(value, dict) and "lat" in value and "lng" in value:
            try:
                return (float(value["lat"]), float(value["lng"]))
            except Exception:
                return None
        return None

    def _destination_to_api_value(self, destination: Any) -> Optional[str]:
        if destination is None:
            return None
        coord = self._normalize_coordinate(destination)
        if coord:
            return f"{coord[0]},{coord[1]}"
        value = str(destination).strip()
        return value or None

    def _normalize_mode(self, mode: str) -> str:
        normalized = str(mode or "driving").strip().lower()
        if normalized in {"walk", "walking"}:
            return "walking"
        if normalized in {"bike", "bicycle", "bicycling", "cycling"}:
            return "bicycling"
        if normalized in {"drive", "car", "driving"}:
            return "driving"
        if normalized in {"transit", "public transport"}:
            return "transit"
        return "driving"

    def _project_to_meters(self, lat: float, lng: float, ref_lat: float) -> Coordinate:
        earth_radius_m = 6371000.0
        x = math.radians(lng) * earth_radius_m * math.cos(math.radians(ref_lat))
        y = math.radians(lat) * earth_radius_m
        return x, y

    def _format_distance(self, meters: int) -> str:
        if meters is None:
            return "unknown distance"
        meters_value = float(meters)
        if meters_value < 1000:
            rounded = int(5 * round(meters_value / 5.0)) if meters_value >= 20 else int(round(meters_value))
            return f"{max(rounded, 1)} meters"
        return f"{meters_value / 1000.0:.1f} kilometers"

    def _format_duration(self, seconds: int) -> str:
        if seconds is None:
            return "unknown time"
        minutes = int(round(seconds / 60.0))
        if minutes <= 0:
            return "less than a minute"
        if minutes < 60:
            return f"{minutes} minutes"
        hours = minutes // 60
        remaining_minutes = minutes % 60
        if remaining_minutes == 0:
            return f"{hours} hours"
        return f"{hours} hours {remaining_minutes} minutes"

    def _decode_polyline(self, encoded: str) -> List[Coordinate]:
        if not encoded:
            return []

        points: List[Coordinate] = []
        index = 0
        lat = 0
        lng = 0

        while index < len(encoded):
            shift = 0
            result = 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta_lat = ~(result >> 1) if (result & 1) else (result >> 1)
            lat += delta_lat

            shift = 0
            result = 0
            while True:
                b = ord(encoded[index]) - 63
                index += 1
                result |= (b & 0x1F) << shift
                shift += 5
                if b < 0x20:
                    break
            delta_lng = ~(result >> 1) if (result & 1) else (result >> 1)
            lng += delta_lng

            points.append((lat / 1e5, lng / 1e5))

        return points

    # Compatibility helpers for existing integration points.
    def update_location(self):
        location = self._get_live_location()
        if location:
            self.current_location = location
        return location

    def check_next_turn(self, lat: float, lng: float):
        with self._route_lock:
            if not self.steps or self.current_step_index >= len(self.steps):
                return
            step = self.steps[self.current_step_index]
        distance_to_step = self._distance_to_step((lat, lng), step)
        if distance_to_step is not None:
            self._process_step_guidance(step, distance_to_step)

    def _arrive(self):
        logger.info("[NAV] Arrival detected")
        self._enqueue_speech("You have arrived at your destination.", SpeechPriority.NAVIGATION)
        self._stop_event.set()

    def _haversine_m(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        return haversine_distance(lat1, lng1, lat2, lng2)


# ==========================================
# FROM: navigation_live.py
# ==========================================
import logging
import threading
import time
from math import radians, sin, cos, sqrt, atan2

from app.services.speech.speech_manager import SpeechPriority, speech_manager

logger = logging.getLogger("smartvision.navigation_live")


class NavigationLive:
    def __init__(
        self,
        get_current_location,
        vision_engine=None,
        speech_manager_instance=None,
        obstacle_cooldown: float = 10.0,
    ):
        self.get_current_location = get_current_location
        self.vision_engine = vision_engine
        self.speech_manager = speech_manager_instance or speech_manager
        self.obstacle_cooldown = float(obstacle_cooldown)

        self.navigation_active = False
        self.navigation_stop_event = threading.Event()
        self._thread = None
        self.last_obstacle_time = 0.0
        self.last_obstacle_key = None

    def start(self, textual_instructions, step_coordinates, destination_name: str = None):
        if self.navigation_active:
            return False
        self.navigation_active = True
        self.navigation_stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker,
            args=(textual_instructions, step_coordinates, destination_name),
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self):
        self.navigation_stop_event.set()
        self.navigation_active = False

    def _speak(self, text: str, priority: SpeechPriority = SpeechPriority.NAVIGATION):
        if not text or not text.strip():
            return
        try:
            self.speech_manager.speak(text, priority, source="navigation")
        except Exception:
            logger.exception("Navigation speech failed")

    def _calculate_distance(self, lat1, lng1, lat2, lng2):
        r = 6371000
        phi1, phi2 = radians(lat1), radians(lat2)
        dphi = radians(lat2 - lat1)
        dlambda = radians(lng2 - lng1)
        a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlambda / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        return r * c

    def _worker(self, textual_instructions, step_coordinates, destination_name):
        try:
            logger.info("Navigation live started")
            if not textual_instructions:
                return

            current_step = 0
            last_step_spoken = -1
            turn_threshold_m = 15.0

            step_text = textual_instructions[current_step]
            self._speak(f"Step {current_step + 1}. {step_text}", priority=SpeechPriority.NAVIGATION)
            last_step_spoken = current_step

            while current_step < len(textual_instructions):
                if self.navigation_stop_event.is_set():
                    break

                current_loc = self.get_current_location()
                if not current_loc:
                    time.sleep(5)
                    continue

                user_lat, user_lng = current_loc[0], current_loc[1]
                if current_step >= len(step_coordinates):
                    break

                step_lat, step_lng, _ = step_coordinates[current_step]
                distance_to_turn = self._calculate_distance(user_lat, user_lng, step_lat, step_lng)

                if distance_to_turn <= turn_threshold_m:
                    current_step += 1
                    if current_step >= len(textual_instructions):
                        break
                    if current_step != last_step_spoken:
                        step_text = textual_instructions[current_step]
                        self._speak(
                            f"Step {current_step + 1}. {step_text}",
                            priority=SpeechPriority.NAVIGATION,
                        )
                        last_step_spoken = current_step

                if self.vision_engine:
                    frame = self.vision_engine.capture_frame()
                    if frame is not None:
                        self.vision_engine.detect_objects(frame)

                if self.vision_engine and hasattr(self.vision_engine, "get_detected_obstacles"):
                    try:
                        obstacles = self.vision_engine.get_detected_obstacles()
                        if obstacles:
                            obstacle_text = ", ".join(sorted(obstacles))
                            now_ts = time.time()
                            if (
                                self.last_obstacle_key == obstacle_text
                                and (now_ts - self.last_obstacle_time) < self.obstacle_cooldown
                            ):
                                time.sleep(5)
                                continue
                            self.last_obstacle_key = obstacle_text
                            self.last_obstacle_time = now_ts
                            self._speak(
                                f"Caution: {obstacle_text} ahead",
                                priority=SpeechPriority.NAVIGATION,
                            )
                    except Exception:
                        pass

                time.sleep(5)

            if not self.navigation_stop_event.is_set():
                destination = destination_name or "your destination"
                completion_msg = f"You have arrived at {destination}. Navigation complete!"
                self._speak(completion_msg, priority=SpeechPriority.NAVIGATION)
            else:
                self._speak("Navigation stopped.", priority=SpeechPriority.NAVIGATION)
        except Exception:
            logger.exception("Navigation live error")
            self._speak("Navigation interrupted due to an error.", priority=SpeechPriority.SYSTEM)
        finally:
            self.navigation_active = False
            self.navigation_stop_event.clear()

