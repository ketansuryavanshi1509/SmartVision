import threading
import time
import subprocess
from enum import IntEnum
from typing import Callable, Optional, Dict, Any

from flask import request, jsonify

# Import voice command parser
try:
    from app.services.speech.voice_command_parser import (
        VoiceCommandParser, 
        CommandRouter,
        CommandType,
        get_command_router
    )
    VOICE_COMMAND_AVAILABLE = True
except ImportError:
    VOICE_COMMAND_AVAILABLE = False


class SpeechPriority(IntEnum):
    BACKGROUND = 20
    REGULAR = 40
    SYSTEM = 60
    NAVIGATION = 80
    HIGH = 80
    EMERGENCY = 100


speech_lock = threading.Lock()


def _normalize_priority(priority) -> int:
    if isinstance(priority, SpeechPriority):
        return int(priority.value)
    try:
        return int(priority)
    except Exception:
        return int(SpeechPriority.REGULAR)


class SpeechManager:
    def __init__(self) -> None:
        self._queue = []
        self._queue_lock = threading.Lock()
        self._queue_event = threading.Event()
        self._counter = 0
        self._cancel_flag = threading.Event()
        self._navigation_suppression = False
        self._pause_listening_cb: Optional[Callable[[], None]] = None
        self._resume_listening_cb: Optional[Callable[[], None]] = None

        self.is_speaking = False
        self.last_spoken_text: Optional[str] = None
        self.last_spoken_id: int = 0
        self._current_speech: Optional[Dict[str, Any]] = None
        self._last_speech_ended_ts: float = 0.0

        # Pending speech for web clients (when local speech is suppressed)
        self._pending_web_speech: list = []
        self._pending_web_lock = threading.Lock()
        
        # Voice command processing
        self.command_router = get_command_router() if VOICE_COMMAND_AVAILABLE else None
        self.voice_listening_enabled = False

        self._worker = threading.Thread(target=self._speech_worker, daemon=True)
        self._worker.start()

    def set_speech_callbacks(self, pause_cb: Callable[[], None], resume_cb: Callable[[], None]) -> None:
        self._pause_listening_cb = pause_cb
        self._resume_listening_cb = resume_cb

    def set_navigation_suppression(self, enabled: bool) -> None:
        self._navigation_suppression = bool(enabled)

    def speak(self, text: str, priority: SpeechPriority, source: str = "system", callback=None) -> bool:
        if not text or not str(text).strip():
            return False

        priority_value = _normalize_priority(priority)
        if self._navigation_suppression:
            if source != "navigation" and priority_value < SpeechPriority.NAVIGATION:
                return False

        with self._queue_lock:
            self._counter += 1
            item = {
                "id": self._counter,
                "text": str(text),
                "priority": int(priority_value),
                "source": source or "system",
                "callback": callback,
                "queued_at": time.time(),
            }
            self._queue.append(item)
            self._queue.sort(key=lambda x: (-x["priority"], x["queued_at"]))
            self._queue_event.set()
        return True

    def enqueue(
        self,
        text: str,
        priority: SpeechPriority = SpeechPriority.REGULAR,
        source: str = "system",
        callback=None,
    ) -> bool:
        return self.speak(text=text, priority=priority, source=source, callback=callback)
    
    def process_voice_command(self, transcribed_text: str) -> bool:
        """
        Process a voice command from speech-to-text.
        """
        if not VOICE_COMMAND_AVAILABLE or self.command_router is None:
            return False
        
        if not self.voice_listening_enabled:
            return False
        
        try:
            success = self.command_router.process_command(transcribed_text)
            return success
        except Exception:
            return False
    
    def enable_voice_commands(self):
        self.voice_listening_enabled = True
    
    def disable_voice_commands(self):
        self.voice_listening_enabled = False

    def speak_blocking(self, text: str, priority: SpeechPriority = SpeechPriority.REGULAR, source: str = "system") -> None:
        done_event = threading.Event()

        def _on_done():
            done_event.set()

        ok = self.speak(text, priority, source, _on_done)
        if ok:
            done_event.wait()

    def cancel_all_speech(self) -> None:
        with self._queue_lock:
            should_cancel_current = bool(self.is_speaking)
            self._queue.clear()
        if should_cancel_current:
            self._cancel_flag.set()
        else:
            self._cancel_flag.clear()
        self._queue_event.set()

    def cancel_all(self) -> None:
        self.cancel_all_speech()

    def clear_queue(self) -> None:
        with self._queue_lock:
            self._queue.clear()
        self._queue_event.clear()

    def cancel_background(self) -> None:
        with self._queue_lock:
            self._queue = [q for q in self._queue if q.get("priority", 0) > SpeechPriority.BACKGROUND]
            if not self._queue:
                self._queue_event.clear()

    def stop(self) -> None:
        self.cancel_all_speech()

    def is_currently_speaking(self) -> bool:
        return bool(self.is_speaking)

    def _speak_sync(self, text: str) -> None:
        from app.config import config
        if getattr(config, "DISABLE_LOCAL_SPEECH", False):
            from app.core.logger import logger
            logger.info(f"[SPEECH QUEUE] Local speech suppressed: {text}")
            # Queue for web client pickup instead
            with self._pending_web_lock:
                self._pending_web_speech.append(text)
            return

        try:
            safe_text = text.replace("'", "''")
            cmd = (
                "powershell -Command \""
                "Add-Type -AssemblyName System.Speech; "
                "(New-Object System.Speech.Synthesis.SpeechSynthesizer)"
                f".Speak('{safe_text}');\""
            )
            subprocess.run(cmd, shell=True, capture_output=True)
        except Exception:
            pass

    def _speech_worker(self) -> None:
        while True:
            self._queue_event.wait()
            while True:
                with self._queue_lock:
                    if self._cancel_flag.is_set():
                        self._queue.clear()
                        self._cancel_flag.clear()
                    if not self._queue:
                        self._queue_event.clear()
                        self._current_speech = None
                        self.is_speaking = False
                        break
                    item = self._queue.pop(0)

                self._current_speech = {
                    "text": item.get("text"),
                    "priority": item.get("priority"),
                    "source": item.get("source"),
                    "started_at": time.time(),
                }
                self.is_speaking = True

                if self._pause_listening_cb:
                    try:
                        self._pause_listening_cb()
                    except Exception:
                        pass

                if not self._cancel_flag.is_set():
                    self._speak_sync(item.get("text", ""))

                if self._resume_listening_cb:
                    try:
                        self._resume_listening_cb()
                    except Exception:
                        pass

                self.last_spoken_text = item.get("text")
                self.last_spoken_id = item.get("id", self.last_spoken_id + 1)
                self.is_speaking = False
                self._last_speech_ended_ts = time.time()

                cb = item.get("callback")
                if callable(cb):
                    try:
                        cb()
                    except Exception:
                        pass

    def get_status(self) -> Dict[str, Any]:
        with self._queue_lock:
            queue_size = len(self._queue)
        return {
            "is_speaking": bool(self.is_speaking),
            "current_speech": self._current_speech,
            "queue_size": queue_size,
            "last_spoken_text": self.last_spoken_text,
            "last_spoken_id": self.last_spoken_id,
        }

    def get_pending_web_speech(self) -> list:
        """Return and clear any pending speech texts for web clients."""
        with self._pending_web_lock:
            items = list(self._pending_web_speech)
            self._pending_web_speech.clear()
        return items


speech_manager = SpeechManager()


def create_speech_api(app):
    @app.route("/api/speak", methods=["POST"])
    def api_speak():
        data = request.json or {}
        text = str(data.get("text", "") or "").strip()
        priority = data.get("priority", SpeechPriority.REGULAR)
        source = data.get("source", "web")
        if not text:
            return jsonify({"success": True, "ignored": True, "reason": "empty_text"}), 200
        ok = speech_manager.speak(text, priority, source)
        if ok:
            return jsonify({"success": True}), 200
        return jsonify({"success": True, "ignored": True, "reason": "speech_rejected"}), 200

    @app.route("/api/speech/status", methods=["GET"])
    def api_speech_status():
        return jsonify(speech_manager.get_status()), 200

    @app.route("/api/speech/cancel", methods=["POST"])
    def api_speech_cancel():
        speech_manager.cancel_all_speech()
        return jsonify({"success": True}), 200

    @app.route("/api/speech/pending", methods=["GET"])
    def api_speech_pending():
        """Return any pending speech texts for the web client to speak."""
        items = speech_manager.get_pending_web_speech()
        return jsonify({"success": True, "pending_speech": items}), 200
