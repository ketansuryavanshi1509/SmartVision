import io
import queue
import threading
import time
import wave

import numpy as np
import sounddevice as sd
from groq import Groq
from scipy.signal import resample

from app.config import config
from app.core.logger import logger
from app.services.speech.speech_manager import speech_manager


class GroqWhisperSTT:
    """Streaming microphone capture with sentence-end hold before transcription."""

    def __init__(self, sample_rate_hz: int = 16000, language_code: str = "en"):
        if not config.GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not configured")

        self.sample_rate_hz = int(sample_rate_hz)
        self.language_code = language_code
        self._client = Groq(api_key=config.GROQ_API_KEY)
        self._model = config.GROQ_WHISPER_MODEL or "whisper-large-v3-turbo"
        self._stop_event = threading.Event()
        self._paused = threading.Event()
        self._listening = False
        self._listening_lock = threading.Lock()
        self._restart_delay_s = 1.0
        self._audio_queue = None
        self._capture_rate = None
        self._last_voice_ts = 0.0
        self._voice_detected = threading.Event()
        self._voice_frame_count = 0

        self._vad_rms_threshold = config.VAD_RMS_THRESHOLD
        self._vad_silence_s = config.VAD_SILENCE_SECONDS
        self._vad_min_utterance_s = config.VAD_MIN_UTTERANCE_SECONDS
        self._vad_max_utterance_s = config.VAD_MAX_UTTERANCE_SECONDS
        self._vad_start_frames = config.VAD_START_FRAMES
        self._chunk_log_interval_s = config.STT_CHUNK_LOG_INTERVAL
        self._sentence_end_hold_s = max(0.0, float(config.STT_SENTENCE_END_HOLD_SECONDS))

        logger.info("Groq Whisper STT initialized")

    # Whisper transcription prompt — gives the model vocabulary hints
    _TRANSCRIPTION_PROMPT = (
        "Navigate, navigation, destination, directions, stop navigation, cancel, "
        "describe, location, emergency, walking, driving, two wheeler, public transport, "
        "nearest, nearby, closest, hospital, pharmacy, railway station, bus stop, "
        "Pune, Mumbai, Delhi, Bengaluru, Hyderabad, Chennai, Kolkata, Ahmedabad, "
        "Jaipur, Lucknow, Thane, Vengurla, Navi Mumbai, Pimpri, Chinchwad, Hadapsar, Kothrud, "
        "Hinjewadi, Wakad, Baner, Aundh, Shivajinagar, Deccan, Swargate, Katraj, "
        "Indiranagar, Koramangala, Brigade Road, MG Road, Whitefield, "
        "yes, no, first, second, third, "
        "navigate to, go to, take me to, find nearest"
    )

    def transcribe_audio_bytes(self, audio_bytes: bytes) -> str:
        if not audio_bytes:
            return ""

        wav_file = io.BytesIO(self._audio_to_wav_bytes(audio_bytes, sample_rate=16000))
        wav_file.name = "audio.wav"
        result = self._client.audio.transcriptions.create(
            model=self._model,
            file=wav_file,
            language=self.language_code,
            response_format="text",
            prompt=self._TRANSCRIPTION_PROMPT,
        )
        return (result or "").strip()

    def stop(self):
        self._stop_event.set()
        if self._audio_queue is not None:
            try:
                self._audio_queue.put_nowait(None)
            except Exception:
                pass

    def pause(self):
        self._paused.set()

    def resume(self):
        self._paused.clear()

    def listen(self, on_transcript):
        if not callable(on_transcript):
            return

        with self._listening_lock:
            if self._listening:
                return
            self._listening = True

        self._stop_event.clear()
        audio_queue = queue.Queue()
        self._audio_queue = audio_queue
        audio_stop_event = threading.Event()

        def audio_callback(indata, frames, time_info, status):
            if status:
                logger.debug("Audio status: %s", status)
            if self._paused.is_set():
                return
            if speech_manager and getattr(speech_manager, "is_speaking", False):
                return
            try:
                rms = self._rms_energy(indata)
                now = time.time()
                if rms >= self._vad_rms_threshold:
                    self._voice_frame_count += 1
                    self._last_voice_ts = now
                    if self._voice_frame_count >= self._vad_start_frames:
                        self._voice_detected.set()
                else:
                    self._voice_frame_count = 0
                    if (now - self._last_voice_ts) > self._vad_silence_s:
                        self._voice_detected.clear()
                audio_queue.put_nowait((indata.copy(), rms))
            except Exception:
                logger.exception("Audio callback failed")

        def audio_capture_worker():
            while not self._stop_event.is_set() and not audio_stop_event.is_set():
                try:
                    device_info = self._get_input_device_info()
                    if not device_info:
                        logger.warning("No input device available")
                        time.sleep(self._restart_delay_s)
                        continue

                    default_rate = int(device_info.get("default_samplerate") or self.sample_rate_hz)
                    max_channels = int(device_info.get("max_input_channels") or 1)
                    channels = 1 if max_channels >= 1 else 1
                    attempt_rates = [self.sample_rate_hz]
                    if default_rate != self.sample_rate_hz:
                        attempt_rates.append(default_rate)

                    opened = False
                    for device_rate in attempt_rates:
                        try:
                            with sd.InputStream(
                                samplerate=device_rate,
                                blocksize=max(1, int(device_rate / 10)),
                                dtype="int16",
                                channels=channels,
                                device=None,
                                callback=audio_callback,
                            ):
                                self._capture_rate = device_rate
                                opened = True
                                logger.info(
                                    "STT audio capture started (rate=%s, channels=%s, device=%s)",
                                    device_rate,
                                    channels,
                                    device_info.get("name"),
                                )
                                while not self._stop_event.is_set() and not audio_stop_event.is_set():
                                    time.sleep(0.2)
                            if opened:
                                break
                        except Exception as exc:
                            logger.warning("STT audio capture failed at %s Hz: %s", device_rate, exc)
                            if self._stop_event.is_set() or audio_stop_event.is_set():
                                break
                            time.sleep(self._restart_delay_s)

                    if not opened:
                        time.sleep(self._restart_delay_s)
                except Exception as exc:
                    logger.exception("Audio capture failed: %s", exc)
                    if self._stop_event.is_set() or audio_stop_event.is_set():
                        break
                    time.sleep(self._restart_delay_s)

        def stt_worker():
            utterance_buffer = bytearray()
            utterance_start_ts = 0.0
            collecting = False
            last_chunk_log_ts = 0.0
            silence_hold_started_ts = 0.0

            while not self._stop_event.is_set():
                try:
                    if self._paused.is_set():
                        if collecting:
                            utterance_buffer.clear()
                            collecting = False
                            silence_hold_started_ts = 0.0
                        time.sleep(0.1)
                        continue

                    try:
                        chunk = audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        if collecting and not self._voice_detected.is_set():
                            if silence_hold_started_ts == 0.0:
                                silence_hold_started_ts = time.time()
                            if (time.time() - silence_hold_started_ts) < self._sentence_end_hold_s:
                                continue
                            self._flush_utterance(utterance_buffer, utterance_start_ts, on_transcript)
                            utterance_buffer.clear()
                            collecting = False
                            silence_hold_started_ts = 0.0
                        continue

                    if chunk is None:
                        break

                    indata, rms = chunk if isinstance(chunk, tuple) else (chunk, 0.0)
                    now = time.time()

                    if speech_manager and getattr(speech_manager, "is_speaking", False):
                        if collecting:
                            utterance_buffer.clear()
                            collecting = False
                            silence_hold_started_ts = 0.0
                        continue

                    effective_rate = self._capture_rate or self.sample_rate_hz
                    resampled = self._resample_chunk(indata, effective_rate, 16000)
                    if not resampled:
                        continue

                    if (now - last_chunk_log_ts) >= self._chunk_log_interval_s:
                        logger.debug("Audio chunk: %d bytes, RMS: %.0f", len(resampled), rms)
                        last_chunk_log_ts = now

                    if self._voice_detected.is_set():
                        silence_hold_started_ts = 0.0
                        if not collecting:
                            collecting = True
                            utterance_start_ts = now
                            utterance_buffer.clear()
                            logger.info("Voice activity detected, collecting utterance...")
                        utterance_buffer.extend(resampled)

                        if (now - utterance_start_ts) >= self._vad_max_utterance_s:
                            logger.warning("Max utterance length reached, force-transcribing")
                            self._flush_utterance(utterance_buffer, utterance_start_ts, on_transcript)
                            utterance_buffer.clear()
                            collecting = False
                            silence_hold_started_ts = 0.0
                    elif collecting:
                        if silence_hold_started_ts == 0.0:
                            silence_hold_started_ts = now
                        if (now - silence_hold_started_ts) < self._sentence_end_hold_s:
                            continue
                        self._flush_utterance(utterance_buffer, utterance_start_ts, on_transcript)
                        utterance_buffer.clear()
                        collecting = False
                        silence_hold_started_ts = 0.0
                except Exception as exc:
                    logger.exception("Groq STT worker failed: %s", exc)
                    utterance_buffer.clear()
                    collecting = False
                    silence_hold_started_ts = 0.0
                    if self._stop_event.is_set():
                        break
                    time.sleep(self._restart_delay_s)

        audio_thread = threading.Thread(target=audio_capture_worker, daemon=True)
        stt_thread = threading.Thread(target=stt_worker, daemon=True)
        audio_thread.start()
        stt_thread.start()

        while not self._stop_event.is_set():
            time.sleep(0.2)

        audio_stop_event.set()
        try:
            audio_queue.put_nowait(None)
        except Exception:
            pass
        self._audio_queue = None

        with self._listening_lock:
            self._listening = False

    # Known system TTS phrases that the mic might pick up (substrings to check)
    _SYSTEM_ECHO_PHRASES = [
        "walking driving two wheeler public transport",
        "walking driving cycling public transport",
        "how would you like to travel",
        "where would you like to go",
        "tell me where you want to go",
        "please tell me your destination",
        "say the destination again",
        "drive to travel walking driving",
        "navigation cancelled",
        "starting navigation",
        "calculating route",
        "no navigation is active",
        "analyzing scene",
        "say yes or no",
    ]

    def _flush_utterance(self, utterance_buffer: bytearray, utterance_start_ts: float, on_transcript):
        duration = time.time() - utterance_start_ts
        if duration < self._vad_min_utterance_s or not utterance_buffer:
            return

        # Post-speech cooldown: discard utterances that started within 1.5s
        # after the speech manager finished speaking (likely mic echo)
        if speech_manager:
            current_speech = getattr(speech_manager, "_current_speech", None)
            if current_speech:
                started_at = current_speech.get("started_at", 0)
                # If the utterance started during an active speech, discard
                if utterance_start_ts >= started_at:
                    logger.debug("Discarding utterance that overlapped with system speech")
                    return
            # Also check if speech just finished (within 1.5s)
            last_spoken = getattr(speech_manager, "last_spoken_text", None)
            if last_spoken and getattr(speech_manager, "_last_speech_ended_ts", 0):
                gap = utterance_start_ts - speech_manager._last_speech_ended_ts
                if gap < 1.5:
                    logger.debug("Discarding utterance started %.1fs after speech ended (echo buffer)", gap)
                    return

        logger.info(
            "Utterance complete (%.1fs, %d bytes), transcribing...",
            duration,
            len(utterance_buffer),
        )
        try:
            transcript = self.transcribe_audio_bytes(bytes(utterance_buffer))
        except Exception as exc:
            logger.error("Groq Whisper transcription failed: %s", exc)
            transcript = ""

        if transcript:
            cleaned = transcript.strip().lower()
            for punc in [".", ",", "!", "?", ":", ";"]:
                cleaned = cleaned.replace(punc, "")
            cleaned = " ".join(cleaned.split())  # normalize whitespace
            
            # 1. Filler/empty check
            invalid_fillers = {"", "uh", "um", "hmm", "mm", "mm-hmm", "ah", "oh", "huh"}
            if cleaned in invalid_fillers or len(cleaned.split()) < 1:
                logger.debug("Discarding filler/invalid utterance: '%s'", transcript)
                transcript = ""

            # 2. System echo detection — check if the transcript contains TTS prompt text
            if transcript:
                for echo_phrase in self._SYSTEM_ECHO_PHRASES:
                    if echo_phrase in cleaned:
                        logger.debug("Discarding system echo: '%s'", transcript[:80])
                        transcript = ""
                        break

            # 3. Detect Whisper hallucinations (repetitive noise artifacts)
            if transcript:
                words = cleaned.split()
                word_count = len(words)
                # Whisper often generates very long repetitive output from noise
                if word_count > 12:
                    logger.debug("Discarding overly long hallucination (%d words): '%s'", word_count, transcript[:80])
                    transcript = ""
                elif word_count >= 3:
                    # Check if any word repeats excessively (>50% of transcript)
                    from collections import Counter
                    counts = Counter(words)
                    most_common_word, most_common_count = counts.most_common(1)[0]
                    if most_common_count / word_count > 0.5:
                        logger.debug("Discarding repetitive hallucination: '%s'", transcript[:80])
                        transcript = ""

            # 4. Foreign script / non-ASCII artifacts (Whisper sometimes outputs Korean, Chinese, etc.)
            if transcript:
                import re as _re
                ascii_ratio = sum(1 for c in cleaned if ord(c) < 128) / max(len(cleaned), 1)
                if ascii_ratio < 0.8:
                    logger.debug("Discarding non-ASCII hallucination: '%s'", transcript[:80])
                    transcript = ""

            # 5. Discard known Whisper noise-to-text artifacts
            if transcript:
                noise_phrases = {
                    "thank you for watching", "thanks for watching",
                    "please subscribe", "like and subscribe",
                    "you", "bye", "bye bye", "good bye",
                    "i win", "i win i win",
                    "yeah", "yep", "irene", "ok",
                }
                if cleaned in noise_phrases:
                    logger.debug("Discarding noise phrase: '%s'", transcript)
                    transcript = ""

        if transcript and not (speech_manager and getattr(speech_manager, "is_speaking", False)):
            logger.info("[VOICE RAW] %s", transcript)
            try:
                on_transcript(transcript)
            except Exception:
                logger.exception("Transcript handler failed")

    def _get_input_device_info(self):
        try:
            return sd.query_devices(None, "input")
        except Exception:
            try:
                return sd.query_devices(kind="input")
            except Exception:
                return None

    def _audio_to_wav_bytes(self, raw_audio: bytes, sample_rate: int = 16000) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(raw_audio)
        return buf.getvalue()

    @staticmethod
    def _rms_energy(indata) -> float:
        if indata is None:
            return 0.0
        try:
            mono = np.mean(indata.astype(np.float32), axis=1) if indata.ndim > 1 else indata.astype(np.float32)
            if mono.size == 0:
                return 0.0
            return float(np.sqrt(np.mean(np.square(mono))))
        except Exception:
            return 0.0

    @staticmethod
    def _resample_chunk(indata, input_rate, target_rate):
        if indata is None:
            return b""
        mono = np.mean(indata.astype(np.float32), axis=1) if indata.ndim > 1 else indata.astype(np.float32)
        if input_rate != target_rate:
            target_length = int(round(len(mono) * float(target_rate) / float(input_rate)))
            if target_length <= 0:
                return b""
            mono = resample(mono, target_length)
        mono = np.clip(mono, -32768, 32767).astype(np.int16)
        return mono.tobytes()
