import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Firebase
    FIREBASE_CREDENTIALS = os.getenv("FIREBASE_CREDENTIALS")
    FIREBASE_API_KEY = os.getenv("FIREBASE_API_KEY")
    FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID")
    FIREBASE_AUTH_DOMAIN = os.getenv("FIREBASE_AUTH_DOMAIN")
    FIREBASE_STORAGE_BUCKET = os.getenv("FIREBASE_STORAGE_BUCKET")
    
    # Google Maps
    GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
    
    # Groq
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo")
    
    # Flask
    FLASK_SECRET_KEY = os.getenv("FLASK_SECRET_KEY")
    
    # Audio
    MIC_SAMPLE_RATE = int(os.getenv("MIC_SAMPLE_RATE", "16000"))
    
    # VAD & STT
    VAD_RMS_THRESHOLD = float(os.getenv("VAD_RMS_THRESHOLD", "300"))
    VAD_SILENCE_DURATION = float(os.getenv("VAD_SILENCE_DURATION", "2.2"))
    VAD_SILENCE_SECONDS = float(os.getenv("VAD_SILENCE_SECONDS", "2.2"))
    VAD_MIN_UTTERANCE_SECONDS = float(os.getenv("VAD_MIN_UTTERANCE_SECONDS", "0.5"))
    VAD_MAX_UTTERANCE_SECONDS = float(os.getenv("VAD_MAX_UTTERANCE_SECONDS", "10.0"))
    VAD_START_FRAMES = int(os.getenv("VAD_START_FRAMES", "3"))
    STT_MODEL_SIZE = os.getenv("STT_MODEL_SIZE", "base")
    STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en")
    STT_CHUNK_LOG_INTERVAL = float(os.getenv("STT_CHUNK_LOG_INTERVAL", "5.0"))
    STT_SENTENCE_END_HOLD_SECONDS = float(os.getenv("STT_SENTENCE_END_HOLD_SECONDS", "0.8"))
    STT_FRAGMENT_MERGE_SECONDS = float(os.getenv("STT_FRAGMENT_MERGE_SECONDS", "5.0"))

    # Thresholds
    PLACE_MATCH_THRESHOLD = float(os.getenv("PLACE_MATCH_THRESHOLD", "0.62"))

    DEFAULT_COUNTRY_CODE = os.getenv("DEFAULT_COUNTRY_CODE", "IN")
    
    # Headless/API Mode
    DISABLE_LOCAL_SPEECH = os.getenv("DISABLE_LOCAL_SPEECH", "True").lower() in ("true", "1", "yes")

config = Config()
