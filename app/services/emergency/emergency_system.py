"""
Enhanced Emergency Safety System for SmartVision
Implements "Help" keyword detection, audio recording, live location tracking,
and automatic emergency contact notifications.
"""

import threading
import time
import wave
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Callable, Any
import pyaudio
from app.core.logger import logger
from app.database.firebase_client import get_db, get_optimizer
from app.services.emergency.sms_service import send_emergency_sms


class EmergencySystem:
    """
    Advanced emergency response system with voice activation,
    audio recording, location tracking, and contact notifications.
    """
    
    def __init__(self):
        self.is_active = False
        self.emergency_in_progress = False
        self.current_emergency_id: Optional[str] = None
        self.user_id: Optional[str] = None
        
        # Audio recording configuration
        self.audio_recording = False
        self.audio_thread: Optional[threading.Thread] = None
        self.audio_frames: List[bytes] = []
        self.recording_start_time: Optional[float] = None
        
        # Location tracking
        self.location_tracking = False
        self.last_known_location: Optional[Dict[str, float]] = None
        self.location_updates: List[Dict] = []
        
        # Emergency contacts
        self.emergency_contacts: List[Dict] = []
        self.contacts_notified: List[str] = []
        
        # Callbacks
        self.on_emergency_triggered: Optional[Callable] = None
        self.on_contact_notified: Optional[Callable] = None
        
        # Detection sensitivity
        self.help_keyword_sensitivity = 0.85  # High sensitivity for "help"
        self.fall_detection_enabled = True
        
        # Cooldown to prevent false alarms
        self.last_trigger_time = 0
        self.trigger_cooldown_seconds = 30.0
        
        logger.info("Emergency system initialized")
    
    def initialize_for_user(self, user_id: str):
        """
        Initialize emergency system for a specific user.
        
        Args:
            user_id: Firebase user ID
        """
        self.user_id = user_id
        self._load_emergency_contacts()
        logger.info(f"Emergency system initialized for user {user_id}")
    
    def trigger_emergency(self, 
                         emergency_type: str = "help",
                         transcribed_text: str = None,
                         auto_detected: bool = False,
                         location: Optional[Dict[str, float]] = None) -> bool:
        """
        Trigger emergency response.
        
        Args:
            emergency_type: Type of emergency (help, medical, fall_detected, danger)
            transcribed_text: Voice command that triggered this (if applicable)
            auto_detected: True if automatically detected (not voice command)
            location: Optional GPS location to use
            
        Returns:
            True if emergency was triggered successfully
        """
        # Check cooldown
        now = time.time()
        if (now - self.last_trigger_time) < self.trigger_cooldown_seconds:
            logger.warning(f"Emergency trigger on cooldown")
            return False
        
        if not self.user_id:
            logger.error("User ID not set for emergency system")
            return False
        
        # Reset tracking for new emergency
        self.contacts_notified = []
        self.location_updates = []
        
        try:
            logger.warning(f"EMERGENCY TRIGGERED: {emergency_type} (auto={auto_detected})")
            
            # Get current location (use provided or fallback)
            current_location = location or self._get_current_location()
            normalized_location = self._normalize_location(current_location)
            
            # Store emergency event in Firestore
            try:
                optimizer = get_optimizer()
                if optimizer:
                    emergency_id = optimizer.store_emergency_event_optimized(
                        user_id=self.user_id,
                        emergency_type=emergency_type,
                        location=normalized_location,
                        audio_recording_url="",  # Will be updated after recording
                        contacts_notified=[]
                    )
                    self.current_emergency_id = emergency_id
            except Exception as e:
                logger.error(f"Error storing emergency event: {e}")
            
            # Start audio recording
            self._start_audio_recording()
            
            # Start continuous location tracking
            self._start_location_tracking()
            
            # Notify emergency contacts
            self._notify_emergency_contacts(emergency_type, current_location)
            
            # Set flags
            self.emergency_in_progress = True
            self.last_trigger_time = now
            
            # Call callback if registered
            if self.on_emergency_triggered:
                self.on_emergency_triggered({
                    'type': emergency_type,
                    'location': current_location,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'auto_detected': auto_detected,
                    'transcribed_text': transcribed_text
                })
            
            return True
            
        except Exception as e:
            logger.error(f"Error triggering emergency: {e}")
            return False
    
    def check_help_keyword(self, transcribed_text: str) -> bool:
        """
        Check if transcribed text contains help keywords.
        
        Args:
            transcribed_text: Text from speech-to-text
            
        Returns:
            True if help keyword detected
        """
        if not transcribed_text:
            return False
        
        text_lower = transcribed_text.lower()
        
        # High-priority help phrases
        help_phrases = [
            'help', 'help me', 'i need help', 'somebody help',
            'emergency', 'urgent', 'danger', 'trouble',
            'call 911', 'call police', 'call ambulance',
            'i fell', 'fall down', 'can\'t get up',
            'medical emergency', 'hurt', 'injured'
        ]
        
        # Check for exact matches
        for phrase in help_phrases:
            if phrase in text_lower:
                logger.warning(f"HELP KEYWORD DETECTED: '{phrase}' in '{transcribed_text}'")
                
                # Trigger emergency
                emergency_type = self._classify_emergency(text_lower)
                threading.Thread(
                    target=self.trigger_emergency,
                    args=(emergency_type, transcribed_text, False),
                    daemon=True
                ).start()
                
                return True
        
        return False
    
    def _classify_emergency(self, text: str) -> str:
        """
        Classify emergency type from text.
        
        Args:
            text: Lowercase transcribed text
            
        Returns:
            Emergency type string
        """
        if any(word in text for word in ['fell', 'fall', 'drop']):
            return 'fall_detected'
        elif any(word in text for word in ['medical', 'hurt', 'injured', 'pain', 'heart']):
            return 'medical'
        elif any(word in text for word in ['police', 'danger', 'threat', 'attack']):
            return 'security'
        else:
            return 'help'  # Default
    
    def _start_audio_recording(self):
        """Start recording audio during emergency."""
        if self.audio_recording:
            return
        
        try:
            self.audio_recording = True
            self.audio_frames = []
            self.recording_start_time = time.time()
            
            self.audio_thread = threading.Thread(
                target=self._record_audio_worker,
                daemon=True
            )
            self.audio_thread.start()
            
            logger.info("Audio recording started")
            
        except Exception as e:
            logger.error(f"Failed to start audio recording: {e}")
    
    def _record_audio_worker(self):
        """Worker thread to record audio."""
        try:
            # Audio configuration
            CHUNK = 1024
            FORMAT = pyaudio.paInt16
            CHANNELS = 1
            RATE = 16000
            
            p = pyaudio.PyAudio()
            stream = p.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )
            
            logger.info("Recording audio...")
            
            # Record for up to 60 seconds or until stopped
            start_time = time.time()
            max_duration = 60.0  # seconds
            
            while self.audio_recording and (time.time() - start_time) < max_duration:
                data = stream.read(CHUNK, exception_on_overflow=False)
                self.audio_frames.append(data)
            
            # Stop recording
            stream.stop_stream()
            stream.close()
            p.terminate()
            
            logger.info(f"Audio recording completed ({len(self.audio_frames)} chunks)")
            
            # Save and upload recording
            if self.audio_frames:
                self._save_audio_recording()
            
        except Exception as e:
            logger.error(f"Audio recording error: {e}")
    
    def _save_audio_recording(self):
        """Save audio recording to file and upload to Firebase Storage."""
        try:
            # Create recordings directory
            os.makedirs('emergency_recordings', exist_ok=True)
            
            # Generate filename
            timestamp = int(time.time())
            filename = f'emergency_{self.user_id}_{timestamp}.wav'
            filepath = os.path.join('emergency_recordings', filename)
            
            # Save WAV file
            wf = wave.open(filepath, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(b''.join(self.audio_frames))
            wf.close()
            
            logger.info(f"Audio saved to {filepath}")
            
            # TODO: Upload to Firebase Storage
            # For now, just log the local path
            # In production, upload to Firebase Storage and get URL
            
        except Exception as e:
            logger.error(f"Error saving audio: {e}")
    
    def _stop_audio_recording(self):
        """Stop audio recording."""
        self.audio_recording = False
        if self.audio_thread:
            self.audio_thread.join(timeout=2.0)
        logger.info("Audio recording stopped")
    
    def _start_location_tracking(self):
        """Start continuous location tracking during emergency."""
        if self.location_tracking:
            return
        
        try:
            self.location_tracking = True
            self.location_updates = []
            
            # Start location tracking thread
            location_thread = threading.Thread(
                target=self._track_location_worker,
                daemon=True
            )
            location_thread.start()
            
            logger.info("Location tracking started")
            
        except Exception as e:
            logger.error(f"Failed to start location tracking: {e}")
    
    def _track_location_worker(self):
        """Worker thread to track location continuously."""
        try:
            update_interval = 5.0  # Update every 5 seconds
            
            while self.location_tracking:
                location = self._get_current_location()
                
                if location:
                    self.last_known_location = location
                    self.location_updates.append({
                        'lat': location.get('lat'),
                        'lng': location.get('lng'),
                        'timestamp': time.time(),
                        'accuracy': location.get('accuracy', 'unknown')
                    })
                    
                    # Keep only last 20 locations
                    if len(self.location_updates) > 20:
                        self.location_updates = self.location_updates[-20:]
                
                time.sleep(update_interval)
            
        except Exception as e:
            logger.error(f"Location tracking error: {e}")
    
    def _stop_location_tracking(self):
        """Stop location tracking."""
        self.location_tracking = False
        logger.info("Location tracking stopped")
    
    def _notify_emergency_contacts(self, 
                                   emergency_type: str,
                                   location: Optional[Dict[str, float]]):
        """
        Notify all emergency contacts.
        
        Args:
            emergency_type: Type of emergency
            location: Current GPS location
        """
        # We proceed even if self.emergency_contacts is empty to notify the hardcoded number
        if not self.emergency_contacts:
            logger.info("No dynamic emergency contacts configured in database, but proceeding with hardcoded number.")
        
        try:
            normalized_location = self._normalize_location(location)
            location_str = self._format_location_link(normalized_location) if location else "Location unavailable"
            
            # Get verified number from environment for Twilio testing/development
            verified_number = os.getenv('VERIFIED_EMERGENCY_NUMBER')
            
            # Try to get user name for better message
            user_name = "User"
            try:
                db = get_db()
                if db:
                    user_doc = db.collection("users").document(self.user_id).get()
                    if user_doc.exists:
                        user_name = user_doc.to_dict().get('full_name', user_doc.to_dict().get('name', 'User'))
            except Exception:
                pass

            message = (
                f"SMARTVISION EMERGENCY ALERT\n\n"
                f"User: {user_name} ({self.user_id})\n"
                f"Status: Triggered an emergency alert\n"
                f"Type: {emergency_type.upper()}\n"
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Location: {location_str}\n\n"
                f"Please check on them immediately."
            )
            
            # 1. ALWAYS notify the hardcoded verified number
            if verified_number:
                logger.info(f"Sending emergency SMS to hardcoded verified number: {verified_number}")
                if send_emergency_sms(verified_number, message):
                    self.contacts_notified.append(verified_number)
            else:
                logger.warning("VERIFIED_EMERGENCY_NUMBER not set in environment. Skipping hardcoded notification.")

            # 2. Notify other emergency contacts from database
            if self.emergency_contacts:
                for contact in self.emergency_contacts:
                    contact_phone = contact.get('phone')
                    if not contact_phone or contact_phone == verified_number:
                        continue
                        
                    contact_name = contact.get('name', 'Emergency Contact')
                    
                    logger.warning(f"Notifying dynamic contact {contact_name}: {contact_phone}")
                    if send_emergency_sms(contact_phone, message):
                        self.contacts_notified.append(contact_phone)
                        
                        if self.on_contact_notified:
                            try:
                                self.on_contact_notified(contact)
                            except Exception:
                                pass
            
            logger.info(f"Total contacts successfully notified: {len(self.contacts_notified)}")
            
        except Exception as e:
            logger.error(f"Error notifying contacts: {e}")
    
    def resolve_emergency(self, status: str = "resolved"):
        """
        Mark current emergency as resolved.
        
        Args:
            status: Resolution status (resolved, false_alarm)
        """
        if not self.current_emergency_id:
            logger.warning("No active emergency to resolve")
            return
        
        try:
            # Stop recordings and tracking
            self._stop_audio_recording()
            self._stop_location_tracking()
            
            # Update Firestore
            db = get_db()
            if db:
                emergency_ref = db.collection("users").document(self.user_id)\
                               .collection("emergencies").document(self.current_emergency_id)
                
                emergency_ref.update({
                    'status': status,
                    'resolved_at': datetime.now(timezone.utc),
                    'contacts_notified': self.contacts_notified,
                    'location_history': self.location_updates
                })
            
            logger.info(f"Emergency {self.current_emergency_id} marked as {status}")
            
            # Reset state
            self.emergency_in_progress = False
            self.current_emergency_id = None
            self.contacts_notified = []
            
        except Exception as e:
            logger.error(f"Error resolving emergency: {e}")
    
    def _load_emergency_contacts(self):
        """Load emergency contacts from Firestore."""
        if not self.user_id:
            return
        
        try:
            db = get_db()
            contacts_ref = db.collection("users").document(self.user_id)\
                           .collection("emergency_contacts")
            
            docs = contacts_ref.order_by("priority").stream()
            
            self.emergency_contacts = []
            for doc in docs:
                contact = doc.to_dict()
                contact['id'] = doc.id
                self.emergency_contacts.append(contact)
            
            logger.info(f"Loaded {len(self.emergency_contacts)} emergency contacts")
            
        except Exception as e:
            logger.error(f"Error loading emergency contacts: {e}")
    
    def _get_current_location(self) -> Optional[Dict[str, float]]:
        """
        Get current GPS location.
        Implementation depends on platform (web/mobile).
        
        Returns:
            Location dict with lat, lng, accuracy
        """
        # TODO: Implement based on platform
        # For web: Use browser's geolocation API
        # For mobile: Use native GPS
        
        # Placeholder - in production, get from location service
        return self.last_known_location
    
    def _format_location_link(self, location: Dict[str, float]) -> str:
        if not location:
            return "Location unavailable"
        
        # Location is expected to be normalized to dict here
        lat = location.get('lat', 0)
        lng = location.get('lng', 0)
        
        if lat == 0 and lng == 0:
            return "Location unavailable"
            
        return f"https://www.google.com/maps?q={lat},{lng}"

    def _normalize_location(self, location: Any) -> Dict[str, float]:
        """Normalize location from tuple or dict to standard dict format."""
        if not location:
            return {"lat": 0.0, "lng": 0.0}
            
        if isinstance(location, dict):
            return {
                "lat": float(location.get('lat', 0.0)),
                "lng": float(location.get('lng', 0.0))
            }
            
        if isinstance(location, (tuple, list)):
            try:
                # Handle (lat, lng) or (lat, lng, city)
                return {
                    "lat": float(location[0]),
                    "lng": float(location[1])
                }
            except (IndexError, ValueError, TypeError):
                pass
                
        return {"lat": 0.0, "lng": 0.0}
    
    def add_emergency_contact(self, name: str, phone: str, 
                             email: str = None, priority: int = 1) -> bool:
        """
        Add emergency contact to Firestore.
        
        Args:
            name: Contact name
            phone: Phone number for SMS
            email: Email address (optional)
            priority: Notification priority (1=highest)
            
        Returns:
            True if successful
        """
        try:
            db = get_db()
            contacts_ref = db.collection("users").document(self.user_id)\
                           .collection("emergency_contacts")
            
            contact_data = {
                'name': name,
                'phone': phone,
                'email': email or '',
                'priority': priority,
                'created_at': datetime.now(timezone.utc)
            }
            
            contacts_ref.add(contact_data)
            self._load_emergency_contacts()  # Reload
            
            logger.info(f"Added emergency contact: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding contact: {e}")
            return False
    
    def remove_emergency_contact(self, contact_id: str) -> bool:
        """Remove emergency contact."""
        try:
            db = get_db()
            contacts_ref = db.collection("users").document(self.user_id)\
                           .collection("emergency_contacts")
            
            contacts_ref.document(contact_id).delete()
            self._load_emergency_contacts()
            
            logger.info(f"Removed emergency contact: {contact_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing contact: {e}")
            return False
    
    def test_emergency_system(self) -> Dict:
        """
        Test emergency system components.
        
        Returns:
            Test results dict
        """
        results = {
            'contacts_loaded': len(self.emergency_contacts) > 0,
            'contacts_count': len(self.emergency_contacts),
            'audio_ready': True,  # PyAudio available
            'location_available': self._get_current_location() is not None,
            'firestore_connected': get_db() is not None
        }
        
        logger.info(f"Emergency system test: {results}")
        return results


# Singleton instance
_emergency_system = None


def normalize_phone_number(phone: str, default_country: str = 'US') -> str:
    """
    Normalize a phone number to international format.
    
    Args:
        phone: Phone number string
        default_country: Default country code (e.g., 'US', 'IN')
        
    Returns:
        Normalized phone number with country code
    """
    # Remove all non-digit characters except +
    cleaned = ''.join(c for c in phone if c.isdigit() or c == '+')
    
    # If already has country code, return as is
    if cleaned.startswith('+'):
        return cleaned
    elif cleaned.startswith('1') and len(cleaned) >= 11:
        # US/Canada numbers
        return '+' + cleaned
    elif cleaned.startswith('91') and len(cleaned) >= 12:
        # India numbers
        return '+' + cleaned
    else:
        # Add default country code
        country_codes = {
            'US': '+1',
            'IN': '+91',
            'UK': '+44'
        }
        prefix = country_codes.get(default_country, '+1')
        return prefix + cleaned.lstrip('0')


class EmergencyAlertManager:
    """Manager for emergency alerts and notifications."""
    
    def __init__(self, cooldown_seconds: int = 30):
        self.emergency_system = get_emergency_system()
        self.cooldown_seconds = cooldown_seconds
        self.last_trigger_time = 0
    
    def trigger_emergency(self, user_id: str, trigger_type: str, location_provider: Callable = None, 
                         voice_engine = None, access_token: str = "") -> tuple[bool, str]:
        """
        Trigger an emergency alert.
        
        Args:
            user_id: Firebase user ID
            trigger_type: Type of trigger (voice, manual, auto)
            location_provider: Function to get current location
            voice_engine: Voice engine instance for announcements
            access_token: User's access token
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        # Check cooldown
        import time
        now = time.time()
        if (now - self.last_trigger_time) < self.cooldown_seconds:
            logger.warning(f"Emergency trigger on cooldown ({int(self.cooldown_seconds - (now - self.last_trigger_time))}s remaining)")
            return False, f"Emergency trigger on cooldown. Try again in {int(self.cooldown_seconds - (now - self.last_trigger_time))} seconds."
        
        try:
            # Initialize emergency system for user
            self.emergency_system.initialize_for_user(user_id)
            
            # Get current location if provider available
            current_location = None
            if location_provider:
                try:
                    current_location = location_provider()
                except Exception as e:
                    logger.error(f"Error getting location for emergency: {e}")

            # Trigger the emergency system with location
            success = self.emergency_system.trigger_emergency(
                emergency_type=trigger_type,
                transcribed_text=f"Emergency triggered via {trigger_type}",
                auto_detected=(trigger_type != "manual"),
                location=current_location
            )
            
            if success:
                self.last_trigger_time = now
                
                # Announce emergency
                if voice_engine and hasattr(voice_engine, 'speak'):
                    voice_engine.speak("Emergency alert sent. Help is on the way.")
                
                return True, "Emergency alert sent successfully. Help is being notified."
            else:
                return False, "Failed to trigger emergency. Please try again."
                
        except Exception as e:
            logger.error(f"Error triggering emergency: {e}")
            import traceback
            traceback.print_exc()
            return False, f"Emergency handler failed: {e}"
    
    def send_alert(self, user_id: str, alert_type: str, message: str):
        """Send an emergency alert."""
        logger.info(f"Emergency alert sent: {alert_type} - {message}")
        return True


def contains_emergency_keyword(text: str) -> bool:
    """Check if text contains emergency keywords."""
    emergency_keywords = ['help', 'emergency', 'danger', 'accident', 'hurt', 'pain', 'fall']
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in emergency_keywords)


def get_emergency_system() -> EmergencySystem:
    """Get or create singleton EmergencySystem instance."""
    global _emergency_system
    if _emergency_system is None:
        _emergency_system = EmergencySystem()
    return _emergency_system
