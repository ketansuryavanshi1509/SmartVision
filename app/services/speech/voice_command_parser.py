"""
Voice Command Parser for SmartVision
Fully voice-controlled system with intent routing
Supports navigation, object search, emergency, and general commands
"""

import re
from typing import Dict, List, Optional, Tuple, Callable
from enum import Enum
from app.core.logger import logger


class CommandType(Enum):
    """Types of voice commands."""
    NAVIGATION = "navigation"
    FIND_NEARBY = "find_nearby"
    SPECIFIC_PLACE = "specific_place"
    OBJECT_SEARCH = "object_search"
    PERSONAL_OBJECT = "personal_object"
    EMERGENCY = "emergency"
    SCENE_DESCRIPTION = "scene_description"
    SYSTEM_CONTROL = "system_control"
    UNKNOWN = "unknown"


class VoiceCommand:
    """Represents a parsed voice command with intent."""
    
    def __init__(self, 
                 command_type: CommandType,
                 intent: str,
                 parameters: Dict[str, str],
                 confidence: float = 1.0,
                 raw_text: str = ""):
        self.command_type = command_type
        self.intent = intent
        self.parameters = parameters or {}
        self.confidence = confidence
        self.raw_text = raw_text
    
    def __repr__(self):
        return f"VoiceCommand(type={self.command_type.value}, intent={self.intent}, params={self.parameters})"


class VoiceCommandParser:
    """
    Parse transcribed speech into structured commands with intents.
    Routes commands to appropriate handlers (navigation, emergency, etc.)
    """
    
    def __init__(self):
        # Navigation-related keywords and patterns
        self.navigation_keywords = {
            'navigate', 'go', 'take me', 'direction', 'route', 'way',
            'lead', 'guide', 'show me how', 'get to'
        }
        
        self.mode_keywords = {
            'walking': ['walk', 'walking', 'on foot'],
            'driving': ['drive', 'driving', 'car'],
            'bicycling': ['bike', 'bicycle', 'cycling'],
            'transit': ['public transport', 'bus', 'train', 'subway']
        }
        
        # Object search patterns
        self.search_patterns = [
            r'(?:where is|where are|find|look for|search for)\s+(?:my\s+)?(.+)',
            r'(?:do you see|can you find|have you seen)\s+(?:my\s+)?(.+)',
            r'i\'?m looking for\s+(?:my\s+)?(.+)',
            r'(?:show me|locate)\s+(?:my\s+)?(.+)'
        ]
        
        # Personal object enrollment patterns
        self.personal_object_patterns = [
            r'(?:remember|learn|save|store|memorize)\s+(?:this|my)\s+(.+)',
            r'(?:this is my|i want to save|add)\s+(?:my\s+)?(.+)',
            r'(?:capture|record|take)\s+(?:a picture of|photo of)\s+(?:my\s+)?(.+)'
        ]
        
        # Emergency trigger phrases
        self.emergency_triggers = {
            'help': ['help', 'help me', 'i need help', 'assistance'],
            'emergency': ['emergency', 'urgent', 'danger', 'trouble'],
            'call_police': ['call police', 'call 911', 'call emergency'],
            'fall_detected': ['i fell', 'i dropped', 'fall down'],
            'medical': ['medical emergency', 'hurt', 'injured', 'pain']
        }
        
        # Scene description requests
        self.scene_patterns = [
            r'(?:describe|tell me about|what do you see|what\'?s around)',
            r'(?:where am i|what place|what location)',
            r'(?:what\'?s here|what\'?s in front|what\'?s behind)'
        ]
        
        # System control commands
        self.system_commands = {
            'stop': ['stop', 'halt', 'cease', 'quit'],
            'start': ['start', 'begin', 'resume', 'continue'],
            'pause': ['pause', 'hold on', 'wait', 'stop talking'],
            'louder': ['louder', 'speak up', 'volume up'],
            'quieter': ['quieter', 'softer', 'volume down'],
            'faster': ['faster', 'speed up', 'talk faster'],
            'slower': ['slower', 'slow down', 'talk slower']
        }
        
        # Prepositions for destination extraction
        self.prepositions = ['to', 'toward', 'towards', 'near', 'by', 'at']
        
        # Common filler words to remove
        self.filler_words = {
            'the', 'a', 'an', 'please', 'can', 'could', 'would', 'you',
            'i', 'want', 'need', 'like', 'just', 'really', 'actually'
        }
    
    def parse_command(self, transcribed_text: str) -> VoiceCommand:
        """
        Parse transcribed speech into structured command.
        
        Args:
            transcribed_text: Text from speech-to-text
            
        Returns:
            VoiceCommand with type, intent, and parameters
        """
        if not transcribed_text:
            return VoiceCommand(
                command_type=CommandType.UNKNOWN,
                intent="no_input",
                parameters={},
                raw_text=""
            )
        
        text_lower = transcribed_text.lower().strip()
        logger.debug(f"Parsing command: '{text_lower}'")
        
        # Check for emergency first (highest priority)
        emergency_result = self._check_emergency(text_lower)
        if emergency_result:
            return emergency_result
        
        # Check for nearby place discovery (before generic navigation)
        nearby_result = self._check_find_nearby(text_lower)
        if nearby_result:
            return nearby_result

        specific_result = self._check_specific_place(text_lower)
        if specific_result:
            return specific_result
        
        # Check for navigation commands
        nav_result = self._check_navigation(text_lower)
        if nav_result:
            return nav_result
        
        # Check for object search
        search_result = self._check_object_search(text_lower)
        if search_result:
            return search_result
        
        # Check for personal object enrollment
        personal_result = self._check_personal_object(text_lower)
        if personal_result:
            return personal_result
        
        # Check for scene description
        scene_result = self._check_scene_description(text_lower)
        if scene_result:
            return scene_result
        
        # Check for system control
        system_result = self._check_system_control(text_lower)
        if system_result:
            return system_result
        
        # Unknown command
        logger.warning(f"Could not parse command: '{text_lower}'")
        return VoiceCommand(
            command_type=CommandType.UNKNOWN,
            intent="unrecognized",
            parameters={'original_text': text_lower},
            raw_text=transcribed_text
        )
    
    def _check_emergency(self, text: str) -> Optional[VoiceCommand]:
        """Check if command is emergency-related."""
        for trigger_type, phrases in self.emergency_triggers.items():
            if any(phrase in text for phrase in phrases):
                logger.info(f"EMERGENCY TRIGGER DETECTED: {trigger_type}")
                
                # Extract location if mentioned
                location = self._extract_location(text)
                
                return VoiceCommand(
                    command_type=CommandType.EMERGENCY,
                    intent=trigger_type,
                    parameters={
                        'urgency': 'high',
                        'location': location or 'unknown'
                    },
                    confidence=0.95,
                    raw_text=text
                )
        
        return None
    
    def _check_find_nearby(self, text: str) -> Optional[VoiceCommand]:
        """Detect generic nearby queries e.g. 'nearest medical shop near me'."""
        patterns = [
            r'\b(nearest|nearby|closest|near\s+me)\b',
            r'\bfind\s+(a\s+|the\s+|some\s+|any\s+)?(nearest|nearby|closest)\b',
            r'\bwhere\s+(is|are)\s+the\s+(nearest|closest)\b',
        ]
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                query = re.sub(r'^(find|show|tell|get|give)\s+(me\s+)?', '', text)
                query = re.sub(r'\b(the\s+)?(nearest|nearby|closest)\s+', '', query)
                query = re.sub(r'\bnear\s+(me|my\s+(current\s+)?location)\b', '', query)
                query = re.sub(r'\baround\s+me\b', '', query).strip(' .,')
                return VoiceCommand(
                    command_type=CommandType.FIND_NEARBY,
                    intent="find_nearby",
                    parameters={"query": query, "raw_text": text},
                    confidence=0.92,
                    raw_text=text,
                )
        return None

    def _check_specific_place(self, text: str) -> Optional[VoiceCommand]:
        """Detect named destination commands e.g. 'I want to visit Mamledar Misal in Bhandup'."""
        patterns = [
            (r'\bi\s+want\s+to\s+(?:visit|go\s+to|go)\s+(.+)', 1),
            (r'\bwanna\s+(?:visit|go\s+to|go)\s+(.+)', 1),
            (r'\b(?:take\s+me\s+to|navigate\s+to|go\s+to|directions?\s+to)\s+(.+)', 1),
        ]
        for pattern, group in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                destination = m.group(group).strip(' ,.')
                if len(destination) > 2:
                    return VoiceCommand(
                        command_type=CommandType.SPECIFIC_PLACE,
                        intent="specific_place",
                        parameters={"destination": destination, "raw_text": text},
                        confidence=0.90,
                        raw_text=text,
                    )
        return None
    
    def _check_navigation(self, text: str) -> Optional[VoiceCommand]:
        """Check if command is navigation-related."""
        # Look for navigation keywords
        has_nav_keyword = any(
            keyword in text for keyword in self.navigation_keywords
        )
        
        if not has_nav_keyword:
            return None
        
        # Extract destination
        destination = self._extract_destination(text)
        
        # Extract mode (walking, driving, etc.)
        mode = self._extract_mode(text)
        
        if destination:
            return VoiceCommand(
                command_type=CommandType.NAVIGATION,
                intent="start_navigation",
                parameters={
                    'destination': destination,
                    'mode': mode or 'walking'  # Default to walking for blind users
                },
                confidence=0.9,
                raw_text=text
            )
        else:
            # Navigation command without destination
            return VoiceCommand(
                command_type=CommandType.NAVIGATION,
                intent="request_destination",
                parameters={'mode': mode or 'walking'},
                confidence=0.8,
                raw_text=text
            )
    
    def _check_object_search(self, text: str) -> Optional[VoiceCommand]:
        """Check if command is searching for an object."""
        for pattern in self.search_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                object_name = match.group(1).strip()
                
                # Remove filler words
                object_name = self._clean_object_name(object_name)
                
                if object_name:
                    return VoiceCommand(
                        command_type=CommandType.OBJECT_SEARCH,
                        intent="find_object",
                        parameters={
                            'object_name': object_name,
                            'is_personal': 'my' in text.lower()
                        },
                        confidence=0.85,
                        raw_text=text
                    )
        
        return None
    
    def _check_personal_object(self, text: str) -> Optional[VoiceCommand]:
        """Check if command is to save/remember a personal object."""
        for pattern in self.personal_object_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                object_name = match.group(1).strip()
                object_name = self._clean_object_name(object_name)
                
                if object_name:
                    return VoiceCommand(
                        command_type=CommandType.PERSONAL_OBJECT,
                        intent="enroll_object",
                        parameters={
                            'object_name': object_name,
                            'action': 'save'
                        },
                        confidence=0.85,
                        raw_text=text
                    )
        
        return None
    
    def _check_scene_description(self, text: str) -> Optional[VoiceCommand]:
        """Check if command is requesting scene description."""
        for pattern in self.scene_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # Determine what they want to know about
                focus = self._extract_scene_focus(text)
                
                return VoiceCommand(
                    command_type=CommandType.SCENE_DESCRIPTION,
                    intent="describe_scene",
                    parameters={
                        'focus': focus or 'general'
                    },
                    confidence=0.8,
                    raw_text=text
                )
        
        return None
    
    def _check_system_control(self, text: str) -> Optional[VoiceCommand]:
        """Check if command is system control."""
        for cmd_type, phrases in self.system_commands.items():
            if any(phrase in text for phrase in phrases):
                return VoiceCommand(
                    command_type=CommandType.SYSTEM_CONTROL,
                    intent=cmd_type,
                    parameters={},
                    confidence=0.9,
                    raw_text=text
                )
        
        return None
    
    def _extract_destination(self, text: str) -> Optional[str]:
        """Extract destination from navigation command."""
        # Remove common phrases
        cleaned = text
        
        # Look for text after prepositions
        for prep in self.prepositions:
            pattern = rf'{prep}\s+(.+?)(?:\s+(?:by|via|through|using)|$)'
            match = re.search(pattern, cleaned, re.IGNORECASE)
            if match:
                destination = match.group(1).strip()
                if destination and len(destination) > 2:
                    return destination
        
        # Try to extract location-like phrases
        location_patterns = [
            r'(?:to|toward)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # Capitalized words
            r'(?:the)\s+(\w+\s+(?:street|st|avenue|ave|road|rd|boulevard|blvd))',
            r'(\d+\s+\w+\s+(?:street|st|avenue|ave|road|rd))'
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_mode(self, text: str) -> Optional[str]:
        """Extract transportation mode from command."""
        for mode, keywords in self.mode_keywords.items():
            if any(keyword in text for keyword in keywords):
                return mode
        return None
    
    def _extract_location(self, text: str) -> Optional[str]:
        """Extract current location context from emergency command."""
        location_indicators = [
            'at', 'near', 'by', 'in front of', 'behind', 'next to'
        ]
        
        for indicator in location_indicators:
            pattern = rf'{indicator}\s+(.+?)(?:\s+(?:and|but|because|please)|$)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_scene_focus(self, text: str) -> Optional[str]:
        """Extract what aspect of scene user wants to know about."""
        if 'in front' in text or 'ahead' in text:
            return 'front'
        elif 'behind' in text or 'back' in text:
            return 'back'
        elif 'left' in text:
            return 'left'
        elif 'right' in text:
            return 'right'
        elif 'around' in text or 'surrounding' in text:
            return 'surroundings'
        elif 'place' in text or 'location' in text:
            return 'location'
        return None
    
    def _clean_object_name(self, name: str) -> str:
        """Remove filler words from object name."""
        words = name.split()
        cleaned_words = [
            word for word in words 
            if word.lower() not in self.filler_words
        ]
        return ' '.join(cleaned_words) if cleaned_words else name
    
    def get_help_text(self) -> str:
        """Return help text showing available voice commands."""
        return """
Available Voice Commands:

**Navigation:**
- "Navigate to [destination]"
- "Take me to [place]"
- "How do I get to [location]?"
- "Guide me to [destination]"

**Object Search:**
- "Where is my wallet?"
- "Find my keys"
- "Look for my phone"
- "Do you see my bag?"

**Personal Objects:**
- "Remember this as my wallet"
- "Save this as my keys"
- "This is my phone"

**Scene Description:**
- "Describe what you see"
- "Where am I?"
- "What's around me?"
- "What's in front of me?"

**Emergency:**
- "Help!"
- "I need help"
- "Call emergency services"
- "Medical emergency"

**System Control:**
- "Stop" / "Pause"
- "Speak louder" / "Speak softer"
- "Talk faster" / "Talk slower"
"""


class CommandRouter:
    """
    Routes parsed commands to appropriate handlers.
    Acts as central dispatcher for voice-controlled system.
    """
    
    def __init__(self):
        self.parser = VoiceCommandParser()
        self.handlers: Dict[CommandType, Callable] = {}
        self.current_state = "idle"
        self.state_stack = []
    
    def register_handler(self, 
                        command_type: CommandType, 
                        handler: Callable[[VoiceCommand], None]):
        """Register a handler function for a command type."""
        self.handlers[command_type] = handler
        logger.info(f"Registered handler for {command_type.value}")
    
    def process_command(self, transcribed_text: str) -> bool:
        """
        Process a voice command from start to finish.
        
        Args:
            transcribed_text: Raw speech-to-text output
            
        Returns:
            True if command was successfully processed
        """
        # Parse the command
        command = self.parser.parse_command(transcribed_text)
        
        logger.info(f"Processing command: {command}")
        
        # Route to appropriate handler
        handler = self.handlers.get(command.command_type)
        
        if handler:
            try:
                handler(command)
                return True
            except Exception as e:
                logger.error(f"Handler execution failed: {e}")
                return False
        else:
            logger.warning(f"No handler registered for {command.command_type.value}")
            
            # Provide feedback to user
            if command.command_type == CommandType.UNKNOWN:
                self._handle_unknown_command(command)
            
            return False
    
    def _handle_unknown_command(self, command: VoiceCommand):
        """Handle unrecognized commands with helpful suggestions."""
        raw_text = command.raw_text
        
        # Try to provide contextual help
        if 'help' in raw_text:
            suggestion = "Here are some things you can say..."
        elif any(word in raw_text for word in ['go', 'navigate', 'direction']):
            suggestion = "Try saying: 'Navigate to [destination]' or 'Take me to [place]'"
        elif any(word in raw_text for word in ['find', 'where', 'search']):
            suggestion = "Try saying: 'Find my [object]' or 'Where is my [item]?'"
        else:
            suggestion = "You can ask for navigation, search for objects, or request scene descriptions."
        
        logger.info(f"Suggestion for unknown command: {suggestion}")
        
        # Speak suggestion (if speech handler available)
        if 'speak' in self.handlers:
            self.handlers['speak'](suggestion)
    
    def set_state(self, state: str):
        """Set current conversation state."""
        self.current_state = state
        logger.debug(f"Command router state: {state}")
    
    def push_state(self, state: str):
        """Push state onto stack (for nested conversations)."""
        self.state_stack.append(self.current_state)
        self.current_state = state
        logger.debug(f"Pushed state: {state}")
    
    def pop_state(self):
        """Pop state from stack (return to previous state)."""
        if self.state_stack:
            self.current_state = self.state_stack.pop()
            logger.debug(f"Popped to state: {self.current_state}")


# Singleton instance
_command_router = None


def get_command_router() -> CommandRouter:
    """Get or create singleton CommandRouter instance."""
    global _command_router
    if _command_router is None:
        _command_router = CommandRouter()
    return _command_router
