# SmartVision
### AI-Powered Assistive Navigation & Safety System for the Visually Impaired

SmartVision is an AI-powered assistive system designed to enhance the independence and safety of visually impaired individuals. The application combines Computer Vision, Speech Intelligence, Navigation Services, and Emergency Assistance into a single voice-first platform that helps users navigate, recognize objects, understand their surroundings, authenticate securely, and request emergency help when needed.

---

## 🌟 Project Overview

According to the World Health Organization (WHO), over **285 million people worldwide** live with visual impairment, with nearly **39 million being completely blind**. Existing assistive technologies often address only a single problem, such as navigation or screen reading.

SmartVision provides a unified AI-driven solution capable of:

- Safe navigation using voice guidance
- Real-time obstacle detection
- Scene understanding
- Personal object recognition
- Face authentication
- Emergency SOS alerts
- Voice-based interaction without requiring visual input

The objective of SmartVision is to provide visually impaired individuals with greater independence, confidence, and accessibility in their daily lives.

---

# ✨ Features

## 👁️ Computer Vision

- Real-time Object Detection using YOLOv8
- Scene Description using BLIP
- Personal Object Recognition using CLIP
- Depth Estimation using MiDaS
- Smart obstacle filtering during navigation

---

## 🎤 Voice Assistant

- Voice Command Recognition
- Speech-to-Text using Groq Whisper
- Text-to-Speech feedback
- Voice Activity Detection (VAD)
- Intelligent command parsing

---

## 🧭 Smart Navigation

- Turn-by-turn voice navigation
- Google Maps Directions API
- Nearby Places Search
- Live GPS tracking
- Route recalculation
- Walking guidance

---

## 🔐 Face Authentication

- FaceNet based facial recognition
- Liveness Detection
- Eye Blink Detection
- Head Pose Verification
- Secure user authentication

---

## 🚨 Emergency Assistance

- Voice-triggered SOS
- Automatic GPS location sharing
- SMS alerts via Twilio
- Firestore emergency contact management
- Reverse geocoding of user location

---

## 🧠 AI Models Used

| Model | Purpose |
|--------|----------|
| YOLOv8 | Real-time object detection |
| BLIP | Scene caption generation |
| CLIP | Personal object recognition |
| FaceNet | Face authentication |
| MiDaS | Depth estimation |
| Groq Whisper | Speech recognition |
| MediaPipe | Liveness detection |

---

# 🏗 System Architecture

```
                Mobile Browser
             Camera • Mic • GPS
                     │
                     ▼
             Flask REST Server
                     │
        ┌────────────┼────────────┐
        │            │            │
        ▼            ▼            ▼
 Vision Engine   Voice Engine  Navigation
        │            │            │
        └───────┬────┴────────────┘
                ▼
         Emergency Manager
                │
      Firebase • Twilio • Google Maps
```

---

# 🔄 Workflow

1. User opens SmartVision.
2. Camera, microphone, and GPS are initialized.
3. Voice commands are captured using Whisper.
4. Commands are analyzed and routed.
5. Depending on the request:
   - Detect objects
   - Describe surroundings
   - Navigate to destination
   - Recognize personal objects
   - Authenticate face
   - Trigger emergency SOS
6. Audio feedback is provided through the speech engine.

---

# ⚙️ Technology Stack

### Programming Language

- Python

### Backend

- Flask
- Flask-CORS

### Computer Vision

- OpenCV
- Ultralytics YOLOv8
- Pillow

### Deep Learning

- PyTorch
- Transformers
- CLIP
- BLIP
- MiDaS
- FaceNet
- DeepFace

### Speech Processing

- Groq Whisper
- SpeechRecognition
- pyttsx3
- sounddevice

### Navigation

- Google Maps API
- Google Places API

### Database

- Firebase Authentication
- Firebase Firestore

### Cloud Services

- Twilio SMS API

---

# 📂 Project Structure

```
SmartVision
│
├── app
│   ├── api
│   ├── auth
│   ├── core
│   ├── database
│   ├── models
│   ├── services
│   ├── utils
│   └── vision
│
├── requirements.txt
├── run.py
├── README.md
└── .gitignore
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/ketansuryavanshi1509/SmartVision.git
```

Move into the project

```bash
cd SmartVision
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file and configure all required API keys.

Run the application

```bash
python run.py
```

---

# 🔑 Required Configuration

The project requires credentials for:

- Firebase
- Google Maps API
- Groq API
- Twilio API

These credentials are **not included** in the repository for security reasons.

---

# 🎯 Key Capabilities

- Voice-first interaction
- AI-powered object detection
- Real-time navigation assistance
- Intelligent scene understanding
- Emergency safety system
- Personal object recognition
- Secure biometric authentication

---

# 📊 Highlights

- 7+ AI Models
- 25+ REST APIs
- Modular architecture
- Voice-first interface
- Real-time navigation
- Emergency SOS support
- Firebase integration
- Google Maps integration
- Twilio SMS alerts

---

# 🔮 Future Improvements

- Android Application
- Offline AI inference
- Multi-language support
- Smart wearable integration
- OCR for reading text
- Currency recognition
- Medicine identification
- Cloud deployment
- Performance optimization

---

# 📖 Technical Documentation

A detailed technical report explaining the system architecture, workflows, AI models, algorithms, and implementation is available in the project documentation.

---

# 🤝 Contributing

Contributions are welcome.

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

---

# 📄 License

This project is developed for academic and educational purposes.

---

# 👨‍💻 Author

**Ketan Suryavanshi**

AI/ML Engineer | Backend Developer | Computer Vision Enthusiast

If you found this project helpful, consider giving it a ⭐ on GitHub.
