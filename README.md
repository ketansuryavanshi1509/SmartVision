# SmartVision

An AI-powered assistive system for visually impaired users that provides real-time object detection, voice assistance, navigation, and emergency support.

## Features

- 🎯 Real-time Object Detection (YOLOv8)
- 🎤 Voice Commands & Speech Recognition
- 🗣️ Text-to-Speech Assistance
- 🚨 Emergency Alert System
- 🔥 Firebase Authentication
- 🧭 Navigation Assistance

## Tech Stack

- Python
- FastAPI
- OpenCV
- YOLOv8
- Firebase
- SpeechRecognition
- pyttsx3
- NumPy

## Project Structure

```
SmartVision/
│── app/
│── run.py
│── requirements.txt
│── README.md
│── .gitignore
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/ketansuryavanshi1509/SmartVision.git
cd SmartVision
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Create a `.env` file and add the required API keys and configuration values.

### 4. Firebase Setup

Download your Firebase service account JSON file and place it in the project root. This file is not included in the repository for security reasons.

### 5. Run the project

```bash
python run.py
```

## Future Improvements

- Mobile application support
- Multi-language voice assistant
- Cloud deployment
- Improved indoor navigation

## License

This project is for educational and research purposes.
