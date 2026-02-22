# NOVA - Navigation and Obstacle Visual Assistant

NOVA is an Android application designed to help visually impaired users navigate their surroundings safely. It uses real-time object detection and depth estimation to identify obstacles and announce them through audio feedback.

## How It Works

```
Android Camera → JPEG Frame → Python Server (YOLO + MiDaS) → Detected Objects + Distances → Text-to-Speech → Audio Output
```

1. The app captures frames from the phone's camera and sends them to a Python server
2. The server runs **YOLOv8** object detection to identify objects in the frame
3. **MiDaS** depth estimation calculates the distance to each detected object
4. Results are sent back to the app and converted to speech (e.g. *"person at 0.2 meters detected"*) using the **ElevenLabs** TTS API

## Tech Stack

| Component | Technologies |
|-----------|-------------|
| **Android App** | Kotlin, Jetpack Compose, CameraX, Retrofit, OkHttp, Coroutines |
| **Backend Server** | Python, FastAPI, Uvicorn |
| **AI Models** | YOLOv8 (ultralytics), Intel MiDaS (depth estimation) |
| **Audio** | ElevenLabs TTS API, Android MediaPlayer |

## Project Structure

```
NOVA/
├── app/                        # Android application
│   └── app/src/main/java/com/example/impairedapp/
│       └── MainActivity.kt     # Main app logic (camera, networking, TTS)
└── server/                     # Python backend
    ├── server.py               # FastAPI server with YOLO detection
    ├── distanceCalculator.py   # MiDaS depth-to-distance conversion
    ├── yolo.py                 # Standalone YOLO test script
    └── *.pt                    # YOLO model weights (nano, medium, large)
```

## Setup

### Prerequisites

- Android Studio
- Python 3.8+
- Android device with camera (API 24+)

### Server

```bash
cd server
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install fastapi uvicorn ultralytics torch torchvision opencv-python numpy matplotlib

# Create config.py with your server IP
echo 'SERVER_IP = "0.0.0.0"' > config.py

python server.py
# Server runs on http://0.0.0.0:8000
```

### Android App

1. Open the `app/` directory in Android Studio
2. Update the server URL in `MainActivity.kt` to point to your server's IP address
3. Add your ElevenLabs API key
4. Build and run on a physical device:
   ```bash
   ./gradlew installDebug
   ```

> **Note:** A physical Android device is recommended since the app relies on the camera for real-time object detection.

## Permissions

The app requires:
- **Camera** - for real-time environment scanning
- **Internet** - for server communication and TTS API calls

## API Endpoint

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/send` | Accepts an image upload, returns detected objects with distances |

**Response format:**
```json
{
  "objects": [["person", 0.2], ["car", 3.6]]
}
```
