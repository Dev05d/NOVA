import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors 
from flask import Flask, render_template_string
from flask_socketio import SocketIO
from concurrent.futures import ThreadPoolExecutor
import torch
import math
import time

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# --- 1. Load Models ---
print("Loading YOLO models...")
# Automatically use Apple Silicon (MPS), Nvidia GPU (CUDA), or fallback to CPU
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

model1 = YOLO("yolo26m.pt")
model2 = YOLO("yolo26n_run.pt")

print(f"Loading MiDaS depth model on {device}...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform
print("All models loaded successfully.")


# --- 2. Configuration & Calibration ---
# True Distance (Z) = DEPTH_A * (1 / Raw_MiDaS_Value) + DEPTH_B
DEPTH_A = 400.0  # Placeholder: Replace with your calculated A
DEPTH_B = 0.1    # Placeholder: Replace with your calculated B

FOV_DEGREES = 60.0
HUMAN_HALF_WIDTH_METERS = 0.35 # 70cm total width clearance (a bit wider than average shoulders for safety)
ALERT_COOLDOWN = 4.0 # Seconds between audio alerts
WARNING_DISTANCE = 4.0 # Start warning when objects are closer than 4 meters

last_alert_time = 0


# --- 3. Embedded HTML & JS Frontend ---
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>NOVA</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.1/socket.io.js"></script>
    <style>
        body { margin: 0; padding: 0; background-color: #000; overflow: hidden; font-family: sans-serif; }
        .media-container { position: relative; width: 100vw; height: 100vh; }
        #video { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; z-index: 1; }
        #overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; object-fit: cover; z-index: 2; pointer-events: none; }
        #capture-canvas { display: none; }
        .ui-overlay { position: absolute; top: 30px; left: 0; width: 100%; text-align: center; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.8); z-index: 10; pointer-events: none; }
        .ui-overlay h2 { margin: 0; font-size: 28px; }
        
        /* Toggle Buttons */
        .btn-container { margin-top: 15px; display: flex; justify-content: center; gap: 10px; flex-wrap: wrap; }
        .ui-btn { 
            pointer-events: auto; padding: 12px 20px; color: white; border: 2px solid white; 
            border-radius: 25px; font-size: 16px; font-weight: bold; cursor: pointer; transition: 0.3s;
        }
        .lens-btn { background-color: rgba(50, 50, 50, 0.7); }
        .audio-off { background-color: rgba(220, 53, 69, 0.9); } /* Red */
        .audio-on { background-color: rgba(40, 167, 69, 0.9); }  /* Green */
        .box-on { background-color: rgba(0, 123, 255, 0.9); }    /* Blue */
        .box-off { background-color: rgba(108, 117, 125, 0.9); } /* Gray */
    </style>
</head>
<body>
    <div class="ui-overlay">
        <h2>NOVA</h2>
        <div class="btn-container">
            <button id="switch-btn" class="ui-btn lens-btn">Switch Lens</button>
            <button id="box-toggle-btn" class="ui-btn box-on">Boxes: ON</button>
            <button id="audio-toggle-btn" class="ui-btn audio-off">Audio: OFF</button>
        </div>
    </div>
    
    <div class="media-container">
        <video id="video" autoplay playsinline></video>
        <canvas id="overlay"></canvas>
    </div>
    <canvas id="capture-canvas"></canvas>

    <script>
        const video = document.getElementById('video');
        const overlay = document.getElementById('overlay');
        const overlayCtx = overlay.getContext('2d');
        const captureCanvas = document.getElementById('capture-canvas');
        const captureCtx = captureCanvas.getContext('2d', { willReadFrequently: true });
        
        const switchBtn = document.getElementById('switch-btn');
        const audioToggleBtn = document.getElementById('audio-toggle-btn');
        const boxToggleBtn = document.getElementById('box-toggle-btn');
        const socket = io();

        // --- Camera Setup ---
        let isProcessing = false; 
        let currentStream = null;
        let videoDevices = [];
        let currentDeviceIndex = 0;

        async function startCamera(deviceId = null) {
            if (currentStream) currentStream.getTracks().forEach(track => track.stop());
            let constraints = { video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } } };
            if (deviceId) { constraints = { video: { deviceId: { exact: deviceId } } }; }
            try {
                const stream = await navigator.mediaDevices.getUserMedia(constraints);
                video.srcObject = stream;
                currentStream = stream;
                if (videoDevices.length === 0) {
                    const devices = await navigator.mediaDevices.enumerateDevices();
                    videoDevices = devices.filter(device => device.kind === 'videoinput');
                }
            } catch (err) { console.error("Camera access denied:", err); }
        }

        switchBtn.addEventListener('click', () => {
            if (videoDevices.length > 0) {
                currentDeviceIndex = (currentDeviceIndex + 1) % videoDevices.length;
                startCamera(videoDevices[currentDeviceIndex].deviceId);
            }
        });

        // --- Box Visibility Toggle Logic ---
        let showBoxes = true;
        boxToggleBtn.addEventListener('click', () => {
            showBoxes = !showBoxes;
            if (showBoxes) {
                boxToggleBtn.innerText = "Boxes: ON";
                boxToggleBtn.className = "ui-btn box-on";
            } else {
                boxToggleBtn.innerText = "Boxes: OFF";
                boxToggleBtn.className = "ui-btn box-off";
                overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
            }
        });

        // --- Audio Toggle Logic ---
        let isAudioEnabled = false;
        audioToggleBtn.addEventListener('click', () => {
            isAudioEnabled = !isAudioEnabled;
            if (isAudioEnabled) {
                audioToggleBtn.innerText = "Audio: ON";
                audioToggleBtn.className = "ui-btn audio-on";
                const utterance = new SpeechSynthesisUtterance("Audio navigation active.");
                window.speechSynthesis.speak(utterance);
            } else {
                audioToggleBtn.innerText = "Audio: OFF";
                audioToggleBtn.className = "ui-btn audio-off";
                window.speechSynthesis.cancel(); 
            }
        });

        // --- Socket Listeners ---
        socket.on('detections', (data) => {
            isProcessing = false; 
            
            overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
            
            if (!showBoxes) return;
            
            data.forEach(box => {
                overlayCtx.strokeStyle = box.color;
                overlayCtx.lineWidth = 3;
                overlayCtx.strokeRect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
                
                overlayCtx.fillStyle = box.color;
                overlayCtx.font = "bold 16px sans-serif";
                const textWidth = overlayCtx.measureText(box.label).width;
                overlayCtx.fillRect(box.x1, box.y1 - 22, textWidth + 10, 22);
                
                overlayCtx.fillStyle = '#ffffff'; 
                overlayCtx.fillText(box.label, box.x1 + 5, box.y1 - 6);
            });
        });

        socket.on('audio_alert', (data) => {
            if (isAudioEnabled) {
                const utterance = new SpeechSynthesisUtterance(data.text);
                utterance.rate = 1.15; 
                window.speechSynthesis.speak(utterance);
            }
        });

        // --- Video Stream Loop ---
        const getCanvasBlob = (quality) => {
            return new Promise((resolve) => { captureCanvas.toBlob((blob) => { resolve(blob); }, 'image/jpeg', quality); });
        };

        setInterval(async () => {
            if (!isProcessing && video.videoWidth > 0 && socket.connected) {
                isProcessing = true; 
                if (captureCanvas.width !== video.videoWidth) {
                    captureCanvas.width = video.videoWidth;
                    captureCanvas.height = video.videoHeight;
                    overlay.width = video.videoWidth;
                    overlay.height = video.videoHeight;
                }
                captureCtx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
                const imageBlob = await getCanvasBlob(0.3); 
                const arrayBuffer = await imageBlob.arrayBuffer();
                socket.emit('process_frame', arrayBuffer);
            }
        }, 100); 

        startCamera();
    </script>
</body>
</html>
"""

# --- 4. Helper Functions ---
def get_object_depth(depth_map, box):
    x1, y1, x2, y2 = map(int, box)
    depth_roi = depth_map[y1:y2, x1:x2]
    if depth_roi.size == 0:
        return None
    h, w = depth_roi.shape
    center_roi = depth_roi[h//4:3*h//4, w//4:3*w//4]
    return float(np.median(center_roi)) 

def check_collision_course(x1, x2, frame_width, depth_z):
    """
    Checks if an object horizontally intersects with the user's path.
    Returns True if it's in the way, False otherwise.
    """
    # 1. Map pixel width to real-world meters at distance Z
    fov_rad = math.radians(FOV_DEGREES)
    frame_width_meters = 2 * depth_z * math.tan(fov_rad / 2)
    meters_per_px = frame_width_meters / frame_width
    
    # 2. Shift origin to center screen (User is at X=0)
    center_x = frame_width / 2
    obj_left_m = (x1 - center_x) * meters_per_px
    obj_right_m = (x2 - center_x) * meters_per_px
    
    # 3. Collision logic: Does object intersect with user's body path?
    if obj_left_m < HUMAN_HALF_WIDTH_METERS and obj_right_m > -HUMAN_HALF_WIDTH_METERS:
        return True
            
    return False 


# --- 5. Flask & SocketIO Routes ---
@app.route('/')
def index():
    return render_template_string(html_content)

@socketio.on('process_frame')
def handle_frame(image_bytes):
    global last_alert_time
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return

        frame_height, frame_width = frame.shape[:2]

        # 1. Run inference concurrently for speed
        with ThreadPoolExecutor(max_workers=2) as executor:
            future1 = executor.submit(model1, frame, conf=0.60, verbose=False, device=device)
            future2 = executor.submit(model2, frame, conf=0.10, verbose=False, device=device)
            results1 = future1.result()
            results2 = future2.result()

        # 2. Run MiDaS Depth
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_batch = midas_transforms(img_rgb).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=(frame_height, frame_width), mode="bicubic", align_corners=False
            ).squeeze()
        depth_map = prediction.cpu().numpy()

        detections = []
        closest_threat = None
        min_threat_depth = float('inf')

        # Helper to process boxes uniformly
        def process_box(box, label_name, hex_color):
            nonlocal min_threat_depth, closest_threat
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            raw_depth_val = get_object_depth(depth_map, (x1, y1, x2, y2))
            
            if raw_depth_val:
                z_distance = DEPTH_A * (1.0 / raw_depth_val) + DEPTH_B
                
                # Check if it's in our path
                in_path = check_collision_course(x1, x2, frame_width, z_distance)
                
                if in_path and z_distance < min_threat_depth:
                    min_threat_depth = z_distance
                    closest_threat = {'name': label_name, 'z': z_distance}

                detections.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'label': f"{label_name} {z_distance:.1f}m",
                    'color': hex_color
                })

        # Process Model 1 (General)
        for box in results1[0].boxes:
            cls_id = int(box.cls[0])
            label_name = model1.names[cls_id]
            b, g, r = colors(cls_id, True)
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            process_box(box, label_name, hex_color)

        # Process Model 2 (Custom Obstacles)
        for box in results2[0].boxes:
            cls_id = int(box.cls[0])
            class_name = model2.names[cls_id].lower()
            if class_name in ['obstacle', 'pole', 'stairs']:
                hex_color = '#FF0000' if class_name == 'obstacle' else '#FF1493'
                process_box(box, class_name, hex_color)

        # 3. Audio Alert Logic
        current_time = time.time()
        if closest_threat and min_threat_depth < WARNING_DISTANCE: 
            if current_time - last_alert_time > ALERT_COOLDOWN:
                # Simplified Audio Alert
                alert_text = f"{closest_threat['name']} is {closest_threat['z']:.1f} meters ahead."
                socketio.emit('audio_alert', {'text': alert_text})
                last_alert_time = current_time

        socketio.emit('detections', detections)

    except Exception as e:
        print(f"Server processing error: {e}")

if __name__ == '__main__':
    print("Starting Nav Assist Server on port 7800...")
    socketio.run(app, host='0.0.0.0', port=7800, debug=False, ssl_context='adhoc')