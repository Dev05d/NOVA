import config
import cv2
from ultralytics import YOLO
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import torch
import urllib.request
import matplotlib.pyplot as plt

app = FastAPI()
model = YOLO("/server/yolo26l.pt")
count = 0
prv_class_id = -999
prv_time = datetime(1, 1, 1, 1, 1, 1)

midas_model = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", midas_model, trust_repo=True)

# ---------------------------- #
# Preparing the model #
# ---------------------------- #
device = torch.device("cpu")
midas.to(device)
midas.eval()

# ---------------------------- #
# Transforming the data to resize/normalize the input image for the model #
# ---------------------------- #
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform
SCALE_FACTOR = 0.00034610144793987274

# This is the function to get the object depth
def get_object_depth(depth_map, box):
    x1, y1, x2, y2 = map(int, box)
    
    depth_roi = depth_map[y1:y2, x1:x2]

    if depth_roi.size == 0:
        return None

    h, w = depth_roi.shape
    center_roi = depth_roi[h//4:3*h//4, w//4:3*w//4]

    return float(np.median(center_roi)) 

def get_object_distance(result, frame):
    #result = model_result[0]
    boxes = result.boxes.xyxy.cpu().numpy()

    #Guard: no boxes detected
    if len(boxes) == 0:
        return 0.0

    # MiDaS Depth Estimation
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    # ---------------------------- #
    # The inference stage that creates the heat map #
    # ---------------------------- #
    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False
        ).squeeze()

    depth_map = prediction.cpu().numpy()

    # ---------------------------- #
    # Now that an inference is made, we can get depth value #
    # ---------------------------- #

    # for box in boxes:
    #     depth_value = get_object_depth(depth_map, box)
    #     if depth_value is None:
    #         continue
    #     metric_distance = depth_value * SCALE_FACTOR
    
    # return metric_distance

    depth_value = get_object_depth(depth_map, boxes[0])
    if depth_value is None:
        return 0.0
    
    return float(depth_value * SCALE_FACTOR)

@app.post("/send")
async def get_object(file: UploadFile = File(...)):
    global count, prv_class_id, prv_time

    # 1. Read bytes from the uploaded file
    contents = await file.read()
    
    # 2. Convert bytes to a numpy array for OpenCV
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"objects": []}

    results = model(frame, conf=0.20, verbose=False)
    objects = []
    class_distance = 0.0
    
    for r in results: 
        for box in r.boxes:
            class_id = int(box.cls[0])
            class_name = r.names[class_id]
            now = datetime.now()
            differece = now - prv_time     # Get the numerical ID (e.g., 0) 
            if ((class_id != prv_class_id) or ((differece.total_seconds() / 60) > 2)): # Look up the name (e.g., 'person')
                if ((differece.total_seconds() / 60) > 2):
                    objects = []
                if (class_name not in [obj[0] for obj in objects]):
                    class_distance = get_object_distance(r, frame)
                    objects.append((class_name, class_distance))
                count = 0
                prv_time = now
            if(count == 0):
                prv_class_id = class_id
                print(f"Detected {class_name} at distance {class_distance:.2f} meters")

    count += 1
    if (count >= 5):
        count = 1

    objects = {"objects": objects}  # Example array of detected objects
    return objects

if __name__ == "__main__":
    # Run the server
    uvicorn.run(app, host=config.SERVER_IP, port=8000)