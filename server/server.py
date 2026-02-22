import config
import distanceCalculator
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
model = YOLO("yolo26l.pt")

count = 0
prv_class_id = -999
prv_time = datetime(1, 1, 1, 1, 1, 1)

@app.post("/send")
async def get_object(file: UploadFile = File(...)):
    global count, prv_class_id, prv_time
    # 1. Read bytes from the uploaded file
    contents = await file.read()

    # 2. Convert bytes to a numpy array for OpenCV
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return {"error": "Invalid image format"}
    results = model(frame, conf=0.20, verbose=False)
    objects = []

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
                    class_distance = distanceCalculator.get_object_distance(r, frame)
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