import cv2
from ultralytics import YOLO
from datetime import datetime
from fastapi import FastAPI

# Load a pretrained YOLO26 nano model
model = YOLO("yolo26l.pt")

vid = cv2.VideoCapture(0)

count = 0
prv_class_id = -999
prv_time = datetime(1, 1, 1, 1, 1, 1)
while True: # Capture frame-by-frame
    ret, frame = vid.read()
    if not ret:
        break

    # Run YOLO inference on the frame
    results = model(frame, conf=0.20, verbose=False)

    # Display the results
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO26 Nano Detection", annotated_frame)

    for r in results: 
        for box in r.boxes:
            class_id = int(box.cls[0])
            now = datetime.now()
            differece = now - prv_time     # Get the numerical ID (e.g., 0)
            class_name = r.names[class_id] 
            if ((class_id != prv_class_id) or ((differece.total_seconds() / 60) > 2)): # Look up the name (e.g., 'person')
                count = 0
                prv_time = now
            if(count == 0):
                prv_class_id = class_id
                print(f"Detected {class_name}")

    count += 1
    if (count >= 5):
        count = 1

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


