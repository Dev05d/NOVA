#We will attempt to use a Midas small (v2.1) model to get a depth map.
import cv2
import torch
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# ---------------------------- #
# Initializing the model #
# ---------------------------- #
model_type = "MiDaS_small"
midas = torch.hub.load("intel-isl/MiDaS", model_type)

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

# ---------------------------- #
# Load the image that the model will create depth map for #
# ---------------------------- #
# The image will be from YOLO
yolo_model = YOLO("yolo26n.pt")
SCALE_FACTOR = 0.020119065418839455

# This is the function to get the object depth
def get_object_depth(depth_map, box):
    x1, y1, x2, y2 = map(int, box)
    
    depth_roi = depth_map[y1:y2, x1:x2]

    if depth_roi.size == 0:
        return None

    h, w = depth_roi.shape
    center_roi = depth_roi[h//4:3*h//4, w//4:3*w//4]

    return float(np.median(center_roi))

vid = cv2.VideoCapture(0)
if not vid.isOpened():
    print("Failed to Open Camera")
    exit()

# ---------------------------- #
# Main loop to get things started #
# ---------------------------- #
while True:
    ret, frame = vid.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # YOLO Model Detection #
    results = yolo_model(frame)
    result = results[0]

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()

    # MiDaS Depth Estimation
    img_rgb = cv2.cvtColor(frame, cv2.BGR2RGB)
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

    for box, cls, confidence in zip(boxes, classes, confidences):
        if confidence < 0.25: # CHANGE THIS WHEN NECESSARY #
            continue

        depth_value = get_object_depth(depth_map, box)
        if depth_value is None:
            continue
        metric_distance = depth_value * SCALE_FACTOR

        x1, y1, x2, y2 = map(int, box)
        class_name = yolo_model.names[int(cls)]

        label = f"{class_name} | {metric_distance:.2f} m"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("YOLO26n + MiDaS Depth (Meters)", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

vid.release()
cv2.destroyAllWindows()
    
#depth_roi = depth_map[y1:y2, x1:x2]
#depth_value = np.median(depth_roi) # This is depth value.

# Finally, we can use this depth value and known distance (1 m) to get scale factor
#scale = 3.0 / depth_value # THIS IS THE SCALE.
#print(f"Scale factor: {scale}")
# In the future:


# Scale: (0.028350001201033592 + 0.011888129636645317) / 2
# Scale = 0.020119065418839455
