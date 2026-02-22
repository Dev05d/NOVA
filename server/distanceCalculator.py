import torch
import cv2
import numpy as np

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
SCALE_FACTOR = -0.001896733388315722
bias = 2.6160199698946247

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
    for box in boxes:
        depth_value = get_object_depth(depth_map, box)
        if depth_value is None:
            continue
        metric_distance = (SCALE_FACTOR * depth_value) + bias
    return metric_distance