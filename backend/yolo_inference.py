from ultralytics import YOLO
import cv2
import torch
import numpy as np

# Use the currently deployed model (by default, this points to yolov5s.pt; it will be updated when a custom model is deployed)
YOLO_MODEL_PATH = "yolov5s.pt"  # initial model; will be replaced later
yolo_model = YOLO(YOLO_MODEL_PATH)

def run_yolo_inference(image_path):
    # Load image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image from " + image_path)
    
    # Convert from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize image and convert to float32
    image = image.astype(np.float32) / 255.0
    
    # Convert image to torch tensor with shape [1, C, H, W]
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    # Run inference
    results = yolo_model.model(image_tensor)
    
    # If the result is a tuple, extract the first element.
    if isinstance(results, tuple):
        results = results[0]
    
    # If the result is a tensor, use it directly; otherwise, get results.pred if available.
    if isinstance(results, torch.Tensor):
        pred_tensor = results
    elif hasattr(results, "pred"):
        pred_tensor = results.pred
    else:
        raise ValueError("Inference result does not have 'pred' attribute or is not a tensor: " + str(results))
    
    predictions = []
    # Process detections for the first (and only) image in the batch.
    for det in pred_tensor[0]:
        det_list = det.tolist()
        # Take the first 6 elements in case there are more
        if len(det_list) < 6:
            continue  # skip if detection doesn't have at least 6 elements
        x1, y1, x2, y2, conf, cls = det_list[:6]
        # Retrieve class label if available
        label = yolo_model.model.names.get(int(cls), str(int(cls))) if hasattr(yolo_model.model, "names") else str(int(cls))
        predictions.append({
            "box": [x1, y1, x2, y2],
            "confidence": conf,
            "label": label
        })
    return predictions
