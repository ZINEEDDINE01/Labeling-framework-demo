import os
import shutil
import yaml
import random
import json
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from keypoints_model import runtest

from yolo_inference import run_yolo_inference, yolo_model  # Ensure yolo_model is imported
from ultralytics import YOLO

# Fix Pydantic protected namespace warnings
BaseModel.model_config = {"protected_namespaces": ()}

app = FastAPI()

origins = [
    "http://localhost:4200",
    "https://your-angular-app-url.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Use BASE_DIR (directory where main.py resides)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print("BASE_DIR:", BASE_DIR)

# Define folders using absolute paths
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
print("UPLOAD_DIR:", UPLOAD_DIR)

ANNOTATION_DIR = os.path.join(BASE_DIR, "dataset")
os.makedirs(ANNOTATION_DIR, exist_ok=True)
print("ANNOTATION_DIR:", ANNOTATION_DIR)

CUSTOM_MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(CUSTOM_MODELS_DIR, exist_ok=True)
print("CUSTOM_MODELS_DIR:", CUSTOM_MODELS_DIR)

# Annotated folder for manual JSONs
ANNOTATED_DIR = os.path.join(ANNOTATION_DIR, "annotated_batch")
os.makedirs(ANNOTATED_DIR, exist_ok=True)
print("ANNOTATED_DIR:", ANNOTATED_DIR)

# Create YOLO subfolders (images and labels for train and val)
IMAGES_DIR = os.path.join(ANNOTATED_DIR, "images")
LABELS_DIR = os.path.join(ANNOTATED_DIR, "labels")
for sub in ["train", "val"]:
    os.makedirs(os.path.join(IMAGES_DIR, sub), exist_ok=True)
    os.makedirs(os.path.join(LABELS_DIR, sub), exist_ok=True)
    print(f"Created folder for {sub}: {os.path.join(IMAGES_DIR, sub)} and {os.path.join(LABELS_DIR, sub)}")

def clear_folder(folder):
    print("Clearing folder:", folder)
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print("Removed file:", file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    print("Removed folder:", file_path)
            except Exception as e:
                print(f"Error clearing {file_path}: {e}")

# 1. Upload Endpoint
@app.post("/upload/")
async def upload_images(files: List[UploadFile] = File(...)):
    print("upload_images: Clearing UPLOAD_DIR")
    clear_folder(UPLOAD_DIR)
    file_paths = []
    for file in files:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        print("Saving file:", file_path)
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            file_paths.append({"filename": file.filename, "filepath": file_path})
        except Exception as e:
            print("Error saving file:", file.filename, e)
            raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")
    print("Upload complete, files:", file_paths)
    return {"uploaded_files": file_paths}

# 2. Manual Annotation Save Endpoint
class AnnotationData(BaseModel):
    filename: str
    annotations: List[dict]

@app.post("/annotations/save")
async def save_annotations(data: AnnotationData):
    os.makedirs(ANNOTATED_DIR, exist_ok=True)
    json_path = os.path.join(ANNOTATED_DIR, f"{data.filename}.json")
    print("Saving annotations to:", json_path)
    try:
        with open(json_path, "w") as f:
            json.dump(data.dict(), f)
        print("Annotations saved for:", data.filename)
        return {"message": "Annotations saved", "path": json_path}
    except Exception as e:
        print("Error saving annotations for", data.filename, e)
        raise HTTPException(status_code=500, detail=str(e))

# NEW ENDPOINT: Prepare YOLO dataset
@app.post("/annotations/prepare-yolo")
async def prepare_yolo_dataset():
    os.makedirs(ANNOTATED_DIR, exist_ok=True)
    print("Preparing YOLO dataset...")
    try:
        json_files = [f for f in os.listdir(ANNOTATED_DIR) if f.endswith(".json")]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cannot list annotated files: {str(e)}")
    print("Found JSON files:", json_files)
    if not json_files:
        raise HTTPException(status_code=404, detail="No annotation JSON files found.")

    random.shuffle(json_files)
    split_index = int(0.8 * len(json_files))
    train_files = json_files[:split_index]
    val_files = json_files[split_index:]
    print(f"Split {len(json_files)} files: {len(train_files)} train, {len(val_files)} val")

    for sub in ["train", "val"]:
        os.makedirs(os.path.join(IMAGES_DIR, sub), exist_ok=True)
        os.makedirs(os.path.join(LABELS_DIR, sub), exist_ok=True)

    class_set = set()

    def convert_box_to_yolo(box, img_w, img_h):
        x = box["x"]
        y = box["y"]
        w = box["width"]
        h = box["height"]
        x = max(x, 0)
        y = max(y, 0)
        if x + w > img_w:
            w = img_w - x
        if y + h > img_h:
            h = img_h - y
        w = max(w, 0)
        h = max(h, 0)
        x_center = (x + w / 2) / img_w
        y_center = (y + h / 2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        class_id = 0
        return f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"

    def process_json_file(json_file, subset):
        print(f"Processing {json_file} for {subset}")
        json_path = os.path.join(ANNOTATED_DIR, json_file)
        with open(json_path, "r") as f:
            data = json.load(f)
        filename = os.path.basename(data["filename"].strip())
        print(f"Expected image filename from JSON: '{filename}'")
        boxes = data["annotations"]
        for box in boxes:
            if "label" in box:
                class_set.add(box["label"])
        src_image_path = os.path.join(UPLOAD_DIR, filename)
        if not os.path.exists(src_image_path):
            print(f"Warning: image not found: {src_image_path}")
            return
        dst_image_path = os.path.join(IMAGES_DIR, subset, filename)
        print(f"Copying image from {src_image_path} to {dst_image_path}")
        shutil.copyfile(src_image_path, dst_image_path)
        from PIL import Image
        with Image.open(src_image_path) as im:
            img_w, img_h = im.size
        yolo_lines = []
        for box in boxes:
            line = convert_box_to_yolo(box, img_w, img_h)
            yolo_lines.append(line)
        base_name, _ = os.path.splitext(filename)
        txt_path = os.path.join(LABELS_DIR, subset, base_name + ".txt")
        print(f"Writing YOLO labels to {txt_path}")
        with open(txt_path, "w") as txt_f:
            for line in yolo_lines:
                txt_f.write(line + "\n")

    clear_folder(os.path.join(IMAGES_DIR, "train"))
    clear_folder(os.path.join(IMAGES_DIR, "val"))
    clear_folder(os.path.join(LABELS_DIR, "train"))
    clear_folder(os.path.join(LABELS_DIR, "val"))

    for jf in train_files:
        process_json_file(jf, "train")
    for jf in val_files:
        process_json_file(jf, "val")

    print("Train images folder exists:", os.path.exists(os.path.join(IMAGES_DIR, "train")))
    print("Val images folder exists:", os.path.exists(os.path.join(IMAGES_DIR, "val")))
    print("Train labels folder exists:", os.path.exists(os.path.join(LABELS_DIR, "train")))
    print("Val labels folder exists:", os.path.exists(os.path.join(LABELS_DIR, "val")))

    return {
        "message": "YOLO dataset prepared",
        "train_count": len(train_files),
        "val_count": len(val_files),
        "classes": list(class_set)
    }

# 3. YOLO Inference Endpoint
def get_allowed_labels():
    try:
        yaml_path = os.path.join(BASE_DIR, "temp_zINGg.yaml")
        with open(yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f)
        allowed_labels = set(yaml_data.get("names", []))
        print("Allowed classes from YAML:", allowed_labels)
        return allowed_labels
    except Exception as e:
        print("Error reading YAML file for allowed classes:", e)
        return set()

@app.post("/api/ai/detect/yolo")
async def detect_yolo(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        detections = run_yolo_inference(file_path)
        allowed_labels = get_allowed_labels()
        detections = [det for det in detections if det.get("label") in allowed_labels]
    except Exception as e:
        print("YOLO Inference error:", e)
        raise HTTPException(status_code=500, detail=str(e))
    return {"detections": detections}

# 4. Auto Annotate Dataset Endpoint (with COCO JSON generation)
import cv2
from PIL import Image

RESULTS_DIR = os.path.join(ANNOTATION_DIR, "results_batch")
os.makedirs(RESULTS_DIR, exist_ok=True)
print("RESULTS_DIR:", RESULTS_DIR)

def draw_detections(image_path, detections, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not read image at", image_path)
        return
    for det in detections:
        try:
            x1, y1, x2, y2 = map(int, det.get("box", [0, 0, 0, 0]))
            label = det.get("label", "unknown")
            conf = det.get("confidence", 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            print("Error drawing detection:", e)
    cv2.imwrite(output_path, image)
    print("Annotated image saved to", output_path)

class AnnotateDatasetRequest(BaseModel):
    model_path: str
    confidence_threshold: float = 0.5  # Frontend sends a normalized threshold (0-1)

@app.post("/api/ai/annotate/dataset")
async def annotate_dataset(request: AnnotateDatasetRequest = Body(...)):
    model_path = request.model_path.strip()
    print("Received model_path:", model_path)
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model not found")
    
    os.makedirs(ANNOTATED_DIR, exist_ok=True)
    annotated_bases = {os.path.splitext(f)[0] for f in os.listdir(ANNOTATED_DIR) if f.endswith(".json")}
    print("Already annotated (base names):", annotated_bases)
    
    upload_files = os.listdir(UPLOAD_DIR)
    print("Files in UPLOAD_DIR:", upload_files)
    
    results_list = []
    allowed_labels = get_allowed_labels()
    
    for filename in upload_files:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ['.png', '.jpg', '.jpeg', '.bmp']:
            print(f"Skipping non-image file: {filename}")
            continue
        base = os.path.splitext(filename)[0]
        if base in annotated_bases:
            print(f"Skipping already annotated image: {filename}")
            continue
        image_path = os.path.join(UPLOAD_DIR, filename)
        print("Processing image:", filename, "at path:", image_path)
        try:
            detections = run_yolo_inference(image_path)
            print("Detections before filtering for", filename, ":", detections)
            def normalize_confidence(conf):
                return conf / 100 if conf > 1 else conf

            filtered_detections = []
            for det in detections:
                label = det.get("label", "").lower()
                norm_conf = normalize_confidence(det.get("confidence", 0))
                # Store the original label for mapping later.
                if (label in allowed_labels or label.isdigit()) and norm_conf >= request.confidence_threshold:
                    det["confidence"] = norm_conf
                    det["stored_label"] = label
                    filtered_detections.append(det)
            detections = filtered_detections
            print("Detections after filtering for", filename, ":", detections)
            result_image_path = os.path.join(RESULTS_DIR, filename)
            draw_detections(image_path, detections, result_image_path)
            print("Annotated image saved at", result_image_path)
            results_list.append({
                "filename": filename,
                "detections": detections,
                "annotated_image": result_image_path
            })
        except Exception as e:
            print("Error processing image", filename, ":", e)
            results_list.append({"filename": filename, "error": str(e)})
    print("Auto annotation results:", results_list)
    
    # --- COCO JSON Generation ---
    # Create a new dataset folder in RESULTS_DIR.
    dataset_folder_name = f"annotated_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset_folder_path = os.path.join(RESULTS_DIR, dataset_folder_name)
    os.makedirs(dataset_folder_path, exist_ok=True)
    print("Created new dataset folder:", dataset_folder_path)
    
    coco_images = []
    coco_annotations = []
    annotation_id = 0  # start at 0
    image_id = 0       # start at 0
    # Fixed date for exported JSON:
    fixed_date = "2025-02-22T14:24:33+00:00"
    for result in results_list:
        if "error" in result:
            continue
        filename = result["filename"]
        src_annotated_path = result["annotated_image"]
        if not os.path.exists(src_annotated_path):
            print(f"Warning: Annotated image not found: {src_annotated_path}. Skipping.")
            continue
        dst_image_path = os.path.join(dataset_folder_path, filename)
        shutil.copy(src_annotated_path, dst_image_path)
        print(f"Copied {src_annotated_path} to {dst_image_path}")
        try:
            with Image.open(dst_image_path) as im:
                img_w, img_h = im.size
        except Exception as e:
            print("Error opening image for COCO metadata:", e)
            img_w, img_h = 0, 0
        coco_images.append({
            "id": image_id,
            "license": 1,
            "file_name": filename,
            "width": img_w,
            "height": img_h,
            "date_captured": fixed_date
        })
        for det in result.get("detections", []):
            box = det.get("box", [0, 0, 0, 0])
            try:
                x1, y1, x2, y2 = map(int, box)
            except Exception as e:
                print("Error converting box coordinates for", filename, ":", e)
                continue
            width = x2 - x1
            height = y2 - y1
            area = width * height
            segmentation = []  # Empty list per specification
            stored_label = det.get("stored_label", "")
            # Map the label to fixed category ids.
            category_mapping = {"hand": 0, "left hand": 1, "right hand": 2}
            category_id = category_mapping.get(stored_label, 0)
            coco_annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, width, height],
                "area": area,
                "segmentation": segmentation,
                "iscrowd": 0
            })
            annotation_id += 1
        image_id += 1

    info = {
        "year": "2025",
        "version": "1",
        "description": "Exported from roboflow.com",
        "contributor": "",
        "url": "https://app.roboflow.com/datasets/data-labeling-task/1",
        "date_created": fixed_date
    }
    licenses = [{
        "id": 1,
        "url": "",
        "name": "Unknown"
    }]
    categories = [
        {"id": 0, "name": "hand", "supercategory": "none"},
        {"id": 1, "name": "left hand", "supercategory": "hand"},
        {"id": 2, "name": "right hand", "supercategory": "hand"}
    ]
    
    coco_json = {
        "info": info,
        "licenses": licenses,
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": categories
    }
    
    coco_json_path = os.path.join(dataset_folder_path, "annotations.json")
    with open(coco_json_path, "w") as f:
        json.dump(coco_json, f, indent=2)
    print("COCO JSON file created at", coco_json_path)
    
    return {"annotated_images": results_list, "coco_json_path": coco_json_path, "dataset_folder": dataset_folder_path}

# 5. Custom YOLO Training Endpoint (using prepared dataset)
class YOLOTrainRequest(BaseModel):
    epochs: int = 100
    model_name: str = "yolov5_custom"

import glob

@app.post("/api/models/train/yolo")
async def train_yolo(request: YOLOTrainRequest):
    try:
        print("Starting training endpoint...")
        print("Preparing YOLO dataset before training...")
        prep_result = await prepare_yolo_dataset()
        print("Dataset preparation result:", prep_result)
        
        train_dir = os.path.join(IMAGES_DIR, "train")
        val_dir = os.path.join(IMAGES_DIR, "val")
        labels_train_dir = os.path.join(LABELS_DIR, "train")
        labels_val_dir = os.path.join(LABELS_DIR, "val")
        print("Train dir exists:", os.path.exists(train_dir))
        print("Val dir exists:", os.path.exists(val_dir))
        print("Labels train dir exists:", os.path.exists(labels_train_dir))
        print("Labels val dir exists:", os.path.exists(labels_val_dir))
        
        if not os.listdir(val_dir):
            print("Validation directory is empty. Copying images from train to validation directory as fallback.")
            for file in os.listdir(train_dir):
                shutil.copy(os.path.join(train_dir, file), os.path.join(val_dir, file))
        
        if not (os.path.exists(train_dir) and os.path.exists(val_dir) and 
                os.path.exists(labels_train_dir) and os.path.exists(labels_val_dir)):
            raise HTTPException(status_code=400, detail="YOLO dataset not prepared. Check dataset preparation logs.")

        unique_labels = set()
        for file in os.listdir(ANNOTATED_DIR):
            if file.endswith(".json"):
                json_path = os.path.join(ANNOTATED_DIR, file)
                print("Reading JSON file for classes:", json_path)
                with open(json_path, "r") as f:
                    data = json.load(f)
                    for box in data.get("annotations", []):
                        if "label" in box:
                            unique_labels.add(box["label"])
        class_list = sorted(list(unique_labels))
        nc = len(class_list)
        print("Detected classes:", class_list)
        if nc == 0:
            raise HTTPException(status_code=400, detail="No annotation classes found.")

        data_yaml = {
            "train": train_dir,
            "val": val_dir,
            "nc": nc,
            "names": class_list
        }
        temp_yaml_path = os.path.join(BASE_DIR, f"temp_{request.model_name}.yaml")
        print("Writing temporary YAML to:", temp_yaml_path)
        with open(temp_yaml_path, "w") as f:
            yaml.dump(data_yaml, f)

        print("Loading model 'yolov5su.pt'")
        model = YOLO("yolov5su.pt")
        
        runs_dir = os.path.join(BASE_DIR, "runs")
        if not os.path.exists(runs_dir):
            print("Folder 'runs' does not exist in backend. Creating it.")
            os.makedirs(runs_dir, exist_ok=True)
        runs_detect_dir = os.path.join(runs_dir, "detect")
        if not os.path.exists(runs_detect_dir):
            print("Folder 'runs/detect' does not exist in backend. Creating it.")
            os.makedirs(runs_detect_dir, exist_ok=True)
        print("runs_detect_dir:", runs_detect_dir)
        
        print("Starting training with data.yaml:", temp_yaml_path)
        results = model.train(
            data=temp_yaml_path,
            epochs=request.epochs,
            name=request.model_name,
            project=runs_detect_dir
        )

        weights_pattern = os.path.join(runs_detect_dir, f"{request.model_name}*", "weights", "best.pt")
        weight_files = glob.glob(weights_pattern)
        if weight_files:
            trained_model_path = max(weight_files, key=os.path.getmtime)
            print(f"Found trained model file: {trained_model_path}")
        else:
            weights_pattern = os.path.join(runs_detect_dir, f"{request.model_name}*", "weights", "last.pt")
            weight_files = glob.glob(weights_pattern)
            if weight_files:
                trained_model_path = max(weight_files, key=os.path.getmtime)
                print(f"Found trained model file (last.pt): {trained_model_path}")
            else:
                raise Exception("Trained model file not found in expected location.")

        custom_model_path = os.path.join(CUSTOM_MODELS_DIR, f"{request.model_name}.pt")
        print("Copying trained model from", trained_model_path, "to", custom_model_path)
        shutil.copy(trained_model_path, custom_model_path)
        os.remove(temp_yaml_path)
        from yolo_inference import yolo_model
        print("Deploying the trained model...")
        new_model = YOLO(custom_model_path)
        yolo_model.model = new_model.model
        print("Training and deployment successful.")
        return {"message": "Training completed and model deployed", "model_path": custom_model_path}
    except Exception as e:
        print("Training endpoint error:", e)
        raise HTTPException(status_code=500, detail=str(e))

# 6. List Available Models Endpoint
@app.get("/api/models/available")
async def available_models():
    models = []
    if os.path.exists(CUSTOM_MODELS_DIR):
        for file in os.listdir(CUSTOM_MODELS_DIR):
            if file.endswith(".pt"):
                models.append({
                    "name": file,
                    "path": os.path.join(CUSTOM_MODELS_DIR, file)
                })
    models.append({"name": "YOLOv5s (default)", "path": "yolov5s.pt"})
    return {"models": models}

# 7. YOLO Deploy Endpoint (using JSON payload)
class DeployModelRequest(BaseModel):
    model_path: str
    yaml_path: str = ""  # Optional: full path to the YAML file

@app.post("/api/models/deploy/yolo")
async def deploy_yolo(request: DeployModelRequest):
    model_path = request.model_path.strip()
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="Model weights not found")
    
    if request.yaml_path:
        config_path = request.yaml_path.strip()
    else:
        model_basename = os.path.splitext(os.path.basename(model_path))[0]
        config_filename = f"temp_{model_basename}.yaml"
        config_path = os.path.join(BASE_DIR, config_filename)
    
    try:
        new_model = YOLO(model_path)
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                data = yaml.safe_load(f)
            if "names" in data and isinstance(data["names"], list):
                new_names = {i: name for i, name in enumerate(data["names"])}
                new_model.names = new_names
                print(f"Loaded custom class names from {config_path}: {new_names}")
            else:
                print(f"No 'names' field found in {config_path}. Using model defaults.")
        else:
            print(f"YAML config {config_path} not found; using default model names.")
        yolo_model.model = new_model.model
        yolo_model.model.names = new_model.names
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"message": f"Deployed new YOLO model from {model_path}", "names": new_model.names}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
