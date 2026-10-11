"""
FastAPI Server for Face Recognition Machine Learning Studio.
Powered strictly by Scikit-Learn using .npy data from the ./data/ directory:
- Model: sklearn.neighbors.KNeighborsClassifier
- Metrics: sklearn.metrics
- Dataset Storage: ./data/<Name>.npy
"""

import os
import io
import base64
import json
import cv2
import numpy as np
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.neighbors import KNeighborsClassifier

import ml_engine

app = FastAPI(title="Face Recognition ML Studio (KNN)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Workspace directories
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# Global dataset manager pointing to ./data
dataset_manager = ml_engine.DatasetManager(dataset_path=DATA_DIR)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Cached active Scikit-Learn model state
ACTIVE_MODEL: Optional[KNeighborsClassifier] = None
ACTIVE_CLASSES: Dict[int, str] = {}
ACTIVE_FEATURE_DIM: int = 30000
ACTIVE_TRAIN_Y: Optional[np.ndarray] = None


class EvaluationRequest(BaseModel):
    k: int = 5
    test_size: float = 0.25


class EnrollRequest(BaseModel):
    name: str
    images: List[str]  # List of base64 data URLs
    mode: Optional[str] = "replace"  # "replace" or "append"


class PredictRequest(BaseModel):
    image: str  # Base64 data URL
    k: Optional[int] = 5


def get_current_dataset():
    npy_files = dataset_manager.get_local_npy_files()
    if len(npy_files) > 0:
        X, y, names = dataset_manager.load_dataset()
        source = f"./data/ ({len(npy_files)} people registered: {', '.join(names.values())})"
        return X, y, names, source
    return None, None, {}, "No .npy files in ./data/ yet"


@app.get("/api/status")
def get_status():
    npy_files = dataset_manager.get_local_npy_files()
    registered_names = [f[:-4] for f in npy_files]
    return {
        "status": "online",
        "has_data": len(npy_files) > 0,
        "local_file_count": len(npy_files),
        "registered_people": registered_names,
        "data_directory": "./data",
        "has_active_model": ACTIVE_MODEL is not None,
        "cascade_loaded": not face_cascade.empty(),
        "framework": "K-Nearest Neighbors (KNN)"
    }


@app.get("/api/dataset/summary")
def get_dataset_summary():
    try:
        X, y, names, source = get_current_dataset()
        if X is None:
            return {
                "source": "./data/",
                "total_samples": 0,
                "feature_dim": 0,
                "num_classes": 0,
                "class_distribution": {}
            }

        counts = {}
        for c_id, name in names.items():
            count = int(np.sum(y == c_id))
            counts[name] = count

        return {
            "source": source,
            "total_samples": int(X.shape[0]),
            "feature_dim": int(X.shape[1]),
            "num_classes": len(names),
            "class_distribution": counts
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/evaluate")
def evaluate_model(req: EvaluationRequest):
    global ACTIVE_MODEL, ACTIVE_CLASSES, ACTIVE_FEATURE_DIM, ACTIVE_TRAIN_Y
    try:
        X, y, names, source = get_current_dataset()
        if X is None or len(X) == 0:
            raise HTTPException(
                status_code=400,
                detail="No .npy face datasets found in ./data/. Please enroll your face first via webcam or run 'py getFaceData.py'!"
            )

        # Run comprehensive evaluation with Scikit-Learn
        results = ml_engine.run_full_evaluation(
            X, y, names, k=req.k, test_size=req.test_size
        )

        # Cache active Scikit-Learn model for live webcam recognition
        actual_k = max(1, min(req.k, len(y)))
        ACTIVE_MODEL = KNeighborsClassifier(n_neighbors=actual_k, metric="euclidean")
        ACTIVE_MODEL.fit(X, y)
        ACTIVE_CLASSES = names
        ACTIVE_FEATURE_DIM = int(X.shape[1])
        ACTIVE_TRAIN_Y = y

        results["dataset_source"] = source
        return results
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/dataset/enroll")
def enroll_user_faces(req: EnrollRequest):
    """
    Saves webcam frames sent from the browser directly into ./data/<name>.npy.
    Follows 100x100 resolution convention.
    """
    if not req.name.strip():
        raise HTTPException(status_code=400, detail="Name cannot be empty.")
    if not req.images:
        raise HTTPException(status_code=400, detail="No face images provided.")

    processed_faces = []

    for img_b64 in req.images:
        try:
            if "," in img_b64:
                img_b64 = img_b64.split(",")[1]
            img_bytes = base64.b64decode(img_b64)
            np_arr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            if len(faces) == 0:
                h, w = frame.shape[:2]
                min_dim = min(h, w)
                face_crop = frame[(h-min_dim)//2:(h+min_dim)//2, (w-min_dim)//2:(w+min_dim)//2]
            else:
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                x, y, w, h = faces[0]
                offset = 5
                y1, y2 = max(0, y - offset), min(frame.shape[0], y + h + offset)
                x1, x2 = max(0, x - offset), min(frame.shape[1], x + w + offset)
                face_crop = frame[y1:y2, x1:x2]

            resized = cv2.resize(face_crop, (100, 100))
            # Shape (100, 100, 3) flattened -> 30,000 features matching getFaceData.py
            processed_faces.append(resized.flatten())
        except Exception as e:
            print(f"Error processing enroll frame: {e}")

    if not processed_faces:
        raise HTTPException(status_code=400, detail="Failed to detect faces in captured frames. Please ensure your face is clearly visible.")

    face_matrix = np.array(processed_faces, dtype=np.uint8)
    filepath = dataset_manager.save_face_data(req.name, face_matrix, mode=req.mode or "replace")

    # Invalidate active model cache so next prediction retrains on updated data
    global ACTIVE_MODEL
    ACTIVE_MODEL = None

    return {
        "status": "success",
        "name": req.name,
        "samples_enrolled": len(processed_faces),
        "saved_path": filepath
    }


@app.delete("/api/dataset/{name}")
def delete_identity(name: str):
    """Deletes an enrolled identity .npy file from ./data/ and clears model cache."""
    clean_name = "".join(c for c in name if c.isalnum() or c in ("_", "-")).strip()
    target_file = os.path.join(DATA_DIR, f"{clean_name}.npy")
    if os.path.exists(target_file):
        try:
            os.remove(target_file)
            global ACTIVE_MODEL
            ACTIVE_MODEL = None
            return {"status": "success", "message": f"Deleted {clean_name}.npy"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete file: {e}")
    raise HTTPException(status_code=404, detail=f"Identity '{clean_name}' not found in ./data/")


@app.post("/api/predict-image")
def predict_image(req: PredictRequest):
    """
    Takes a base64 image frame from client webcam,
    detects face using Haar Cascade, crops and flattens,
    and runs Scikit-Learn KNeighborsClassifier prediction on data in ./data.
    """
    global ACTIVE_MODEL, ACTIVE_CLASSES, ACTIVE_FEATURE_DIM, ACTIVE_TRAIN_Y

    # Ensure model is ready
    if ACTIVE_MODEL is None:
        X, y, names, _ = get_current_dataset()
        if X is None:
            return {"detected": False, "message": "No dataset found in ./data/. Enroll your face first!"}
        actual_k = max(1, min(req.k or 5, len(y)))
        ACTIVE_MODEL = KNeighborsClassifier(n_neighbors=actual_k, metric="euclidean")
        ACTIVE_MODEL.fit(X, y)
        ACTIVE_CLASSES = names
        ACTIVE_FEATURE_DIM = int(X.shape[1])
        ACTIVE_TRAIN_Y = y

    try:
        img_b64 = req.image
        if "," in img_b64:
            img_b64 = img_b64.split(",")[1]
        img_bytes = base64.b64decode(img_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"detected": False, "message": "Failed to decode frame"}

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return {"detected": False, "message": "No face detected"}

        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]
        offset = 5
        y1, y2 = max(0, y - offset), min(frame.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(frame.shape[1], x + w + offset)

        if ACTIVE_FEATURE_DIM == 30000:
            face_crop = frame[y1:y2, x1:x2]
            target_size = (100, 100)
            face_resized = cv2.resize(face_crop, target_size)
            face_vec = face_resized.flatten().astype(np.float64)
        else:
            face_crop = gray[y1:y2, x1:x2]
            target_size = (100, 100)
            face_resized = cv2.resize(face_crop, target_size)
            face_vec = face_resized.flatten().astype(np.float64)

        face_query = face_vec.reshape(1, -1)

        # Scikit-Learn Prediction & Confidence Probability
        pred_class = int(ACTIVE_MODEL.predict(face_query)[0])
        pred_name = ACTIVE_CLASSES.get(pred_class, f"Person {pred_class}")

        probs = ACTIVE_MODEL.predict_proba(face_query)[0]
        confidence = float(np.max(probs))

        # Scikit-Learn Nearest Neighbors and Distances
        n_neighbors_count = min(ACTIVE_MODEL.n_neighbors, len(ACTIVE_TRAIN_Y))
        distances, indices = ACTIVE_MODEL.kneighbors(face_query, n_neighbors=n_neighbors_count)

        enriched_neighbors = []
        for dist, idx in zip(distances[0], indices[0]):
            n_class = int(ACTIVE_TRAIN_Y[idx])
            enriched_neighbors.append({
                "label": n_class,
                "name": ACTIVE_CLASSES.get(n_class, f"Person {n_class}"),
                "distance": round(float(dist), 4)
            })

        return {
            "detected": True,
            "bbox": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
            "predicted_class": int(pred_class),
            "predicted_name": pred_name,
            "confidence": confidence,
            "votes": int(round(confidence * n_neighbors_count)),
            "k": n_neighbors_count,
            "neighbors": enriched_neighbors
        }
    except Exception as e:
        return {"detected": False, "error": str(e)}


@app.get("/api/export-report")
def export_academic_report():
    """Generates an academic markdown evaluation report strictly from ./data."""
    X, y, names, source = get_current_dataset()
    if X is None:
        return Response(content="# No Data Found in ./data/\nPlease enroll faces first.", media_type="text/markdown")

    res = ml_engine.run_full_evaluation(X, y, names, k=5, test_size=0.25)
    m = res["metrics"]
    
    report_md = f"""# Face Recognition System - Machine Learning Evaluation Report
**Classifier:** `K-Nearest Neighbors (KNN)`  
**Dataset Directory:** `./data/`  
**Total Samples in ./data:** {res['dataset_info']['total_samples']} | **Classes:** {res['dataset_info']['num_classes']} | **Dimensions:** {res['dataset_info']['feature_dimension']}

---

## 1. Executive Summary & Core Metrics (KNN)

| Metric | Value | What it measures |
| :--- | :--- | :--- |
| **Accuracy** | **{m['accuracy_percent']}%** | Overall correct predictions |
| **Error Rate** | **{m['error_rate_percent']}%** | Fraction of incorrect predictions |
| **Misclassifications** | **{m['misclassifications']} / {m['total_test_samples']}** | Test Misclassification Count |
| **Macro Precision** | **{m['macro_precision']}** | Average class precision |
| **Macro Recall** | **{m['macro_recall']}** | Average class recall / sensitivity |
| **Macro F1-Measure** | **{m['macro_f1']}** | Balance between precision & recall |
| **Weighted F1-Measure** | **{m['weighted_f1']}** | Weighted F1 measure |

---

## 2. Multi-Model Benchmark Comparison

| Algorithm / Classifier | Library | Accuracy (%) | Error Rate (%) | Macro F1 |
| :--- | :--- | :---: | :---: | :---: |
"""
    for bm in res.get("model_benchmarks", []):
        report_md += f"| {bm['model_name']} | `{bm['library']}` | {bm['accuracy']:.2f}% | {bm['error_rate']:.2f}% | {bm['f1_score']:.4f} |\n"

    report_md += f"""
---

## 3. Class-by-Class Metric Breakdown

| Identity | Support | Precision | Recall | F1-Score | Error Count |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for row in res["class_breakdown"]:
        report_md += f"| {row['name']} | {row['support']} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1_score']:.4f} | {row['error_count']} |\n"

    report_md += f"""
---

## 4. Hyperparameter Sensitivity Analysis (K-Sweep)

| K-Value | Accuracy (%) | Error Rate (%) | Macro F1 |
| :---: | :---: | :---: | :---: |
"""
    for row in res["k_tuning_curve"]:
        report_md += f"| k = {row['k']} | {row['accuracy']:.2f}% | {row['error_rate']:.2f}% | {row['f1_score']:.4f} |\n"

    report_md += f"""
*Optimal K value found:* **k = {res['model_config']['best_k_found']}**
"""
    return Response(content=report_md, media_type="text/markdown", headers={
        "Content-Disposition": "attachment; filename=Face_Recognition_ML_Report.md"
    })


# Mount static assets
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
