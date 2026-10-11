# 🎭 Face Recognition Studio (KNN)

A complete end-to-end Machine Learning Face Recognition system powered by **K-Nearest Neighbors (KNN)**, **OpenCV**, and **FastAPI**, featuring a web dashboard with real-time webcam recognition, dataset enrollment, and metric evaluations.

All face data is strictly stored as NumPy arrays (`.npy`) inside the **`./data/`** directory.

---

## 🏗️ Architecture & System Workflow

```mermaid
graph TD
    subgraph Frontend ["Frontend (HTML5 / CSS3 / Vanilla JS)"]
        UI_METRICS["Tab 1: Metrics & Evaluation<br/>KPI Cards, Confusion Matrix, K-Curve, Breakdown"]
        UI_LIVE["Tab 2: Live Camera Recognition<br/>Webcam Viewport, Canvas Overlay, Top Neighbors"]
        UI_ENROLL["Tab 3: Dataset & Enroll<br/>Sample Slider (25-200), Guidance, Identity Cards"]
    end

    subgraph Server ["FastAPI Backend (server.py)"]
        API_EVAL["/api/evaluate"]
        API_PRED["/api/predict-image"]
        API_ENROLL["/api/dataset/enroll"]
        API_SUMM["/api/dataset/summary"]
        API_DEL["/api/dataset/{name}"]
    end

    subgraph MLEngine ["Machine Learning Engine (ml_engine.py)"]
        DM["DatasetManager<br/>Loads ./data/*.npy"]
        KNN_TRAIN["KNN Training & Inference<br/>KNeighborsClassifier(metric='euclidean')"]
        METRICS_CALC["Metric Evaluation<br/>Accuracy, Precision, Recall, Specificity, F1, Error Rate"]
        K_CURVE["K-Curve Hyperparameter Sweep"]
    end

    subgraph Storage ["Dataset Directory (./data/)"]
        NPY_FILES["./data/<Identity>.npy<br/>(N samples × 30,000 features)"]
    end

    UI_METRICS -->|POST /api/evaluate| API_EVAL
    UI_LIVE -->|POST /api/predict-image (Base64)| API_PRED
    UI_ENROLL -->|POST /api/dataset/enroll| API_ENROLL
    UI_ENROLL -->|GET /api/dataset/summary| API_SUMM
    UI_ENROLL -->|DELETE /api/dataset/{name}| API_DEL

    API_EVAL --> DM
    API_EVAL --> KNN_TRAIN
    API_EVAL --> METRICS_CALC
    API_EVAL --> K_CURVE

    API_PRED --> KNN_TRAIN
    API_ENROLL --> DM
    API_ENROLL --> NPY_FILES
    DM --> NPY_FILES
```

---

## 🔄 End-to-End Workflow Explained

1. **Face Data Enrollment**:
   - The user opens the **Dataset & Enroll** tab (or runs `getFaceData.py`).
   - The webcam captures frames $\rightarrow$ OpenCV's Haar Cascade detects face regions.
   - Faces are cropped with padding, normalized to $100 \times 100 \times 3$ pixels, flattened into a $30,000$-dimensional vector, and saved into `./data/<Name>.npy`.
   - Supports 25 to 200 samples per person (100 recommended) with replace or append modes.

2. **Model Training & Evaluation**:
   - When the dashboard opens or the user clicks **Re-Run Evaluation** (or runs `evaluateModel.py`):
   - `ml_engine.DatasetManager` scans `./data/*.npy`, stacks all feature vectors into matrix $X$ of shape $(N, 30000)$, and assigns integer label vector $y$.
   - Data is split into Train/Test sets using stratified splitting.
   - The KNN classifier is fitted with Euclidean distance ($k=5$ by default).
   - Predictions are evaluated against ground truth test labels to calculate:
     - **Accuracy**: $\frac{TP + TN}{TP + TN + FP + FN}$
     - **Precision**: $\frac{TP}{TP + FP}$
     - **Recall / Sensitivity**: $\frac{TP}{TP + FN}$
     - **Specificity**: $\frac{TN}{TN + FP}$
     - **F1-Score**: $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
     - **Error Rate**: $1 - \text{Accuracy}$
     - **Confusion Matrix**: Full actual vs. predicted matrix.

3. **Live Camera Recognition**:
   - The user starts the camera in the **Live Recognition** tab (or runs `recognizeFace.py`).
   - Every video frame is sent to `/api/predict-image`.
   - The Haar cascade finds the face, crops, resizes, flattens to 30,000 features, and queries the KNN model.
   - The model computes the nearest neighbors via Euclidean distance, determines voting probability, and returns:
     - Predicted identity and confidence percentage.
     - Top nearest neighbors and their exact distances.
     - Face bounding box coordinates for real-time canvas overlays.

---

## 📁 File-by-File Purpose & Module Reference

| File | Purpose | Modules / Libraries Used | How / Where It Is Used |
| :--- | :--- | :--- | :--- |
| **`app.py`** | Main entry launcher | `uvicorn`, `webbrowser`, `server` | Run `py app.py`. Launches server and automatically opens browser at `http://127.0.0.1:8000`. |
| **`server.py`** | REST API & web server | `fastapi`, `uvicorn`, `cv2`, `numpy`, `sklearn.neighbors.KNeighborsClassifier`, `pydantic`, `ml_engine`, `base64`, `os` | Backend API for the web UI; serves static assets, processes webcam frames, runs predictions, handles enrollment and dataset deletion. |
| **`ml_engine.py`** | Machine learning engine & dataset layer | `sklearn.neighbors.KNeighborsClassifier`, `sklearn.model_selection.train_test_split`, `sklearn.metrics`, `numpy`, `os`, `typing` | Core business logic module imported by `server.py` and `evaluateModel.py`. Handles loading `.npy` files, fitting KNN, metric formulas, and K sweeps. |
| **`getFaceData.py`** | CLI face enrollment script | `cv2` (OpenCV), `numpy`, `os` | Run `py getFaceData.py`. Terminal alternative to enroll faces via webcam (25–200 samples) directly into `./data/<Name>.npy`. |
| **`recognizeFace.py`** | CLI real-time recognition | `cv2` (OpenCV), `numpy`, `sklearn.neighbors.KNeighborsClassifier`, `os` | Run `py recognizeFace.py`. Standalone OpenCV window showing live bounding boxes and labels without a browser. |
| **`evaluateModel.py`** | CLI model evaluation tool | `argparse`, `sys`, `numpy`, `ml_engine` | Run `py evaluateModel.py [--k 5]`. Prints formatted terminal tables for all 6 metrics, confusion matrix, and per-class metrics. |
| **`checkCam.py`** | Camera diagnostic utility | `cv2` (OpenCV) | Run `py checkCam.py`. Quick test to verify that your camera (index 0) works properly. |
| **`haarcascade_frontalface_default.xml`** | Pre-trained face detector | OpenCV Cascade Classifier format | Loaded by OpenCV in `server.py`, `getFaceData.py`, and `recognizeFace.py` to localize faces in images. |
| **`static/index.html`** | Web dashboard markup | HTML5 semantic structure | Loaded by the browser when visiting `http://127.0.0.1:8000`. Contains the 3 tabs: Metrics, Live, and Enroll. |
| **`static/style.css`** | Dashboard design system | Vanilla CSS3 (glassmorphism, CSS grid, variables) | Styles the entire web dashboard with dark neon aesthetics, custom scrollbars, and badges. |
| **`static/app.js`** | Frontend client controller | Vanilla JavaScript (WebRTC, Fetch API, Canvas 2D) | Handles tab navigation, webcam stream capture, API communication, canvas bounding boxes, and SVG charts. |
| **`data/`** | Dataset storage folder | Filesystem directory | Stores all enrolled identity files (`./data/<Name>.npy`). |

---

## ⚡ Quick Start Commands

```bash
# 1. Launch the complete Web Studio (Recommended)
py app.py

# 2. Standalone terminal face enrollment
py getFaceData.py

# 3. Standalone terminal model evaluation
py evaluateModel.py --k 5 --test-size 0.25

# 4. Standalone OpenCV desktop live recognizer
py recognizeFace.py
```
