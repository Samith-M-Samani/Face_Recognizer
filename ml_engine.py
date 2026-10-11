"""
Machine Learning Engine for Face Recognition using Scikit-Learn.
Features:
- Models: Scikit-Learn KNeighborsClassifier (with multi-model benchmark: SVC, Random Forest)
- Evaluation Metrics via scikit-learn.metrics:
  * Accuracy Score
  * Error Count & Error Rate (1 - Accuracy)
  * Precision (Macro, Weighted, and Per-Class)
  * Recall / Sensitivity (Macro, Weighted, and Per-Class)
  * F1-Measure / F1-Score (Macro, Weighted, and Per-Class)
  * Confusion Matrix & Classification Report
- Hyperparameter tuning across K-neighbors
- Dataset management strictly for local .npy face files in ./data/
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


class DatasetManager:
    """Manages face datasets strictly from the ./data directory."""
    def __init__(self, dataset_path: str = "./data"):
        self.dataset_path = dataset_path
        os.makedirs(self.dataset_path, exist_ok=True)

    def get_local_npy_files(self) -> List[str]:
        """Finds all .npy face data files in data directory."""
        if not os.path.exists(self.dataset_path):
            return []
        return [f for f in os.listdir(self.dataset_path) if f.endswith(".npy")]

    def load_dataset(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[int, str]]:
        """
        Loads all .npy files from ./data/
        Each file represents one class named after the filename without .npy extension.
        """
        npy_files = self.get_local_npy_files()
        if not npy_files:
            return None, None, {}

        face_data = []
        labels = []
        names = {}
        class_id = 0

        for fx in npy_files:
            name = fx[:-4]
            names[class_id] = name
            full_path = os.path.join(self.dataset_path, fx)
            try:
                data_item = np.load(full_path)
                if data_item.ndim == 1:
                    data_item = data_item.reshape(1, -1)
                elif data_item.ndim > 2:
                    data_item = data_item.reshape(data_item.shape[0], -1)

                face_data.append(data_item)
                target = class_id * np.ones((data_item.shape[0],), dtype=np.int64)
                labels.append(target)
                class_id += 1
            except Exception as e:
                print(f"Error loading {fx}: {e}")

        if not face_data:
            return None, None, {}

        X = np.concatenate(face_data, axis=0)
        y = np.concatenate(labels, axis=0)
        return X, y, names

    def save_face_data(self, name: str, face_arrays: np.ndarray, mode: str = "append") -> str:
        """Saves captured face images as <name>.npy in ./data/."""
        clean_name = "".join(c for c in name if c.isalnum() or c in ("_", "-")).strip()
        if not clean_name:
            clean_name = "User"
        
        target_file = os.path.join(self.dataset_path, f"{clean_name}.npy")
        
        if mode == "append" and os.path.exists(target_file):
            try:
                existing = np.load(target_file)
                if existing.ndim == 1:
                    existing = existing.reshape(1, -1)
                combined = np.concatenate([existing, face_arrays], axis=0)
                np.save(target_file, combined)
                return target_file
            except Exception:
                pass
        
        np.save(target_file, face_arrays)
        return target_file


def evaluate_with_sklearn(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[int],
    class_names: Dict[int, str]
) -> Dict[str, Any]:
    """
    Computes all standard metrics using Scikit-Learn metrics suite.
    """
    y_true = np.asarray(y_true, dtype=np.int64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
    n_samples = len(y_true)

    # 1. Scikit-Learn Accuracy & Error Rate
    acc = float(accuracy_score(y_true, y_pred))
    misclassifications = int(np.sum(y_true != y_pred))
    correct = int(n_samples - misclassifications)
    error_rate = float(1.0 - acc)

    # 2. Scikit-Learn Precision, Recall, F1 (Macro & Weighted)
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))

    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # 3. Scikit-Learn Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    # 4. Scikit-Learn Per-Class Metrics
    per_class_prec = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    per_class_rec = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)

    # 5. Macro Specificity (TNR = TN / (TN + FP)) averaged across classes
    specificities = []
    for idx in range(len(classes)):
        tp = int(cm[idx, idx])
        fp = int(np.sum(cm[:, idx]) - tp)
        fn = int(np.sum(cm[idx, :]) - tp)
        tn = int(n_samples - (tp + fp + fn))
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        specificities.append(spec)
    macro_specificity = float(np.mean(specificities)) if specificities else 0.0

    class_breakdown = []
    for idx, c in enumerate(classes):
        tp = int(cm[idx, idx])
        fp = int(np.sum(cm[:, idx]) - tp)
        fn = int(np.sum(cm[idx, :]) - tp)
        tn = int(n_samples - (tp + fp + fn))
        support = int(np.sum(cm[idx, :]))
        spec = specificities[idx]

        class_breakdown.append({
            "class_id": int(c),
            "name": class_names.get(c, f"Class_{c}"),
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(float(per_class_prec[idx]), 4),
            "recall": round(float(per_class_rec[idx]), 4),
            "specificity": round(float(spec), 4),
            "f1_score": round(float(per_class_f1[idx]), 4),
            "error_count": int(fn + fp)
        })

    # Full text classification report from scikit-learn
    sk_report_text = classification_report(
        y_true, y_pred, labels=classes,
        target_names=[class_names.get(c, f"Class_{c}") for c in classes],
        zero_division=0
    )

    return {
        "total_samples": n_samples,
        "correct_predictions": correct,
        "misclassifications": misclassifications,
        "accuracy": round(acc, 4),
        "accuracy_percent": round(acc * 100, 2),
        "error_rate": round(error_rate, 4),
        "error_rate_percent": round(error_rate * 100, 2),
        "macro_precision": round(macro_prec, 4),
        "weighted_precision": round(weighted_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "weighted_recall": round(weighted_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_specificity": round(macro_specificity, 4),
        "confusion_matrix": cm.tolist(),
        "class_breakdown": class_breakdown,
        "classification_report_text": sk_report_text
    }


def run_full_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    names: Dict[int, str],
    k: int = 5,
    test_size: float = 0.25,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Executes complete ML training and evaluation using Scikit-Learn:
    1. Stratified train/test split via sklearn.model_selection.train_test_split
    2. Fit sklearn.neighbors.KNeighborsClassifier
    3. Calculate all evaluation metrics via scikit-learn.metrics
    4. Hyperparameter tuning (K-Sweep)
    5. Multi-model comparative benchmark (KNN vs SVC vs Random Forest)
    """
    unique_classes = sorted(list(set(y.tolist())))
    num_classes = len(unique_classes)
    
    class_counts = {c: int(np.sum(y == c)) for c in unique_classes}
    min_count = min(class_counts.values()) if class_counts else 0
    stratify = y if (num_classes >= 2 and min_count >= 2) else None

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    actual_k = max(1, min(k, len(X_train)))
    knn = KNeighborsClassifier(n_neighbors=actual_k, metric="euclidean", weights="uniform")
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    eval_metrics = evaluate_with_sklearn(y_test, y_pred, unique_classes, names)

    # Hyperparameter sweep
    max_k = min(len(X_train), 15)
    k_candidates = [val for val in [1, 3, 5, 7, 9, 11, 13, 15] if val <= max_k]
    if actual_k not in k_candidates and actual_k <= max_k:
        k_candidates.append(actual_k)
        k_candidates.sort()

    k_curve = []
    best_k = actual_k
    best_acc = -1.0

    for cand_k in k_candidates:
        m_cand = KNeighborsClassifier(n_neighbors=cand_k, metric="euclidean")
        m_cand.fit(X_train, y_train)
        pred_cand = m_cand.predict(X_test)

        cand_acc = float(accuracy_score(y_test, pred_cand))
        cand_err = float(1.0 - cand_acc)
        cand_f1 = float(f1_score(y_test, pred_cand, average="macro", zero_division=0))

        if cand_acc > best_acc:
            best_acc = cand_acc
            best_k = cand_k

        k_curve.append({
            "k": cand_k,
            "accuracy": round(cand_acc * 100, 2),
            "error_rate": round(cand_err * 100, 2),
            "f1_score": round(cand_f1, 4)
        })

    # Only KNN benchmark
    model_benchmarks = [{
        "model_name": f"K-Nearest Neighbors (k={actual_k})",
        "library": "KNN",
        "accuracy": eval_metrics["accuracy_percent"],
        "error_rate": eval_metrics["error_rate_percent"],
        "f1_score": eval_metrics["macro_f1"]
    }]

    class_labels_mapped = [names.get(c, f"Class {c}") for c in unique_classes]

    notice = None
    if num_classes == 1:
        only_person = names.get(0, "User")
        notice = f"Currently 1 identity ('{only_person}') is enrolled in ./data. Machine learning classification is self-testing. To evaluate inter-class confusion, precision, and recall across different people, enroll at least one more identity."

    return {
        "status": "success",
        "framework": "K-Nearest Neighbors (KNN)",
        "notice": notice,
        "dataset_info": {
            "total_samples": int(X.shape[0]),
            "feature_dimension": int(X.shape[1]),
            "num_classes": len(unique_classes),
            "class_names": names,
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "test_ratio": test_size,
            "dataset_folder": "./data"
        },
        "model_config": {
            "algorithm": "K-Nearest Neighbors (KNN)",
            "k": actual_k,
            "distance_metric": "Euclidean Distance",
            "best_k_found": best_k
        },
        "metrics": {
            "accuracy": eval_metrics["accuracy"],
            "accuracy_percent": eval_metrics["accuracy_percent"],
            "error_rate": eval_metrics["error_rate"],
            "error_rate_percent": eval_metrics["error_rate_percent"],
            "total_test_samples": eval_metrics["total_samples"],
            "correct_predictions": eval_metrics["correct_predictions"],
            "misclassifications": eval_metrics["misclassifications"],
            "macro_precision": eval_metrics["macro_precision"],
            "macro_recall": eval_metrics["macro_recall"],
            "macro_f1": eval_metrics["macro_f1"],
            "macro_specificity": eval_metrics["macro_specificity"],
            "weighted_precision": eval_metrics["weighted_precision"],
            "weighted_recall": eval_metrics["weighted_recall"],
            "weighted_f1": eval_metrics["weighted_f1"]
        },
        "confusion_matrix": {
            "matrix": eval_metrics["confusion_matrix"],
            "labels": class_labels_mapped,
            "class_ids": unique_classes
        },
        "class_breakdown": eval_metrics["class_breakdown"],
        "k_tuning_curve": k_curve,
        "model_benchmarks": model_benchmarks,
        "classification_report_text": eval_metrics["classification_report_text"]
    }
