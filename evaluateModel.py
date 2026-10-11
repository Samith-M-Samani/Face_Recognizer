"""
Command-Line Evaluation Script for Face Recognition Model.
Evaluates strictly using .npy data from the ./data/ directory.

Run:
    py evaluateModel.py [--k 5] [--test-size 0.25]
"""

import sys
import argparse
import numpy as np
import ml_engine

def print_separator(char="=", length=75):
    print(char * length)

def format_table_row(cols, widths, aligns=None):
    if aligns is None:
        aligns = ["<"] * len(cols)
    out = []
    for col, w, a in zip(cols, widths, aligns):
        val = str(col)
        if a == ">":
            out.append(val.rjust(w))
        elif a == "^":
            out.append(val.center(w))
        else:
            out.append(val.ljust(w))
    return " | ".join(out)

def main():
    parser = argparse.ArgumentParser(description="Evaluate Face Recognizer on ./data using Scikit-Learn")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors (default: 5)")
    parser.add_argument("--test-size", type=float, default=0.25, help="Test set fraction (default: 0.25)")
    args = parser.parse_args()

    dm = ml_engine.DatasetManager(dataset_path="./data")
    npy_files = dm.get_local_npy_files()

    if not npy_files:
        print("\n" + "=" * 75)
        print("[!] No .npy face datasets found in the './data/' directory.")
        print("    -> Run 'py getFaceData.py' to enroll your face, OR")
        print("    -> Open the web dashboard ('py app.py') to enroll faces via webcam.")
        print("=" * 75 + "\n")
        sys.exit(1)

    print(f"\nLoading face dataset from ./data/ ({len(npy_files)} files: {', '.join(npy_files)})...")
    X, y, names = dm.load_dataset()

    if X is None or len(X) == 0:
        print("[ERROR] Failed to load data from ./data/.")
        sys.exit(1)

    print_separator("=")
    print("        FACE RECOGNITION ML MODEL - SCIKIT-LEARN EVALUATION")
    print("        Classifier: sklearn.neighbors.KNeighborsClassifier")
    print_separator("=")
    print(f" Dataset Folder     : ./data/")
    print(f" Registered People  : {', '.join(names.values())}")
    print(f" Total Samples      : {X.shape[0]}")
    print(f" Feature Dimensions : {X.shape[1]} pixels per face")
    print(f" Number of Classes  : {len(names)}")
    print(f" Hyperparameter K   : {args.k}")
    print(f" Test Split Ratio   : {args.test_size * 100:.1f}%")
    print_separator("-")

    # Run Scikit-Learn evaluation pipeline
    results = ml_engine.run_full_evaluation(
        X, y, names, k=args.k, test_size=args.test_size
    )

    if results.get("notice"):
        print(f"\n[NOTE] {results['notice']}\n")

    metrics = results["metrics"]
    cm = results["confusion_matrix"]["matrix"]
    cm_labels = results["confusion_matrix"]["labels"]
    class_table = results["class_breakdown"]
    k_curve = results["k_tuning_curve"]
    benchmarks = results.get("model_benchmarks", [])

    # 1. Primary Metrics Card (Scikit-Learn)
    print("\n[1] OVERALL PERFORMANCE METRICS (SCIKIT-LEARN EVALUATION)")
    print_separator("-")
    print(f"  [+] Accuracy (accuracy_score) : {metrics['accuracy_percent']:6.2f}%  ({metrics['correct_predictions']}/{metrics['total_test_samples']} correct)")
    print(f"  [+] Error Count               : {metrics['misclassifications']} misclassifications")
    print(f"  [+] Error Rate (1 - Accuracy) : {metrics['error_rate_percent']:6.2f}%")
    print(f"  [+] Macro Precision           : {metrics['macro_precision']:6.4f}  (Unweighted average)")
    print(f"  [+] Macro Recall              : {metrics['macro_recall']:6.4f}  (Sensitivity / TPR)")
    print(f"  [+] Macro F1-Measure          : {metrics['macro_f1']:6.4f}  (Harmonic mean of Prec & Rec)")
    print(f"  [+] Weighted Precision        : {metrics['weighted_precision']:6.4f}  (Support weighted)")
    print(f"  [+] Weighted Recall           : {metrics['weighted_recall']:6.4f}")
    print(f"  [+] Weighted F1-Measure       : {metrics['weighted_f1']:6.4f}")
    print_separator("-")

    # 2. Per-Class Classification Report Table
    print("\n[2] DETAILED PER-CLASS EVALUATION BREAKDOWN (SCIKIT-LEARN)")
    print_separator("-")
    headers = ["Class / Identity", "Support", "TP", "FP", "FN", "Precision", "Recall", "F1-Score"]
    widths = [18, 8, 5, 5, 5, 10, 10, 10]
    aligns = ["<", ">", ">", ">", ">", ">", ">", ">"]
    print(format_table_row(headers, widths, aligns))
    print_separator("-")

    for row in class_table:
        cols = [
            row["name"][:18],
            row["support"],
            row["tp"],
            row["fp"],
            row["fn"],
            f"{row['precision']:.4f}",
            f"{row['recall']:.4f}",
            f"{row['f1_score']:.4f}"
        ]
        print(format_table_row(cols, widths, aligns))

    print_separator("-")
    avg_cols = [
        "Macro Average",
        metrics["total_test_samples"],
        "-", "-", "-",
        f"{metrics['macro_precision']:.4f}",
        f"{metrics['macro_recall']:.4f}",
        f"{metrics['macro_f1']:.4f}"
    ]
    print(format_table_row(avg_cols, widths, aligns))

    # 3. Confusion Matrix
    print("\n[3] CONFUSION MATRIX (sklearn.metrics.confusion_matrix)")
    print_separator("-")
    max_label_len = min(12, max(len(l) for l in cm_labels))
    col_w = max(5, max(len(str(val)) for row in cm for val in row) + 2)
    
    header_str = "Actual \\ Pred".ljust(max_label_len + 2) + "| " + " ".join(l[:col_w-1].rjust(col_w) for l in cm_labels)
    print(header_str)
    print("-" * len(header_str))
    for idx, (label, row) in enumerate(zip(cm_labels, cm)):
        row_vals = " ".join(str(val).rjust(col_w) for val in row)
        print(f"{label[:max_label_len].ljust(max_label_len + 2)}| {row_vals}")

    # 4. Multi-Model Benchmark Comparison
    if benchmarks:
        print("\n[4] MULTI-MODEL BENCHMARK COMPARISON (SCIKIT-LEARN CLASSIFIERS)")
        print_separator("-")
        b_headers = ["Algorithm / Classifier", "Accuracy (%)", "Error Rate (%)", "Macro F1"]
        b_widths = [32, 14, 15, 10]
        b_aligns = ["<", ">", ">", ">"]
        print(format_table_row(b_headers, b_widths, b_aligns))
        print_separator("-")
        for b in benchmarks:
            b_cols = [
                b["model_name"][:32],
                f"{b['accuracy']:.2f}%",
                f"{b['error_rate']:.2f}%",
                f"{b['f1_score']:.4f}"
            ]
            print(format_table_row(b_cols, b_widths, b_aligns))

    # 5. Hyperparameter Tuning (K-Sweep)
    print("\n[5] HYPERPARAMETER OPTIMIZATION (K-NEIGHBOR SWEEP)")
    print_separator("-")
    k_headers = ["K-Value", "Accuracy (%)", "Error Rate (%)", "Macro F1"]
    k_widths = [10, 15, 16, 12]
    k_aligns = ["^", ">", ">", ">"]
    print(format_table_row(k_headers, k_widths, k_aligns))
    print_separator("-")
    for row in k_curve:
        marker = " *" if row["k"] == results["model_config"]["best_k_found"] else ""
        cols = [
            f"k = {row['k']}{marker}",
            f"{row['accuracy']:.2f}%",
            f"{row['error_rate']:.2f}%",
            f"{row['f1_score']:.4f}"
        ]
        print(format_table_row(cols, k_widths, k_aligns))
    print_separator("=")
    print(f"Optimal K determined: k = {results['model_config']['best_k_found']}")
    print_separator("=")
    print("\nEvaluation complete. To launch the interactive web dashboard, run: py app.py\n")

if __name__ == "__main__":
    main()
