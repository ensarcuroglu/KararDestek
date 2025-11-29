import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.data_preprocessing.preprocess_diabetes import preprocess_diabetes_dataset


def find_best_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    return float(best_threshold), float(f1s[best_idx])


def main():
    print("[INFO] Preprocess çalıştırılıyor...")
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feature_names,   # ekstra dönüş
    ) = preprocess_diabetes_dataset()

    print("[INFO] Random Forest modeli oluşturuluyor...")
    model = RandomForestClassifier(
        n_estimators=600,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42
    )

    print("[INFO] Model eğitiliyor...")
    model.fit(X_train, y_train)

    val_probs = model.predict_proba(X_val)[:, 1]
    best_threshold, best_f1 = find_best_threshold(y_val, val_probs)
    print(f"[INFO] Valid için en iyi threshold: {best_threshold:.4f} (F1={best_f1:.4f})")

    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)

    roc_auc = roc_auc_score(y_test, test_probs)
    pr_auc = average_precision_score(y_test, test_probs)
    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds)
    rec = recall_score(y_test, test_preds)
    f1 = f1_score(y_test, test_preds)

    print("\n📊 Test Sonuçları (Random Forest + optimized threshold):")
    print(f"roc_auc   : {roc_auc:.4f}")
    print(f"pr_auc    : {pr_auc:.4f}")
    print(f"accuracy  : {acc:.4f}")
    print(f"precision : {prec:.4f}")
    print(f"recall    : {rec:.4f}")
    print(f"f1        : {f1:.4f}")
    print(f"threshold : {best_threshold:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_preds))

    print("\nClassification Report:")
    print(classification_report(y_test, test_preds, digits=4))

    os.makedirs("models/rf_preprocessed", exist_ok=True)
    joblib.dump(model, "models/rf_preprocessed/model.joblib")

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "threshold": best_threshold
    }

    with open("models/rf_preprocessed/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n💾 Model ve metrikler kaydedildi: models/rf_preprocessed/")


if __name__ == "__main__":
    main()
