# src/modeling/train_xgboost_preprocessed.py

import os
import json
import joblib
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from xgboost import XGBClassifier

# preprocess_diabetes_dataset fonksiyonunu kullanıyoruz
from src.data_preprocessing.preprocess_diabetes import preprocess_diabetes_dataset


def find_best_threshold(y_true, y_prob):
    """
    Precision-Recall eğrisi üzerinden F1'i maksimize eden threshold'u bulur.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

    # thresholds dizisi, precisions/recalls'tan 1 eleman daha kısadır
    best_idx = np.nanargmax(f1_scores[:-1])
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def main():
    print("[INFO] Preprocess başlatılıyor...")
    (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        feature_names,
    ) = preprocess_diabetes_dataset()

    print("[INFO] XGBoost modeli oluşturuluyor...")

    # Dikkat: Train zaten SMOTE ile DENGELİ, o yüzden scale_pos_weight=1 bırakıyoruz.
    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=1.0,
        min_child_weight=3,
        reg_lambda=1.0,
        reg_alpha=0.0,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",  # varsa GPU kullanmıyorsan da sorun değil
    )

    print("[INFO] Model eğitiliyor...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )


    # -------- VALID ---------- #
    y_valid_prob = model.predict_proba(X_valid)[:, 1]
    best_thr, best_f1 = find_best_threshold(y_valid, y_valid_prob)
    print(f"[INFO] Valid set için en iyi threshold (F1'e göre): {best_thr:.4f} (F1={best_f1:.4f})")

    # -------- TEST ---------- #
    y_test_prob = model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= best_thr).astype(int)

    roc_auc = roc_auc_score(y_test, y_test_prob)
    pr_auc = average_precision_score(y_test, y_test_prob)
    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, zero_division=0)
    rec = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)

    print("\n📊 Test Sonuçları (threshold optimize edilmiş):")
    print(f"roc_auc   : {roc_auc:.4f}")
    print(f"pr_auc    : {pr_auc:.4f}")
    print(f"accuracy  : {acc:.4f}")
    print(f"precision : {prec:.4f}")
    print(f"recall    : {rec:.4f}")
    print(f"f1        : {f1:.4f}")
    print(f"threshold : {best_thr:.4f}")

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "threshold": best_thr,
    }

    # -------- KAYDET ---------- #
    os.makedirs("models/xgb_preprocessed", exist_ok=True)

    model_path = "models/xgb_preprocessed/model.joblib"
    metrics_path = "models/xgb_preprocessed/metrics.json"
    threshold_path = "models/xgb_preprocessed/threshold.json"

    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
        },
        model_path,
    )

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    with open(threshold_path, "w") as f:
        json.dump({"threshold": best_thr}, f, indent=4)

    print(f"\n💾 Model kaydedildi: {model_path}")
    print(f"💾 Metrikler kaydedildi: {metrics_path}")
    print(f"💾 Threshold kaydedildi: {threshold_path}")


if __name__ == "__main__":
    main()
