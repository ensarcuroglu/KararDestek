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

# Preprocess fonksiyonunu içe aktar
# (Bu dosyanın src/modeling içinde olduğunu varsayıyoruz)
from src.data_preprocessing.preprocess_diabetes import preprocess_diabetes_dataset


def find_best_threshold(y_true, y_prob):
    """F1 skorunu maksimize eden en iyi eşik değerini bulur."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # F1 hesapla (0'a bölünme hatasını önlemek için 1e-9 ekledik)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

    best_idx = np.argmax(f1s)
    # thresholds dizisi precision/recall dizisinden 1 eleman kısadır
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    return float(best_threshold), float(f1s[best_idx])


def main():
    # ---------------------------------------------------------
    # 1. DOSYA YOLLARINI DİNAMİK OLARAK BELİRLEME (HATA ÇÖZÜMÜ)
    # ---------------------------------------------------------
    # Şu anki dosyanın (train_random_forest...) bulunduğu klasör: .../src/modeling
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Proje ana dizinine (root) çıkmak için 2 kez yukarı gidiyoruz: .../KararDestek2
    project_root = os.path.dirname(os.path.dirname(current_dir))

    # CSV dosyasının tam yolu
    csv_path_full = os.path.join(project_root, "data", "raw", "diabetic_data.csv")

    print(f"[INFO] Proje Ana Dizini: {project_root}")
    print(f"[INFO] CSV Dosya Yolu: {csv_path_full}")

    # ---------------------------------------------------------

    print("[INFO] Preprocess çalıştırılıyor...")
    # csv_path parametresini elle veriyoruz
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feature_names,
    ) = preprocess_diabetes_dataset(csv_path=csv_path_full)

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

    # --- Validation Seti ile Threshold Belirleme ---
    val_probs = model.predict_proba(X_val)[:, 1]
    best_threshold, best_f1 = find_best_threshold(y_val, val_probs)
    print(f"[INFO] Valid için en iyi threshold: {best_threshold:.4f} (F1={best_f1:.4f})")

    # --- Test Seti Değerlendirmesi ---
    test_probs = model.predict_proba(X_test)[:, 1]
    test_preds = (test_probs >= best_threshold).astype(int)

    roc_auc = roc_auc_score(y_test, test_probs)
    pr_auc = average_precision_score(y_test, test_probs)
    acc = accuracy_score(y_test, test_preds)
    prec = precision_score(y_test, test_preds, zero_division=0)
    rec = recall_score(y_test, test_preds, zero_division=0)
    f1 = f1_score(y_test, test_preds, zero_division=0)

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
    print(classification_report(y_test, test_preds, digits=4, zero_division=0))

    # --- Kaydetme İşlemleri ---
    # Kayıt yollarını da proje ana dizinine göre ayarlayalım
    output_dir = os.path.join(project_root, "models", "rf_preprocessed")
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "model.joblib")
    metrics_path = os.path.join(output_dir, "metrics.json")

    joblib.dump(model, model_path)

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "threshold": best_threshold
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n💾 Model kaydedildi: {model_path}")
    print(f"💾 Metrikler kaydedildi: {metrics_path}")


if __name__ == "__main__":
    main()