# src/modeling/train_xgboost_preprocessed_optuna.py

import os
import json
import joblib
import optuna
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    auc
)

from xgboost import XGBClassifier
from src.data_preprocessing.preprocess_diabetes import preprocess_diabetes_dataset


def find_best_threshold(y_true, y_prob):
    """Precision-Recall üzerinden F1'i maksimize eden threshold'u bulur."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.nanargmax(f1_scores[:-1])
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def generate_plots(model, X_test, y_test, y_prob, best_thr, feature_names, output_dir):
    """
    Rapor için profesyonel kalitede 4 adet grafik üretir ve kaydeder.
    """
    print(f"[INFO] Grafikler hazırlanıyor: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")

    # 1. Confusion Matrix (Optimize edilmiş threshold ile)
    y_pred = (y_prob >= best_thr).astype(int)
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"size": 14})
    plt.title(f"Confusion Matrix (Threshold={best_thr:.2f})", fontsize=16)
    plt.ylabel("Gerçek Durum", fontsize=12)
    plt.xlabel("Tahmin Edilen Durum", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "1_confusion_matrix.png"), dpi=300)
    plt.close()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Yanlış Alarm Oranı)', fontsize=12)
    plt.ylabel('True Positive Rate (Duyarlılık)', fontsize=12)
    plt.title('ROC Eğrisi', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "2_roc_curve.png"), dpi=300)
    plt.close()

    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='green', lw=2, label=f'PR Curve (AP = {pr_auc:.4f})')
    plt.xlabel('Recall (Kaçını Yakaladık?)', fontsize=12)
    plt.ylabel('Precision (Ne Kadarı Doğru?)', fontsize=12)
    plt.title('Precision-Recall Eğrisi', fontsize=16)
    plt.legend(loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "3_pr_curve.png"), dpi=300)
    plt.close()

    # 4. Feature Importance (En önemli 20 özellik)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]  # İlk 20

        plt.figure(figsize=(10, 8))
        sns.barplot(x=importances[indices], y=[feature_names[i] for i in indices], palette="viridis")
        plt.title("Model İçin En Önemli 20 Özellik", fontsize=16)
        plt.xlabel("Önem Skoru (Importance)", fontsize=12)
        plt.ylabel("Özellikler", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "4_feature_importance.png"), dpi=300)
        plt.close()

    print(f"[INFO] ✅ Tüm grafikler '{output_dir}' klasörüne kaydedildi.")


def main(n_trials: int = 50):
    # Proje ana dizinini dinamik olarak buluyoruz
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    # CSV dosyasının tam yolunu oluşturuyoruz
    csv_path_full = os.path.join(project_root, "data", "raw", "diabetic_data.csv")

    print(f"[INFO] CSV Dosya Yolu: {csv_path_full}")
    print("[INFO] Preprocess başlatılıyor...")
    (
        X_train, y_train,
        X_valid, y_valid,
        X_test, y_test,
        feature_names,
    ) = preprocess_diabetes_dataset(csv_path=csv_path_full)

    print("[INFO] Optuna için XGBoost hiperparametre arama başlıyor...")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 700),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
        }

        model = XGBClassifier(
            **params,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )

        y_valid_prob = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, y_valid_prob)
        return auc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\n✅ En iyi ROC-AUC (validation):", study.best_value)

    # ---- En iyi parametrelerle final modeli eğit ----
    best_params = study.best_params
    final_model = XGBClassifier(
        **best_params,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    print("\n[INFO] En iyi parametrelerle final model eğitiliyor...")
    final_model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    # Threshold'u VALID set üzerinde optimize et
    y_valid_prob = final_model.predict_proba(X_valid)[:, 1]
    best_thr, best_f1 = find_best_threshold(y_valid, y_valid_prob)
    print(f"[INFO] Valid set için en iyi threshold: {best_thr:.4f} (F1={best_f1:.4f})")

    # ---- TEST sonuçları ----
    y_test_prob = final_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= best_thr).astype(int)

    roc_auc = roc_auc_score(y_test, y_test_prob)
    pr_auc = average_precision_score(y_test, y_test_prob)
    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, zero_division=0)
    rec = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)

    print("\n📊 Test Sonuçları (Optuna + threshold optimize):")
    print(f"roc_auc   : {roc_auc:.4f}")
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
        "best_params": best_params,
    }

    # ---- Kaydetme İşlemleri ----
    out_dir = "models/xgb_preprocessed_optuna"
    os.makedirs(out_dir, exist_ok=True)

    # Modeli Kaydet
    joblib.dump(
        {"model": final_model, "feature_names": feature_names},
        os.path.join(out_dir, "model.joblib"),
    )

    # Metrikleri Kaydet
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # ---------------------------------------------------------
    # 🔥 YENİ EKLENEN KISIM: Grafikleri Oluştur ve Kaydet
    # ---------------------------------------------------------
    plots_dir = os.path.join(out_dir, "plots")
    generate_plots(
        model=final_model,
        X_test=X_test,
        y_test=y_test,
        y_prob=y_test_prob,  # Test olasılıkları
        best_thr=best_thr,  # Optimize ettiğimiz threshold
        feature_names=feature_names,
        output_dir=plots_dir
    )

    print(f"\n💾 Model ve Rapor Grafikleri kaydedildi: {out_dir}")


if __name__ == "__main__":
    main(n_trials=50)