# eski v2 modeli

import os
import sys
import json
import joblib
import optuna
import numpy as np
import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    fbeta_score,
    confusion_matrix
)

from xgboost import XGBClassifier

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data_preprocessing.preprocess_diabetes import preprocess_diabetes_dataset_for_training



def find_best_threshold_f2(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # F-Beta Formülü: (1 + beta^2) * (P * R) / ((beta^2 * P) + R)
    beta = 2
    numerator = (1 + beta ** 2) * (precisions * recalls)
    denominator = (beta ** 2 * precisions) + recalls + 1e-9

    f2_scores = numerator / denominator

    best_idx = np.nanargmax(f2_scores[:-1])

    return float(thresholds[best_idx]), float(f2_scores[best_idx])


def main(n_trials: int = 50):
    print("\n" + "=" * 60)
    print("🚀 XGBOOST MODEL EĞİTİM SÜRECİ BAŞLATILIYOR (V2 - API Uyumlu)")
    print("=" * 60 + "\n")

    csv_path_full = os.path.join(project_root, "data", "raw", "diabetic_data.csv")

    if not os.path.exists(csv_path_full):
        print(f"[HATA] CSV dosyası bulunamadı: {csv_path_full}")
        return

    print(f"[INFO] CSV Dosya Yolu: {csv_path_full}")
    print("[INFO] Veri Ön İşleme (Preprocessing) başlatılıyor...")

    try:
        (
            X_train, y_train,
            X_valid, y_valid,
            X_test, y_test,
            feature_names,
            scaler,
            num_cols,
            num_imputer,
            cat_imputer
        ) = preprocess_diabetes_dataset_for_training(csv_path=csv_path_full)
    except ValueError as e:
        print("\n[KRİTİK HATA] Preprocess fonksiyonu beklenen değerleri döndürmedi.")
        print(f"Hata Detayı: {e}")
        return

    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()

    scale_pos_weight = neg_count / pos_count

    print(f"\n[INFO] Sınıf Dağılımı (Train - SMOTE Sonrası): Negatif={neg_count}, Pozitif={pos_count}")

    # OPTUNA İLE HİPERPARAMETRE OPTİMİZASYONU
    print("\n[INFO] Optuna ile en iyi parametreler aranıyor...")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 700),
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
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )

        y_valid_prob = model.predict_proba(X_valid)[:, 1]
        return roc_auc_score(y_valid, y_valid_prob)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print("\nEn iyi Validation AUC:", study.best_value)
    print("En iyi parametreler:", study.best_params)

    # MODEL EĞİTİMİ
    print("\n[INFO] Final model eğitiliyor...")

    best_params = study.best_params
    final_model = XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
    )

    final_model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    # THRESHOLD
    y_valid_prob = final_model.predict_proba(X_valid)[:, 1]
    best_thr, best_f2_score = find_best_threshold_f2(y_valid, y_valid_prob)
    print(f"[STRATEGY] Optimize Edilen Threshold: {best_thr:.4f} (Max F2={best_f2_score:.4f})")

    # SONUÇLAR
    y_test_prob = final_model.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= best_thr).astype(int)

    roc_auc = roc_auc_score(y_test, y_test_prob)
    pr_auc = average_precision_score(y_test, y_test_prob)
    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, zero_division=0)
    rec = recall_score(y_test, y_test_pred, zero_division=0)
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    f2_test = fbeta_score(y_test, y_test_pred, beta=2, zero_division=0)

    print("\nTEST SETİ PERFORMANSI:")
    print("-" * 30)
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f} ")
    print(f"F1 Score  : {f1:.4f}")
    print(f"F2 Score  : {f2_test:.4f}")
    print("-" * 30)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    # KAYDETME
    out_dir = os.path.join(project_root, "models", "xgb_weighted_f2")
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "model.joblib")
    metrics_path = os.path.join(out_dir, "metrics.json")

    joblib.dump(
        {
            "model": final_model,
            "feature_names": feature_names,
            "threshold": best_thr,
            "scaler": scaler,
            "num_cols": num_cols,
            "num_imputer": num_imputer,
            "cat_imputer": cat_imputer
        },
        model_path,
    )

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "f2": f2_test,
        "threshold": best_thr,
        "scale_pos_weight": scale_pos_weight,
        "best_params": best_params,
    }

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nModel ve Pipeline başarıyla kaydedildi: {model_path}")
    print(f"Paket içeriği: Model, Scaler, Imputer, Threshold, Feature Names")


if __name__ == "__main__":
    main(n_trials=50)  # Optuna deneme sayısı