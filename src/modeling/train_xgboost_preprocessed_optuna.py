# src/modeling/train_xgboost_preprocessed_optuna.py

import os
import json
import joblib
import optuna
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
from src.data_preprocessing.preprocess_diabetes import preprocess_diabetes_dataset


def find_best_threshold(y_true, y_prob):
    """Precision-Recall üzerinden F1'i maksimize eden threshold'u bulur."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)

    # thresholds, precision/recall'dan 1 eleman kısa
    best_idx = np.nanargmax(f1_scores[:-1])
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def main(n_trials: int = 50):
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
            tree_method="hist",   # CPU'da da çalışır
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
    print("✅ En iyi parametreler:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

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
        "best_params": best_params,
        "best_valid_auc": study.best_value,
    }

    # ---- Kaydet ----
    out_dir = "models/xgb_preprocessed_optuna"
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "model.joblib")
    metrics_path = os.path.join(out_dir, "metrics.json")

    joblib.dump(
        {
            "model": final_model,
            "feature_names": feature_names,
        },
        model_path,
    )

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n💾 Model kaydedildi: {model_path}")
    print(f"💾 Metrikler kaydedildi: {metrics_path}")


if __name__ == "__main__":
    main(n_trials=50)   # daha fazla deneme istersen burayı 100–150 yapabilirsin
