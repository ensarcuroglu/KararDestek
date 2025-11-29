# src/decision/run_topsis_on_test.py

import joblib
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

from src.data_preprocessing.preprocess_diabetes import preprocess_diabetes_dataset
from src.decision.topsis_decision import (
    compute_total_visits,
    compute_diag_severity,
    compute_topsis_scores,
    assign_risk_category,
)


def main():
    # 1) Preprocess: train/valid/test + feature names
    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        feature_names,
    ) = preprocess_diabetes_dataset()

    # 2) XGBoost modelini yükle
    bundle = joblib.load("models/xgb_preprocessed/model.joblib")
    xgb_model = bundle["model"]

    # 3) Test set için ML risk skoru (XGBoost olasılığı)
    ml_risk = xgb_model.predict_proba(X_test)[:, 1]

    # 4) Test özelliklerini DataFrame'e çevir
    df_test = pd.DataFrame(X_test, columns=feature_names)

    # 5) Karar kriterlerini oluştur

    # 5.1. ml_risk
    df_test["ml_risk"] = ml_risk

    # 5.2. total_visits = inpatient + emergency + outpatient
    df_test["total_visits"] = compute_total_visits(df_test)

    # 5.3. diag_severity = number_diagnoses (komorbidite proxy)
    df_test["diag_severity"] = compute_diag_severity(df_test)

    # 5.4. A1Cresult_ord ve time_in_hospital zaten df_test içinde olmalı
    # Eğer sütun isimleri farklıysa burada uyarlaman gerekir.
    if "A1Cresult_ord" not in df_test.columns:
        raise ValueError("A1Cresult_ord sütunu df_test içinde yok.")
    if "time_in_hospital" not in df_test.columns:
        raise ValueError("time_in_hospital sütunu df_test içinde yok.")

    # 6) TOPSIS skorlarını hesapla
    criteria_cols = [
        "ml_risk",
        "total_visits",
        "diag_severity",
        "A1Cresult_ord",
        "time_in_hospital",
    ]

    df_test["topsis_score"] = compute_topsis_scores(
        df_test,
        criteria_cols=criteria_cols,
    )

    # 7) Risk kategorisi ata (Low / Medium / High)
    df_test["topsis_risk_group"] = assign_risk_category(df_test["topsis_score"])

    # 8) Birkaç örnek göster
    print(df_test[["ml_risk", "topsis_score", "topsis_risk_group"]].head(10))

    # 9) İstersen basit değerlendirme:
    # High risk grubunu "pozitif" (yeniden yatacak) gibi düşünelim
    high_risk_pred = (df_test["topsis_risk_group"] == "High").astype(int)

    print("\nConfusion Matrix (High risk = 1):")
    print(confusion_matrix(y_test, high_risk_pred))

    print("\nClassification Report (High risk = 1):")
    print(classification_report(y_test, high_risk_pred, digits=4))

    # 10) Sonuçları dosyaya kaydet (rapor için işine yarar)
    df_test_out = df_test.copy()
    df_test_out["true_label"] = y_test

    df_test_out.to_csv("models/xgb_preprocessed/topsis_test_results.csv", index=False)
    print("\n💾 TOPSIS sonuçları kaydedildi: models/xgb_preprocessed/topsis_test_results.csv")


if __name__ == "__main__":
    main()
