import os
import sys
import pandas as pd
import numpy as np
import joblib


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Ortak dönüşüm fonksiyonu (dosya yolu hatası için scr kaynak seçilecek - çağrı)
from src.data_preprocessing.preprocess_diabetes import transform_features_base


class DiabetesPredictor:
    def __init__(self, model_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

        print(f"[INFO] Model yükleniyor: {model_path}")

        # .joblib dosyasından dictionary yapısını oku !
        try:
            artifacts = joblib.load(model_path)
        except Exception as e:
            raise RuntimeError(f"Model dosyası bozuk veya okunamıyor: {e}")

        self.model = artifacts["model"]
        self.scaler = artifacts["scaler"]
        self.num_cols = artifacts["num_cols"]
        self.feature_names = artifacts["feature_names"]
        self.threshold = artifacts["threshold"]

        self.num_imputer = artifacts.get("num_imputer")
        self.cat_imputer = artifacts.get("cat_imputer")

        print(f"[INFO] Model servisi başarıyla başlatıldı. Threshold: {self.threshold:.4f}")

    def predict(self, input_data: dict) -> dict:
        df_raw = pd.DataFrame([input_data])

        df_clean = transform_features_base(df_raw)

        if self.num_imputer:
            cols_to_impute = [c for c in self.num_cols if c in df_clean.columns]
            if cols_to_impute:
                df_clean[cols_to_impute] = self.num_imputer.transform(df_clean[cols_to_impute])

        # One-Hot Encoding
        df_encoded = pd.get_dummies(df_clean)

        # Feature için
        df_final = df_encoded.reindex(columns=self.feature_names, fill_value=0)

        # Scaling (Standardscaler)

        if self.scaler:
            df_final[self.num_cols] = self.scaler.transform(df_final[self.num_cols])

        # Tahmin
        prob = self.model.predict_proba(df_final)[:, 1][0]

        prediction_class = 1 if prob >= self.threshold else 0

        return {
            "prediction": int(prediction_class),
            "probability": float(prob),
            "threshold_used": float(self.threshold),
            "risk_level": "YÜKSEK RİSK" if prediction_class == 1 else "Düşük Risk",
            "message": "Hasta 30 gün içinde tekrar yatış riski taşıyor." if prediction_class == 1 else "Tekrar yatış riski düşük."
        }