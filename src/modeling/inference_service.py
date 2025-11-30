import os
import sys
import pandas as pd
import numpy as np
import joblib

# Proje kök dizinini path'e ekle (Import hatası almamak için)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Ortak dönüşüm fonksiyonunu çağırıyoruz (Stateless pre-processing)
from src.data_preprocessing.preprocess_diabetes import transform_features_base


class DiabetesPredictor:
    def __init__(self, model_path: str):
        """
        Modeli ve eğitim sırasında oluşturulan tüm yardımcı araçları (Scaler, Imputer vb.) yükler.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

        print(f"[INFO] Model yükleniyor: {model_path}")

        # .joblib dosyasından dictionary yapısını oku
        try:
            artifacts = joblib.load(model_path)
        except Exception as e:
            raise RuntimeError(f"Model dosyası bozuk veya okunamıyor: {e}")

        # Artifacts içinden gerekli parçaları al
        self.model = artifacts["model"]
        self.scaler = artifacts["scaler"]  # StandardScaler
        self.num_cols = artifacts["num_cols"]  # Scale edilecek sayısal sütun listesi
        self.feature_names = artifacts["feature_names"]  # Eğitimdeki One-Hot sonrası tüm sütun isimleri
        self.threshold = artifacts["threshold"]  # F2 Score ile optimize edilmiş threshold

        # Eğitim setinden öğrenilen eksik veri doldurucular
        self.num_imputer = artifacts.get("num_imputer")
        self.cat_imputer = artifacts.get("cat_imputer")

        print(f"[INFO] Model servisi başarıyla başlatıldı. Threshold: {self.threshold:.4f}")

    def predict(self, input_data: dict) -> dict:
        """
        API'den gelen tekil JSON veriyi (dict) alır, işler ve tahmin döner.
        """
        # 1. Dict -> DataFrame (Tek satırlık)
        df_raw = pd.DataFrame([input_data])

        # 2. Ortak Temizlik ve Feature Engineering (Stateless)
        # Yaş gruplama, ID mapping, ilaç sayma vb. işlemleri yapar.
        df_clean = transform_features_base(df_raw)

        # 3. Imputing (Eksik Veri Doldurma) - Eğitimdeki medyanları kullanır
        # API'den gelen veride bazı sayısal alanlar eksik olabilir.
        if self.num_imputer:
            # Sadece mevcut olan num_col sütunlarını bul
            cols_to_impute = [c for c in self.num_cols if c in df_clean.columns]
            if cols_to_impute:
                # Transform işlemi yap ve dataframe'e geri ata
                df_clean[cols_to_impute] = self.num_imputer.transform(df_clean[cols_to_impute])

        # 4. One-Hot Encoding
        # Kategorik verileri 0-1 sütunlarına çevirir.
        # DİKKAT: Tek satır olduğu için sadece o anki değerin sütunu oluşur (Örn: Race_Asian).
        df_encoded = pd.get_dummies(df_clean)

        # 5. Feature Alignment (Hizalama) - EN KRİTİK ADIM
        # Model eğitimde 60+ sütun gördü ama df_encoded'da belki 15 sütun var.
        # 'reindex' komutu eksik sütunları oluşturur ve içini 0 (fill_value) ile doldurur.
        # Fazla sütun varsa (eğitimde olmayan yeni bir kategori) onları da atar.
        df_final = df_encoded.reindex(columns=self.feature_names, fill_value=0)

        # 6. Scaling (StandardScaler) - Eğitimdeki ölçeği kullanır
        # Sayısal verileri modelin tanıdığı aralığa (örn: -1 ile 1 arasına) çeker.
        if self.scaler:
            df_final[self.num_cols] = self.scaler.transform(df_final[self.num_cols])

        # 7. Tahmin (Prediction)
        # Olasılık değerini al (Sınıf 1 olma olasılığı)
        prob = self.model.predict_proba(df_final)[:, 1][0]

        # Eğitimde belirlediğimiz en iyi Threshold'a göre karar ver
        prediction_class = 1 if prob >= self.threshold else 0

        return {
            "prediction": int(prediction_class),
            "probability": float(prob),
            "threshold_used": float(self.threshold),
            "risk_level": "YÜKSEK RİSK" if prediction_class == 1 else "Düşük Risk",
            "message": "Hasta 30 gün içinde tekrar yatış riski taşıyor." if prediction_class == 1 else "Tekrar yatış riski düşük."
        }