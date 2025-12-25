import joblib
import pandas as pd
import numpy as np
import shap
import io
import base64
import os
import matplotlib
import warnings

# Backend ayarları
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Uyarıları loglamak için
warnings.filterwarnings('ignore')

app = FastAPI(title="Diabetes Risk Prediction API v2", version="2.0")

# AYARLAR VE MODEL YÜKLEME
MODEL_DIR = os.path.join("models", "xgb_v5_percentile")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")

POSSIBLE_PATHS = [
    MODEL_PATH,
    os.path.join("..", MODEL_PATH),
    os.path.join("src", "modeling", MODEL_PATH),
    "model.joblib"
]

# Global Değişkenler
model_artifacts = {}
model = None
feature_names = []
scaler = None
num_cols = []
thr_low = 0.3
thr_high = 0.7


def load_model_artifacts():
    global model, feature_names, scaler, num_cols, thr_low, thr_high, model_artifacts

    found_path = None
    for path in POSSIBLE_PATHS:
        if os.path.exists(path):
            found_path = path
            break

    if not found_path:
        print(f"[KRİTİK HATA] Model dosyası bulunamadı! Aranan yollar: {POSSIBLE_PATHS}")
        return False

    try:
        print(f"[INFO] Model yükleniyor: {found_path}")
        model_artifacts = joblib.load(found_path)

        model = model_artifacts["model"]
        feature_names = model_artifacts["feature_names"]
        scaler = model_artifacts["scaler"]
        num_cols = model_artifacts["num_cols"]
        thr_low = model_artifacts.get("threshold_low", 0.3)
        thr_high = model_artifacts.get("threshold_high", 0.7)

        print(f"✅ Model Başarıyla Yüklendi.")
        print(f"   -> Thresholds: Yeşil < {thr_low:.3f} <= Sarı < {thr_high:.3f} <= Kırmızı")
        print(f"   -> Feature Sayısı: {len(feature_names)}")
        return True
    except Exception as e:
        print(f"[HATA] Model yüklenirken hata oluştu: {e}")
        return False


# Başlangıçta yükle
load_model_artifacts()


# GİRDİ ŞEMASI
class PatientInput(BaseModel):
    age: int
    gender: str
    race: str
    admission_type: str
    admission_source: str
    discharge_disposition: str
    time_in_hospital: int
    num_lab_procedures: int
    num_procedures: int
    num_medications: int
    number_diagnoses: int
    primary_diagnosis: str
    number_outpatient: int
    number_emergency: int
    number_inpatient: int
    max_glu_serum: str
    A1Cresult: str
    change: str
    diabetesMed: str
    insulin: str = "No"


# VERİ İŞLEME VE FEATURE ENGINEERING
def preprocess_input(data: PatientInput):
    # Boş DataFrame
    df_processed = pd.DataFrame(0, index=[0], columns=feature_names)

    # Sayısal Değerleri Mapping
    # Modelden gelen verileri sözlüğe al
    raw_payload = data.dict()

    # Ordinal Mapping (Eğitimdeki mantıkla birebir aynı olmalı)
    glu_map = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
    a1c_map = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}

    # Sayısal alanları doldurma
    df_processed['time_in_hospital'] = data.time_in_hospital
    df_processed['num_lab_procedures'] = data.num_lab_procedures
    df_processed['num_procedures'] = data.num_procedures
    df_processed['num_medications'] = data.num_medications
    df_processed['number_outpatient'] = data.number_outpatient
    df_processed['number_emergency'] = data.number_emergency
    df_processed['number_inpatient'] = data.number_inpatient
    df_processed['number_diagnoses'] = data.number_diagnoses

    if 'age_mid' in feature_names:
        df_processed['age_mid'] = data.age
    elif 'age' in feature_names:
        df_processed['age'] = data.age

    df_processed["max_glu_serum_ord"] = glu_map.get(data.max_glu_serum, 0)
    df_processed["A1Cresult_ord"] = a1c_map.get(data.A1Cresult, 0)

    # KATEGORİK EŞLEŞTİRME (One-Hot Fix)
    categorical_map = {
        data.race: "race_",
        data.gender: "gender_",
        data.change: "change_",
        data.diabetesMed: "diabetesMed_",
        data.admission_type: "admission_type_grp_",  # C#'tan "Emergency", "Elective" vb. gelmeli
        data.admission_source: "admission_source_grp_",  # C#'tan "Emergency", "Referral" vb. gelmeli
        data.discharge_disposition: "discharge_disposition_grp_",  # C#'tan "Home", "Other" vb. gelmeli
        data.primary_diagnosis: "diag_1_group_",  # C#'tan "Diabetes", "Circulatory" vb. gelmeli
        data.insulin: "insulin_"
    }

    for val, prefix in categorical_map.items():
        target_col = f"{prefix}{val}"

        if target_col in df_processed.columns:
            df_processed[target_col] = 1
        else:
            print(f"[UYARI] '{target_col}' adında bir sütun modelde bulunamadı. Bu özellik 0 olarak kaldı.")

    # SCALING
    if scaler and num_cols:
        try:
            # Sadece dataframe'de var olan num_cols'ları alma
            valid_num_cols = [c for c in num_cols if c in df_processed.columns]

            input_vals = df_processed[valid_num_cols].values
            scaled_vals = scaler.transform(input_vals)
            df_processed[valid_num_cols] = scaled_vals
        except Exception as e:
            print(f"[SCALING ERROR] {e}")

    # FEATURE ENGINEERING (Scale edilmiş veri üzerinden)
    if 'service_utilization_score' in feature_names:
        df_processed['service_utilization_score'] = (
                df_processed.get('number_inpatient', 0) +
                df_processed.get('number_emergency', 0) +
                df_processed.get('number_outpatient', 0)
        )

    return df_processed[feature_names]


# SHAP ANALİZİ
def generate_shap_plots(df_input):
    images = {"waterfall": None, "bar": None}
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(df_input)

        # Türkçe Mapping
        mapping = {
            "num_medications": "İlaç Sayısı",
            "time_in_hospital": "Yatış Süresi",
            "num_lab_procedures": "Lab Testi",
            "num_procedures": "Tıbbi İşlem",
            "number_diagnoses": "Tanı Sayısı",
            "number_inpatient": "Geçmiş Yatan",
            "number_emergency": "Geçmiş Acil",
            "age_mid": "Yaş",
            "service_utilization_score": "Top. Sağlık Hizmeti (Skor)",
            "medical_complexity_score": "Tıbbi Karmaşıklık",
            "emergency_intensity": "Acil Yoğunluğu",
            "max_glu_serum_ord": "Glikoz Seviyesi",
            "A1Cresult_ord": "A1C Sonucu"
        }

        # Feature isimlerini güncelle
        new_names = [mapping.get(col, col) for col in feature_names]
        shap_values.feature_names = new_names

        # 1. Waterfall Plot
        plt.clf()
        fig_w = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        images["waterfall"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close(fig_w)

        # 2. Bar Plot
        plt.clf()
        fig_b = plt.figure(figsize=(10, 6))
        shap.plots.bar(shap_values[0], max_display=10, show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        images["bar"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close(fig_b)

    except Exception as e:
        print(f"[SHAP HATASI] {e}")

    return images


# ENDPOINT
@app.post("/predict")
def predict_risk(data: PatientInput):
    if model is None:
        # Tekrar yüklemeyi deneme
        if not load_model_artifacts():
            raise HTTPException(status_code=500, detail="Model sunucuda yüklü değil.")

    try:
        print("\n" + "-" * 50)
        print("⚡ [API v2] Yeni İstek Alındı")

        df_ready = preprocess_input(data)

        prob = model.predict_proba(df_ready)[:, 1][0]
        risk_pct = float(round(prob * 100, 2))

        if prob < thr_low:
            status = "DÜŞÜK RİSK (Yeşil)"
            risk_color = "green"
        elif prob < thr_high:
            status = "ORTA RİSK (Sarı)"
            risk_color = "yellow"
        else:
            status = "YÜKSEK RİSK (Kırmızı)"
            risk_color = "red"

        print(f"   📊 Tahmin: %{risk_pct}")
        print(f"   🚦 Durum : {status}")
        print(f"   📉 Eşikler: <{thr_low:.2f} | {thr_low:.2f}-{thr_high:.2f} | >{thr_high:.2f}")

        # 4. SHAP Açıklaması
        imgs = generate_shap_plots(df_ready)

        return {
            "risk_score": risk_pct,
            "risk_status": status,
            "risk_color": risk_color,
            "thresholds": {
                "low": thr_low,
                "high": thr_high
            },
            "shap_waterfall": imgs["waterfall"],
            "shap_bar": imgs["bar"]
        }

    except Exception as e:
        print(f"[API HATASI] Beklenmeyen hata: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    print("🚀 API v2 Başlatılıyor...")
    uvicorn.run(app, host="0.0.0.0", port=8000)