import joblib
import pandas as pd
import numpy as np
import shap
import io
import base64
import os
import matplotlib

# Sunucu tarafında GUI hatası almamak için Agg backend
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# --- 1. MODEL YÜKLEME VE SCALER ONARIMI ---
POSSIBLE_PATHS = [
    "src/modeling/models/models/xgb_weighted_f2/model.joblib",
    "src/modeling/models/xgb_weighted_f2/model.joblib",
    "models/xgb_weighted_f2/model.joblib",
    "../models/xgb_weighted_f2/model.joblib"
]

model = None
feature_names = []
scaler = None
num_cols = []
saved_threshold = 0.5

# Manuel İstatistikler (Diabetes 130-US Veri Seti Ortalamaları)
# Scaler bozuksa bunlar devreye girecek
MANUAL_STATS = {
    "time_in_hospital": {"mean": 4.4, "std": 3.0},
    "num_lab_procedures": {"mean": 43.0, "std": 19.6},
    "num_procedures": {"mean": 1.3, "std": 1.7},
    "num_medications": {"mean": 16.0, "std": 8.1},
    "number_outpatient": {"mean": 0.37, "std": 1.26},
    "number_emergency": {"mean": 0.20, "std": 0.93},
    "number_inpatient": {"mean": 0.63, "std": 1.26},
    "number_diagnoses": {"mean": 7.4, "std": 1.93},
    "age_mid": {"mean": 65.0, "std": 15.0},  # Tahmini
    "total_meds_active": {"mean": 16.0, "std": 8.1},
    "service_utilization": {"mean": 1.2, "std": 2.0}
}

for path in POSSIBLE_PATHS:
    if os.path.exists(path):
        try:
            saved_data = joblib.load(path)
            model = saved_data["model"]
            feature_names = saved_data["feature_names"]
            scaler = saved_data.get("scaler")
            num_cols = saved_data.get("num_cols")
            saved_threshold = saved_data.get("threshold", 0.5)

            print(f"\n[INFO] ✅ Model yüklendi: {path}")

            # SCALER KONTROLÜ
            if scaler:
                try:
                    # Scaler boş mu dolu mu test edelim
                    test_val = scaler.transform([[65] * len(num_cols)])
                    if test_val[0][0] == 65:
                        print("⚠️ [UYARI] Yüklenen Scaler BOZUK (Fit edilmemiş). Manuel moda geçiliyor.")
                        scaler = None  # Bozuksa iptal et
                    else:
                        print("✅ [BİLGİ] Yüklenen Scaler SAĞLAM.")
                except:
                    print("⚠️ [UYARI] Scaler test edilemedi. Manuel moda geçiliyor.")
                    scaler = None

            break
        except Exception as e:
            print(f"[HATA] {e}")


# --- 2. GİRDİ ŞEMASI ---
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


# --- 3. VERİ İŞLEME ---
def process_input(data: PatientInput):
    df_input = pd.DataFrame(0, index=[0], columns=feature_names)

    # Bebek Kontrolü
    adm_type = data.admission_type
    if data.age > 2 and adm_type == "Newborn":
        adm_type = "Emergency"

    # Ham Değerler
    df_input["time_in_hospital"] = data.time_in_hospital
    df_input["num_lab_procedures"] = data.num_lab_procedures
    df_input["num_procedures"] = data.num_procedures
    df_input["num_medications"] = data.num_medications
    df_input["number_outpatient"] = data.number_outpatient
    df_input["number_emergency"] = data.number_emergency
    df_input["number_inpatient"] = data.number_inpatient
    df_input["number_diagnoses"] = data.number_diagnoses
    df_input["age_mid"] = data.age

    # Feature Engineering
    if "service_utilization" in df_input.columns:
        df_input["service_utilization"] = data.number_outpatient + data.number_emergency + data.number_inpatient
    if "total_meds_active" in df_input.columns:
        df_input["total_meds_active"] = data.num_medications

    # Ordinal
    glu_map = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
    a1c_map = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
    if "max_glu_serum_ord" in df_input.columns:
        df_input["max_glu_serum_ord"] = glu_map.get(data.max_glu_serum, 0)
    if "A1Cresult_ord" in df_input.columns:
        df_input["A1Cresult_ord"] = a1c_map.get(data.A1Cresult, 0)

    # --- KRİTİK: MANUEL VEYA OTOMATİK SCALING ---
    cols_to_scale = num_cols if num_cols else list(MANUAL_STATS.keys())

    for col in cols_to_scale:
        if col in df_input.columns:
            val = df_input[col].values[0]

            if scaler:
                # Scaler sağlamsa onu kullan (Toplu işlem zor olduğu için tek tek yapmıyoruz, aşağıda toplu yapacağız)
                pass
            else:
                # Scaler bozuksa MANUEL hesapla: (Değer - Ortalama) / StdSapma
                stats = MANUAL_STATS.get(col)
                if stats:
                    scaled_val = (val - stats["mean"]) / stats["std"]
                    df_input[col] = scaled_val
                    # print(f"   🔧 Manuel Scale: {col} ({val} -> {scaled_val:.4f})")

    # Eğer scaler sağlamsa toplu transform (Daha güvenli)
    if scaler and num_cols:
        try:
            df_input[num_cols] = scaler.transform(df_input[num_cols])
            print("   ✅ Otomatik Scaler kullanıldı.")
        except:
            pass

    # One-Hot Encoding
    targets = [
        f"race_{data.race}",
        f"gender_{data.gender}",
        f"change_{data.change}",
        f"diabetesMed_{data.diabetesMed}",
        f"admission_type_grp_{adm_type}",
        f"admission_source_grp_{data.admission_source}",
        f"discharge_disposition_grp_{data.discharge_disposition}",
        f"diag_1_group_{data.primary_diagnosis}",
        f"insulin_{data.insulin}"
    ]

    for col in targets:
        if col in df_input.columns:
            df_input[col] = 1

    return df_input


def generate_shap_plots(df_input):
    images = {"waterfall": "", "bar": ""}
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer(df_input)

        mapping = {
            "num_medications": "İlaç Sayısı", "time_in_hospital": "Yatış Süresi",
            "num_lab_procedures": "Lab Testi", "num_procedures": "Tıbbi İşlem",
            "number_diagnoses": "Tanı Sayısı", "number_inpatient": "Geçmiş Yatış",
            "number_emergency": "Geçmiş Acil", "age_mid": "Yaş",
            "service_utilization": "Top. Sağlık Hizmeti", "total_meds_active": "Aktif İlaç",
            "discharge_disposition_grp_SNF_ICF": "Taburcu: Bakımevi",
            "discharge_disposition_grp_Home": "Taburcu: Ev"
        }
        new_names = [mapping.get(col, col) for col in feature_names]
        shap_values.feature_names = new_names

        # Waterfall
        plt.clf()
        plt.figure(figsize=(10, 6))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        images["waterfall"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close('all')

        # Bar
        plt.figure(figsize=(10, 6))
        shap.plots.bar(shap_values[0], max_display=10, show=False)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
        buf.seek(0)
        images["bar"] = base64.b64encode(buf.getvalue()).decode("utf-8")
        plt.close('all')
    except Exception as e:
        print(f"SHAP Hatası: {e}")
    return images


@app.post("/predict")
def predict_risk(data: PatientInput):
    if model is None: return {"error": "Model Yüklenemedi!"}

    try:
        print("\n" + "-" * 40)
        print("⚡ YENİ İSTEK (API v3 - Manuel Scaler Korumalı)")

        # Veriyi İşle
        df_ready = process_input(data)

        # Casusluk: Yaş kontrolü
        print(f"   🔍 Modele Giren Yaş Değeri: {df_ready['age_mid'].values[0]:.4f}")
        if df_ready['age_mid'].values[0] > 10:
            print("   ⚠️ UYARI: Değer hala yüksek (Scale edilmemiş olabilir)!")
        else:
            print("   ✅ Değer Scale Edilmiş (Normal)")

        # Tahmin
        prob = model.predict_proba(df_ready)[:, 1][0]
        risk_pct = float(round(prob * 100, 2))

        if prob >= saved_threshold:
            status = "ACİL MÜDAHALE" if prob > 0.80 else "YÜKSEK RİSK"
        else:
            status = "DÜŞÜK RİSK"

        print(f"🎯 SONUÇ: %{risk_pct} ({status})")
        print("-" * 40)

        imgs = generate_shap_plots(df_ready)

        return {
            "risk_score": risk_pct,
            "risk_status": status,
            "shap_waterfall": imgs["waterfall"],
            "shap_bar": imgs["bar"]
        }
    except Exception as e:
        print(f"[API HATASI] {e}")
        return {"error": str(e)}