# Dosya: src/modeling/explain_model_shap.py

import os
import sys
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# --- 1. AYARLAR VE YOL BULMA ---
def setup_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))

    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.append(src_path)

    return project_root


project_root = setup_paths()

# Preprocess fonksiyonunu import et
try:
    from data_preprocessing.preprocess_diabetes import preprocess_diabetes_dataset
except ImportError:
    print("[HATA] Preprocess modülü bulunamadı.")
    sys.exit(1)


def get_professional_feature_names(columns):
    """
    Teknik sütun isimlerini rapor için profesyonel Türkçe isimlere çevirir.
    """
    mapping = {
        # Sayısal Değişkenler
        "num_medications": "Kullanılan İlaç Sayısı",
        "time_in_hospital": "Hastanede Kalış Süresi (Gün)",
        "num_lab_procedures": "Laboratuvar Test Sayısı",
        "num_procedures": "Tıbbi İşlem Sayısı (Prosedürler)",
        "number_diagnoses": "Konulan Tanı Sayısı",
        "number_inpatient": "Geçmiş Yatan Hasta Başvurusu",
        "number_emergency": "Geçmiş Acil Servis Başvurusu",
        "number_outpatient": "Geçmiş Poliklinik Başvurusu",
        "age_mid": "Hasta Yaşı (Ortalama)",
        "total_meds_active": "Aktif İlaç Değişimi",
        "service_utilization": "Toplam Sağlık Hizmeti Kullanımı",

        # Kategorik - Demografik
        "gender_Female": "Cinsiyet: Kadın",
        "gender_Male": "Cinsiyet: Erkek",
        "race_Caucasian": "Irk: Beyaz",
        "race_AfricanAmerican": "Irk: Afro-Amerikan",
        "race_Asian": "Irk: Asyalı",
        "race_Hispanic": "Irk: Hispanik",

        # Kategorik - Klinik Durum
        "diabetesMed_Yes": "Diyabet İlacı Kullanımı: Evet",
        "diabetesMed_No": "Diyabet İlacı Kullanımı: Hayır",
        "change_Ch": "İlaç Değişikliği: Var",
        "change_No": "İlaç Değişikliği: Yok",
        "A1Cresult_ord": "HbA1c Test Sonucu (Sıralı)",
        "max_glu_serum_ord": "Glikoz Serum Testi (Sıralı)",

        # Taburcu Durumları (En Kritik Olanlar)
        "discharge_disposition_grp_Home": "Taburcu: Eve Gönderildi",
        "discharge_disposition_grp_SNF_ICF": "Taburcu: Bakımevine Sevk",
        "discharge_disposition_grp_HomeHealth": "Taburcu: Evde Sağlık Hizmeti",
        "discharge_disposition_grp_Other": "Taburcu: Diğer",

        # Başvuru Kaynağı
        "admission_source_grp_EmergencyRoom": "Giriş: Acil Servis",
        "admission_source_grp_Referral": "Giriş: Doktor Sevkli",
        "admission_source_grp_Transfer": "Giriş: Başka Hastaneden Transfer",

        # Tanı Grupları
        "diag_1_group_Circulatory": "Birincil Tanı: Dolaşım Sistemi",
        "diag_1_group_Diabetes": "Birincil Tanı: Diyabet",
        "diag_1_group_Respiratory": "Birincil Tanı: Solunum Yolu",
        "diag_1_group_Digestive": "Birincil Tanı: Sindirim Sistemi",
        "diag_1_group_Injury": "Birincil Tanı: Yaralanma/Zehirlenme",
        "diag_1_group_Musculoskeletal": "Birincil Tanı: Kas-İskelet",
        "diag_1_group_Neoplasms": "Birincil Tanı: Tümör/Kanser",
    }

    new_names = []
    for col in columns:
        if col in mapping:
            new_names.append(mapping[col])
        else:
            # Haritada yoksa temizle: 'admission_type_id' -> 'Admission Type Id'
            clean_name = col.replace("_", " ").title()
            new_names.append(clean_name)
    return new_names


def main():
    print("--- SHAP ANALİZİ (Explainable AI) BAŞLATILIYOR ---")

    # 1. Kayıtlı Modeli Yükle
    model_path = os.path.join(project_root, "src", "modeling", "models", "models", "xgb_weighted_f2", "model.joblib")
    if not os.path.exists(model_path):
        model_path = os.path.join(project_root, "src", "modeling", "models", "xgb_weighted_f2", "model.joblib")

    print(f"[INFO] Model yükleniyor: {model_path}")
    saved_data = joblib.load(model_path)
    model = saved_data["model"]
    feature_names = saved_data["feature_names"]

    # 2. Veriyi Hazırla (Test Seti Üzerinden Analiz Yapacağız)
    csv_path = os.path.join(project_root, "data", "raw", "diabetic_data.csv")
    print("[INFO] Veri hazırlanıyor...")
    _, _, _, _, X_test, _, _ = preprocess_diabetes_dataset(csv_path)

    # Sütunları modelin beklediği hale getir (Feature Alignment)
    if hasattr(X_test, "columns"):
        X_test_aligned = pd.DataFrame(0, index=X_test.index, columns=feature_names)
        common_cols = list(set(X_test.columns) & set(feature_names))
        X_test_aligned[common_cols] = X_test[common_cols]
        X_test = X_test_aligned

    print(f"[INFO] Analiz edilecek veri boyutu: {X_test.shape}")

    # 3. SHAP Explainer Oluştur
    print("[INFO] SHAP değerleri hesaplanıyor (Bu işlem biraz sürebilir)...")

    # XGBoost için TreeExplainer kullanıyoruz
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    # --- KRİTİK ADIM: İsimleri Türkçeleştirme ---
    # SHAP nesnesinin içindeki özellik isimlerini değiştiriyoruz.
    # Böylece grafiklerde "time_in_hospital" yerine "Hastanede Kalış Süresi" yazacak.
    professional_names = get_professional_feature_names(feature_names)
    shap_values.feature_names = professional_names

    # Raporlama Klasörü
    output_dir = os.path.join(project_root, "reports", "shap_plots")
    os.makedirs(output_dir, exist_ok=True)

    # --- GRAFİK 1: SUMMARY PLOT (Beeswarm) ---
    print("[INFO] Grafik 1: Summary Plot oluşturuluyor...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_values, X_test, feature_names=professional_names, show=False)
    plt.title("Özelliklerin Risk Üzerindeki Etkisi (SHAP Özeti)", fontsize=16)
    # Etiketleri büyüt ve netleştir
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- GRAFİK 2: BAR PLOT (Mutlak Önem) ---
    print("[INFO] Grafik 2: Bar Plot oluşturuluyor...")
    plt.figure(figsize=(12, 10))
    shap.plots.bar(shap_values, max_display=15, show=False)
    plt.title("Değişkenlerin Ortalama Etki Düzeyi (Global Önem)", fontsize=16)
    plt.xlabel("Ortalama SHAP Değeri (Risk Katkısı)", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_bar_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # --- GRAFİK 3: WATERFALL PLOT (En Riskli Hasta) ---
    # Modelin en yüksek risk verdiği hastayı bulalım
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    riskiest_idx = np.argmax(y_pred_prob)

    print(f"[INFO] Grafik 3: En Riskli Hasta (Index: {riskiest_idx}) için Waterfall Plot...")

    plt.figure(figsize=(12, 8))
    # Waterfall plot
    shap.plots.waterfall(shap_values[riskiest_idx], max_display=12, show=False)

    # Başlığı ve eksenleri düzenle
    current_risk = y_pred_prob[riskiest_idx]
    plt.title(f"En Riskli Hastanın Karar Analizi (Tahmin: %{current_risk * 100:.1f} Risk)", fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_waterfall_patient.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[BAŞARILI] ✅ Tüm SHAP grafikleri profesyonel formatta kaydedildi: {output_dir}")
    print("1. shap_summary_plot.png -> Değişkenlerin yönü (Pozitif/Negatif Etki).")
    print("2. shap_bar_plot.png     -> Değişkenlerin genel önem sıralaması.")
    print("3. shap_waterfall_patient.png -> Tek bir hastanın neden yüksek riskli çıktığının hikayesi.")


if __name__ == "__main__":
    main()