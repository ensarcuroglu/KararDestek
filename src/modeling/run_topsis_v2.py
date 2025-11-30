import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# --- 1. AYARLAR VE YOL BULMA ---
def setup_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))  # src/modeling
    project_root = os.path.dirname(os.path.dirname(current_dir))  # KararDestek2

    # Preprocessing modülünü bulabilmek için sys.path'e ekle
    src_path = os.path.join(project_root, "src")
    if src_path not in sys.path:
        sys.path.append(src_path)

    return project_root


project_root = setup_paths()

# Preprocess fonksiyonunu import et
try:
    from data_preprocessing.preprocess_diabetes import preprocess_diabetes_dataset
except ImportError:
    print("[HATA] 'preprocess_diabetes_dataset' fonksiyonu bulunamadı. Lütfen sys.path ayarını kontrol et.")
    sys.exit(1)


# --- 2. TOPSIS FONKSİYONU ---
def calculate_topsis(df, criteria_cols, weights, benefit_mask):
    """
    df: Değerlendirilecek veri seti (DataFrame)
    criteria_cols: Kriter sütun isimleri listesi
    weights: Ağırlıklar listesi (Toplamı 1 olmalı)
    benefit_mask: [True, True, False...] (True: Yüksek değer iyi/riskli, False: Düşük değer iyi/riskli)
    """
    # Veriyi seç ve normalize et (Vector Normalization)
    data = df[criteria_cols].values

    # Paydayı hesapla (Karekök toplamı)
    norm = np.sqrt((data ** 2).sum(axis=0))
    normalized_data = data / (norm + 1e-9)  # 0'a bölünmeyi önle

    # Ağırlıklandır
    weighted_data = normalized_data * weights

    # İdeal ve Anti-İdeal Çözümleri Bul
    ideal_solution = []
    anti_ideal_solution = []

    for i, is_benefit in enumerate(benefit_mask):
        if is_benefit:  # Yüksek değer riskliyse (Benefit)
            ideal_solution.append(np.max(weighted_data[:, i]))
            anti_ideal_solution.append(np.min(weighted_data[:, i]))
        else:  # Düşük değer riskliyse (Cost)
            ideal_solution.append(np.min(weighted_data[:, i]))
            anti_ideal_solution.append(np.max(weighted_data[:, i]))

    ideal_solution = np.array(ideal_solution)
    anti_ideal_solution = np.array(anti_ideal_solution)

    # Uzaklıkları Hesapla (Öklid)
    dist_to_ideal = np.sqrt(((weighted_data - ideal_solution) ** 2).sum(axis=1))
    dist_to_anti_ideal = np.sqrt(((weighted_data - anti_ideal_solution) ** 2).sum(axis=1))

    # Skor Hesapla
    topsis_score = dist_to_anti_ideal / (dist_to_ideal + dist_to_anti_ideal + 1e-9)
    return topsis_score


# --- 3. GRAFİK OLUŞTURMA FONKSİYONU ---
def generate_plots(df, output_dir):
    """
    Rapor için gerekli grafikleri oluşturur ve kaydeder.
    """
    print("[INFO] Grafikler oluşturuluyor...")
    sns.set_style("whitegrid")

    # Klasör yoksa oluştur
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Veriyi Görselleştirme İçin Hazırla
    df['Readmitted_Binary'] = df['actual_readmission_text'].apply(lambda x: 'Hayır' if x == 'NO' else 'Evet')

    # --- GRAFİK 1: XGBoost Risk vs TOPSIS Skor (Scatter Plot) ---
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x='xgboost_risk_score',
        y='topsis_score',
        hue='Readmitted_Binary',
        palette={'Evet': '#d62728', 'Hayır': '#1f77b4'},  # Kırmızı ve Mavi
        alpha=0.7,
        s=60
    )
    plt.title("Yapay Zeka Riski vs. Karar Destek Puanı (AHP Destekli)", fontsize=14)
    plt.xlabel("XGBoost Risk Olasılığı", fontsize=12)
    plt.ylabel("AHP-TOPSIS Öncelik Skoru", fontsize=12)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "1_scatter_risk_vs_topsis_ahp.png"), dpi=300)
    plt.close()

    # --- GRAFİK 2: Top N Başarı Analizi (Bar Chart) ---
    top_n = 100

    # Sadece Model ile sıralama
    top_xgb = df.sort_values(by='xgboost_risk_score', ascending=False).head(top_n)
    hits_xgb = top_xgb[top_xgb['Readmitted_Binary'] == 'Evet'].shape[0]

    # TOPSIS ile sıralama
    top_topsis = df.sort_values(by='topsis_score', ascending=False).head(top_n)
    hits_topsis = top_topsis[top_topsis['Readmitted_Binary'] == 'Evet'].shape[0]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(['Sadece Yapay Zeka (XGBoost)', 'Hibrit Sistem (AHP+TOPSIS)'], [hits_xgb, hits_topsis],
                   color=['gray', '#2ca02c'])
    plt.title(f"En Riskli {top_n} Hastada Yakalanan Gerçek Vaka Sayısı", fontsize=14)
    plt.ylabel("Doğru Tespit Sayısı", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, int(yval), ha='center', va='bottom', fontweight='bold',
                 fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "2_success_comparison_ahp.png"), dpi=300)
    plt.close()

    # --- GRAFİK 3: TOPSIS Skorunun Duruma Göre Dağılımı (Box Plot) ---
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='Readmitted_Binary', y='topsis_score', palette={'Evet': '#d62728', 'Hayır': '#1f77b4'})
    plt.title("Gerçek Duruma Göre TOPSIS Skor Dağılımı", fontsize=14)
    plt.xlabel("Gerçekte Tekrar Yattı mı?", fontsize=12)
    plt.ylabel("AHP-TOPSIS Skoru", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "3_boxplot_topsis_distribution_ahp.png"), dpi=300)
    plt.close()

    print(f"[INFO] Grafikler kaydedildi: {plot_dir}")


def main():
    print("--- KLİNİK KARAR DESTEK SİSTEMİ (XGBoost + AHP + TOPSIS) ---")

    # A. Dosya Yolları
    model_path = os.path.join(project_root, "src", "modeling", "models", "models", "xgb_weighted_f2", "model.joblib")
    csv_path = os.path.join(project_root, "data", "raw", "diabetic_data.csv")

    if not os.path.exists(model_path):
        model_path = os.path.join(project_root, "src", "modeling", "models", "xgb_weighted_f2", "model.joblib")

    print(f"[INFO] Model Yolu: {model_path}")

    # B. Modeli Yükle
    if not os.path.exists(model_path):
        print(f"[HATA] Model dosyası bulunamadı! Path: {model_path}")
        return

    saved_data = joblib.load(model_path)
    model = saved_data["model"]
    feature_names = saved_data["feature_names"]

    print("[INFO] Model başarıyla yüklendi.")

    # C. Veriyi Hazırla
    print("[INFO] Model girdileri hazırlanıyor (Preprocess)...")
    _, _, _, _, X_test_proc, y_test, _ = preprocess_diabetes_dataset(csv_path)
    print(f"[INFO] İşlenmiş Test Seti Boyutu: {X_test_proc.shape[0]} satır")

    # 2. Orijinal Veri Eşleştirme
    print("[INFO] Orijinal hasta verileri eşleştiriliyor...")
    df_raw = pd.read_csv(csv_path)
    df_raw = df_raw.replace('?', np.nan)
    df_raw = df_raw[~df_raw['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21])]
    df_raw = df_raw.sort_values('encounter_id')
    df_raw = df_raw.drop_duplicates(subset=['patient_nbr'], keep='first')

    df_train, df_temp = train_test_split(df_raw, test_size=0.3, random_state=42, stratify=df_raw['readmitted'])
    df_valid, df_test_original = train_test_split(df_temp, test_size=0.5, random_state=42,
                                                  stratify=df_temp['readmitted'])

    print(f"[INFO] Eşleştirilen Orijinal Veri Boyutu: {df_test_original.shape[0]} satır")

    if X_test_proc.shape[0] != df_test_original.shape[0]:
        print("\n[KRİTİK HATA] Satır sayıları uyuşmuyor!")
        return

    # Tahmin Al
    print("[INFO] Risk skorları hesaplanıyor...")
    if hasattr(X_test_proc, "columns"):
        for col in feature_names:
            if col not in X_test_proc.columns:
                X_test_proc[col] = 0
        X_test_aligned = X_test_proc[feature_names]
        y_pred_prob = model.predict_proba(X_test_aligned)[:, 1]
    else:
        y_pred_prob = model.predict_proba(X_test_proc)[:, 1]

    # D. TOPSIS Veri Seti
    # D. TOPSIS Veri Seti
    # C1–C5 için gerekli kolonları raw test veri setinden al
    cols_needed = [
        'encounter_id',
        'patient_nbr',
        'time_in_hospital',
        'number_emergency',
        'number_inpatient',
        'number_outpatient',
        'num_procedures',
        'number_diagnoses',
        'A1Cresult',
        # Eğer daha önce preprocess'te oluşturduğun diag_1_group_ord kolonları raw'a eklenmişse:
        # 'diag_1_group_ord', 'diag_2_group_ord', 'diag_3_group_ord'
        # yoksa aşağıda basit bir proxy kullanacağız.
    ]
    # Sadece var olan kolonları al (eksik varsa hata vermesin diye filtreliyoruz)
    cols_existing = [c for c in cols_needed if c in df_test_original.columns]
    df_topsis = df_test_original[cols_existing].copy()

    # C1: XGBoost risk skoru
    df_topsis['xgboost_risk_score'] = y_pred_prob

    # Gerçek readmission label'ını da saklayalım
    df_topsis['actual_readmission_text'] = df_test_original['readmitted']

    # C2: Total visits = inpatient + emergency + outpatient
    for col in ['number_inpatient', 'number_emergency', 'number_outpatient']:
        if col not in df_topsis.columns:
            raise ValueError(f"TOPSIS için {col} sütunu eksik, lütfen df_test_original içinde olduğundan emin olun.")

    df_topsis['total_visits'] = (
            df_topsis['number_inpatient'] +
            df_topsis['number_emergency'] +
            df_topsis['number_outpatient']
    )

    # C4: A1Cresult_ord (0: None, 1: Norm, 2: >7, 3: >8)
    if 'A1Cresult' in df_topsis.columns:
        a1c_map = {
            'None': 0,
            'Norm': 1,
            '>7': 2,
            '>8': 3
        }
        df_topsis['A1Cresult_ord'] = df_topsis['A1Cresult'].map(a1c_map).fillna(0).astype(int)
    else:
        raise ValueError("TOPSIS için 'A1Cresult' sütunu bulunamadı.")

    # C5: Tanı ağırlığı (diag_1_group_ord + diag_2_group_ord + diag_3_group_ord)
    # Eğer raw içinde diag_*_group_ord yoksa, basit proxy olarak number_diagnoses kullanabiliriz.
    if all(col in df_topsis.columns for col in ['diag_1_group_ord', 'diag_2_group_ord', 'diag_3_group_ord']):
        df_topsis['diag_severity'] = (
                df_topsis['diag_1_group_ord'] +
                df_topsis['diag_2_group_ord'] +
                df_topsis['diag_3_group_ord']
        )
    else:
        # Proxy çözüm: tanı sayısını komorbidite göstergesi olarak kullan
        if 'number_diagnoses' not in df_topsis.columns:
            raise ValueError(
                "Tanı ağırlığı için ne diag_*_group_ord ne de number_diagnoses bulundu. "
                "Lütfen preprocess ile bu alanları ekleyin."
            )
        df_topsis['diag_severity'] = df_topsis['number_diagnoses']

    print("[INFO] TOPSIS hesaplanıyor (AHP Ağırlıkları ile)...")

    # --- AHP ENTEGRASYONU ---
    # C1: xgboost_risk_score
    # C2: total_visits
    # C3: time_in_hospital
    # C4: A1Cresult_ord
    # C5: diag_severity

    criteria = [
        'xgboost_risk_score',
        'total_visits',
        'time_in_hospital',
        'A1Cresult_ord',
        'diag_severity'
    ]

    # AHP'den gelen ağırlıklar (örnek: w1=0.49, w2=0.23, w3=0.06, w4=0.09, w5=0.13)
    # Burada daha önce konuştuğumuz:
    # C1: 0.49 (ML risk)
    # C2: 0.23 (total visits)
    # C3: 0.06 (time in hospital)
    # C4: 0.09 (A1C)
    # C5: 0.13 (diag severity)
    weights = np.array([0.49, 0.23, 0.06, 0.09, 0.13], dtype=float)
    weights = weights / weights.sum()  # normalize, güvenlik için

    # Bu kriterlerin hepsi "daha büyük = daha riskli" olduğu için benefit=True
    benefit = [True, True, True, True, True]

    scores = calculate_topsis(df_topsis, criteria, weights, benefit)
    df_topsis['topsis_score'] = scores
    # E. Sıralama ve Rapor
    df_sorted = df_topsis.sort_values(by='topsis_score', ascending=False)

    print("\n" + "=" * 90)
    print("🚨 AHP-TOPSIS SONUCU: ACİL MÜDAHALE GEREKTİREN EN RİSKLİ 15 HASTA 🚨")
    print("=" * 90)

    top_15 = df_sorted.head(15)

    print(
        f"{'Patient ID':<12} | {'Risk(%)':<8} | {'Acil':<5} | {'Gün':<4} | {'İşlem':<5} | {'AHP-TOPSIS':<10} | {'Durum'}")
    print("-" * 100)
    for index, row in top_15.iterrows():
        print(
            f"{row['patient_nbr']:<12} | {row['xgboost_risk_score']:.4f}   | {row['number_emergency']:<5} | {row['time_in_hospital']:<4} | {row['num_procedures']:<5} | {row['topsis_score']:.4f}       | {row['actual_readmission_text']}")

    # F. Kaydet
    report_dir = os.path.join(project_root, "reports")
    output_path = os.path.join(report_dir, "patient_risk_ranking_ahp.xlsx")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_sorted.to_excel(output_path, index=False)
    print(f"\n[INFO] Rapor kaydedildi: {output_path}")

    # G. Grafikleri Oluştur
    generate_plots(df_sorted, report_dir)


if __name__ == "__main__":
    main()