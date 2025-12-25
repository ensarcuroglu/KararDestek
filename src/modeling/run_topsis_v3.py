import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Uyarıları kapatma
warnings.filterwarnings('ignore')

# DİZİN AYARLARI
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))

if project_root not in sys.path:
    sys.path.append(project_root)

print(f"[DEBUG] Proje Kök Dizini: {project_root}")

# Preprocess modülü
try:
    from src.data_preprocessing.preprocess_diabetes import preprocess_diabetes_dataset_for_training
except ImportError:
    sys.path.append(os.path.join(project_root, 'src'))
    from data_preprocessing.preprocess_diabetes import preprocess_diabetes_dataset_for_training

# Görselleştirme Ayarları
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['figure.figsize'] = (12, 8)


# V5 İÇİN GEREKLİ FEATURE ENGINEERING (UNUTMA!!!!!!)
def enhance_features(X, feature_names):
    X_enhanced = X.copy()
    try:
        fn_list = list(feature_names)
        idx_map = {
            'number_inpatient': -1, 'number_emergency': -1, 'number_outpatient': -1,
            'num_medications': -1, 'number_diagnoses': -1
        }
        for k in idx_map.keys():
            if k in fn_list: idx_map[k] = fn_list.index(k)

        new_feats = []
        if idx_map['number_inpatient'] != -1 and idx_map['number_emergency'] != -1 and idx_map[
            'number_outpatient'] != -1:
            service_util = X[:, idx_map['number_inpatient']] + X[:, idx_map['number_emergency']] + X[:, idx_map[
                                                                                                            'number_outpatient']]
            new_feats.append(service_util.reshape(-1, 1))
        if idx_map['num_medications'] != -1 and idx_map['number_diagnoses'] != -1:
            complexity = X[:, idx_map['num_medications']] * X[:, idx_map['number_diagnoses']]
            new_feats.append(complexity.reshape(-1, 1))
        if idx_map['number_emergency'] != -1:
            emergency_sq = X[:, idx_map['number_emergency']] ** 2
            new_feats.append(emergency_sq.reshape(-1, 1))

        if new_feats:
            X_new = np.hstack(new_feats)
            X_enhanced = np.hstack([X_enhanced, X_new])

    except Exception as e:
        print(f"[UYARI] Feature Engineering hatası: {e}")

    return X_enhanced


# TOPSIS
def calculate_topsis(df, weights, impacts):

    data_matrix = df.values.astype(np.float64)
    weights_arr = np.array(weights, dtype=np.float64)

    rms = np.sqrt((data_matrix ** 2).sum(axis=0))
    norm_matrix = np.divide(data_matrix, rms, out=np.zeros_like(data_matrix), where=rms != 0)

    weighted_matrix = norm_matrix * weights_arr

    # İdeal Çözümleri aldık
    ideal_best = []
    ideal_worst = []
    num_cols = weighted_matrix.shape[1]

    for i in range(num_cols):
        col_data = weighted_matrix[:, i]
        if impacts[i] == '+':
            ideal_best.append(col_data.max())
            ideal_worst.append(col_data.min())
        else:
            ideal_best.append(col_data.min())
            ideal_worst.append(col_data.max())

    ideal_best_arr = np.array(ideal_best, dtype=np.float64)
    ideal_worst_arr = np.array(ideal_worst, dtype=np.float64)

    # Uzaklıklar
    s_plus = np.sqrt(((weighted_matrix - ideal_best_arr) ** 2).sum(axis=1))
    s_minus = np.sqrt(((weighted_matrix - ideal_worst_arr) ** 2).sum(axis=1))

    # Skor
    denominator = s_plus + s_minus
    scores = np.divide(s_minus, denominator, out=np.zeros_like(s_minus), where=denominator != 0)

    return scores


# GÖRSELLEŞTİRME
def create_advanced_visuals(df, output_dir, feature_cols):

    # Sunum için grafikler.

    df['Model_Probability'] = df['Model_Probability'].astype(float)
    df['TOPSIS_Score'] = df['TOPSIS_Score'].astype(float)
    df['Actual_Label'] = df['Actual_Label'].astype(int)

    # Kümülatif kazanç
    plt.figure(figsize=(12, 7))

    df_model = df.sort_values(by='Model_Probability', ascending=False).reset_index()
    df_topsis = df.sort_values(by='TOPSIS_Score', ascending=False).reset_index()

    cum_model = df_model['Actual_Label'].cumsum()
    cum_topsis = df_topsis['Actual_Label'].cumsum()

    total_pos = df['Actual_Label'].sum()
    x_axis = np.arange(1, len(df) + 1)
    random_line = x_axis * (total_pos / len(df))

    limit = 1000
    plt.plot(x_axis[:limit], cum_topsis[:limit], label='TOPSIS Sıralaması', color='#2ecc71', linewidth=3)
    plt.plot(x_axis[:limit], cum_model[:limit], label='Saf Model Sıralaması', color='#3498db', linestyle='--',
             linewidth=2.5)
    plt.plot(x_axis[:limit], random_line[:limit], label='Rastgele Seçim', color='gray', linestyle=':', alpha=0.7)

    plt.title('Klinik Etki Analizi: TOPSIS vs Model (İlk 1000 Hasta)', fontsize=16, fontweight='bold')
    plt.xlabel('Kontrol Edilen Hasta Sayısı', fontsize=12)
    plt.ylabel('Yakalanan Gerçek Hasta Sayısı', fontsize=12)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.savefig(os.path.join(output_dir, "1_Performance_Comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Heatmap
    top_100_model_idx = df_model.head(100).index
    top_100_topsis_idx = df_topsis.head(100).index

    # sadece sayısal karşılaştırma sütunlarını aldık
    mean_model = df_model.iloc[:100][feature_cols].astype(float).mean()
    mean_topsis = df_topsis.iloc[:100][feature_cols].astype(float).mean()

    comparison_df = pd.DataFrame({'Saf Model (Top 100)': mean_model, 'TOPSIS (Top 100)': mean_topsis})

    comparison_df = comparison_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    # Normalizasyon
    normalized_df = (comparison_df - comparison_df.min()) / (comparison_df.max() - comparison_df.min() + 1e-9)
    normalized_df = normalized_df.fillna(0).astype(float)

    plt.figure(figsize=(10, 6))
    sns.heatmap(normalized_df, annot=comparison_df.round(2), fmt=".2f", cmap="YlOrRd", linewidths=.5)

    plt.title('Neden TOPSIS? - En Riskli 100 Hastanın Özellik Karşılaştırması', fontsize=15)
    plt.ylabel('Klinik Özellikler')
    plt.savefig(os.path.join(output_dir, "2_Feature_Comparison_Heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # Skor Dağılımı
    plt.figure(figsize=(12, 6))
    sns.kdeplot(data=df, x='Model_Probability', fill=True, label='Saf Model Olasılığı', color='blue', alpha=0.3)
    sns.kdeplot(data=df, x='TOPSIS_Score', fill=True, label='TOPSIS Skoru', color='green', alpha=0.3)
    plt.title('Risk Skoru Dağılımı: Model vs TOPSIS', fontsize=16)
    plt.xlabel('Hesaplanan Skor (0-1 Arası)')
    plt.ylabel('Yoğunluk (Density)')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "3_Score_Distribution.png"), dpi=300, bbox_inches='tight')
    plt.close()


# MAIN --------------------------
def main():
    print("\n" + "=" * 70)
    print("TOPSIS KARAR DESTEK ANALİZİ - SUNUM MODU (V5 Final)")
    print("=" * 70 + "\n")

    model_path = os.path.join(project_root, "models", "xgb_v5_percentile", "model.joblib")
    csv_path = os.path.join(project_root, "data", "raw", "diabetic_data.csv")
    output_dir = os.path.join(project_root, "reports", "topsis_final_presentation")
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(model_path):
        print(f"[HATA] Model bulunamadı: {model_path}")
        return

    print(f"[INFO] Model Yükleniyor... OK")
    packet = joblib.load(model_path)
    model = packet["model"]
    feature_names = packet.get("feature_names", model.get_booster().feature_names)

    print(f"[INFO] Veri Seti İşleniyor...")
    _, _, _, _, X_test, y_test, _, _, _, _, _ = preprocess_diabetes_dataset_for_training(csv_path=csv_path)

    # FEATURE ENGINEERING
    if hasattr(model, "n_features_in_") and X_test.shape[1] != model.n_features_in_:
        base_feature_names = [f for f in feature_names if "score" not in f and "intensity" not in f]
        if not base_feature_names: base_feature_names = feature_names
        X_test_enhanced = enhance_features(X_test, base_feature_names)
    else:
        X_test_enhanced = X_test

    # TAHMİN
    print("[INFO] Model Olasılıkları Hesaplanıyor...")
    y_prob = model.predict_proba(X_test_enhanced)[:, 1]

    # TOPSIS
    target_cols = ['number_inpatient', 'number_emergency', 'time_in_hospital', 'num_medications']

    col_indices = []
    found_cols = []
    for col in target_cols:
        for i, feat in enumerate(feature_names):
            if feat == col:
                col_indices.append(i)
                found_cols.append(col)
                break

    selected_data = X_test_enhanced[:, col_indices]
    topsis_df = pd.DataFrame(selected_data, columns=found_cols)

    # DÜZELTME: float hatası için eklendi(ensar)
    topsis_df = topsis_df.astype(float)

    topsis_df['Model_Probability'] = y_prob
    topsis_df['Actual_Label'] = y_test

    # TOPSIS
    print("[INFO] TOPSIS Algoritması Çalıştırılıyor...")
    weights = [0.15, 0.15, 0.10, 0.10, 0.50]
    impacts = ['+', '+', '+', '+', '+']

    cols_for_topsis = found_cols + ['Model_Probability']
    topsis_df['TOPSIS_Score'] = calculate_topsis(topsis_df[cols_for_topsis], weights, impacts)

    # GÖRSELLEŞTİRME
    print(f"[INFO] Profesyonel Grafikler Üretiliyor: {output_dir}")
    create_advanced_visuals(topsis_df, output_dir, found_cols)

    print("\n" + "=" * 50)
    print("ÖZET")
    print("=" * 50)

    df_sorted_model = topsis_df.sort_values(by='Model_Probability', ascending=False)
    df_sorted_topsis = topsis_df.sort_values(by='TOPSIS_Score', ascending=False)

    k_values = [10, 50, 100, 200]
    print(f"{'Metric':<20} | {'Model (AI Only)':<15} | {'TOPSIS (AI + MCDM)':<15} | {'Fark (Lift)':<10}")
    print("-" * 70)

    for k in k_values:
        prec_model = df_sorted_model.head(k)['Actual_Label'].sum()
        prec_topsis = df_sorted_topsis.head(k)['Actual_Label'].sum()
        diff = prec_topsis - prec_model
        print(f"Top {k:<16} | {prec_model:<15} | {prec_topsis:<15} | {diff:+d}")

    print("-" * 70)

    result_path = os.path.join(output_dir, "final_risk_ranking.csv")
    df_sorted_topsis.to_csv(result_path, index=False)
    print(f"\nTüm işlemler tamamlandı.")
    print(f"Raporlar ve Grafikler: {output_dir}")


if __name__ == "__main__":
    main()