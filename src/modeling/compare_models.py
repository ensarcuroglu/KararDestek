# Dosya: src/modeling/compare_models.py

import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# DÜZELTME BURADA: roc_auc_score EKLENDİ
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score
)


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


def main():
    print("--- MODEL KARŞILAŞTIRMA ANALİZİ BAŞLIYOR ---")

    # 1. Modellerin Yolları
    xgb_path = os.path.join(project_root, "src", "modeling", "models", "xgb_preprocessed_optuna", "model.joblib")
    rf_path = os.path.join(project_root, "models", "rf_preprocessed", "model.joblib")

    # Eğer RF ana dizindeyse oraya bak (Alternatif yol)
    if not os.path.exists(rf_path):
        rf_path = os.path.join(project_root, "models", "rf_preprocessed", "model.joblib")

    # 2. Veriyi Hazırla
    print("[INFO] Test verisi hazırlanıyor...")
    csv_path = os.path.join(project_root, "data", "raw", "diabetic_data.csv")
    _, _, _, _, X_test, y_test, feature_names = preprocess_diabetes_dataset(csv_path)

    # 3. Modelleri Yükle ve Tahmin Al
    results = {}

    # --- XGBoost ---
    if os.path.exists(xgb_path):
        print("[INFO] XGBoost modeli yükleniyor...")
        xgb_data = joblib.load(xgb_path)
        xgb_model = xgb_data["model"]

        # Sütun hizalama (XGBoost için kritik)
        if hasattr(X_test, "columns"):
            X_test_xgb = pd.DataFrame(0, index=X_test.index, columns=feature_names)
            common = list(set(X_test.columns) & set(feature_names))
            X_test_xgb[common] = X_test[common]
        else:
            X_test_xgb = X_test

        results['XGBoost'] = xgb_model.predict_proba(X_test_xgb)[:, 1]
    else:
        print(f"[UYARI] XGBoost modeli bulunamadı: {xgb_path}")

    # --- Random Forest ---
    if os.path.exists(rf_path):
        print("[INFO] Random Forest modeli yükleniyor...")
        rf_model = joblib.load(rf_path)
        # RF genelde sütun sırasına bakar, veri aynı preprocess'ten geçtiği için direkt veriyoruz
        results['Random Forest'] = rf_model.predict_proba(X_test)[:, 1]
    else:
        print(f"[UYARI] Random Forest modeli bulunamadı: {rf_path}")

    # 4. Grafikleri Oluştur
    output_dir = os.path.join(project_root, "reports", "model_comparison")
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")

    # --- GRAFİK 1: ROC Curve Overlay (Üst Üste) ---
    print("[INFO] Grafik 1: ROC Karşılaştırması çiziliyor...")
    plt.figure(figsize=(10, 8))

    colors = {'XGBoost': '#d62728', 'Random Forest': '#1f77b4'}  # Kırmızı vs Mavi

    for name, y_prob in results.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, color=colors.get(name, 'green'), label=f'{name} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Yanlış Alarm)', fontsize=12)
    plt.ylabel('True Positive Rate (Duyarlılık)', fontsize=12)
    plt.title('Model Karşılaştırması: ROC Eğrisi', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_roc_curve.png"), dpi=300)
    plt.close()

    # --- GRAFİK 2: Precision-Recall Curve Overlay ---
    print("[INFO] Grafik 2: PR Karşılaştırması çiziliyor...")
    plt.figure(figsize=(10, 8))

    for name, y_prob in results.items():
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        plt.plot(recall, precision, lw=2, color=colors.get(name, 'green'), label=f'{name} (AP = {pr_auc:.4f})')

    plt.xlabel('Recall (Yakalanan Hasta Oranı)', fontsize=12)
    plt.ylabel('Precision (Tahmin Keskinliği)', fontsize=12)
    plt.title('Model Karşılaştırması: Precision-Recall Eğrisi', fontsize=16)
    plt.legend(loc="upper right", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_pr_curve.png"), dpi=300)
    plt.close()

    # --- GRAFİK 3: Metrik Bazlı Bar Grafiği ---
    print("[INFO] Grafik 3: Metrik Tablosu çiziliyor...")

    metrics_data = []

    # Thresholdlar (Senin loglarından aldım)
    thresholds = {'XGBoost': 0.1077, 'Random Forest': 0.2004}

    for name, y_prob in results.items():
        thr = thresholds.get(name, 0.5)
        y_pred = (y_prob >= thr).astype(int)

        # Recall (Duyarlılık)
        rec = np.sum((y_test == 1) & (y_pred == 1)) / np.sum(y_test == 1)
        # Precision
        prec = np.sum((y_test == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
        # F1
        f1 = 2 * (prec * rec) / (prec + rec + 1e-9)
        # AUC
        roc_val = roc_auc_score(y_test, y_prob)

        metrics_data.append({'Model': name, 'Metrik': 'AUC Score', 'Değer': roc_val})
        metrics_data.append({'Model': name, 'Metrik': 'Recall (Duyarlılık)', 'Değer': rec})
        metrics_data.append({'Model': name, 'Metrik': 'F1 Score', 'Değer': f1})

    df_metrics = pd.DataFrame(metrics_data)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_metrics, x='Metrik', y='Değer', hue='Model', palette=colors)
    plt.title("Performans Metriklerinin Karşılaştırması", fontsize=16)
    plt.ylim(0, 0.8)

    # Barların üzerine değer yaz
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.3f', padding=3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_metrics_bar.png"), dpi=300)
    plt.close()

    print(f"\n[BAŞARILI] ✅ Karşılaştırma grafikleri kaydedildi: {output_dir}")
    print("1. comparison_roc_curve.png -> XGBoost'un eğrisinin daha üstte olduğunu gösterecek.")
    print("2. comparison_metrics_bar.png -> Recall farkını net bir şekilde gösterecek.")


if __name__ == "__main__":
    main()