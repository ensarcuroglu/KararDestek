import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, '../data/raw/diabetic_data.csv')

print(f"Veri seti yükleniyor: {DATA_PATH}")

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print("HATA: 'diabetic_data.csv' dosyası bulunamadı.")
    sys.exit(1)

# Hedef Değişken Dönüşümü
df['target'] = (df['readmitted'] == '<30').astype(int)

cols_to_drop = ['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'readmitted']
df_clean = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

for col in df_clean.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))

df_clean.fillna(0, inplace=True)

X = df_clean.drop('target', axis=1)
y = df_clean['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# SMOTE Öncesi Sayılar
counts_before = y_train.value_counts().to_dict()
neg_before = counts_before.get(0, 0)
pos_before = counts_before.get(1, 0)
total_before = neg_before + pos_before


# SMOTE UYGULAMA

print("SMOTE uygulanıyor... (Lütfen bekleyiniz)")
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

counts_after = y_train_res.value_counts().to_dict()
neg_after = counts_after.get(0, 0)
pos_after = counts_after.get(1, 0)
total_after = neg_after + pos_after

data_viz = pd.DataFrame({
    'Durum': ['SMOTE Öncesi\n(Orijinal)', 'SMOTE Öncesi\n(Orijinal)',
              'SMOTE Sonrası\n(Sentetik)', 'SMOTE Sonrası\n(Sentetik)'],
    'Sınıf': ['Negatif (0)', 'Pozitif (1)', 'Negatif (0)', 'Pozitif (1)'],
    'Sayı': [neg_before, pos_before, neg_after, pos_after],
    'Yüzde': [neg_before / total_before * 100, pos_before / total_before * 100,
              neg_after / total_after * 100, pos_after / total_after * 100]
})

sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

custom_palette = ["#5DADE2", "#EC7063"]

fig, ax = plt.subplots(figsize=(12, 7))

# Bar Plot
bar_plot = sns.barplot(
    data=data_viz,
    x='Durum',
    y='Sayı',
    hue='Sınıf',
    palette=custom_palette,
    edgecolor='white',
    linewidth=1.5,
    width=0.6,
    alpha=0.9
)

plt.title('Şekil 3.2: Eğitim Veri Setinde SMOTE Uygulamasının Sınıf Dağılımına Etkisi',
          fontsize=16, fontweight='bold', pad=20, color='#2c3e50')
plt.xlabel('', fontsize=12)
plt.ylabel('Örnek Sayısı (Hasta Kaydı)', fontsize=12, labelpad=10)

plt.legend(title='Sınıf Durumu', title_fontsize='11', fontsize='10',
           loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)

for container in ax.containers:
    for bar in container:
        height = bar.get_height()
        if height > 0:
            center_x = bar.get_x() + bar.get_width() / 2
            is_after = center_x > 0.5

            total = total_after if is_after else total_before
            percentage = (height / total) * 100

            label_text = f"{int(height):,}\n(%{percentage:.1f})"

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + (height * 0.02),
                label_text,
                ha='center',
                va='bottom',
                fontsize=11,
                fontweight='bold',
                color='#34495e'
            )

ax.yaxis.grid(True, linestyle='--', alpha=0.7, color='#bdc3c7')
ax.xaxis.grid(False)
sns.despine(left=True, bottom=False)

# not eklendi metin hazırlandı (ömer)
note_text = (
    "Not: SMOTE (Synthetic Minority Over-sampling Technique) algoritması, azınlık sınıfı olan 'Pozitif (1)' sınıfını, \n"
    "çoğunluk sınıfı sayısına eşitleyecek şekilde sentetik veriler üreterek dengelemiştir. \n"
    "Bu işlem, modelin yanlı (bias) öğrenmesini engellemek amacıyla sadece eğitim setine uygulanmıştır."
)
plt.figtext(0.5, -0.02, note_text, ha="center", fontsize=10, style='italic', color='#7f8c8d')

plt.tight_layout()
plt.subplots_adjust(bottom=0.25)

print("Grafik oluşturuldu...")
plt.show()
