import sys
import os

# DÜZELTME (ömer)
# Eski isim: preprocess_diabetes_dataset
# Yeni isim: preprocess_diabetes_dataset_for_training
from src.data_preprocessing.preprocess_diabetes import preprocess_diabetes_dataset_for_training

if __name__ == "__main__":
    # Dosya yolunu kendine göre ayarla
    my_csv_path = r"D:\Ensar Dosya\KararDestek\KararDestek2\data\raw\diabetic_data.csv"

    print("🚀 Preprocessing işlemi başlatılıyor...")

    # Fonksiyon ismi aşağıda da güncellendi (unutma)
    (
        X_train_res,
        y_train_res,
        X_valid,
        y_valid,
        X_test,
        y_test,
        feature_names,
        scaler,
        num_cols,
        num_imputer,
        cat_imputer
    ) = preprocess_diabetes_dataset_for_training(csv_path=my_csv_path)

    print("\nPreprocess başarıyla tamamlandı!")
    print("-" * 30)
    print(f"Train (SMOTE)  : {X_train_res.shape}")
    print(f"Valid          : {X_valid.shape}")
    print(f"Test           : {X_test.shape}")
    print("-" * 30)
    print(f"Scaler Durumu  : {'Yüklendi' if scaler else 'Yok'}")
    print(f"Num Imputer    : {'Yüklendi' if num_imputer else 'Yok'}")
    print(f"Cat Imputer    : {'Yüklendi' if cat_imputer else 'Yok'}")