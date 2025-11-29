"""
train_xgboost_optuna_smote.py

- UCI Diabetes 130-US Hospitals datası ile:
  - Veri sızıntısı önleme
  - Özellik mühendisliği
  - SMOTENC ile sınıf dengesi
  - Optuna ile XGBoost hiperparametre optimizasyonu
  - F1, Accuracy, Precision, Recall ve ROC-AUC metrikleri
"""

import os
import numpy as np
import pandas as pd
import joblib
import optuna

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)

from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTENC


# === Dosya yolları ===
RAW_PATH = r"C:\Users\omerc\PycharmProjects\PythonProject1\data\raw\diabetic_data.csv"


def build_preprocessor(num_cols, cat_cols):
    """
    Model Pipeline'ında kullanılacak son preprocess (OneHot vs.).
    - Sayısal: median ile imputasyon
    - Kategorik: most_frequent imputasyon + OneHotEncoder
    """
    return ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols)
        ]
    )


def prepare_for_smote(X_train, num_cols, cat_cols):
    """
    SMOTENC öncesi sayısal ve kategorik sütunları dönüştürür.
    - Sayısallar: median ile NaN doldurulur
    - Kategorikler: "MISSING" ile doldurulur, sonra .codes ile integer'a çevrilir
    """
    Xc = X_train.copy()

    # Eksik değerleri doldur
    if num_cols:
        medians = Xc[num_cols].median()
        Xc[num_cols] = Xc[num_cols].fillna(medians)
    if cat_cols:
        Xc[cat_cols] = Xc[cat_cols].fillna("MISSING")

    # Kategorikleri integer koda çevir
    code_maps = {}
    for col in cat_cols:
        cat = pd.Categorical(Xc[col].astype(str))
        Xc[col] = cat.codes
        code_maps[col] = list(cat.categories)

    cat_indices = [Xc.columns.get_loc(c) for c in cat_cols]
    return Xc, cat_indices, code_maps


def restore_from_smote(X_res_df, cat_cols, code_maps):
    """
    SMOTENC sonrası integer kodları tekrar orijinal string etiketlere çevirir.
    (Pipeline için tekrar kategorik forma döndürmemiz gerekiyor.)
    """
    Xr = X_res_df.copy()
    for col in cat_cols:
        cats = code_maps[col]
        Xr[col] = Xr[col].astype(int).map(
            lambda i: cats[i] if 0 <= i < len(cats) else "MISSING"
        )
    return Xr


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Özellik mühendisliği:
     - age_gender_combo
     - total_visits
     - med_lab_ratio
     - long_stay
    """
    df = df.copy()
    # Bazı sütunlar string gelebilir, garantiye alalım
    df["age"] = df["age"].astype(str)
    df["gender"] = df["gender"].astype(str)

    df["age_gender_combo"] = df["age"] + "_" + df["gender"]
    df["total_visits"] = (
        df["number_inpatient"] +
        df["number_outpatient"] +
        df["number_emergency"]
    )
    df["med_lab_ratio"] = df["num_medications"] / (df["num_lab_procedures"] + 1)
    df["long_stay"] = (df["time_in_hospital"] > 10).astype(int)
    return df


def load_and_prepare_raw():
    """
    Ham csv'den tüm veriyi yükler, temel temizlik + feature engineering yapar.
    Train/val split'i bu fonksiyondan sonra yapılır.
    """
    print("[INFO] Ham veri yükleniyor...")
    df = pd.read_csv(RAW_PATH)

    print(f"[INFO] Ham veri şekli: {df.shape}")

    # Ölüm/hospice vb. taburcu olanları çıkaralım (discharge_disposition_id 11,13,14,19,20,21)
    remove_discharge_codes = [11, 13, 14, 19, 20, 21]
    before_rows = df.shape[0]
    df = df[~df["discharge_disposition_id"].isin(remove_discharge_codes)].copy()
    print(f"[INFO] discharge filtresi: {before_rows} -> {df.shape[0]} satır kaldı")

    # Hedef sütun: readmitted_binary (<30 -> 1, diğerleri 0)
    TARGET_COL = "readmitted"
    df["readmitted_binary"] = df[TARGET_COL].apply(
        lambda x: 1 if x == "<30" else 0
    )

    # Özellik mühendisliği
    print("[INFO] Özellik mühendisliği uygulanıyor...")
    df = add_feature_engineering(df)

    # Veri sızıntısı olabilecek sütunları çıkar
    leakage_cols = [
        c for c in df.columns
        if ("readmit" in c.lower()) or ("target" in c.lower())
    ]
    print(f"[INFO] Veri sızıntısı önleme aktif. {len(leakage_cols)} sütun çıkarılacak: {leakage_cols}")

    # Çok eksik ve anlamsız bulduğumuz bazı sütunları çıkarabiliriz (opsiyonel)
    drop_cols = [
        "weight",
        "payer_code",
        "medical_specialty"
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]

    # X / y oluştur
    X = df.drop(columns=leakage_cols + drop_cols)
    y = df["readmitted_binary"]

    print(f"[INFO] Özellik sayısı (feature engineering sonrası, leakage drop sonrası): {X.shape[1]}")
    return X, y


def main(n_trials: int = 50):
    """
    Optuna + SMOTENC + XGBoost eğitim pipeline'ı.
    n_trials: Optuna deneme sayısı (daha yüksek -> daha iyi ama daha yavaş)
    """
    X, y = load_and_prepare_raw()

    # Train/Val böl
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Kategorik ve sayısal sütunlar
    cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
    num_cols = [c for c in X_train.columns if X_train[c].dtype != "object"]

    print(f"[INFO] Kategorik sütun sayısı: {len(cat_cols)}")
    print(f"[INFO] Sayısal sütun sayısı  : {len(num_cols)}")

    # SMOTENC hazırlığı
    X_train_smote, cat_indices, code_maps = prepare_for_smote(X_train, num_cols, cat_cols)
    preprocessor = build_preprocessor(num_cols, cat_cols)

    # === Optuna Objective ===
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 6.0),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
        }

        # SMOTENC ile sınıf dengeleme
        smote = SMOTENC(
            categorical_features=cat_indices,
            sampling_strategy=0.3,   # minority:majority oranı
            random_state=42
        )
        X_res, y_res = smote.fit_resample(X_train_smote.values, y_train.values)

        # Integer kodları geri kategorik string'e çevir
        X_res_df = pd.DataFrame(X_res, columns=X_train_smote.columns)
        X_res_restored = restore_from_smote(X_res_df, cat_cols, code_maps)

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                **params,
                random_state=42,
                eval_metric="auc",
                use_label_encoder=False
            ))
        ])

        model.fit(X_res_restored, y_res)
        preds = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, preds)
        return auc

    # === Optuna çalıştır ===
    study = optuna.create_study(direction="maximize")
    print("[INFO] Optuna + SMOTENC + Feature Engineering optimizasyonu başlatıldı...")
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=True)

    print("\n✅ En iyi parametreler:")
    print(study.best_params)
    print(f"✅ En iyi ROC-AUC (validation): {study.best_value:.4f}")

    # === Final model eğitimi ===
    best_params = study.best_params
    smote_final = SMOTENC(
        categorical_features=cat_indices,
        sampling_strategy=0.3,
        random_state=42
    )
    X_res_f, y_res_f = smote_final.fit_resample(X_train_smote.values, y_train.values)

    X_res_f_df = pd.DataFrame(X_res_f, columns=X_train_smote.columns)
    X_res_f_restored = restore_from_smote(X_res_f_df, cat_cols, code_maps)

    final_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            **best_params,
            random_state=42,
            eval_metric="auc",
            use_label_encoder=False
        ))
    ])

    print("[INFO] En iyi parametrelerle final model eğitiliyor...")
    final_model.fit(X_res_f_restored, y_res_f)

    val_preds_prob = final_model.predict_proba(X_val)[:, 1]
    val_preds = (val_preds_prob >= 0.5).astype(int)

    val_auc = roc_auc_score(y_val, val_preds_prob)
    val_f1 = f1_score(y_val, val_preds)
    val_acc = accuracy_score(y_val, val_preds)
    val_prec = precision_score(y_val, val_preds)
    val_rec = recall_score(y_val, val_preds)

    print("\n📊 Final Model Değerlendirme Sonuçları (Validation set):")
    print(f"ROC-AUC:   {val_auc:.4f}")
    print(f"Accuracy:  {val_acc:.4f}")
    print(f"F1 Score:  {val_f1:.4f}")
    print(f"Precision: {val_prec:.4f}")
    print(f"Recall:    {val_rec:.4f}")

    os.makedirs("models", exist_ok=True)
    out_path = "models/xgb_model_optuna_smote.joblib"
    joblib.dump(final_model, out_path)
    print(f"\n💾 Model kaydedildi: {out_path}")


if __name__ == "__main__":
    # n_trials değerini artırırsan (ör: 100, 150) daha iyi parametre bulur ama daha uzun sürer.
    main(n_trials=50)
