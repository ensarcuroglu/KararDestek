# src/data_preprocessing/preprocess_diabetes.py

import os
from typing import Tuple, List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from .icd_mapping import map_icd9_to_group


def _map_age_to_midpoint(age_str: str) -> float:
    """[40-50) -> 45 gibi yaş aralıklarını midpoint'e çevirir."""
    age_map = {
        "[0-10)": 5,
        "[10-20)": 15,
        "[20-30)": 25,
        "[30-40)": 35,
        "[40-50)": 45,
        "[50-60)": 55,
        "[60-70)": 65,
        "[70-80)": 75,
        "[80-90)": 85,
        "[90-100)": 95,
    }
    return age_map.get(age_str, np.nan)


def _map_admission_type(v) -> str:
    """
    admission_type_id'yi makaledeki gibi daha az kategoriye indirger.
    1,2 -> Emergency / Urgent
    3   -> Elective
    4   -> Newborn
    diğer -> Other
    """
    try:
        v = int(v)
    except (TypeError, ValueError):
        return "Other"

    if v in (1, 2):
        return "Emergency"
    elif v == 3:
        return "Elective"
    elif v == 4:
        return "Newborn"
    else:
        return "Other"


def _map_admission_source(v) -> str:
    """
    admission_source_id'yi benzer kategorilerde toplar.
    1,2,3 -> Referral
    7     -> EmergencyRoom
    4,5,6,10,18,19,20,21 -> Transfer
    diğer -> Other
    """
    try:
        v = int(v)
    except (TypeError, ValueError):
        return "Other"

    if v in (1, 2, 3):
        return "Referral"
    elif v == 7:
        return "EmergencyRoom"
    elif v in (4, 5, 6, 10, 18, 19, 20, 21):
        return "Transfer"
    else:
        return "Other"


def _map_discharge_disposition(v) -> str:
    """
    discharge_disposition_id'yi daha az kategoriye indirger.
    1                -> Home
    6,8              -> HomeHealth
    3,4,5,12,15,16,
    23,24            -> SNF/ICF (bakım merkezleri)
    diğer            -> Other
    """
    try:
        v = int(v)
    except (TypeError, ValueError):
        return "Other"

    if v == 1:
        return "Home"
    elif v in (6, 8):
        return "HomeHealth"
    elif v in (3, 4, 5, 12, 15, 16, 23, 24):
        return "SNF_ICF"
    else:
        return "Other"


def preprocess_diabetes_dataset(
    csv_path: str = "data/raw/diabetic_data.csv",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Liu (2024) makalesindeki tanıma yakın olacak şekilde
    Diabetes 130-US hospitals datasını işler ve SMOTE uygulanmış train set döner.

    Dönüş:
        X_train_res, y_train_res, X_valid, y_valid, X_test, y_test, feature_names
    """

    print(f"[INFO] Veri yükleniyor: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"[INFO] Ham veri şekli: {df.shape}")

    # '?' -> NaN
    df = df.replace("?", np.nan)

    # 1) Hedef değişken: <30 vs (>30 + NO)
    df["target"] = (df["readmitted"] == "<30").astype(int)

    # 2) Strack ve diğer çalışmalardaki gibi bazı discharge tiplerini tamamen çıkar
    invalid_discharge = [11, 13, 14, 19, 20, 21]
    before = df.shape[0]
    df = df[~df["discharge_disposition_id"].isin(invalid_discharge)].copy()
    print(f"[INFO] Discharge filtresi: {before} -> {df.shape[0]} satır kaldı")

    # 3) Aynı hastadan birden fazla kayıt varsa, encounter_id'ye göre son kaydı tut
    df = df.sort_values(["patient_nbr", "encounter_id"])
    before = df.shape[0]
    df = df.drop_duplicates(subset="patient_nbr", keep="last").copy()
    print(f"[INFO] Duplicate patient_nbr temizliği: {before} -> {df.shape[0]} satır kaldı")

    # 4) Yaş aralıklarını midpoint'e çevir
    df["age_mid"] = df["age"].apply(_map_age_to_midpoint).astype(float)

    # 5) Admission / Discharge / Source grupları
    df["admission_type_grp"] = df["admission_type_id"].apply(_map_admission_type)
    df["admission_source_grp"] = df["admission_source_id"].apply(_map_admission_source)
    df["discharge_disposition_grp"] = df["discharge_disposition_id"].apply(_map_discharge_disposition)

    # 6) Tanı sütunlarını grupla ve ordinal koda çevir
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[f"{col}_group"] = df[col].apply(map_icd9_to_group)

    for col in ["diag_1_group", "diag_2_group", "diag_3_group"]:
        cat = pd.Categorical(df[col])
        df[f"{col}_ord"] = cat.codes  # -1 NaN demek; birazdan imputing yapılacak

    # 7) Toplam aktif ilaç sayısı
    med_cols = [
        "metformin",
        "repaglinide",
        "nateglinide",
        "chlorpropamide",
        "glimepiride",
        "acetohexamide",
        "glipizide",
        "glyburide",
        "tolbutamide",
        "pioglitazone",
        "rosiglitazone",
        "acarbose",
        "miglitol",
        "troglitazone",
        "tolazamide",
        "examide",
        "citoglipton",
        "insulin",
        "glyburide-metformin",
        "glipizide-metformin",
        "glimepiride-pioglitazone",
        "metformin-rosiglitazone",
        "metformin-pioglitazone",
    ]
    med_cols = [c for c in med_cols if c in df.columns]
    if med_cols:
        med_tmp = (
            df[med_cols]
            .replace({"No": 0, "Steady": 1, "Up": 1, "Down": 1})
            .astype(float)
        )
        df["total_meds_active"] = med_tmp.sum(axis=1)
    else:
        df["total_meds_active"] = df["num_medications"]

    # 8) Service utilization: outpatient + emergency + inpatient
    df["service_utilization"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )

    # 9) max_glu_serum & A1C ordinal encoding
    glu_map = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
    a1c_map = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
    df["max_glu_serum_ord"] = df["max_glu_serum"].map(glu_map)
    df["A1Cresult_ord"] = df["A1Cresult"].map(a1c_map)

    # 10) Çok eksik veya anlamsız sütunları drop et
    drop_cols = [
        "weight",
        "payer_code",
        "medical_specialty",
        "age",
        "max_glu_serum",
        "A1Cresult",
        "diag_1",
        "diag_2",
        "diag_3",
    ]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)

    # İlaçların granular hallerini de kaldır (sadece total_meds_active kalsın)
    drop_med_cols = [c for c in med_cols if c in df.columns]
    if drop_med_cols:
        df = df.drop(columns=drop_med_cols)

    # 11) Modelde kullanılmayacak ID vb. sütunlar
    non_feature_cols = [
        "encounter_id",
        "patient_nbr",
        "readmitted",
        "target",
        "admission_type_id",
        "discharge_disposition_id",
        "admission_source_id",
    ]
    non_feature_cols = [c for c in non_feature_cols if c in df.columns]

    feature_cols = [c for c in df.columns if c not in non_feature_cols]
    print(f"[INFO] Feature sütun sayısı (ham): {len(feature_cols)}")

    X = df[feature_cols].copy()
    y = df["target"].values

    # 12) Sayısal / kategorik ayrımı
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]

    print("[INFO] Kategorik sütunlar:")
    print(f"       {cat_cols}")
    print("[INFO] Sayısal sütun sayısı:", len(num_cols))

    # 13) Eksik değer imputing
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    X_num = pd.DataFrame(
        num_imputer.fit_transform(X[num_cols]),
        columns=num_cols,
        index=X.index,
    )
    if cat_cols:
        X_cat = pd.DataFrame(
            cat_imputer.fit_transform(X[cat_cols]),
            columns=cat_cols,
            index=X.index,
        )
    else:
        X_cat = pd.DataFrame(index=X.index)

    # 14) One-Hot Encoding
    if not X_cat.empty:
        X_cat_oh = pd.get_dummies(X_cat, columns=cat_cols, drop_first=False)
        X_processed = pd.concat([X_num, X_cat_oh], axis=1)
    else:
        X_processed = X_num

    feature_names = list(X_processed.columns)
    print(f"[INFO] Nihai (one-hot sonrası) feature sayısı: {len(feature_names)}")

    # 15) Train / Valid / Test böl
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_processed,
        y,
        test_size=0.15,
        random_state=random_state,
        stratify=y,
    )

    # valid oranı: toplamın %15'i olacak şekilde ayarla
    valid_size_rel = 0.15 / (1.0 - 0.15)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_val,
        y_train_val,
        test_size=valid_size_rel,
        random_state=random_state,
        stratify=y_train_val,
    )

    print(f"[INFO] Train şekli (SMOTE öncesi): {X_train.shape}")
    print(f"[INFO] Valid şekli:               {X_valid.shape}")
    print(f"[INFO] Test şekli:                {X_test.shape}")

    # 16) Sayısal sütunlara StandardScaler (makale ile uyumlu)
    scaler = StandardScaler()
    X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
    X_valid.loc[:, num_cols] = scaler.transform(X_valid[num_cols])
    X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])

    # 17) SMOTE sadece train set üzerinde
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"[INFO] SMOTE sonrası train şekli: {X_train_res.shape}")
    print("[INFO] SMOTE sonrası sınıf dağılımı (train):")
    print(pd.Series(y_train_res).value_counts())

    return (
        X_train_res.values,
        y_train_res,
        X_valid.values,
        y_valid,
        X_test.values,
        y_test,
        feature_names,
    )
