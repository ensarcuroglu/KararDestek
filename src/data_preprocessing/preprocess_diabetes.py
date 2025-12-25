# src/data_preprocessing/preprocess_diabetes.py

import numpy as np
import pandas as pd
from typing import Tuple, List, Any
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from src.data_preprocessing.icd_mapping import map_icd9_to_group

pd.set_option('future.no_silent_downcasting', True)


def _map_age_to_midpoint(age_str: str) -> float:
    age_map = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
        "[80-90)": 85, "[90-100)": 95,
    }
    return age_map.get(age_str, np.nan)


def _map_admission_type(v) -> str:
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


def transform_features_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # '?' -> NaN
    df = df.replace("?", np.nan)

    if 'age' in df.columns:
        df["age_mid"] = df["age"].apply(_map_age_to_midpoint).astype(float)

    # ID Gruplamaları
    if 'admission_type_id' in df.columns:
        df["admission_type_grp"] = df["admission_type_id"].apply(_map_admission_type)
    if 'admission_source_id' in df.columns:
        df["admission_source_grp"] = df["admission_source_id"].apply(_map_admission_source)
    if 'discharge_disposition_id' in df.columns:
        df["discharge_disposition_grp"] = df["discharge_disposition_id"].apply(_map_discharge_disposition)

    # ICD Kodları ve Gruplama
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[f"{col}_group"] = df[col].astype(str).apply(map_icd9_to_group)

    # İlaç Sayısı Hesaplama
    med_cols = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
        "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
        "examide", "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
        "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
    ]
    available_meds = [c for c in med_cols if c in df.columns]

    if available_meds:
        med_tmp = df[available_meds].replace({"No": 0, "Steady": 1, "Up": 1, "Down": 1}).infer_objects(copy=False)
        med_tmp = med_tmp.astype(float)
        df["total_meds_active"] = med_tmp.sum(axis=1)
    elif "num_medications" in df.columns:
        df["total_meds_active"] = df["num_medications"]

    df["service_utilization"] = (
            df.get("number_outpatient", 0) +
            df.get("number_emergency", 0) +
            df.get("number_inpatient", 0)
    )

    glu_map = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
    a1c_map = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}

    if "max_glu_serum" in df.columns:
        df["max_glu_serum_ord"] = df["max_glu_serum"].map(glu_map)
    if "A1Cresult" in df.columns:
        df["A1Cresult_ord"] = df["A1Cresult"].map(a1c_map)


    potential_drop_cols = [
                              "weight", "payer_code", "medical_specialty", "age", "max_glu_serum", "A1Cresult",
                              "diag_1", "diag_2", "diag_3", "encounter_id", "patient_nbr",
                              "admission_type_id", "discharge_disposition_id", "admission_source_id"
                          ] + available_meds

    existing_drop = [c for c in potential_drop_cols if c in df.columns]
    df = df.drop(columns=existing_drop)

    return df

def preprocess_diabetes_dataset_for_training(
        csv_path: str = "data/raw/diabetic_data.csv",
        random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], Any, Any, Any]:
    print(f"[INFO] Veri yükleniyor: {csv_path}")
    df = pd.read_csv(csv_path)

    df["target"] = (df["readmitted"] == "<30").astype(int)

    invalid_discharge = [11, 13, 14, 19, 20, 21]
    df = df[~df["discharge_disposition_id"].isin(invalid_discharge)]
    df = df.sort_values(["patient_nbr", "encounter_id"])
    df = df.drop_duplicates(subset="patient_nbr", keep="last")

    y = df["target"].values

    df_features = transform_features_base(df)

    cols_to_drop = ["readmitted", "target"]
    df_features = df_features.drop(columns=[c for c in cols_to_drop if c in df_features.columns])

    cat_cols = df_features.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in df_features.columns if c not in cat_cols]

    num_imputer = SimpleImputer(strategy="median")
    cat_imputer = SimpleImputer(strategy="most_frequent")

    X_num = pd.DataFrame(num_imputer.fit_transform(df_features[num_cols]), columns=num_cols, index=df_features.index)

    if cat_cols:
        X_cat = pd.DataFrame(cat_imputer.fit_transform(df_features[cat_cols]), columns=cat_cols,
                             index=df_features.index)
        X_cat_oh = pd.get_dummies(X_cat, columns=cat_cols, drop_first=False)
    else:
        X_cat_oh = pd.DataFrame(index=df_features.index)

    X_processed = pd.concat([X_num, X_cat_oh], axis=1)
    feature_names = list(X_processed.columns)

    print(f"[DEBUG] X shape: {X_processed.shape}, y shape: {y.shape}")
    if len(X_processed) != len(y):
        raise ValueError(f"Feature ve Target boyutları uyuşmuyor! X: {len(X_processed)}, y: {len(y)}")

    X_train_val, X_test, y_train_val, y_test = train_test_split(X_processed, y, test_size=0.15,
                                                                random_state=random_state, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train_val, y_train_val, test_size=0.176,
                                                          random_state=random_state, stratify=y_train_val)

    # Scaling
    scaler = StandardScaler()
    X_train.loc[:, num_cols] = scaler.fit_transform(X_train[num_cols])
    X_valid.loc[:, num_cols] = scaler.transform(X_valid[num_cols])
    X_test.loc[:, num_cols] = scaler.transform(X_test[num_cols])

    # SMOTE
    smote = SMOTE(random_state=random_state)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    return (
        X_train_res.values, y_train_res,
        X_valid.values, y_valid,
        X_test.values, y_test,
        feature_names,
        scaler,
        num_cols,
        num_imputer,
        cat_imputer
    )