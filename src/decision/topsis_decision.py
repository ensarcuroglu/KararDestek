import numpy as np
import pandas as pd

# AHP ile belirlediğimiz ağırlıklar (C1..C5)
TOPSIS_WEIGHTS = {
    "ml_risk": 0.49,          # XGBoost'un verdiği olasılık
    "total_visits": 0.23,     # number_inpatient + number_emergency + number_outpatient
    "diag_severity": 0.13,    # burada number_diagnoses'i kullanacağız (komorbidite proxy)
    "A1Cresult_ord": 0.09,    # ordinal A1C sonucu
    "time_in_hospital": 0.06, # yatış süresi
}


def compute_total_visits(df: pd.DataFrame) -> pd.Series:
    """
    number_inpatient + number_emergency + number_outpatient ---> total_visits
    """
    cols = ["number_inpatient", "number_emergency", "number_outpatient"]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"DataFrame'de {c} sütunu yok, TOPSIS için gerekli.")
    return df[cols].sum(axis=1)


def compute_diag_severity(df: pd.DataFrame) -> pd.Series:
    """
    Komorbidite proxy'si olarak number_diagnoses'i kullanacağız.
    İstersen ileride daha sofistike bir skorla değiştirebilirsin.
    """
    if "number_diagnoses" not in df.columns:
        raise ValueError("DataFrame'de 'number_diagnoses' sütunu yok, diag_severity için gerekli.")
    return df["number_diagnoses"]


def compute_topsis_scores(
    df: pd.DataFrame,
    weight_dict=TOPSIS_WEIGHTS,
    criteria_cols=None,
) -> pd.Series:
    """
    Verilen kriter sütunları ve ağırlıklarla TOPSIS risk skorunu hesaplar.
    Bütün kriterler 'cost-type' kabul edilmiştir (büyük = daha riskli).
    """
    if criteria_cols is None:
        criteria_cols = list(weight_dict.keys())

    # Karar matrisi X (N x m)
    for c in criteria_cols:
        if c not in df.columns:
            raise ValueError(f"Kriter sütunu eksik: {c}")
    X = df[criteria_cols].astype(float).values

    # Ağırlık vektörü
    w = np.array([weight_dict[c] for c in criteria_cols], dtype=float)
    w = w / w.sum()  # normalize, güven olması için

    # 1) Normalize et (v_ij)
    denom = np.sqrt((X ** 2).sum(axis=0))
    denom[denom == 0] = 1e-9
    V = X / denom

    # 2) Ağırlıklı normalize matris (y_ij)
    Y = V * w

    # 3) İdeal iyi/kötü çözümler (hepsi cost-type)
    A_plus = Y.min(axis=0)   # en iyi (en düşük risk)
    A_minus = Y.max(axis=0)  # en kötü (en yüksek risk)

    # 4) Uzaklıklar
    S_plus = np.sqrt(((Y - A_plus) ** 2).sum(axis=1))
    S_minus = np.sqrt(((Y - A_minus) ** 2).sum(axis=1))

    # 5) Yakınlık katsayısı CC_i
    # 1'e yaklaştıkça kötü çözüme (riskli profile) daha yakın
    cc = S_minus / (S_plus + S_minus + 1e-9)

    return pd.Series(cc, index=df.index, name="topsis_score")


def assign_risk_category(
    scores: pd.Series,
    low_q: float = 0.33,
    high_q: float = 0.66,
) -> pd.Series:
    """
    TOPSIS skoruna göre 3 risk grubuna ayır:
      Low, Medium, High
    """
    low_thr = scores.quantile(low_q)
    high_thr = scores.quantile(high_q)

    def _cat(s):
        if s < low_thr:
            return "Low"
        elif s < high_thr:
            return "Medium"
        else:
            return "High"

    return scores.apply(_cat)
