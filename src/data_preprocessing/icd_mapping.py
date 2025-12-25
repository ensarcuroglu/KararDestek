# src/data_preprocessing/icd_mapping.py

import pandas as pd

def map_icd9_to_group(code: str) -> str:

    if pd.isna(code) or code == '?' or code == 'Missing':
        return "Missing"

    try:
        if isinstance(code, str) and (code.startswith('V') or code.startswith('E')):
            return "Other"
        val = float(code)
    except ValueError:
        return "Other"

    # Diabetes: 250.xx
    if 250 <= val < 251:
        return "Diabetes"
    # Circulatory: 390–459, 785
    if (390 <= val <= 459) or (val == 785):
        return "Circulatory"
    # Respiratory: 460–519, 786
    if (460 <= val <= 519) or (val == 786):
        return "Respiratory"
    # Digestive: 520–579, 787
    if (520 <= val <= 579) or (val == 787):
        return "Digestive"
    # Injury: 800–999
    if 800 <= val <= 999:
        return "Injury"
    # Musculoskeletal: 710–739
    if 710 <= val <= 739:
        return "Musculoskeletal"
    # Genitourinary: 580–629, 788
    if (580 <= val <= 629) or (val == 788):
        return "Genitourinary"
    # Neoplasms: 140–239
    if 140 <= val <= 239:
        return "Neoplasms"

    return "Other"
