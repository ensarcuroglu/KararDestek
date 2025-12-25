import numpy as np


def calculate_ahp_weights():
    # Kriter İsimleri
    criteria = ['XGBoost Risk', 'Emergency Count', 'Time in Hospital', 'Num Procedures']

    matrix = np.array([
        # Risk  Emerg   Time   Proc
        [1.00, 3.00, 5.00, 5.00],  # risk skoru
        [0.33, 1.00, 3.00, 3.00],  # emergency
        [0.20, 0.33, 1.00, 1.00],  # time
        [0.20, 0.33, 1.00, 1.00]  # procedures
    ])

    print("--- AHP Karşılaştırma Matrisi ---")
    print(matrix)

    col_sums = matrix.sum(axis=0)

    # Matrisi Normalize
    normalized_matrix = matrix / col_sums

    # Satır Ortalamalarını Al
    weights = normalized_matrix.mean(axis=1)

    print("\n--- Hesaplanan AHP Ağırlıkları ---")
    for i, w in enumerate(weights):
        print(f"{criteria[i]}: %{w * 100:.2f}")

    consistency_vector = matrix.dot(weights)
    lambda_max = (consistency_vector / weights).mean()
    n = len(weights)
    ci = (lambda_max - n) / (n - 1)
    ri = 0.90
    cr = ci / ri

    print(f"\nTutarlılık Oranı (CR): {cr:.4f}")
    if cr < 0.1:
        print("Matris Tutarlı! (Mantıklı bir karşılaştırma yapılmış)")
    else:
        print("Matris Tutarsız! Karşılaştırmaları gözden geçir.")

    return weights


if __name__ == "__main__":
    w = calculate_ahp_weights()
    print(f"\nKopyalaman gereken liste: {list(w)}")