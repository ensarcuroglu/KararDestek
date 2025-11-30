import os
import joblib
import pandas as pd


def find_and_read():
    print("--- MODEL ARAMA TARAMASI BAŞLATILIYOR ---")

    # 1. Proje Ana Dizinini Bul
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = current_dir

    # 'KararDestek2' klasörüne kadar yukarı çık
    while "KararDestek2" not in os.path.basename(root_dir):
        parent = os.path.dirname(root_dir)
        if parent == root_dir: break
        root_dir = parent

    # Eğer döngü bulamazsa manuel sabitleme (senin yolun belli)
    if "KararDestek" not in root_dir:
        root_dir = r"D:\Ensar Dosya\KararDestek\KararDestek2"

    print(f"[INFO] Arama Başlangıç Noktası (Root): {root_dir}")
    print("[INFO] 'xgb_weighted_f2' klasörü içindeki 'model.joblib' aranıyor...")

    found_path = None

    # 2. Tüm klasörleri gez (Recursive Search)
    for root, dirs, files in os.walk(root_dir):
        if "model.joblib" in files and "xgb_weighted_f2" in os.path.basename(root):
            found_path = os.path.join(root, "model.joblib")
            print(f"\n✅ BULUNDU! Dosya Yolu: {found_path}")
            break

    if not found_path:
        print("\n❌ HATA: Dosya projenin hiçbir yerinde bulunamadı.")
        print("Lütfen önceki 'train' kodunun gerçekten 'models' klasörü oluşturup oluşturmadığına bak.")
        return

    # 3. Modeli Yükle ve Bilgileri Oku
    print("[INFO] Model yükleniyor...")
    try:
        saved_data = joblib.load(found_path)
        feature_names = saved_data.get("feature_names", [])

        print("\n" + "=" * 50)
        print("TOPSIS İÇİN GEREKLİ LİSTE (Bunu kopyala):")
        print("=" * 50)
        print(feature_names)
        print("=" * 50)

    except Exception as e:
        print(f"Dosya bozuk veya okunamadı: {e}")


if __name__ == "__main__":
    find_and_read()