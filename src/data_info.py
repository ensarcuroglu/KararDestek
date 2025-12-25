import pandas as pd
import matplotlib.pyplot as plt
import os


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.figsize'] = (14, 12)


def load_data(file_path):
    if not os.path.exists(file_path):
        return None
    return pd.read_csv(file_path, na_values=['?', 'NA', 'null', 'MISSING'])


def create_variable_summary_table(df):

    # Detaylı özet tablosu.

    selected_columns = [
        'encounter_id', 'patient_nbr', 'race', 'gender', 'age',
        'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
        'time_in_hospital', 'payer_code', 'medical_specialty',
        'num_lab_procedures', 'num_procedures', 'num_medications',
        'number_outpatient', 'number_emergency', 'number_inpatient',
        'number_diagnoses', 'max_glu_serum', 'A1Cresult',
        'insulin', 'change', 'diabetesMed', 'readmitted'
    ]

    # Filtre kısmı
    valid_cols = [c for c in selected_columns if c in df.columns]

    summary_data = []

    for col in valid_cols:
        col_data = df[col]
        dtype = str(col_data.dtype)

        if 'int' in dtype or 'float' in dtype:
            type_label = "Sayısal"

            example_val = f"Ort: {col_data.mean():.1f}"
        else:
            type_label = "Kategorik"

            try:
                top_val = col_data.mode()[0]
                example_val = f"Mod: {top_val}"
            except:
                example_val = "-"

        # İstatistikler
        unique_count = col_data.nunique()
        missing_perc = (col_data.isnull().sum() / len(df)) * 100
        fill_rate = 100 - missing_perc

        summary_data.append([
            col,
            type_label,
            f"{unique_count:,}",
            f"%{fill_rate:.1f}",
            str(example_val)[:15]
        ])

    # DataFrame oluşturma
    df_summary = pd.DataFrame(summary_data, columns=[
        "Değişken Adı", "Veri Tipi", "Benzersiz Değer", "Doluluk %", "Örnek / İstatistik"
    ])

    # TABLO
    fig, ax = plt.subplots(figsize=(12, len(valid_cols) * 0.5 + 2))  # Satır sayısına göre boyu ayarla
    ax.axis('off')
    table = ax.table(
        cellText=df_summary.values,
        colLabels=df_summary.columns,
        cellLoc='left',
        loc='center',
        colColours=['#404040'] * 5
    )

    # AYARLAR
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.8)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('white')
        cell.set_linewidth(1.5)

        if row == 0:
            cell.set_text_props(weight='bold', color='white', size=12)
            cell.set_facecolor('#2c3e50')
        else:
            cell.set_text_props(color='#333333')
            if row % 2 == 0:
                cell.set_facecolor('#f2f2f2')
            else:
                cell.set_facecolor('#ffffff')

            # İlk sütun kalın
            if col == 0:
                cell.set_text_props(weight='bold')
            if col == 3:
                pass

    plt.title(f"Veri Seti Değişken Özeti (Seçilmiş {len(valid_cols)} Değişken)",
              fontsize=16, fontweight='bold', pad=20, color='#333333')

    plt.tight_layout()

    # Kaydet
    output_path = 'degisken_ozet_tablosu.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Tablo oluşturuldu ve kaydedildi: {output_path}")
    plt.show()


def main():
    # Dosya Yolu
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    FILE_PATH = os.path.join(project_root, 'data', 'raw', 'diabetic_data.csv')

    print("Veri yükleniyor...")
    df = load_data(FILE_PATH)

    if df is not None:
        create_variable_summary_table(df)


if __name__ == "__main__":
    main()