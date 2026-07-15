"""Dokumentasi — dataset, metodologi, dan cara penggunaan."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_APP = Path(__file__).resolve().parent.parent
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from utils.data_loader import load_experiment_results, load_preprocessing_metadata  # noqa: E402

st.set_page_config(page_title="Dokumentasi", page_icon="📚", layout="wide")
st.title("Dokumentasi")

meta = load_preprocessing_metadata()
exp = load_experiment_results()

st.header("1. Tentang proyek")
st.markdown(
    """
**Judul:** Eksperimen Klasifikasi Depresi pada Remaja: Perbandingan Metode
Feature Selection untuk Identifikasi Fitur Gaya Hidup Paling Berpengaruh

**Mahasiswa:** Naf'an Nur'Alim (A11.2024.15651)

**Learning task:** Supervised learning — klasifikasi biner  
**Target:** `depression_label` (0 = tidak terindikasi, 1 = terindikasi)  
**Fokus:** Perbandingan metode feature selection (bukan perbandingan banyak algoritma)
"""
)

st.header("2. Dataset")
st.markdown(
    f"""
| Aspek | Keterangan |
| --- | --- |
| Nama | Teen Mental Health Dataset |
| Sumber | [Kaggle](https://www.kaggle.com/datasets/algozee/teenager-menthal-healy) |
| Observasi | 1.500 baris (hasil augmentasi; rasio kelas ≈ 90:10) |
| Fitur mentah | 12 input + 1 target |
| Fitur setelah preprocessing | {len(meta['feature_cols'])} (encoding + `screen_time_ratio`) |
| Missing values | 0 |
| Split | Train {meta['train_samples']} / Test {meta['test_samples']} (stratified, `random_state={meta['random_state']}`) |

**Kolom mentah:** `age`, `gender`, `daily_social_media_hours`, `platform_usage`,
`sleep_hours`, `screen_time_before_sleep`, `academic_performance`,
`physical_activity`, `social_interaction_level`, `stress_level`,
`anxiety_level`, `addiction_level`, `depression_label`.
"""
)

st.header("3. Metodologi")
st.markdown(
    f"""
### Preprocessing
1. Feature engineering: `screen_time_ratio = screen_time_before_sleep / sleep_hours`
2. Encoding: gender (binary), social interaction (ordinal), platform (one-hot)
3. Scaling: StandardScaler (FS0/FS1/FS3) dan MinMaxScaler (FS2 / Chi-Square)
4. Split stratified 80:20

### Skenario model (klasifikator tetap: Random Forest)
| Kode | Metode | Input | Representasi |
| --- | --- | --- | --- |
| FS0 | Baseline (tanpa seleksi) | StandardScaler | Semua 14 fitur |
| FS1 | PCA (≥95% variansi) | StandardScaler | {exp['n_pca_components']} komponen |
| FS2 | Chi-Square SelectKBest | MinMaxScaler | Top-{exp['best_k_chi2']} |
| **FS3** | **Mutual Information** | **StandardScaler** | **Top-{exp['best_k_mi']}** |

### Model terbaik
- **{exp['best_scenario']}** — {exp['best_method']}
- Fitur MI: {", ".join(f"`{f}`" for f in exp["mi_selected_features"])}
- F1 test = **{exp['test_f1']:.4f}**, ROC-AUC = **{exp['test_roc_auc']:.4f}**

### Interpretabilitas
Analisis SHAP pada model FS3 untuk memverifikasi kontribusi fitur
(lihat halaman **Interpretasi Hasil**).
"""
)

st.header("4. Cara menggunakan aplikasi")
st.markdown(
    """
1. **Dashboard EDA** — eksplorasi distribusi, korelasi, dan boxplot interaktif.
   Gunakan filter sidebar (gender / platform).
2. **Model Demo** — isi form 12 fitur survey (atau klik tombol contoh),
   lalu tekan **Prediksi** untuk melihat kelas dan probabilitas.
3. **Evaluasi Model** — bandingkan metrik FS0–FS3 dan lihat confusion matrix / ROC.
4. **Interpretasi Hasil** — pelajari ranking SHAP dan insight praktis.
5. **Dokumentasi** — halaman ini.

### Menjalankan secara lokal
```bash
cd eksperimen-klasifikasi-depresi
pip install -r requirements.txt
streamlit run app/Home.py
```

### Deploy ke Streamlit Community Cloud
1. Push repository ke GitHub.
2. Buka [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Pilih repo, branch, dan set **Main file path** ke:
   `eksperimen-klasifikasi-depresi/app/Home.py`
4. Pastikan `requirements.txt` di root repository tersedia
   (digunakan oleh Community Cloud).
"""
)

st.header("5. Disclaimer")
st.warning(
    "Aplikasi ini dibuat untuk keperluan tugas akhir / pembelajaran mesin. "
    "Hasil prediksi bukan diagnosis medis. Untuk kekhawatiran kesehatan mental, "
    "konsultasikan tenaga profesional yang berwenang."
)

st.header("6. Artefak teknis")
st.markdown(
    """
| Artefak | Path |
| --- | --- |
| Model terbaik | `artifacts/best_fs_model.joblib` |
| Scaler | `artifacts/standard_scaler.joblib` |
| Metadata preprocessing | `artifacts/preprocessing_metadata.json` |
| Hasil eksperimen | `artifacts/experiment_fs_results.json` |
| Hasil SHAP | `artifacts/shap_analysis_results.json` |
| Notebook | `notebooks/01_eda.ipynb` … `04_xai_shap_analysis.ipynb` |
"""
)
