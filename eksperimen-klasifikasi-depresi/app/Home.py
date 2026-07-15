"""Beranda aplikasi Streamlit — Klasifikasi Depresi Remaja."""

import streamlit as st

from bootstrap import ensure_app_on_path

ensure_app_on_path()

from utils.data_loader import load_experiment_results  # noqa: E402

st.set_page_config(
    page_title="Klasifikasi Depresi Remaja",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

results = load_experiment_results()

st.title("Klasifikasi Depresi Remaja")
st.markdown(
    """
**Eksperimen Feature Selection** untuk mengidentifikasi fitur gaya hidup
paling berpengaruh terhadap indikasi depresi remaja.
"""
)

st.info(
    "Aplikasi ini adalah alat edukasi dan riset pembelajaran mesin — "
    "**bukan** alat diagnosis klinis."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model terbaik", results["best_scenario"])
col2.metric("Metode", "Mutual Information (k=5)")
col3.metric("F1-Score (test)", f"{results['test_f1']:.4f}")
col4.metric("ROC-AUC (test)", f"{results['test_roc_auc']:.4f}")

st.divider()

st.subheader("Navigasi halaman")
st.markdown(
    """
| Halaman | Isi |
| --- | --- |
| **Dashboard EDA** | Visualisasi interaktif analisis data eksploratif |
| **Model Demo** | Input data survey → prediksi model FS3 |
| **Evaluasi Model** | Metrik & visualisasi perbandingan FS0–FS3 |
| **Interpretasi Hasil** | SHAP, ranking fitur, insight bisnis |
| **Dokumentasi** | Dataset, metodologi, dan cara penggunaan |

Gunakan menu di **sidebar** untuk berpindah halaman.
"""
)

st.subheader("Ringkasan model terbaik (FS3)")
st.markdown(
    f"""
- **Algoritma:** Random Forest (`n_estimators=100`, `class_weight='balanced'`)
- **Seleksi fitur:** Mutual Information, top-{results['best_k_mi']} fitur
- **Fitur terpilih:** {", ".join(f"`{f}`" for f in results["mi_selected_features"])}
- **Dataset:** Teen Mental Health (1.500 observasi, rasio kelas ≈ 90:10)
"""
)

st.caption("Mahasiswa: Naf'an Nur'Alim (A11.2024.15651)")
