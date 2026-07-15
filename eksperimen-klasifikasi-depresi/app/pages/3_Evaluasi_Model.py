"""Evaluasi Model — metrik dan visualisasi FS0–FS3."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_APP = Path(__file__).resolve().parent.parent
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from utils.data_loader import load_experiment_results, load_test_metrics  # noqa: E402
from utils.paths import FIGURES_DIR  # noqa: E402

st.set_page_config(page_title="Evaluasi Model", page_icon="📈", layout="wide")
st.title("Evaluasi Model")
st.markdown(
    """
Perbandingan empat skenario feature selection dengan klasifikator tetap
**Random Forest**. Metrik utama: **F1-Score**, **Recall**, dan **ROC-AUC**.
"""
)

results = load_experiment_results()
metrics = load_test_metrics()

st.subheader("Model terbaik")
b1, b2, b3, b4 = st.columns(4)
b1.metric("Skenario", results["best_scenario"])
b2.metric("Metode", results["best_method"])
b3.metric("F1 (test)", f"{results['test_f1']:.4f}")
b4.metric("ROC-AUC (test)", f"{results['test_roc_auc']:.4f}")

st.subheader("Tabel metrik hold-out test (n=300)")
display = metrics.copy()
for col in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
    if col in display.columns:
        display[col] = display[col].map(lambda x: f"{x:.4f}")
st.dataframe(display, use_container_width=True, hide_index=True)

st.caption(
    f"FS3 dipilih dibanding FS2 karena ROC-AUC sedikit lebih tinggi "
    f"({results['test_roc_auc']:.4f}) dengan jumlah fitur lebih sedikit "
    f"(k={results['best_k_mi']} vs k={results['best_k_chi2']})."
)

st.subheader("Visualisasi evaluasi")

fig_specs = [
    ("11_fs_comparison_metrics.png", "Perbandingan metrik antar skenario FS"),
    ("12_confusion_matrices.png", "Confusion matrix FS0–FS3"),
    ("13_roc_curves.png", "Kurva ROC"),
    ("10_k_tuning.png", "Tuning nilai k (Chi-Square & Mutual Information)"),
    ("15_pca_explained_variance.png", "Variansi terjelaskan PCA (FS1)"),
    ("16_chi2_feature_ranking.png", "Ranking fitur Chi-Square (FS2)"),
    ("17_mi_feature_ranking.png", "Ranking fitur Mutual Information (FS3)"),
    ("18_feature_overlap.png", "Overlap fitur Chi-Square vs Mutual Information"),
]

for fname, caption in fig_specs:
    path = FIGURES_DIR / fname
    if path.exists():
        st.markdown(f"**{caption}**")
        st.image(str(path), use_container_width=True)
    else:
        st.warning(f"Gambar tidak ditemukan: `{fname}`")

st.subheader("Ringkasan konfigurasi eksperimen")
st.markdown(
    f"""
| Item | Nilai |
| --- | --- |
| Train / test | 1.200 / 300 (stratified 80:20, `random_state=42`) |
| Validasi | Stratified 5-Fold CV pada train |
| k optimal Chi-Square | {results['best_k_chi2']} |
| k optimal MI | {results['best_k_mi']} |
| Komponen PCA (≥95% var) | {results['n_pca_components']} |
"""
)
