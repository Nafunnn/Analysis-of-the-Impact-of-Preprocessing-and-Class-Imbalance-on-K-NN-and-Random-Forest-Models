"""Interpretasi Hasil — SHAP, ranking fitur, insight bisnis."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

_APP = Path(__file__).resolve().parent.parent
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from utils.data_loader import load_experiment_results, load_shap_results  # noqa: E402
from utils.paths import FIGURES_DIR  # noqa: E402

st.set_page_config(page_title="Interpretasi Hasil", page_icon="🔍", layout="wide")
st.title("Interpretasi Hasil")
st.markdown(
    """
Penjelasan kontribusi fitur (SHAP) dan implikasi praktis terhadap
indikasi depresi remaja berdasarkan model terbaik **FS3**.
"""
)

shap = load_shap_results()
exp = load_experiment_results()

st.subheader("1. Ranking fitur menurut SHAP (model FS3)")
shap_df = pd.DataFrame(shap["fs3_shap_ranking"])
shap_df = shap_df.rename(
    columns={"fitur": "Fitur", "mean_abs_shap": "Mean |SHAP|", "rank_shap": "Peringkat"}
)
fig = px.bar(
    shap_df.sort_values("Mean |SHAP|"),
    x="Mean |SHAP|",
    y="Fitur",
    orientation="h",
    title="Mean Absolute SHAP — FS3",
    text="Peringkat",
)
st.plotly_chart(fig, use_container_width=True)
st.dataframe(shap_df, use_container_width=True, hide_index=True)

c1, c2 = st.columns(2)
with c1:
    st.markdown("**Top-5 SHAP (FS0 — semua fitur)**")
    st.write(", ".join(f"`{f}`" for f in shap["fs0_top5_shap"]))
with c2:
    st.markdown("**Fitur MI terpilih (FS3)**")
    st.write(", ".join(f"`{f}`" for f in exp["mi_selected_features"]))

st.markdown(
    f"""
**Overlap top fitur** (muncul di beberapa metode):  
{", ".join(f"`{f}`" for f in shap["overlap_top5_all_methods"])}

| Alignasi ranking | Spearman ρ |
| --- | ---: |
| SHAP vs Chi-Square | {shap['spearman_shap_chi2']:.4f} |
| SHAP vs Mutual Information | {shap['spearman_shap_mi']:.4f} |

Mutual Information (FS3) paling selaras dengan ranking SHAP,
mendukung pemilihan FS3 sebagai model utama.
"""
)

st.subheader("2. Visualisasi SHAP")
shap_figs = [
    ("20_shap_bar_fs3.png", "SHAP bar plot — FS3"),
    ("19_shap_summary_fs3.png", "SHAP summary — FS3"),
    ("21_shap_dependence_fs3.png", "SHAP dependence — FS3"),
    ("22_shap_waterfall_sample.png", "SHAP waterfall (contoh sampel)"),
    ("24_shap_vs_feature_selection.png", "SHAP vs metode feature selection"),
    ("25_rank_heatmap_comparison.png", "Heatmap perbandingan ranking"),
]
for fname, caption in shap_figs:
    path = FIGURES_DIR / fname
    if path.exists():
        st.markdown(f"**{caption}**")
        st.image(str(path), use_container_width=True)

st.subheader("3. Insight bisnis & praktis")
st.markdown(
    """
Temuan eksperimen menunjukkan bahwa indikasi depresi pada dataset ini
paling terkait dengan kombinasi faktor gaya hidup berikut:

1. **Tingkat stres (`stress_level`)** — kontributor SHAP tertinggi;
   remaja dengan stres tinggi cenderung masuk kelas positif.
2. **Jam tidur (`sleep_hours`)** — tidur pendek berulang terkait risiko lebih tinggi;
   intervensi kebersihan tidur relevan sebagai sinyal pencegahan.
3. **Durasi media sosial harian (`daily_social_media_hours`)** —
   penggunaan intensif relevan, tetapi tidak berdiri sendiri.
4. **Tingkat kecemasan (`anxiety_level`)** — sering bersama stres;
   keduanya redundan sebagian tetapi tetap informatif.
5. **Aktivitas fisik (`physical_activity`)** — aktivitas rendah muncul
   sebagai sinyal tambahan pada seleksi Mutual Information.

**Implikasi praktis (edukasi / monitoring non-klinis):**
- Prioritaskan skrining dini pada kombinasi tidur buruk + stres/kecemasan tinggi
  + media sosial intensif + aktivitas fisik rendah.
- Feature selection membantu menyederhanakan instrumen survei (5 fitur MI)
  tanpa mengorbankan performa (F1 ≈ 0.98 pada test set).
- Model ini **bukan diagnosis** — hasil prediksi hanya indikator statistik
  untuk konteks riset dan pembelajaran.

**Catatan metodologis:** fokus riset adalah membandingkan metode seleksi fitur
(FS0–FS3), bukan membandingkan banyak algoritma klasifikasi.
"""
)
