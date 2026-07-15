"""Dashboard EDA — visualisasi interaktif."""

from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st

_APP = Path(__file__).resolve().parent.parent
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from utils.data_loader import load_raw_dataframe  # noqa: E402

st.set_page_config(page_title="Dashboard EDA", page_icon="📊", layout="wide")
st.title("Dashboard EDA")
st.caption("Analisis eksploratif interaktif pada Teen Mental Health Dataset.")

df = load_raw_dataframe().copy()
df["depression_label"] = df["depression_label"].map(
    {0: "Tidak terindikasi (0)", 1: "Terindikasi (1)"}
)

with st.sidebar:
    st.header("Filter data")
    genders = st.multiselect(
        "Jenis kelamin",
        options=sorted(df["gender"].unique()),
        default=sorted(df["gender"].unique()),
    )
    platforms = st.multiselect(
        "Platform",
        options=sorted(df["platform_usage"].unique()),
        default=sorted(df["platform_usage"].unique()),
    )

filtered = df[df["gender"].isin(genders) & df["platform_usage"].isin(platforms)]
st.write(f"Menampilkan **{len(filtered):,}** dari {len(df):,} observasi.")

# --- Distribusi kelas ---
st.subheader("1. Distribusi kelas target")
class_counts = filtered["depression_label"].value_counts().reset_index()
class_counts.columns = ["Kelas", "Jumlah"]
fig_class = px.bar(
    class_counts,
    x="Kelas",
    y="Jumlah",
    color="Kelas",
    text="Jumlah",
    title="Distribusi depression_label",
)
fig_class.update_traces(textposition="outside")
st.plotly_chart(fig_class, use_container_width=True)

# --- Numerik ---
st.subheader("2. Distribusi fitur numerik")
numeric_cols = [
    "age",
    "daily_social_media_hours",
    "sleep_hours",
    "screen_time_before_sleep",
    "academic_performance",
    "physical_activity",
    "stress_level",
    "anxiety_level",
    "addiction_level",
]
num_feat = st.selectbox("Pilih fitur numerik", numeric_cols, index=2)
fig_hist = px.histogram(
    filtered,
    x=num_feat,
    color="depression_label",
    barmode="overlay",
    opacity=0.65,
    title=f"Histogram: {num_feat}",
    nbins=30,
)
st.plotly_chart(fig_hist, use_container_width=True)

# --- Kategorikal ---
st.subheader("3. Distribusi fitur kategorikal")
cat_cols = ["gender", "platform_usage", "social_interaction_level"]
cat_feat = st.selectbox("Pilih fitur kategorikal", cat_cols)
cat_ct = (
    filtered.groupby([cat_feat, "depression_label"])
    .size()
    .reset_index(name="Jumlah")
)
fig_cat = px.bar(
    cat_ct,
    x=cat_feat,
    y="Jumlah",
    color="depression_label",
    barmode="group",
    title=f"Distribusi {cat_feat} vs target",
)
st.plotly_chart(fig_cat, use_container_width=True)

# --- Korelasi vs target ---
st.subheader("4. Korelasi fitur numerik dengan target")
corr_df = filtered.copy()
corr_df["depression_label_num"] = (
    corr_df["depression_label"].str.contains("Terindikasi").astype(int)
)
corrs = (
    corr_df[numeric_cols + ["depression_label_num"]]
    .corr(numeric_only=True)["depression_label_num"]
    .drop("depression_label_num")
    .sort_values(key=abs, ascending=False)
    .reset_index()
)
corrs.columns = ["Fitur", "Korelasi"]
fig_corr = px.bar(
    corrs,
    x="Korelasi",
    y="Fitur",
    orientation="h",
    title="Korelasi Pearson vs depression_label",
    color="Korelasi",
    color_continuous_scale="RdBu_r",
)
st.plotly_chart(fig_corr, use_container_width=True)

# --- Boxplot ---
st.subheader("5. Boxplot fitur numerik menurut target")
box_feat = st.selectbox(
    "Fitur untuk boxplot",
    numeric_cols,
    index=numeric_cols.index("sleep_hours"),
    key="box_feat",
)
fig_box = px.box(
    filtered,
    x="depression_label",
    y=box_feat,
    color="depression_label",
    title=f"Boxplot {box_feat} menurut kelas",
)
st.plotly_chart(fig_box, use_container_width=True)

# --- Heatmap korelasi ---
st.subheader("6. Heatmap korelasi antar fitur numerik")
heat = corr_df[numeric_cols + ["depression_label_num"]].corr(numeric_only=True)
fig_heat = px.imshow(
    heat,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale="RdBu_r",
    title="Matriks korelasi",
)
st.plotly_chart(fig_heat, use_container_width=True)

with st.expander("Cuplikan data terfilter"):
    st.dataframe(filtered.head(50), use_container_width=True)
