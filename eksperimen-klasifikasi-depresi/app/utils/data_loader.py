"""Loader data, metrik, dan metadata untuk halaman Streamlit."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import streamlit as st

from .paths import (
    EXPERIMENT_RESULTS,
    PREPROCESSING_META,
    RAW_CSV,
    SHAP_RESULTS,
    TEST_METRICS_CSV,
)


@st.cache_data(show_spinner=False)
def load_raw_dataframe() -> pd.DataFrame:
    return pd.read_csv(RAW_CSV)


@st.cache_data(show_spinner=False)
def load_preprocessing_metadata() -> dict[str, Any]:
    with open(PREPROCESSING_META, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_experiment_results() -> dict[str, Any]:
    with open(EXPERIMENT_RESULTS, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_shap_results() -> dict[str, Any]:
    with open(SHAP_RESULTS, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_test_metrics() -> pd.DataFrame:
    return pd.read_csv(TEST_METRICS_CSV)


def sample_raw_row(label: int | None = None) -> dict[str, Any]:
    """Ambil satu baris mentah sebagai contoh form Model Demo."""
    df = load_raw_dataframe()
    if label is not None:
        subset = df[df["depression_label"] == label]
        if not subset.empty:
            df = subset
    row = df.sample(1, random_state=None).iloc[0]
    return row.drop(labels=["depression_label"]).to_dict()
