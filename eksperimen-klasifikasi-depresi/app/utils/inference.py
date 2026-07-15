"""Pipeline inferensi: input survey mentah → encode → FE → scale → prediksi FS3."""

from __future__ import annotations

from typing import Any

import joblib
import pandas as pd
import streamlit as st

from .data_loader import load_preprocessing_metadata
from .paths import BEST_MODEL, STANDARD_SCALER

GENDER_MAP = {"male": 0, "female": 1}
SOCIAL_MAP = {"low": 0, "medium": 1, "high": 2}

MI_TOP_FEATURES = [
    "sleep_hours",
    "stress_level",
    "daily_social_media_hours",
    "anxiety_level",
    "physical_activity",
]

FEATURE_LABELS_ID = {
    "age": "Usia",
    "gender": "Jenis kelamin",
    "daily_social_media_hours": "Jam media sosial / hari",
    "platform_usage": "Platform utama",
    "sleep_hours": "Jam tidur / hari",
    "screen_time_before_sleep": "Screen time sebelum tidur (jam)",
    "academic_performance": "Performa akademik",
    "physical_activity": "Aktivitas fisik",
    "social_interaction_level": "Tingkat interaksi sosial",
    "stress_level": "Tingkat stres (1–10)",
    "anxiety_level": "Tingkat kecemasan (1–10)",
    "addiction_level": "Tingkat kecanduan digital (1–10)",
}


@st.cache_resource(show_spinner=False)
def load_model_and_scaler():
    model = joblib.load(BEST_MODEL)
    scaler = joblib.load(STANDARD_SCALER)
    return model, scaler


def encode_raw_input(raw: dict[str, Any]) -> pd.DataFrame:
    """Ubah input survey mentah menjadi 14 fitur (belum di-scale)."""
    meta = load_preprocessing_metadata()
    feature_cols: list[str] = meta["feature_cols"]

    gender = str(raw["gender"]).strip().lower()
    social = str(raw["social_interaction_level"]).strip().lower()
    platform = str(raw["platform_usage"]).strip()

    if gender not in GENDER_MAP:
        raise ValueError(f"Nilai gender tidak valid: {raw['gender']}")
    if social not in SOCIAL_MAP:
        raise ValueError(f"Nilai social_interaction_level tidak valid: {raw['social_interaction_level']}")

    sleep = float(raw["sleep_hours"])
    screen = float(raw["screen_time_before_sleep"])
    ratio = (screen / sleep) if sleep > 0 else 0.0

    # One-hot tanpa platform_Both (kategori referensi) — selaras artifacts 14 fitur
    platform_instagram = 1 if platform == "Instagram" else 0
    platform_tiktok = 1 if platform == "TikTok" else 0

    row = {
        "age": float(raw["age"]),
        "gender": GENDER_MAP[gender],
        "daily_social_media_hours": float(raw["daily_social_media_hours"]),
        "sleep_hours": sleep,
        "screen_time_before_sleep": screen,
        "academic_performance": float(raw["academic_performance"]),
        "physical_activity": float(raw["physical_activity"]),
        "social_interaction_level": SOCIAL_MAP[social],
        "stress_level": float(raw["stress_level"]),
        "anxiety_level": float(raw["anxiety_level"]),
        "addiction_level": float(raw["addiction_level"]),
        "screen_time_ratio": round(ratio, 3),
        "platform_Instagram": platform_instagram,
        "platform_TikTok": platform_tiktok,
    }

    return pd.DataFrame([row], columns=feature_cols)


def predict_from_raw(raw: dict[str, Any]) -> dict[str, Any]:
    """Prediksi dari dict input mentah.

    Returns
    -------
    dict dengan kunci: label, proba, risk_text, encoded_features
    """
    model, scaler = load_model_and_scaler()
    X = encode_raw_input(raw)
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index,
    )
    label = int(model.predict(X_scaled)[0])
    proba = float(model.predict_proba(X_scaled)[0, 1])

    if label == 1:
        risk_text = (
            f"Model mengindikasikan risiko depresi (kelas positif) "
            f"dengan probabilitas {proba:.1%}."
        )
    else:
        risk_text = (
            f"Model tidak mengindikasikan depresi (kelas negatif) "
            f"- probabilitas kelas positif {proba:.1%}."
        )

    return {
        "label": label,
        "proba": proba,
        "risk_text": risk_text,
        "encoded_features": X.iloc[0].to_dict(),
    }
