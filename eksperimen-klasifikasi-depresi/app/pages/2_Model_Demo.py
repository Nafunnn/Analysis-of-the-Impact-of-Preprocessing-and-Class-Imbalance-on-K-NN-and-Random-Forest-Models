"""Model Demo — prediksi dari input survey mentah memakai model FS3."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

_APP = Path(__file__).resolve().parent.parent
if str(_APP) not in sys.path:
    sys.path.insert(0, str(_APP))

from utils.data_loader import load_raw_dataframe, sample_raw_row  # noqa: E402
from utils.inference import (  # noqa: E402
    FEATURE_LABELS_ID,
    MI_TOP_FEATURES,
    predict_from_raw,
)

st.set_page_config(page_title="Model Demo", page_icon="🔮", layout="wide")
st.title("Model Demo")
st.markdown(
    """
Masukkan nilai fitur survey, lalu dapatkan prediksi dari model terbaik
(**FS3: Mutual Information k=5 + Random Forest**).

Fitur yang paling berpengaruh menurut Mutual Information ditandai dengan ★.
"""
)

df = load_raw_dataframe()

if "demo_defaults" not in st.session_state:
    st.session_state.demo_defaults = {
        "age": 16,
        "gender": "female",
        "daily_social_media_hours": 6.0,
        "platform_usage": "TikTok",
        "sleep_hours": 5.5,
        "screen_time_before_sleep": 1.5,
        "academic_performance": 2.5,
        "physical_activity": 0.5,
        "social_interaction_level": "medium",
        "stress_level": 7,
        "anxiety_level": 8,
        "addiction_level": 6,
    }

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Isi contoh (kelas 0 — tidak terindikasi)", use_container_width=True):
        st.session_state.demo_defaults = sample_raw_row(label=0)
        st.rerun()
with c2:
    if st.button("Isi contoh (kelas 1 — terindikasi)", use_container_width=True):
        st.session_state.demo_defaults = sample_raw_row(label=1)
        st.rerun()
with c3:
    if st.button("Reset nilai default", use_container_width=True):
        st.session_state.demo_defaults = {
            "age": 16,
            "gender": "female",
            "daily_social_media_hours": 6.0,
            "platform_usage": "TikTok",
            "sleep_hours": 5.5,
            "screen_time_before_sleep": 1.5,
            "academic_performance": 2.5,
            "physical_activity": 0.5,
            "social_interaction_level": "medium",
            "stress_level": 7,
            "anxiety_level": 8,
            "addiction_level": 6,
        }
        st.rerun()

d = st.session_state.demo_defaults


def _label(col: str) -> str:
    base = FEATURE_LABELS_ID.get(col, col)
    return f"★ {base}" if col in MI_TOP_FEATURES else base


with st.form("prediction_form"):
    left, right = st.columns(2)

    with left:
        age = st.slider(_label("age"), 13, 19, int(d["age"]))
        gender = st.selectbox(
            _label("gender"),
            ["male", "female"],
            index=0 if str(d["gender"]).lower() == "male" else 1,
        )
        daily_sm = st.slider(
            _label("daily_social_media_hours"),
            1.0,
            8.0,
            float(d["daily_social_media_hours"]),
            0.1,
        )
        platform = st.selectbox(
            _label("platform_usage"),
            ["TikTok", "Instagram", "Both"],
            index=["TikTok", "Instagram", "Both"].index(
                d["platform_usage"] if d["platform_usage"] in ("TikTok", "Instagram", "Both") else "TikTok"
            ),
        )
        sleep = st.slider(
            _label("sleep_hours"), 4.0, 9.0, float(d["sleep_hours"]), 0.1
        )
        screen = st.slider(
            _label("screen_time_before_sleep"),
            0.5,
            3.0,
            float(d["screen_time_before_sleep"]),
            0.1,
        )

    with right:
        academic = st.slider(
            _label("academic_performance"),
            2.0,
            4.0,
            float(d["academic_performance"]),
            0.01,
        )
        physical = st.slider(
            _label("physical_activity"),
            0.0,
            2.0,
            float(d["physical_activity"]),
            0.1,
        )
        social = st.selectbox(
            _label("social_interaction_level"),
            ["low", "medium", "high"],
            index=["low", "medium", "high"].index(
                str(d["social_interaction_level"]).lower()
                if str(d["social_interaction_level"]).lower() in ("low", "medium", "high")
                else "medium"
            ),
        )
        stress = st.slider(_label("stress_level"), 1, 10, int(d["stress_level"]))
        anxiety = st.slider(_label("anxiety_level"), 1, 10, int(d["anxiety_level"]))
        addiction = st.slider(
            _label("addiction_level"), 1, 10, int(d["addiction_level"])
        )

    submitted = st.form_submit_button("Prediksi", type="primary", use_container_width=True)

if submitted:
    raw = {
        "age": age,
        "gender": gender,
        "daily_social_media_hours": daily_sm,
        "platform_usage": platform,
        "sleep_hours": sleep,
        "screen_time_before_sleep": screen,
        "academic_performance": academic,
        "physical_activity": physical,
        "social_interaction_level": social,
        "stress_level": stress,
        "anxiety_level": anxiety,
        "addiction_level": addiction,
    }
    try:
        result = predict_from_raw(raw)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Gagal memprediksi: {exc}")
    else:
        st.divider()
        st.subheader("Hasil prediksi")
        m1, m2 = st.columns(2)
        label_text = (
            "Terindikasi depresi (1)"
            if result["label"] == 1
            else "Tidak terindikasi depresi (0)"
        )
        m1.metric("Prediksi kelas", label_text)
        m2.metric("Probabilitas kelas positif", f"{result['proba']:.1%}")

        if result["label"] == 1:
            st.warning(result["risk_text"])
        else:
            st.success(result["risk_text"])

        st.progress(min(max(result["proba"], 0.0), 1.0))

        with st.expander("Fitur setelah encoding (sebelum scaling)"):
            st.json(result["encoded_features"])

st.caption(
    "★ = fitur terpilih Mutual Information (k=5) di dalam pipeline model. "
    "Semua 12 input tetap diperlukan karena preprocessing menghasilkan 14 fitur "
    "sebelum SelectKBest."
)
