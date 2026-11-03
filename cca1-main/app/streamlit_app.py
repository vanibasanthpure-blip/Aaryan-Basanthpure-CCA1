"""Streamlit demo for ECG Arrhythmia Classification Using Advanced Temporal
Pattern Recognition.

Offline inference demonstration on recorded MIT-BIH signals -- NOT a
clinically validated real-time monitoring system.

Run with:
    streamlit run app/streamlit_app.py
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from predict import AAMI_DESCRIPTIONS, load_models, predict_all, prepare_record_beats  # noqa: E402
from utils import ALL_RECORDS, BEAT_WINDOW_SEC  # noqa: E402

st.set_page_config(page_title="ECG Arrhythmia Classification", layout="wide")

st.title("ECG Arrhythmia Classification Using Advanced Temporal Pattern Recognition")
st.caption(
    "University Pattern Recognition project. Offline demonstration on recorded "
    "MIT-BIH Arrhythmia Database signals -- not a clinically validated "
    "real-time monitoring system."
)

with st.expander("About this project", expanded=False):
    st.markdown(
        "Three independent approaches are trained on the same patient-independent "
        "DS1 (train) / DS2 (test) split of the MIT-BIH Arrhythmia Database and "
        "compared on the AAMI EC57 five-class scheme (N, S, V, F, Q):\n\n"
        "1. **RR interval / HRV features + SVM**\n"
        "2. **Discrete Wavelet Transform (db4) features + Random Forest**\n"
        "3. **1D-CNN + Bidirectional LSTM** on raw beat waveforms\n"
    )


@st.cache_resource
def get_models():
    return load_models()


@st.cache_data
def get_record_data(record_id: int):
    cleaned, beats, labels, rr_prev, rr_next = prepare_record_beats(record_id)
    return cleaned, beats, labels, rr_prev, rr_next


models = get_models()
if not models:
    st.error("No trained models found under models/. Run `python src/train.py --approach all` first.")
    st.stop()

col1, col2 = st.columns([1, 2])
with col1:
    record_id = st.selectbox("MIT-BIH record", ALL_RECORDS, index=ALL_RECORDS.index(100))
    model_options = list(models.keys())
    selected_models = st.multiselect("Models to run", model_options, default=model_options)

cleaned, beats, labels, rr_prev, rr_next = get_record_data(record_id)
n_beats = len(beats)

with col1:
    beat_idx = st.slider("Beat index", 0, n_beats - 1, 0)
    st.metric("Annotated (true) class", f"{labels[beat_idx]} -- {AAMI_DESCRIPTIONS[labels[beat_idx]]}")

active_models = {m: models[m] for m in selected_models}
preds = predict_all(active_models, beats, rr_prev, rr_next)

with col2:
    fig, ax = plt.subplots(figsize=(8, 3))
    before = int(BEAT_WINDOW_SEC[0] * 360)
    t = (np.arange(len(beats[beat_idx])) - before) / 360.0
    ax.plot(t, beats[beat_idx], color="crimson")
    ax.axvline(0, color="gray", linestyle="--", linewidth=1, label="annotated R-peak")
    ax.set_xlabel("Time relative to R-peak (s)")
    ax.set_ylabel("Normalised amplitude")
    ax.set_title(f"Record {record_id} - beat #{beat_idx}")
    ax.legend()
    st.pyplot(fig)

st.subheader("Model predictions for this beat")
cols = st.columns(len(preds) or 1)
for col, (model_name, (pred, conf)) in zip(cols, preds.items()):
    with col:
        st.metric(model_name, f"{pred[beat_idx]} -- {AAMI_DESCRIPTIONS[pred[beat_idx]]}",
                   f"confidence {conf[beat_idx]:.2f}")

st.subheader("Full-record comparison (first 50 beats)")
n_show = min(50, n_beats)
rows = []
for i in range(n_show):
    row = {"#": i, "true": labels[i]}
    for model_name, (pred, conf) in preds.items():
        row[model_name] = f"{pred[i]} ({conf[i]:.2f})"
    rows.append(row)
st.dataframe(rows, use_container_width=True)
