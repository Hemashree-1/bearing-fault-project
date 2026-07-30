import streamlit as st
import numpy as np
import joblib
import os
from scipy.io import loadmat
import librosa

st.set_page_config(page_title="Acoustic Fingerprinting Dashboard", layout="centered")

# Load saved models
fault_model = joblib.load('fault_model.pkl')
severity_model = joblib.load('severity_model.pkl')
scaler = joblib.load('scaler.pkl')

def get_signal(filepath):
    mat = loadmat(filepath)
    de_key = [key for key in mat.keys() if 'DE_time' in key][0]
    return mat[de_key].flatten()

def extract_mfcc_features(signal, sr=12000, n_mfcc=13):
    signal = signal.astype(float)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=512, hop_length=256)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])

def segment_signal(signal, window_size=2048, hop=1024):
    segments = []
    for start in range(0, len(signal) - window_size, hop):
        segments.append(signal[start:start + window_size])
    return np.array(segments)

st.title("🔧 Acoustic Fingerprinting Dashboard")
st.write("Predictive Bearing Failure Detection using ML")

data_files = [f for f in os.listdir('data') if f.endswith('.mat')]
selected_file = st.selectbox("Select a bearing signal file to analyze:", data_files)

if st.button("Analyze Signal"):
    filepath = os.path.join('data', selected_file)
    signal = get_signal(filepath)

    # --- Single-reading snapshot (current status) ---
    segment = signal[:2048]
    features = extract_mfcc_features(segment)
    features_scaled = scaler.transform([features])

    fault_pred = fault_model.predict(features_scaled)[0]
    fault_proba = np.max(fault_model.predict_proba(features_scaled))

    severity_pred = severity_model.predict(features_scaled)[0]
    severity_proba = np.max(severity_model.predict_proba(features_scaled))

    st.subheader("Current Status")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Fault Type", fault_pred)
        st.caption(f"{fault_proba*100:.1f}% confidence")
    with col2:
        st.metric("Severity Stage", severity_pred)
        st.caption(f"{severity_proba*100:.1f}% confidence")

    if severity_pred == "Critical":
        st.error("⚠️ RECOMMENDATION: Schedule maintenance immediately")
    elif severity_pred == "Moderate":
        st.warning("⚠️ RECOMMENDATION: Schedule maintenance soon")
    elif severity_pred == "Early_Wear":
        st.info("ℹ️ RECOMMENDATION: Monitor closely")
    else:
        st.success("✅ Bearing is healthy")

    # --- Trend analysis across the whole file ---
    st.subheader("Trend Over Time")
    st.caption("Simulating a sequence of live readings from this file")

    all_segments = segment_signal(signal)
    stage_order = {'Healthy': 0, 'Early_Wear': 1, 'Moderate': 2, 'Critical': 3}

    all_features = np.array([extract_mfcc_features(s) for s in all_segments])
    all_features_scaled = scaler.transform(all_features)
    severity_preds = severity_model.predict(all_features_scaled)
    severity_probas = severity_model.predict_proba(all_features_scaled)

    severity_scores = []
    for pred, probs in zip(severity_preds, severity_probas):
        stage_num = stage_order[pred]
        confidence = np.max(probs)
        severity_scores.append(stage_num + (confidence - 0.5))

    # Smooth the scores for a clean visual trend
    smooth_window = min(50, len(severity_scores))
    if smooth_window > 1:
        smoothed = np.convolve(severity_scores, np.ones(smooth_window)/smooth_window, mode='valid')
    else:
        smoothed = severity_scores

    st.line_chart(smoothed)

    if len(smoothed) >= 2:
        overall_trend = smoothed[-1] - smoothed[0]
        if overall_trend > 0.05:
            st.warning("📈 Overall trend: RISING — condition worsening over this recording")
        elif overall_trend < -0.05:
            st.success("📉 Overall trend: FALLING — condition improving over this recording")
        else:
            st.info("➡️ Overall trend: STABLE")