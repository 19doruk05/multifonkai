import os
import tempfile
import streamlit as st
import librosa
import numpy as np
import joblib

MODEL_PATH = "audio_model.pkl"

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050)

    if len(audio) < sr:
        audio = np.pad(audio, (0, sr - len(audio)))
    else:
        audio = audio[:sr * 3]

    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfccs)
    delta2 = librosa.feature.delta(mfccs, order=2)

    features = np.hstack([
        np.mean(mfccs.T, axis=0),
        np.std(mfccs.T, axis=0),
        np.mean(delta.T, axis=0),
        np.std(delta.T, axis=0),
        np.mean(delta2.T, axis=0),
        np.std(delta2.T, axis=0)
    ])

    return features

def run_audio_analysis():
    st.header("🎵 Ses Analizi")

    uploaded_audio = st.file_uploader(
        "Bir ses dosyası yükle",
        type=["wav", "mp3"],
        key="audio_uploader"
    )

    if uploaded_audio is not None:
        st.audio(uploaded_audio)

        if not os.path.exists(MODEL_PATH):
            st.error("audio_model.pkl bulunamadı. Önce train_audio_model.py çalıştır.")
            return

        try:
            model = joblib.load(MODEL_PATH)

            suffix = os.path.splitext(uploaded_audio.name)[1]
            if suffix == "":
                suffix = ".wav"

            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_audio.read())
                tmp_path = tmp_file.name

            features = extract_features(tmp_path).reshape(1, -1)
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]

            real_prob = probabilities[0]
            fake_prob = probabilities[1]

            os.remove(tmp_path)

            st.subheader("Analiz Sonucu")
            st.write(f"Gerçek olasılığı: %{real_prob * 100:.2f}")
            st.write(f"AI olasılığı: %{fake_prob * 100:.2f}")

            if prediction == 1:
                st.progress(int(fake_prob * 100))
                st.error(f"Sonuç: Bu ses AI üretimi olabilir (%{fake_prob * 100:.2f})")
            else:
                st.progress(int(real_prob * 100))
                st.success(f"Sonuç: Bu ses gerçek insan sesi olabilir (%{real_prob * 100:.2f})")

        except Exception as e:
            st.error(f"Ses analizinde hata oluştu: {e}")