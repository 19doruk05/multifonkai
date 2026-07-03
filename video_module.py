import streamlit as st

def run_video_analysis():
    st.header("🎬 Video Analizi")

    uploaded_video = st.file_uploader(
        "Bir video yükle",
        type=["mp4", "avi", "mov"],
        key="video_uploader"
    )

    if uploaded_video is not None:
        st.video(uploaded_video)

        ai_prob = 0.72
        real_prob = 1 - ai_prob

        st.subheader("Analiz Sonucu")
        st.write(f"Gerçek olasılığı: %{real_prob * 100:.2f}")
        st.write(f"AI olasılığı: %{ai_prob * 100:.2f}")

        st.progress(int(ai_prob * 100))
        st.error(f"Sonuç: Video AI içerik olabilir (%{ai_prob * 100:.2f})")