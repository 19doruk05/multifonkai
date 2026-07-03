import streamlit as st
from image_module import run_image_analysis
from audio_module import run_audio_analysis
from text_module import run_text_analysis
from video_module import run_video_analysis

st.set_page_config(page_title="AI İçerik Tespit Platformu", layout="wide")

st.title("🧠 AI İçerik Tespit Platformu")
st.write("Görsel, ses, metin ve video içeriklerinde AI üretimi tespiti")

tab1, tab2, tab3, tab4 = st.tabs([
    "🖼 Görsel Analizi",
    "🎵 Ses Analizi",
    "📝 Metin Analizi",
    "🎬 Video Analizi"
])

with tab1:
    run_image_analysis()

with tab2:
    run_audio_analysis()

with tab3:
    run_text_analysis()

with tab4:
    run_video_analysis()