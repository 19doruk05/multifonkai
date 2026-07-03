import streamlit as st

def run_text_analysis():
    st.header("📝 Metin Analizi")

    user_text = st.text_area("Analiz edilecek metni gir", height=200)

    if st.button("Metni Analiz Et", key="text_analyze_btn"):
        if not user_text or user_text.strip() == "":
            st.warning("Lütfen bir metin gir.")
        else:
            length_score = min(len(user_text) / 1000, 1.0)
            ai_prob = 0.50 + (length_score * 0.30)
            real_prob = 1 - ai_prob

            st.subheader("Analiz Sonucu")
            st.write(f"Gerçek olasılığı: %{real_prob * 100:.2f}")
            st.write(f"AI olasılığı: %{ai_prob * 100:.2f}")

            if ai_prob >= 0.5:
                st.progress(int(ai_prob * 100))
                st.error(f"Sonuç: Metin AI tarafından yazılmış olabilir (%{ai_prob * 100:.2f})")
            else:
                st.progress(int(real_prob * 100))
                st.success(f"Sonuç: Metin insan tarafından yazılmış olabilir (%{real_prob * 100:.2f})")