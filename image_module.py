import os
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

MODEL_PATH = "model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(DEVICE)
    model.eval()

    class_names = checkpoint.get("classes", ["fake", "real"])
    return model, class_names

def predict_image(image, model, class_names):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    result = {}
    for i, class_name in enumerate(class_names):
        result[class_name] = float(probs[i].item())

    predicted_index = torch.argmax(probs).item()
    predicted_class = class_names[predicted_index]
    confidence = float(probs[predicted_index].item())

    return predicted_class, confidence, result

def run_image_analysis():
    st.header("🖼 Görsel Analizi")

    uploaded_file = st.file_uploader(
        "Bir görsel yükle",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Yüklenen Görsel", use_container_width=True)

        model, class_names = load_model()

        if model is None:
            st.error("model.pth bulunamadı.")
            return

        try:
            predicted_class, confidence, result = predict_image(image, model, class_names)

            st.subheader("Analiz Sonucu")

            for class_name, prob in result.items():
                st.write(f"{class_name} olasılığı: %{prob * 100:.2f}")

            st.progress(int(confidence * 100))

            if predicted_class.lower() == "fake":
                st.error(f"Sonuç: Görsel AI üretimi olabilir (%{confidence * 100:.2f})")
            elif predicted_class.lower() == "real":
                st.success(f"Sonuç: Görsel gerçek olabilir (%{confidence * 100:.2f})")
            else:
                st.info(f"Sonuç: {predicted_class} (%{confidence * 100:.2f})")

        except Exception as e:
            st.error(f"Görsel analizinde hata oluştu: {e}")