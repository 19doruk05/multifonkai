import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

def get_model():
    model = models.efficientnet_b0(weights="IMAGENET1K_V1")
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model

def predict(model, image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    img = transform(image).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        out = model(img)
        probs = torch.softmax(out, dim=1)[0]
    return probs

st.title("🔍 AI Görsel Doğrulama Sistemi")
st.write("Bir görsel yükle, yapay zeka mı gerçek mi anlayalım.")

@st.cache_resource
def load_model():
    from torchvision import datasets
    from torch.utils.data import DataLoader
    import os

    model = get_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder("dataset", transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    bar = st.progress(0, text="Model eğitiliyor...")
    for epoch in range(5):
        for imgs, labels in loader:
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
        bar.progress((epoch+1)*20, text=f"Epoch {epoch+1}/5 tamamlandı")
    st.success("Model eğitildi ✅")
    return model

model = load_model()

uploaded = st.file_uploader("Görsel yükle", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Yüklenen Görsel", use_container_width=True)
    probs = predict(model, image)
    ai_prob = probs[0].item() * 100
    real_prob = probs[1].item() * 100
    st.subheader("Sonuç:")
    if ai_prob > real_prob:
        st.error(f"🤖 Bu görsel %{ai_prob:.1f} ihtimalle YAPAY ZEKA üretimi")
    else:
        st.success(f"✅ Bu görsel %{real_prob:.1f} ihtimalle GERÇEK")
    st.progress(int(ai_prob))