import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

def get_model():
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model

MODEL_PATH = "model.pth"

def train_model():
    from torchvision import datasets
    from torch.utils.data import DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.ImageFolder("dataset", transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = get_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(3):
        total_loss = 0
        for imgs, labels in loader:
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        st.write(f"Epoch {epoch+1}/3 - Loss: {total_loss:.4f}")
    torch.save(model.state_dict(), MODEL_PATH)
    st.success("Model eğitildi ve kaydedildi!")
    return model

def predict(model, image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    img = transform(image).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        out = model(img)
        probs = torch.softmax(out, dim=1)[0]
    return probs

st.title("🔍 AI Görsel Doğrulama Sistemi")
st.write("Bir görsel yükle, yapay zeka mı gerçek mi anlayalım.")

if not os.path.exists(MODEL_PATH):
    st.warning("Model henüz eğitilmedi.")
    if st.button("Modeli Eğit"):
        model = train_model()
        st.session_state["model"] = model
else:
    model = get_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    st.session_state["model"] = model
    st.success("Model hazır ✅")

uploaded = st.file_uploader("Görsel yükle", type=["jpg", "jpeg", "png"])

if uploaded and "model" in st.session_state:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)
    probs = predict(st.session_state["model"], image)
    ai_prob = probs[0].item() * 100
    real_prob = probs[1].item() * 100
    st.subheader("Sonuç:")
    if ai_prob > real_prob:
        st.error(f"🤖 Bu görsel %{ai_prob:.1f} ihtimalle YAPAY ZEKA üretimi")
    else:
        st.success(f"✅ Bu görsel %{real_prob:.1f} ihtimalle GERÇEK")
    st.progress(int(ai_prob))