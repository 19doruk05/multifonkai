import copy
import random
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import ImageFile
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split


print("1- Kod başladı")

full_dataset = datasets.ImageFolder("dataset/image")
print("2- Dataset okundu:", len(full_dataset))

class_names = full_dataset.classes
print("3- Sınıflar:", class_names)

model = models.efficientnet_b0(weights="IMAGENET1K_V1")
print("4- Model yüklendi")

ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 42
BATCH_SIZE = 16
EPOCHS_HEAD = 6
EPOCHS_FINE = 6
LR_HEAD = 1e-3
LR_FINE = 1e-4
VAL_RATIO = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
torch.manual_seed(SEED)

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(8),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

full_dataset = datasets.ImageFolder("dataset/image")
class_names = full_dataset.classes
print("Sınıf sırası:", class_names)

total_size = len(full_dataset)
val_size = int(total_size * VAL_RATIO)
train_size = total_size - val_size

train_dataset, val_dataset = random_split(
    full_dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(SEED)
)

train_dataset.dataset = copy.deepcopy(full_dataset)
train_dataset.dataset.transform = train_transform

val_dataset.dataset = copy.deepcopy(full_dataset)
val_dataset.dataset.transform = val_transform

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = models.efficientnet_b0(weights="IMAGENET1K_V1")
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_count += labels.size(0)

    acc = total_correct / total_count if total_count else 0
    return total_loss, acc

best_val_acc = 0.0
best_model_state = None

# 1) Sadece classifier eğit
for param in model.features.parameters():
    param.requires_grad = False

optimizer = optim.Adam(model.classifier.parameters(), lr=LR_HEAD)

for epoch in range(EPOCHS_HEAD):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / train_total if train_total else 0
    val_loss, val_acc = evaluate(model, val_loader)

    print(f"[HEAD] Epoch {epoch+1}/{EPOCHS_HEAD} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = copy.deepcopy(model.state_dict())

        print(">>> HEAD bitti, FINE aşaması başlıyor...")


# 2) Son blokları aç, fine-tune et

print(f">>> Fine epoch {epoch+1} başladı")

for param in model.features[-2:].parameters():
    param.requires_grad = True

optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_FINE)

for epoch in range(EPOCHS_FINE):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        preds = outputs.argmax(dim=1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_acc = train_correct / train_total if train_total else 0
    val_loss, val_acc = evaluate(model, val_loader)

    print(f"[FINE] Epoch {epoch+1}/{EPOCHS_FINE} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = copy.deepcopy(model.state_dict())

torch.save({
    "model_state": best_model_state,
    "classes": class_names
}, "model.pth")

print(f"✅ En iyi model kaydedildi. Best Val Acc: {best_val_acc:.4f}")