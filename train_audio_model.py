import os
import librosa
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATASET_PATH = "dataset/audio"

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

X = []
y = []

for label in ["real", "fake"]:
    folder = os.path.join(DATASET_PATH, label)

    if not os.path.exists(folder):
        print(f"Hata: {folder} klasörü bulunamadı.")
        continue

    for filename in os.listdir(folder):
        if filename.endswith(".wav"):
            file_path = os.path.join(folder, filename)
            try:
                features = extract_features(file_path)
                X.append(features)
                y.append(0 if label == "real" else 1)
            except Exception as e:
                print(f"Hata oluştu: {file_path} -> {e}")

X = np.array(X)
y = np.array(y)

if len(X) == 0:
    print("Hiç veri bulunamadı.")
    exit()

print("Toplam örnek sayısı:", len(X))
print("Real sayısı:", np.sum(y == 0))
print("Fake sayısı:", np.sum(y == 1))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("\n🔍 Test Sonucu:")
print(classification_report(y_test, y_pred, target_names=["real", "fake"]))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(clf, "audio_model.pkl")
print("\n✅ Model kaydedildi: audio_model.pkl")