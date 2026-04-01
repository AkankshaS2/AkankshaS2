import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import models, layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

#  Extract MFCC features
def extract_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000, duration=3)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc = np.mean(mfcc.T, axis=0)  # fixed size
        return mfcc
    except:
        return None

#  Load dataset
def load_data(dataset_path):
    X, y = [], []

    classes = {
        "non_scream": 0,
        "scream": 1
    }

    for folder, label in classes.items():
        folder_path = os.path.join(dataset_path, folder)

        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            features = extract_features(file_path)

            if features is not None:
                X.append(features)
                y.append(label)

    return np.array(X), np.array(y)

#  Load data
print("Loading data...")
X, y = load_data("dataset")

print("Total samples:", len(X))

#  Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#  Build model
model = models.Sequential([
    Dense(64, activation='relu', input_shape=(40,)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# compile
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 📌 Train
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# 📌 Evaluate
loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)

# 📌 Save model
model.save("scream_model.keras")

print("✅ Training complete!")