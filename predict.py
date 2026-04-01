import numpy as np
import librosa
import sounddevice as sd
from tensorflow.keras.models import load_model

# load model
model = load_model("scream_model.h5")

# record audio
def record_audio(duration=3, sr=16000):
    print("🎤 Listening...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    return audio.flatten()

# extract MFCC
def extract_features(audio, sr=16000):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc.reshape(1, -1)

# predict
while True:
    audio = record_audio()
    features = extract_features(audio)

    prediction = model.predict(features)[0][0]

    if prediction > 0.5:
        print("🚨 DISTRESS DETECTED!")
    else:
        print("✅ Normal sound")

    stop = input("Press Enter to continue or type 'q' to quit: ")
    if stop.lower() == 'q':
        break