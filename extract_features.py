import numpy as np
import librosa

# Load what we built in the last step
X_raw = np.load('X_raw.npy')
y_raw = np.load('y_raw.npy')

def extract_mfcc_features(signal, sr=12000, n_mfcc=13):
    signal = signal.astype(float)
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=512, hop_length=256)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    return np.concatenate([mfcc_mean, mfcc_std])

# Extract features for ALL segments
print("Extracting features for all segments... this may take a minute or two")

all_features = []
for i, segment in enumerate(X_raw):
    features = extract_mfcc_features(segment)
    all_features.append(features)
    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1} / {len(X_raw)} segments")

X_features = np.array(all_features)

print("\nFinal feature matrix shape:", X_features.shape)

# Save for the training step
np.save('X_features.npy', X_features)
np.save('y_labels.npy', y_raw)
print("Saved X_features.npy and y_labels.npy")