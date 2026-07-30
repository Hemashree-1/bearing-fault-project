from scipy.io import loadmat
import numpy as np

def get_signal(filepath):
    mat = loadmat(filepath)
    de_key = [key for key in mat.keys() if 'DE_time' in key][0]
    return mat[de_key].flatten()

def segment_signal(signal, window_size=2048, hop=1024):
    segments = []
    for start in range(0, len(signal) - window_size, hop):
        segment = signal[start:start + window_size]
        segments.append(segment)
    return np.array(segments)

# Test on our faulty file
signal = get_signal('data/B007_1_123.mat')
segments = segment_signal(signal)

print("Original signal length:", len(signal))
print("Number of segments created:", segments.shape[0])
print("Each segment length:", segments.shape[1])