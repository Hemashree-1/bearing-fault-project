from scipy.io import loadmat
import numpy as np
import os

def get_signal(filepath):
    mat = loadmat(filepath)
    de_key = [key for key in mat.keys() if 'DE_time' in key][0]
    return mat[de_key].flatten()

def segment_signal(signal, window_size=2048, hop=1024):
    segments = []
    for start in range(0, len(signal) - window_size, hop):
        segments.append(signal[start:start + window_size])
    return np.array(segments)

# Map filename prefixes to labels
def get_label(filename):
    # New load-0 fault files
    new_fault_map = {
        '105.mat': 'Inner_Race_Fault',   # IR007_0
        '118.mat': 'Ball_Fault',          # B007_0
        '130.mat': 'Outer_Race_Fault',    # OR007@6_0
        '169.mat': 'Inner_Race_Fault',    # IR014_0
        '185.mat': 'Ball_Fault',          # B014_0
        '197.mat': 'Outer_Race_Fault',    # OR014@6_0
        '209.mat': 'Inner_Race_Fault',    # IR021_0
        '222.mat': 'Ball_Fault',          # B021_0
        '234.mat': 'Outer_Race_Fault',    # OR021@6_0
    }
    if filename in new_fault_map:
        return new_fault_map[filename]

    if filename.startswith('B'):
        return 'Ball_Fault'
    elif filename.startswith('IR'):
        return 'Inner_Race_Fault'
    elif filename.startswith('OR'):
        return 'Outer_Race_Fault'
    elif filename.startswith('Time_Normal'):
        return 'Healthy'
    elif filename in ['97.mat', '98.mat', '99.mat', '100.mat']:
        return 'Healthy'
    else:
        return 'Unknown'

# Process all files in the data folder
all_files = []
all_segments = []
all_labels = []

for filename in os.listdir('data'):
    filepath = os.path.join('data', filename)
    signal = get_signal(filepath)
    segments = segment_signal(signal)
    label = get_label(filename)

    all_segments.append(segments)
    all_labels.extend([label] * len(segments))
    all_files.extend([filename] * len(segments))

    print(f"{filename}: {len(segments)} segments -> label: {label}")

# Combine everything into one big array
X_raw = np.vstack(all_segments)
y_raw = np.array(all_labels)

print("\nTotal segments:", X_raw.shape[0])
print("Segment length:", X_raw.shape[1])
print("\nLabel counts:")
unique, counts = np.unique(y_raw, return_counts=True)
for label, count in zip(unique, counts):
    print(f" {label}: {count}")

# Save this for the next step so we don't have to redo it every time
np.save('X_raw.npy', X_raw)
np.save('y_raw.npy', y_raw)
np.save('file_groups.npy', np.array(all_files))
print("\nSaved X_raw.npy and y_raw.npy")