import numpy as np

# Load what we already have
groups = np.load('file_groups.npy')

# Map every filename to a severity stage
def get_severity(filename):
    healthy_files = ['97.mat', '98.mat', '99.mat', '100.mat', 'Time_Normal_1_098.mat']
    early_wear_files = ['B007_1_123.mat', 'IR007_1_110.mat', 'OR007_6_1_136.mat',
                         '105.mat', '118.mat', '130.mat']  # 0.007" severity
    moderate_files = ['B014_1_190.mat', 'IR014_1_175.mat', 'OR014_6_1_202.mat',
                       '169.mat', '185.mat', '197.mat']  # 0.014" severity
    critical_files = ['B021_1_227.mat', 'IR021_1_214.mat', 'OR021_6_1_239.mat',
                       '209.mat', '222.mat', '234.mat']  # 0.021" severity

    if filename in healthy_files:
        return 'Healthy'
    elif filename in early_wear_files:
        return 'Early_Wear'
    elif filename in moderate_files:
        return 'Moderate'
    elif filename in critical_files:
        return 'Critical'
    else:
        return 'Unknown'

# Apply to every segment based on its source file
y_severity = np.array([get_severity(f) for f in groups])

print("Severity label counts:")
unique, counts = np.unique(y_severity, return_counts=True)
for label, count in zip(unique, counts):
    print(f" {label}: {count}")

# Check for any 'Unknown' - would mean a filename we missed
unknown_count = np.sum(y_severity == 'Unknown')
if unknown_count > 0:
    print(f"\nWARNING: {unknown_count} segments have Unknown severity - check filenames!")

np.save('y_severity.npy', y_severity)
print("\nSaved y_severity.npy")