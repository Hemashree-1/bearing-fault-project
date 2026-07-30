import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load our prepared data
X = np.load('X_features.npy')
y = np.load('y_labels.npy')
groups = np.load('file_groups.npy')

print("Loaded data:", X.shape, y.shape)

# ============================================
# TEST 1: Standard random split (comparable to most published papers)
# ============================================
print("\n" + "="*60)
print("TEST 1: Standard random split (segment-level)")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
model.fit(X_train_scaled, y_train)
preds = model.predict(X_test_scaled)
random_split_acc = accuracy_score(y_test, preds)

print(f"Random-split accuracy: {random_split_acc*100:.2f}%")

# ============================================
# TEST 2: Leave-One-File-Out (strict, no leakage, tests true generalization)
# ============================================
print("\n" + "="*60)
print("TEST 2: Leave-One-File-Out (file-level, strict generalization test)")
print("="*60)

logo = LeaveOneGroupOut()
accuracies = []

for train_idx, test_idx in logo.split(X, y, groups=groups):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    test_file = np.unique(groups[test_idx])[0]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    accuracies.append(acc)
    print(f"Test file: {test_file:25s} Accuracy: {acc*100:.2f}%")

strict_acc = np.mean(accuracies)

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Standard random-split accuracy:     {random_split_acc*100:.2f}%")
print(f"Strict cross-file (LOGO) accuracy:  {strict_acc*100:.2f}%")
print(f"Generalization gap:                 {(random_split_acc - strict_acc)*100:.2f} percentage points")