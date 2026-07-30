import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load features + severity labels
X = np.load('X_features.npy')
y = np.load('y_severity.npy')
groups = np.load('file_groups.npy')

print("Loaded data:", X.shape, y.shape)

# TEST 1: Standard random split
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
print(f"\nRandom-split severity accuracy: {random_split_acc*100:.2f}%")
print(classification_report(y_test, preds, zero_division=0))

# TEST 2: Leave-One-File-Out (strict)
logo = LeaveOneGroupOut()
accuracies = []
for train_idx, test_idx in logo.split(X, y, groups=groups):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)
    m = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    m.fit(X_tr_s, y_tr)
    p = m.predict(X_te_s)
    accuracies.append(accuracy_score(y_te, p))

strict_acc = np.mean(accuracies)
print(f"\nStrict cross-file severity accuracy: {strict_acc*100:.2f}%")