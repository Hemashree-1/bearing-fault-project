import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

X = np.load('X_features.npy')
y = np.load('y_severity.npy')
groups = np.load('file_groups.npy')

# Train a model on everything except one file, then simulate "live readings" from that file
test_file = 'B021_1_227.mat'  # Critical severity, held out entirely
train_mask = groups != test_file
X_train, y_train = X[train_mask], y[train_mask]
X_test = X[groups == test_file]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
model.fit(X_train_scaled, y_train)

# Get severity stage AND confidence for each segment, in order (simulating time-ordered readings)
stage_order = {'Healthy': 0, 'Early_Wear': 1, 'Moderate': 2, 'Critical': 3}
predictions = model.predict(X_test_scaled)
probabilities = model.predict_proba(X_test_scaled)

# Convert predictions to a numeric "severity score" using max probability weighted by stage
severity_scores = []
for pred, probs in zip(predictions, probabilities):
    stage_num = stage_order[pred]
    confidence = np.max(probs)
    severity_scores.append(stage_num + (confidence - 0.5))  # nudge score by confidence

# Simulate a rolling window: look at the trend over the last 5 readings
# Smooth the severity scores using a moving average - reduces noise before checking trend
smooth_window = 50
severity_scores = np.array(severity_scores)
smoothed = np.convolve(severity_scores, np.ones(smooth_window)/smooth_window, mode='valid')

# Compare readings spaced further apart, with a threshold to avoid false "trend" on tiny noise
step = 50
threshold = 0.05
print(f"Simulating live readings from {test_file}:\n")

prev_score = None
for i in range(0, len(smoothed), step):
    current_score = smoothed[i]
    stage_at_i = predictions[i]

    if prev_score is None:
        trend = "N/A (first reading)"
    elif current_score - prev_score > threshold:
        trend = "RISING (getting worse)"
    elif prev_score - current_score > threshold:
        trend = "FALLING (improving)"
    else:
        trend = "STABLE"

    print(f"Reading {i:4d}: Stage={stage_at_i:12s} Score={current_score:.2f} Trend={trend}")
    prev_score = current_score