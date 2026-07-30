import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

X = np.load('X_features.npy')
y_fault = np.load('y_labels.npy')
y_severity = np.load('y_severity.npy')

# Train final fault-type model on ALL data (since we've already validated it properly)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

fault_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
fault_model.fit(X_scaled, y_fault)

severity_model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
severity_model.fit(X_scaled, y_severity)

# Save everything needed to use these models later
joblib.dump(fault_model, 'fault_model.pkl')
joblib.dump(severity_model, 'severity_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Saved fault_model.pkl, severity_model.pkl, and scaler.pkl")
print("These 3 files are everything needed to make predictions on new data later.")