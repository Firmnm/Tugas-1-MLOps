from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import skops.io as sio

# Dummy training data
X = np.random.rand(100, 7)
y = np.random.choice(["Extrovert", "Introvert", "Ambivert"], 100)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

model = RandomForestClassifier()
model.fit(X_scaled, y_encoded)

# Save the model
sio.dump(model, "Model/personality_classifier_new.skops")
sio.dump(encoder, "Model/label_encoder_new.skops")

print("✅ Model trained and saved.")
