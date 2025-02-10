import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 🚀 Load Data
file_path = "blazar_api.csv"
df = pd.read_csv(file_path)

# Convert Timestamp to datetime
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Drop non-numeric columns
df_numeric = df.drop(columns=["hostname", "name", "Timestamp"], errors="ignore")

# ✅ Handle NaN & Infinite Values
df_numeric = df_numeric.replace([np.inf, -np.inf], 0)
df_numeric = df_numeric.fillna(0)

# ✅ Normalize Data using StandardScaler (better stability)
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

# Split Data (First 80% for training, last 20% for testing)
split_idx = int(len(df_scaled) * 0.8)
train_data, test_data = df_scaled[:split_idx], df_scaled[split_idx:]

# ✅ Define Autoencoder Model with LeakyReLU
input_dim = train_data.shape[1]
encoding_dim = 32  # Bottleneck layer

autoencoder = keras.Sequential([
    keras.layers.InputLayer(input_shape=(input_dim,)),
    keras.layers.Dense(256),
    keras.layers.LeakyReLU(alpha=0.1),
    keras.layers.Dense(128),
    keras.layers.LeakyReLU(alpha=0.1),
    keras.layers.Dense(64),
    keras.layers.LeakyReLU(alpha=0.1),
    keras.layers.Dense(encoding_dim),
    keras.layers.LeakyReLU(alpha=0.1),  # Bottleneck layer
    keras.layers.Dense(64),
    keras.layers.LeakyReLU(alpha=0.1),
    keras.layers.Dense(128),
    keras.layers.LeakyReLU(alpha=0.1),
    keras.layers.Dense(256),
    keras.layers.LeakyReLU(alpha=0.1),
    keras.layers.Dense(input_dim, activation="sigmoid")  # Output layer
])

# ✅ Compile Model with Lower Learning Rate
autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="mse")

# ✅ Debug: Check Model Output Before Training
sample_input = np.random.rand(1, train_data.shape[1])
print("Sample Model Output (Before Training):", autoencoder.predict(sample_input))

# 🚀 Train Autoencoder
history = autoencoder.fit(train_data, train_data, 
                          epochs=50, batch_size=64, 
                          validation_data=(test_data, test_data), 
                          verbose=1)

# ✅ Plot Training Loss
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.title("Autoencoder Training Loss")
plt.show()

# ✅ Detect Anomalies
reconstructed = autoencoder.predict(test_data)
mse = np.mean(np.power(test_data - reconstructed, 2), axis=1)

# ✅ Set Anomaly Threshold (Mean + 3 Std)
threshold = mse.mean() + 3 * mse.std()
anomalies = mse > threshold

# ✅ Plot Anomaly Scores
plt.figure(figsize=(12, 5))
plt.plot(mse, label="Reconstruction Error")
plt.axhline(threshold, color="r", linestyle="--", label="Anomaly Threshold")
plt.legend()
plt.title("Reconstruction Error for Anomaly Detection")
plt.show()

# ✅ Find Timestamps of Anomalies
anomaly_indices = np.where(anomalies)[0]
anomaly_timestamps = df["Timestamp"].iloc[split_idx:].iloc[anomaly_indices]

# Print Detected Anomaly Timestamps
print("🚨 Detected Anomalies at:")
print(anomaly_timestamps)
