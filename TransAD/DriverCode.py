import tensorflow as tf
import numpy as np
import pandas as pd

# -----------------------------
# Data Preparation
# -----------------------------
def preprocess_data(file_path, window_size=10):
    """ Load and preprocess the dataset with verbose logging """
    print("Loading dataset...")
    df = pd.read_csv(file_path)
    print(f"Original dataset shape: {df.shape}")
    
    # Drop non-numeric columns explicitly
    df = df.drop(columns=["hostname", "name", "Timestamp"], errors="ignore")
    df = df.select_dtypes(include=[np.number])  # Keep only numeric columns
    
    print(f"Dataset shape after dropping categorical columns: {df.shape}")
    
    # Handle NaN & Infinite Values
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)
    
    data = df.values.astype(np.float32)  # Convert to float32
    print("Data converted to float32")
    
    # Normalize data
    print("Normalizing data...")
    data = (data - np.min(data, axis=0)) / (np.max(data, axis=0) - np.min(data, axis=0) + 1e-8)
    
    # Create sliding windows
    print("Creating sliding windows...")
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])
        Y.append(data[i + window_size])
    
    print(f"Generated {len(X)} training samples")
    return np.array(X), np.array(Y), df.shape[1]  # Return correct feature count

# -----------------------------
# Transformer-based Encoder-Decoder
# -----------------------------
def build_tranad_model(input_dim, hidden_dim=64, num_heads=4):
    print("Building TranAD model...")
    inputs = tf.keras.Input(shape=(None, input_dim))
    x = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim)(inputs, inputs)
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + inputs)
    x = tf.keras.layers.Dense(hidden_dim, activation='relu')(x)
    outputs = tf.keras.layers.Dense(input_dim)(x)
    model = tf.keras.Model(inputs, outputs)
    print("Model built successfully!")
    return model

# -----------------------------
# Training with Adversarial Learning
# -----------------------------
def train_tranad(model, file_path, epochs=10, lr=0.001):
    print("Starting training...")
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.MeanSquaredError()
    
    X_train, _, feature_count = preprocess_data(file_path)
    dataset = tf.data.Dataset.from_tensor_slices((X_train, X_train)).batch(128).shuffle(1000)
    
    for epoch in range(epochs):
        total_loss = 0
        print(f"Epoch {epoch+1}/{epochs}")
        for step, (x, _) in enumerate(dataset):
            with tf.GradientTape() as tape:
                output_1 = model(x, training=True)
                loss_1 = loss_fn(x, output_1)
                focus_score = tf.abs(output_1 - x)
                output_2 = model(x + focus_score, training=True)
                loss_2 = loss_fn(x, output_2)
                loss = 0.5 * loss_1 + 0.5 * loss_2
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            total_loss += loss.numpy()
            if step % 10 == 0:
                print(f"Step {step}, Loss: {loss.numpy():.6f}")
        print(f"Epoch {epoch+1} completed, Average Loss: {total_loss / len(dataset):.6f}")

# -----------------------------
# Anomaly Detection (Inference)
# -----------------------------
def detect_anomalies(model, file_path, threshold=0.05):
    print("Detecting anomalies...")
    X_test, _, _ = preprocess_data(file_path)
    print("Running model inference...")
    reconstructions = model.predict(X_test, verbose=1)
    anomaly_scores = np.mean(np.abs(reconstructions - X_test), axis=(1, 2))
    anomalies = anomaly_scores > threshold
    print(f"Total anomalies detected: {np.sum(anomalies)}")
    return np.where(anomalies)[0], anomaly_scores

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    file_path = "/content/blazar_api.csv"  # Dataset file
    
    print("Initializing model...")
    _, _, feature_count = preprocess_data(file_path)
    model = build_tranad_model(input_dim=feature_count)
    model.compile(optimizer='adam', loss='mse')
    print("Training model...")
    train_tranad(model, file_path, epochs=1)
    
    print("Running anomaly detection...")
    anomalies, scores = detect_anomalies(model, file_path)
    print("Detected Anomalies:", anomalies)
