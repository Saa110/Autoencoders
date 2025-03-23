import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load Dataset
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df["stv"] = df["stv"].apply(lambda x: np.array(eval(x), dtype=np.float32).reshape(-1, 33))
    
    all_values = np.vstack(df["stv"].values)
    min_val, max_val = all_values.min(), all_values.max()
    df["stv"] = df["stv"].apply(lambda x: (x - min_val) / (max_val - min_val))
    
    X = np.stack(df["stv"].values)
    Y = df["anomaly_score"].values
    
    return X, Y

# F1 Score Metric
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name="f1_score", **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
    
    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))
    
    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

# Transformer-based TranAD Model
def build_tranad_model(sequence_length, feature_dim, num_heads=4, dff=64):
    inputs = tf.keras.Input(shape=(sequence_length, feature_dim))
    attention = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=feature_dim)(inputs, inputs)
    x = tf.keras.layers.LayerNormalization()(inputs + attention)
    
    feed_forward = tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(feature_dim)
    ])(x)
    x = tf.keras.layers.LayerNormalization()(x + feed_forward)
    
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x[:, -1, :])
    return tf.keras.Model(inputs, outputs)

# Training Function
def train_model(model, X_train, Y_train, X_test, Y_test, batch_size=32, epochs=20):
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'Precision', 'Recall', 'AUC', F1Score()])
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, Y_test))
    return history

# Plot Results
def plot_results(history):
    plt.figure(figsize=(12, 6))
    for metric in ['accuracy', 'Precision', 'Recall', 'AUC', 'f1_score']:
        plt.plot(history.history[metric], label=f'Train {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Test {metric}')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Model Metrics')
    plt.show()

# Main Execution
file_path = "/content/STVs.csv"
X, Y = load_and_preprocess_data(file_path)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

sequence_length = X.shape[1]
feature_dim = X.shape[2]
model = build_tranad_model(sequence_length, feature_dim)
history = train_model(model, X_train, Y_train, X_test, Y_test)

# Save model
model.save("tranad_model.h5")

# Plot results
plot_results(history)
