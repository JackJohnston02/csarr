#!/usr/bin/env python3

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Residual Plot Function
def plot_residuals_and_save(X_val, y_val_cb, y_pred_cb, plot_path, csv_path):
    """
    Plots a scatter of (altitude, velocity) colored by residuals (predicted - true).
    Saves the plot as a PNG and residual data as a CSV.

    Parameters
    ----------
    X_val : np.ndarray
        Validation set features, i.e., [altitude, velocity].
    y_val_cb : np.ndarray
        True ballistic coefficients in real space.
    y_pred_cb : np.ndarray
        Predicted ballistic coefficients in real space.
    plot_path : str
        Path to save the scatter plot as PNG.
    csv_path : str
        Path to save the residual data as CSV.
    """
    # Calculate residuals
    residuals = y_pred_cb - y_val_cb

    # Extract altitude and velocity
    altitudes = X_val[:, 0]
    velocities = X_val[:, 1]

    # Plot scatter
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(velocities, altitudes, c=residuals, cmap='bwr', edgecolor='k', marker='o')
    cbar = plt.colorbar(sc)
    cbar.set_label("Residuals (Predicted - True)")
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Altitude (m)")
    plt.title("Residuals Scatter Plot")
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Residuals scatter plot saved at: {plot_path}")

    # Save residual data as CSV
    residual_data = pd.DataFrame({
        "Altitude": altitudes,
        "Velocity": velocities,
        "True_Cb": y_val_cb,
        "Predicted_Cb": y_pred_cb,
        "Residuals": residuals
    })
    residual_data.to_csv(csv_path, index=False)
    print(f"Residuals data saved at: {csv_path}")

# Build and train NN
def build_and_train_nn(X_train, y_train, X_val, y_val,
                       epochs=1000, batch_size=32, learning_rate=0.005):
    model = Sequential([
        Dense(32, activation='relu', input_shape=(2,)),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='mean_squared_error', metrics=['mae'])

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                        epochs=epochs, batch_size=batch_size, verbose=0)

    return model, history

# Main workflow
def main():
    # Generate synthetic training and validation data
    np.random.seed(42)
    num_samples = 1000
    altitudes = np.random.uniform(0, 2000, num_samples)
    velocities = np.random.uniform(0, 300, num_samples)
    true_cb = 1.5 * altitudes + 2.5 * velocities + np.random.normal(0, 10, num_samples)

    X = np.column_stack((altitudes, velocities))
    y = np.log(true_cb)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Build and train the model
    model, history = build_and_train_nn(X_train_scaled, y_train, X_val_scaled, y_val,
                                        epochs=1000, batch_size=32, learning_rate=0.005)

    # Evaluate the model
    y_pred_log = model.predict(X_val_scaled).flatten()
    y_pred_cb = np.exp(y_pred_log)
    y_val_cb = np.exp(y_val)

    # Plot and save residuals
    plot_path = "residuals_scatter_plot.png"
    csv_path = "residuals_data.csv"
    plot_residuals_and_save(X_val, y_val_cb, y_pred_cb, plot_path, csv_path)

if __name__ == "__main__":
    main()
