import matplotlib.pyplot as plt
import numpy as np

# Disable LaTeX formatting for matplotlib
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

def plot_gyroscope_measurements(times, gyro_measurements, save_path="plots/measurements/gyroscope_measurements.png"):
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    labels = ['Gyro X', 'Gyro Y', 'Gyro Z']
    for i in range(3):
        axs[i].scatter(times[:len(gyro_measurements)], gyro_measurements[:, i, 0], label=labels[i], color='red', marker='x', s=15)
        axs[i].set_title(f"{labels[i]} vs Time", fontsize=14)
        axs[i].set_ylabel("Angular Velocity (rad/s)", fontsize=12)
        axs[i].grid(True)
        axs[i].legend(fontsize=10)

    axs[2].set_xlabel("Time (s)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_magnetometer_measurements(times, magnetometer_measurements, save_path="plots/measurements/magnetometer_measurements.png"):
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    labels = ['Magnetometer X (μT)', 'Magnetometer Y (μT)', 'Magnetometer Z (μT)']
    for i in range(3):
        valid_indices = np.isfinite(magnetometer_measurements[:, i, 0])
        axs[i].scatter(times[valid_indices], magnetometer_measurements[valid_indices, i, 0], 
                       label=labels[i], color='red', marker='x', s=15)
        axs[i].set_title(f"{labels[i]} vs Time", fontsize=14)
        axs[i].set_ylabel("Magnetometer (μT)", fontsize=12)
        axs[i].grid(True)
        axs[i].legend(fontsize=10)

    axs[2].set_xlabel("Time (s)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_accelerometer_measurements(times, accelerometer_measurements, save_path="plots/measurements/accelerometer_measurements.png"):
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    labels = ['Acceleration X', 'Acceleration Y', 'Acceleration Z']
    for i in range(3):
        valid_indices = np.isfinite(accelerometer_measurements[:, i, 0])
        axs[i].scatter(times[valid_indices], accelerometer_measurements[valid_indices, i, 0], label=labels[i], color='red', marker='x', s=15)
        axs[i].set_title(f"{labels[i]} vs Time", fontsize=14)
        axs[i].set_ylabel("Acceleration (m/s²)", fontsize=12)
        axs[i].grid(True)
        axs[i].legend(fontsize=10)

    axs[2].set_xlabel("Time (s)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_gps_measurements(times, gps_measurements, save_path="plots/measurements/gps_measurements.png"):
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    labels = ['lat', 'lon', 'alt']
    for i in range(3):
        valid_indices = np.isfinite(gps_measurements[:, i, 0])
        axs[i].scatter(times[valid_indices], gps_measurements[valid_indices, i, 0], label=labels[i], color='red', marker='x', s=15)
        axs[i].set_title(f"{labels[i]} vs Time", fontsize=14)
        axs[i].set_ylabel("hmmm", fontsize=12)
        axs[i].grid(True)
        axs[i].legend(fontsize=10)

    axs[2].set_xlabel("Time (s)", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)






def plot_barometer_measurements(times, barometer_measurements, save_path="plots/measurements/barometer_measurements.png"):
    fig, ax = plt.subplots(figsize=(10, 7))
    valid_indices = np.isfinite(barometer_measurements[:, 0])
    ax.scatter(times[valid_indices], barometer_measurements[valid_indices], label='Barometer Measurements', color='red', marker='x', s=15)
    ax.set_title("Barometer vs Time", fontsize=16)
    ax.set_xlabel("Time (s)", fontsize=14)
    ax.set_ylabel("Altitude (m)", fontsize=14)
    ax.grid(True)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_measurements(data):
    """
    Plot all sensor measurements.

    Parameters:
    - data: A dictionary containing times and sensor measurements.
    """
    plot_accelerometer_measurements(data['times'], data['accelerometer_measurements'])
    plot_barometer_measurements(data['times'], data['barometer_measurements'])
    plot_gps_measurements(data['times'], data['gps_measurements'])
    plot_gyroscope_measurements(data['times'], data['gyroscope_measurements'])
    plot_magnetometer_measurements(data['times'], data['magnetometer_measurements'])
