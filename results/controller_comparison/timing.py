import matplotlib.pyplot as plt
import os

# Data: Average time to calculate control signal (in milliseconds) for each controller
controller_times = {
    "Apogee Prediction": 12.3,
    "Dead-Band": 8.7,
    "Neural Network Sliding Surface": 15.2,
    "No Anti-Chattering": 10.5,
    "Soft-Switching": 9.8,
    "Super-Twisting": 11.1
}

# Define consistent colors for controllers (same as scatter plot)
controller_colors = {
    "Apogee Prediction": "blue",
    "Dead-Band": "orange",
    "Neural Network Sliding Surface": "green",
    "No Anti-Chattering": "red",
    "Soft-Switching": "purple",
    "Super-Twisting": "brown"
}

# Set consistent plotting styles
FONT_SCALE = 1.4  # Adjust for scaling
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern'],
    'font.size': 18 * FONT_SCALE,
    'axes.titlesize': 20 * FONT_SCALE,
    'axes.labelsize': 18 * FONT_SCALE,
    'xtick.labelsize': 18 * FONT_SCALE,
    'ytick.labelsize': 18 * FONT_SCALE,
    'legend.fontsize': 18 * FONT_SCALE,
})

# Bar chart
plt.figure(figsize=(12, 8))  # Adjust figure size
controllers = list(controller_times.keys())
times = list(controller_times.values())
colors = [controller_colors[controller] for controller in controllers]  # Use predefined colors

plt.bar(controllers, times, color=colors, alpha=0.8)

# Labels and title
plt.title(r"$\textbf{Average Control Signal Calculation Time}$")
plt.xlabel("Controller")
plt.ylabel("Average Time (ms)")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(axis='y', linestyle='--', alpha=0.6)

# Save plot
output_folder = 'output/plots'
os.makedirs(output_folder, exist_ok=True)
png_path = os.path.join(output_folder, 'control_signal_calc_time.png')
eps_path = os.path.join(output_folder, 'control_signal_calc_time.eps')

plt.savefig(png_path)
plt.savefig(eps_path)

# Show plot
plt.show()

print(f"Bar chart saved as:\n - {png_path}\n - {eps_path}")
