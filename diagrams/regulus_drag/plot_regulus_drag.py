import pandas as pd
import matplotlib.pyplot as plt

# Define a scaling factor
FONT_SCALE = 1.4  # Adjust this value to scale fonts

# Update font sizes using the scaling factor
plt.rcParams.update({
    'text.usetex': True,  # Enable LaTeX for text rendering
    'font.family': 'serif',  # Use serif fonts (like LaTeX's default)
    'font.serif': ['Computer Modern'],  # Set the LaTeX font (optional)
    'font.size': 18 * FONT_SCALE,  # Default text font size
    'axes.titlesize': 20 * FONT_SCALE,  # Font size for plot titles
    'axes.labelsize': 18 * FONT_SCALE,  # Font size for x and y labels
    'xtick.labelsize': 18 * FONT_SCALE,  # Font size for x tick labels
    'ytick.labelsize': 18 * FONT_SCALE,  # Font size for y tick labels
    'legend.fontsize': 18 * FONT_SCALE,  # Font size for legend
})

# File paths for the CSV files
file_power_off = 'powerOffDragCurve.csv'
file_power_on = 'powerOnDragCurve.csv'

# Load the data
power_off_data = pd.read_csv(file_power_off, header=None, names=["Mach", "Drag_Coefficient"])
power_on_data = pd.read_csv(file_power_on, header=None, names=["Mach", "Drag_Coefficient"])


# Create the plot
plt.figure(figsize=(12, 8))  # Match aspect ratio of the contour plot
plt.plot(
    power_off_data["Mach"], power_off_data["Drag_Coefficient"], 
    label=r"Power Off Drag Curve", linestyle='-', linewidth=2, color='blue'
)
plt.plot(
    power_on_data["Mach"], power_on_data["Drag_Coefficient"] - 0.0125, 
    label=r"Power On Drag Curve", linestyle='--', linewidth=2, color='orange'
)

# Customize the plot
plt.xlabel(r'Mach Number, $M$', labelpad=10)
plt.ylabel(r'Regulus Drag Coefficient, $C_d$', labelpad=10)
plt.ylim([0, 0.8])
plt.xlim(0, 1.2)
plt.title(r'Drag Coefficient vs Mach Number')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.5)

# Save as EPS and PNG
plt.tight_layout()
plt.savefig('drag_curves.png', format='png', dpi=300)
plt.savefig('drag_curves.eps', format='eps', dpi=300)
