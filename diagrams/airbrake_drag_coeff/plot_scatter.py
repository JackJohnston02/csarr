import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

# Configure LaTeX-style fonts
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Load the CSV file
file_path = 'air_brakes_cd.csv'  # Update this path to your CSV file
data = pd.read_csv(file_path)

# Clean column names and ensure data consistency
data.columns = data.columns.str.strip()

# Extract the required columns
mach = data['mach']
deployment_level = 100 * data['deployment_level']  # Convert to percentage
cd = data['cd']

# Create a scatter plot with your preferred aspect ratio
plt.figure(figsize=(12, 8))  # Your preferred aspect ratio
scatter = plt.scatter(mach, deployment_level, c=cd, cmap='viridis', s=50, edgecolor='k')

# Add a color bar for `cd` values
cbar = plt.colorbar(scatter, label=r'$C_d$ (Drag Coefficient)')

# Add labels with units
plt.title(r'Airbrake Drag Coefficient Scatter', fontsize=14)
plt.xlabel(r'Mach Number, $M$', fontsize=12)
plt.ylabel(r'Deployment Level, $\%$', fontsize=12)
plt.ylim(0, 100)

# Add grid and save as EPS
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('airbrake_cd_scatter.png', format='png', dpi=300)
plt.savefig('airbrake_cd_scatter.eps', format='eps', dpi=300)
plt.show()
