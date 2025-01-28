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

# Pivot the data to prepare it for contour plotting
pivot_table = data.pivot(index='deployment_level', columns='mach', values='cd')

# Extract grid data for plotting
X, Y = np.meshgrid(pivot_table.columns, pivot_table.index)
Z = pivot_table.values

# Create a contour plot with your preferred aspect ratio
plt.figure(figsize=(12, 8))  # Your preferred aspect ratio
contour = plt.contourf(X, 100*Y, Z, levels=50, cmap='viridis')
cbar = plt.colorbar(contour, label=r'$C_d$ (Drag Coefficient)')

# Add labels with units
plt.title(r'Airbrake Drag Coefficient Contour', fontsize=16)
plt.xlabel(r'Mach Number, $M$', fontsize=14)
plt.ylabel(r'Deployment Level, $\%$', fontsize=14)
plt.ylim(0, 100)

# Add grid and save as EPS
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('airbrake_cd_contour.png', format='png', dpi=300)
plt.savefig('airbrake_cd_contour.eps', format='eps', dpi=300)
plt.show()
