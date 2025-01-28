import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import os
from joblib import Parallel, delayed

from functions import get_gravity, get_air_density, rk4, ballistic_flight_model

def evaluate_target_manifold(reference_apogee, current_altitude, current_Cb):
        
        dt = 0.5  # Time step for simulation
        max_iter = 10  # Maximum iterations for better convergence if necessary
        
        # Expand search bounds initially
        upper_bound = 300
        lower_bound = 0


        # Function to simulate and compute apogee error
        def compute_apogee(altitude, velocity, ballistic_coefficient):
            # Initialize state variables
            x = np.zeros((3, 1))
            x[0, 0] = altitude  # Initial altitude
            x[1, 0] = velocity  # Initial velocity
            x[2, 0] = ballistic_coefficient  # Set current Cb estimate
            
            # Simulate until velocity is zero or less (apogee reached)
            while x[1, 0] > 0:
                x = rk4(ballistic_flight_model, x, dt)
            
            apogee = x[0, 0]
            # Compute the apogee error
            return apogee

        # Secant Method loop
        for iteration in range(max_iter):
            guess_velocity = (upper_bound + lower_bound ) / 2
            apogee = compute_apogee(current_altitude, guess_velocity, current_Cb)

            error = reference_apogee - apogee

            if abs(error) < 0.01:
                return guess_velocity
            
            if apogee > reference_apogee:
                upper_bound = guess_velocity
            
            if apogee < reference_apogee:
                lower_bound = guess_velocity
        
        return guess_velocity


def plot_surface(velocity_grid, altitude_grid, bc_grid):
    # 3D Surface plot
    fig1 = plt.figure(figsize=(12, 9))
    ax1 = fig1.add_subplot(111, projection='3d')
    surf = ax1.plot_surface(velocity_grid, altitude_grid, bc_grid, cmap='viridis')

    # Labels and title
    ax1.set_title(r"\textbf{Target Manifold Through +ve Octant}")
    ax1.set_xlabel(r"Velocity (m/s)")
    ax1.set_ylabel(r"Altitude (m)")
    ax1.set_zlabel(r"Ballistic Coefficient ($kg/m^2$)")
    ax1.set_xlim([0, 300])
    ax1.set_ylim([0, 2500])
    ax1.set_zlim([0, 5000])

    # Adjust label padding
    ax1.xaxis.labelpad = 20
    ax1.yaxis.labelpad = 20
    ax1.zaxis.labelpad = 20
    ax1.tick_params(axis='z', pad=10)  # Move y-axis ticks further from the axis

    ax1.view_init(elev=30, azim=45)  # Adjust these angles as desired
    ax1.set_yticks([500, 1000, 1500, 2000])

    # Save the 3D surface plot
    fig1.savefig(f"{output_folder}/surface_plot.svg", format='svg')
    fig1.savefig(f"{output_folder}/surface_plot.png", format='png', dpi=900)


def plot_contour(velocity_grid, altitude_grid, bc_grid):
    # 2D Top-down view with contour plot
    fig2 = plt.figure(figsize=(12, 9))
    ax2 = fig2.add_subplot(111)
    contour = ax2.contourf(velocity_grid, altitude_grid, bc_grid, cmap='viridis')

    # Labels and title
    ax2.set_title(r"\textbf{Target Manifold Through +ve Octant}")
    ax2.set_xlabel(r"Velocity (m/s)", labelpad=10)
    ax2.set_ylabel(r"Altitude (m)", labelpad=10)

    
    ax2.set_xticks([0, 50, 100, 150, 200, 250, 300])

    # Add colorbar
    cbar2 = fig2.colorbar(contour, ax=ax2, shrink=0.5, aspect=30, location='bottom', ticks=[0, 1000, 2000, 3000, 4000, 5000])
    cbar2.set_label(r"Ballistic Coefficient ($kg/m^2$)")


    # Save the 2D contour plot
    fig2.savefig(f"{output_folder}/contour_plot.svg", format='svg')
    fig2.savefig(f"{output_folder}/contour_plot.png", format='png', dpi=900)



number_of_points = int(input("How many points do you want?:"))

number_of_points_per_axis = int(np.sqrt(number_of_points))
# Reference apogee for calculations
reference_apogee = 2500

altitudes = np.linspace(0, reference_apogee, number_of_points_per_axis)
ballistic_coefficients = np.linspace(10, 5000, number_of_points_per_axis)
altitude_grid, bc_grid = np.meshgrid(altitudes, ballistic_coefficients)

# Initialize a 2D array to store the velocity results
velocity_grid = np.zeros_like(altitude_grid)



print(f"Generating target manifold with reference apogee of {reference_apogee}, and with {number_of_points} points in the grid.")
# Evaluate velocity over the grid
def process_point(i, j, altitude_grid, bc_grid, reference_apogee):
    altitude = altitude_grid[i, j]
    bc = bc_grid[i, j]
    velocity = evaluate_target_manifold(reference_apogee, altitude, bc)
    if velocity > 300:
        velocity = 300
    return i, j, velocity

# Parallel processing
results = Parallel(n_jobs=-1, verbose=5)(
    delayed(process_point)(i, j, altitude_grid, bc_grid, reference_apogee)
    for i in range(altitude_grid.shape[0])
    for j in range(altitude_grid.shape[1])
)

# Populate velocity_grid
for i, j, velocity in results:
    velocity_grid[i, j] = velocity

print("Processing complete, now plotting")

# Ensure the output folder exists
output_folder = 'output'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

FONT_SCALE = 1.4  # Adjust this value to scale fonts

# Update font sizes using the scaling factor
mpl.rcParams.update({
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

plot_surface(velocity_grid, altitude_grid, bc_grid)
plot_contour(velocity_grid, altitude_grid, bc_grid)

print("Finished")