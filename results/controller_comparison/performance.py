import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define a scaling factor
FONT_SCALE = 1.75  # Adjust this value to scale fonts

# Update font sizes using the scaling factor
plt.rcParams.update({
    'text.usetex': True,  # Enable LaTeX for text rendering
    'font.family': 'serif',  # Use serif fonts (like LaTeX's default)
    'font.serif': ['Computer Modern'],  # Set the LaTeX font (optional)
    'font.size': 18 * FONT_SCALE,  # Default text font size
    'axes.titlesize': 20 * FONT_SCALE,  # Font size for plot titles
    'axes.labelsize': 18 * FONT_SCALE,  # Font size for x and y labels
    'xtick.labelsize': 16 * FONT_SCALE,  # Font size for x tick labels
    'ytick.labelsize': 16 * FONT_SCALE,  # Font size for y tick labels
    'legend.fontsize': 16 * FONT_SCALE,  # Font size for legend
})


def import_data(folder_path):
    """Imports the data from the input folder and organizes it by controller and file."""
    controller_folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
    controller_data = {}

    for controller in controller_folders:
        controller_data[controller] = {}
        controller_files = [f for f in os.listdir(f'{folder_path}/{controller}') if f.endswith('.csv')]
        for file in controller_files:
            apogee = float(file.split('m')[0])  # Extract apogee value from filename
            data = pd.read_csv(f'{folder_path}/{controller}/{file}')
            controller_data[controller][apogee] = data

    return controller_data

def calculate_metrics(controller_data):
    """Calculates the apogee error and controller effort for each flight and aggregates them."""
    flight_results = []
    summary_results = []

    for controller, flights in controller_data.items():
        apogee_errors = []
        controller_efforts = []

        print(f"___________________{controller}____________________")
        print(f"Apogee   | Controller Effort | Percentage Apogee Error")
        for apogee, data in flights.items():
            # Compute the percentage apogee error
            max_altitude = data['True Z'].max()
            time = data['Time'].values
            apogee_error = abs(apogee - max_altitude) / apogee * 100
            
            # Compute the total controller effort
            controller_output = data['Controller Output'].values
            controller_output_squared = controller_output ** 2
            controller_effort = np.trapz(controller_output_squared, x=time)

            flight_results.append({
                'Controller Name': controller,
                'Apogee Error': apogee_error,
                'Controller Effort': controller_effort
            })

            print(f"{apogee} & {controller_effort:.3f} & {apogee_error:.3f} \\\\")
            print("\hline")
            apogee_errors.append(apogee_error)
            controller_efforts.append(controller_effort)
    

        # Calculate averages and standard deviations for summary plot
        avg_apogee_error = np.mean(apogee_errors)
        std_apogee_error = np.std(apogee_errors)
        avg_controller_effort = np.mean(controller_efforts)
        std_controller_effort = np.std(controller_efforts)
        print(f"mean_apogee_percentage error: {avg_apogee_error:.3f}")
        print(f"avg_controller_effort: {avg_controller_effort:.3f}")


        summary_results.append({
            'Controller Name': controller,
            'Mean Apogee Error': avg_apogee_error,
            'Std Apogee Error': std_apogee_error,
            'Mean Controller Effort': avg_controller_effort,
            'Std Controller Effort': std_controller_effort
        })
    return pd.DataFrame(flight_results), pd.DataFrame(summary_results)

def plot_results(flight_results, summary_results):
    """
    Generates and saves three plots based on the provided results.
    """
    output_folder = 'output/plots'
    os.makedirs(output_folder, exist_ok=True)

    # Plot 1: Individual flights (scatter) and means
    # Define a colormap or manually assign consistent colors for controllers
    colors = plt.cm.tab10.colors  # Example: Tab10 colormap
    controller_colors = {
        controller: colors[i % len(colors)]
        for i, controller in enumerate(flight_results['Controller Name'].unique())
    }

    plt.figure(figsize=(12, 9))

    # Loop through each controller to plot individual flights and mean values
    for i, controller in enumerate(flight_results['Controller Name'].unique()):
        controller_data = flight_results[flight_results['Controller Name'] == controller]
        
        # Use consistent color for scatter points and mean
        color = controller_colors[controller]
        
        # Scatter individual flights as 'x'
        plt.scatter(
            controller_data['Controller Effort'],
            controller_data['Apogee Error'],
            label=f"{controller}",  # Temporary, will not be shown in the final legend
            s=100,
            alpha=0.6,
            marker='x',
            color=color
        )
        
        # Plot mean as a circle
        mean_effort = controller_data['Controller Effort'].mean()
        mean_error = controller_data['Apogee Error'].mean()
        plt.scatter(
            mean_effort,
            mean_error,
            label=f"{controller} Mean",
            s=200,
            alpha=0.8,
            marker='o',
            color=color
        )

    # Add title and axis labels
    plt.xlim([0, 0.7])
    plt.ylim([0, 0.075])
    plt.title(r"Controller Comparison")
    plt.xlabel(r"Total Controller Effort (rad$^2$/s)")
    plt.ylabel(r"Apogee Error (\%)")

    # Customize legend to show only the means
    handles, labels = plt.gca().get_legend_handles_labels()
    mean_handles = [h for h, l in zip(handles, labels) if "Mean" in l]
    mean_labels = [l for l in labels if "Mean" in l]
    plt.legend(mean_handles, mean_labels, loc='best', frameon=True)

    # Add grid and save the figure
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'flight_level_performance.png'))
    plt.savefig(os.path.join(output_folder, 'flight_level_performance.eps'))
    plt.close()

    # Plot 2: Mean and standard deviation (error bars)
    plt.figure(figsize=(12, 9))
    for i, row in summary_results.iterrows():
        plt.errorbar(
            row['Mean Controller Effort'],
            row['Mean Apogee Error'],
            xerr=row['Std Controller Effort'],
            yerr=row['Std Apogee Error'],
            fmt='o',
            label=row['Controller Name'],
            capsize=5,
            markersize=10
        )
    plt.title(r"Controller Comparison (Mean ± Std)")
    plt.xlabel(r"Total Controller Effort (rad$^2$/s)")
    plt.ylabel(r"Apogee Error (\%)")
    plt.xlim([0, 0.7])
    plt.ylim([0, 0.075])
    plt.legend(loc='best', frameon=True)
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'mean_std_performance.png'))
    plt.savefig(os.path.join(output_folder, 'mean_std_performance.eps'))
    plt.close()

    # Plot 3: Current scatter plot
    plt.figure(figsize=(12, 9))
    for i, row in summary_results.iterrows():
        plt.scatter(
            row['Mean Controller Effort'],
            row['Mean Apogee Error'],
            s=400,
            label=row['Controller Name'],
            alpha=0.8
        )
    plt.xlim([0, 0.7])
    plt.ylim([0, 0.075])
    plt.title(r"Controller Comparison")
    plt.xlabel(r"Mean Total Controller Effort (rad$^2$/s)")
    plt.ylabel(r"Mean Apogee Error (\%)")
    plt.legend(loc='best', frameon=True)
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'controller_evaluation.png'))
    plt.savefig(os.path.join(output_folder, 'controller_evaluation.eps'))
    plt.close()

# Main execution
folder_path = 'input'  # Path to the input folder

# Import data
controller_data = import_data(folder_path)

# Calculate metrics
flight_results, summary_results = calculate_metrics(controller_data)

# Plot the results
plot_results(flight_results, summary_results)
print("Plots saved in 'output/plots'.")