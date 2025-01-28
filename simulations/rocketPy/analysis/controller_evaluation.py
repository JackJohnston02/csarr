"""
Script that takes inputs from past flights and then compares the controllers effort.

Performance Metrics:
    Controller effort, Sum(abs(u))
    Apogee error, apogee_target - apogee_acheived
"""

import numpy as np
import pandas as pd
import glob
import os
import json
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
    'legend.fontsize': 12 * FONT_SCALE,  # Font size for legend
})



def get_input_data():
        # Step 1: Get the directory of the current script
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Set the input directory to be an absolute path
    input_directory = os.path.join(script_directory, 'input')

    # Debug: Print the input directory to confirm it's correct
    print("Input Directory:", input_directory)

    # Get list of folders in the input directory
    try:
        folders = [f for f in os.listdir(input_directory) if os.path.isdir(os.path.join(input_directory, f))]
    except FileNotFoundError:
        print(f"Error: The directory '{input_directory}' was not found.")
        exit()

    # Step 2: Display the folders and prompt user for selection
    print("Select a folder by entering the corresponding number:")
    for idx, folder in enumerate(folders):
        print(f"{idx + 1}: {folder}")

    # Step 3: Get user input
    try:
        selection = int(input("Enter the number of your selected folder: "))
        if 1 <= selection <= len(folders):
            selected_folder = folders[selection - 1]  # Get the selected folder
            folder_path = os.path.join(input_directory, selected_folder)

            # Step 4: Load the JSON file
            json_file_path = os.path.join(folder_path, 'info.json')
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r') as json_file:
                    json_data = json.load(json_file)
                    reference_apogee = json_data.get("reference_apogee")
                    if reference_apogee is None:
                        print("Error: 'reference_apogee' not found in JSON file.")
                        return
                    print(f"Reference Apogee loaded from JSON: {reference_apogee}")
            else:
                print(f"Error: No JSON file found at {json_file_path}.")
                return

            # Step 5: Find all CSV files in the selected folder
            file_paths = glob.glob(os.path.join(folder_path, '*.csv'))

            # Display the found CSV file paths
            if file_paths:
                print("CSV files found in folder:", selected_folder)
                for file_path in file_paths:
                    print(file_path)
            else:
                print("No CSV files found in this folder.")
        else:
            print("Invalid selection. Please run the script again.")
    except ValueError:
        print("Invalid input. Please enter a number.")

    return file_paths, reference_apogee



def controller_evaluation_scatter(file_paths, reference_apogee):
    """
    Evaluate controller effort and percentage apogee error for each file.
    
    Args:
        file_paths (list of str): Paths to CSV files.
        reference_apogee (float): Reference apogee for calculating error.

    Returns:
        np.ndarray: Results with columns [file_path, controller_effort, apogee_percentage_error].
    """
    results = []
    
    # Iterate over file paths and calculate metrics
    for idx, file_path in enumerate(file_paths):
        controller_name = os.path.splitext(os.path.basename(file_path))[0]
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Extract "Time" and "Controller Output" columns
        time = df["Time"].tolist()
        controller_output = df["Controller Output"].tolist()

        # Ensure time and controller output are numeric
        time = [float(t) for t in time]
        controller_output = [float(co) for co in controller_output]

        # Calculate the absolute values of the controller outputs
        controller_output_abs = [co*co for co in controller_output]

        # Integrate using the Euler method
        controller_output_abs_sum = 0
        for i in range(1, len(time)):
            dt = time[i] - time[i - 1]  # Time difference
            controller_output_abs_sum += controller_output_abs[i - 1] * dt  # Accumulate the area of each rectangle

        apogee = df["True Z"].tolist()[-1]

        # Calculate the reference apogee if it is set as "N/A"
        if reference_apogee == "N/A":
            # If the apogee is set as a "N/A" then the filenames will have the format (reference_apogee,"m.csv"), for example the file for the reference apogee of 2600 is "2600m.csv" so we can extract the reference apogee from the filename
            reference_apogee_temp = float(controller_name.split("/")[-1].split("m")[0])
            apogee_percentage_error = abs(((apogee - reference_apogee_temp) / reference_apogee_temp) * 100)

        else:
            # Calculate the percentage error, normal case
            apogee_percentage_error = ((apogee - reference_apogee) / reference_apogee) * 100

        # Append results
        results.append([controller_name, controller_output_abs_sum, apogee_percentage_error])


    results = np.array(results, dtype=object)
    # Extract data for plotting
    controller_efforts = results[:, 1].astype(float)
    apogee_errors = results[:, 2].astype(float)
    labels = results[:, 0]

    # Create scatter plot
    # 10, 6 is normal, 10,10 for square
    plt.figure(figsize=(12, 9))
    for idx, label in enumerate(labels):
        # Plot the controller effort vs. percentage apogee error making the marker larger
        plt.scatter(controller_efforts[idx], apogee_errors[idx], label=label, s=200, marker='o')

    # Customize plot
    plt.title(r"$\textbf{Percentage Apogee Error vs. Controller Effort}$")
    plt.xlabel(r"Total Controller Effort (rad$^2$/s)")
    plt.ylabel(r"Final Percentage Apogee Error ($\%$)")
    # Place the legend at the bottom right of the plot
    plt.legend(loc='best')
    # Limit the x-axis to the maximum controller effort out of all controllers
    plt.xlim(0, (np.max(controller_efforts)+0.05*np.max(controller_efforts)))
    print(np.max(controller_efforts))
    plt.grid()
    
    
    # Get the parent folder path (the folder containing the CSV file)
    parent_folder_path = os.path.dirname(file_path)
    # Extract the last folder name (which is the parent folder)
    folder_name = os.path.basename(parent_folder_path)

    folder_path = f'analysis/output/{folder_name}'
    file_path = os.path.join(folder_path, 'scattter.png')
    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(file_path)

    folder_path = f'analysis/output/{folder_name}'
    file_path = os.path.join(folder_path, 'scattter.eps')
    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(file_path)
    

def controller_control_output_line(file_paths):
    results = []
    
    plt.figure(figsize=(12, 9))

    # Iterate over file paths and calculate metrics
    for idx, file_path in enumerate(file_paths):
        controller_name = os.path.splitext(os.path.basename(file_path))[0]
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Extract "Time" and "Controller Output" columns
        time = df["Time"].tolist()
        controller_output = df["Controller Output"].tolist()

        # Ensure time and controller output are numeric
        time = [float(t) for t in time]
        controller_output = [float(co) for co in controller_output]

        # Plot the lines making them thicker and slightly transparent
        plt.plot(time, controller_output, label=controller_name, linewidth=2, alpha=0.7)

    # Customize plot
    plt.title(r"$\textbf{Control Output vs. Time}$")
    plt.ylabel(r"Control Output (rad/s)")
    plt.xlabel(r"Time (s)")
    plt.xlim(0, np.max(time))
    plt.legend(loc='upper right')
    plt.grid()
    
    
    # Get the parent folder path (the folder containing the CSV file)
    parent_folder_path = os.path.dirname(file_path)
    # Extract the last folder name (which is the parent folder)
    folder_name = os.path.basename(parent_folder_path)

    folder_path = f'analysis/output/{folder_name}'
    file_path = os.path.join(folder_path, 'control_output.png')
    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(file_path)

    folder_path = f'analysis/output/{folder_name}'
    file_path = os.path.join(folder_path, 'control_output.eps')
    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(file_path)


def controller_airbrake_position(file_paths):
    results = []

    plt.figure(figsize=(12, 9))

    # Define distinct markers for the lines
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', 'H', '8']
    num_markers = len(markers)

    # Iterate over file paths and calculate metrics
    for idx, file_path in enumerate(file_paths):
        controller_name = os.path.splitext(os.path.basename(file_path))[0]
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)

        # Extract "Time" and "Controller Output" columns
        time = df["Time"].tolist()
        airbrake_position = df["Airbrake Deployment Level"].tolist()

        # Ensure time and controller output are numeric
        time = [float(t) for t in time]
        airbrake_position = [float(co) * 90 for co in airbrake_position]

        # Calculate staggered marker placement
        total_points = len(time)
        interval = total_points // 10  # Place markers at approximately 10% intervals
        offset = (interval // num_markers) * idx  # Unique offset for each line

        # Plot the lines with staggered markers
        crop = 700
        plt.plot(
            time[crop:],
            airbrake_position[crop:],
            label=controller_name,
            linewidth=2,
            alpha=0.7,
            marker=markers[idx % num_markers],  # Cycle through markers
            markevery=(offset, interval),  # Staggered markers
            markersize=10  # Increase marker size for better visibility
        )
    
    # Customize plot
    plt.title(r"$\textbf{Airbrake Angle vs. Time}$")
    plt.ylabel(r"Airbrake Angle (degrees)")
    plt.xlabel(r"Time (s)")
    plt.xlim(0, np.max(time))
    plt.ylim(0, 90)
    plt.legend(loc='upper left')
    plt.grid(visible=True, linestyle='--', linewidth=1.5)  # Dashed lines with thicker width

    # Get the parent folder path (the folder containing the CSV file)
    parent_folder_path = os.path.dirname(file_path)
    # Extract the last folder name (which is the parent folder)
    folder_name = os.path.basename(parent_folder_path)

    # Save the plot as PNG and EPS
    folder_path = f'analysis/output/{folder_name}'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(os.path.join(folder_path, 'airbrake_position.png'))
    plt.savefig(os.path.join(folder_path, 'airbrake_position.eps'))

file_paths, reference_apogee = get_input_data()

controller_evaluation_scatter(file_paths, reference_apogee)
controller_control_output_line(file_paths)
controller_airbrake_position(file_paths)

print("_-_-_DONE_-_-_")