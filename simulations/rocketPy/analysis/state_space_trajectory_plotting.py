import numpy as np
import plotly.graph_objects as go
import pandas as pd
import glob
import time
import sys
import os
import json
import math

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.environmental_utils import get_gravity, get_air_density

# Function for plotting the state space trajectory of each of the simulations, does both 2D and 3D plots
def state_space_trajectory_plotting():
    # For setting the file names
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    reference_apogee = None 

    # Define parameters and functions g(x) and rho(x)
    def g(x):
        return get_gravity(x)

    def rho(x):
        return get_air_density(x)

    # Define the system of differential equations
    def system(y, C_b):
        x, x_dot = y  # Only x and x_dot are used in the equations
        dxdt = x_dot
        dxdotdt = g(x) - (rho(x) * x_dot**2) / (2 * C_b)
        return np.array([dxdt, dxdotdt])

    # Custom Euler solver function
    def euler_solver(y0, C_b, t_span, dt):
        steps = int((t_span[1] - t_span[0]) / dt)
        y = np.zeros((steps + 1, len(y0)))  # Store results
        y[0] = y0

        for i in range(steps):
            if y[i, 0] < xf:
                y[i + 1, :] = y[i, :]
            elif y[i, 1] > x_dotf:
                y[i + 1, :] = y[i, :]
            else:
                y[i + 1] = y[i] + dt * system(y[i], C_b)

        return y

    # Initialize plot for the 3D surface
    fig_3d = go.Figure()

    # Set constants
    x_dot0 = 0  # Starting velocity
    C_b0 = 500  # Lower limit for Cb values
    xf = 0  # Final velocity - always 0
    x_dotf = 300  # Ending velocity
    C_bf = 5500  # Upper limit for Cb values
    t_span = (0, -30)  # Going backwards in time
    dt = -0.01  # Going backwards in time

    num_lines = 40 
    Cb_values = np.logspace(np.log10(C_b0), np.log10(C_bf), num_lines)

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

    # Process each CSV file
    for idx, file_path in enumerate(file_paths):
        df = pd.read_csv(file_path)

        # Extract flight data from CSV
        flight_data_altitude = df["True Z"].tolist() # Assuming "True Z" is the first column
        flight_data_velocity = df["True VZ"].tolist()  # Assuming "True VZ" is the fourth column
        flight_data_Cb = df["Estimated Cbz"].apply(lambda x: float(x.strip("[]'"))).tolist()

        if idx == 0:
            max_velocity_index = np.argmax(flight_data_velocity)
            burnout_altitude = flight_data_altitude[max_velocity_index]
            burnout_velocity = flight_data_velocity[max_velocity_index]
            bunrout_Cb = flight_data_Cb[max_velocity_index]    
            
            # Get the values for launch conditions
            min_altitude_index = np.argmin(flight_data_altitude)
            launch_altitude = flight_data_altitude[min_altitude_index]
            launch_velocity = flight_data_velocity[min_altitude_index]
            launch_Cb = flight_data_Cb[min_altitude_index]

            # Get the values for the apogee conditions
            max_altitude_index = np.argmax(flight_data_altitude)
            apogee_altitude = flight_data_altitude[max_altitude_index]
            apogee_velocity = flight_data_velocity[max_altitude_index]
            apogee_Cb = flight_data_Cb[max_altitude_index]


        # For the first file, create the surface plot
        if idx == 0:
            # Initialize arrays to store the results for the surface plot
            X_vals, X_dot_vals, C_b_vals = [], [], []

            # Calculate surface for the first file using the defined initial altitude
            for C_b in Cb_values:
                y0 = [reference_apogee, x_dot0]  # Use the defined initial altitude
                sol = euler_solver(y0, C_b, t_span, dt)
                X_vals.append(sol[:, 1])  # Velocity on x-axis
                X_dot_vals.append(sol[:, 0])  # Altitude on y-axis
                C_b_vals.append(np.full_like(sol[:, 1], C_b))

            # Convert lists to 2D arrays for surface plotting
            X_vals = np.array(X_vals)
            X_dot_vals = np.array(X_dot_vals)
            C_b_vals = np.array(C_b_vals)

             # Add green diamond marker for launch point to the 3D figure
            fig_3d.add_trace(go.Scatter3d
            (
                x=[launch_velocity],  # Launch velocity
                y=[launch_altitude],  # Launch altitude
                z=[launch_Cb],  # Launch Cb
                mode='markers',
                marker=dict(size=7, color='green', symbol='diamond'),
                name='Launch' if idx == 0 else None,  # Add "Launch" to the legend only once
                showlegend=(idx == 0)  # Show legend entry only for the first dataset
            ))

            # Add marker for burnout point to the 3D figure
            fig_3d.add_trace(go.Scatter3d(
                x=[burnout_velocity],  # Max velocity
                y=[burnout_altitude],  # Altitude at max velocity
                z=[bunrout_Cb],  # Cb at max velocity
                mode='markers',
                marker=dict(size=7, color='orange', symbol='diamond'),
                name='Burnout' if idx == 0 else None,  # Add "Burnout" to the legend only once
                showlegend=(idx == 0)  # Show legend entry only for the first dataset
            ))



            # Add red diamond marker for apogee point to the 3D figure
            #fig_3d.add_trace(go.Scatter3d(
            #    x=[apogee_velocity],  # Apogee velocity
             #   y=[apogee_altitude],  # Apogee altitude
            #    z=[apogee_Cb],  # Apogee Cb
             #   mode='markers',
             #   marker=dict(size=7, color='red', symbol='diamond'),
             #   name='Apogee' if idx == 0 else None,  # Add "Apogee" to the legend only once
             #   showlegend=(idx == 0)  # Show legend entry only for the first dataset
            #))

            # Add the surface plot
            fig_3d.add_trace(go.Surface(
                x=X_vals,
                y=X_dot_vals,
                z=C_b_vals,
                surfacecolor=np.ones_like(C_b_vals),  # Uniform surfacecolor
                colorscale=[[0, 'yellow'], [1, 'yellow']],  # Single color for the surface
                opacity=0.9,
                showscale=False
            ))

        # Colors for flight data
        colors = ['grey', 'red', 'blue', 'green','white']

        # Add flight data line plot for each file
        file_name = os.path.basename(file_path).replace('.csv', '')  # Extract filename without extension
        fig_3d.add_trace(go.Scatter3d(
            x=flight_data_velocity,  # Velocity on x-axis
            y=flight_data_altitude,  # Altitude on y-axis
            z=flight_data_Cb,
            mode='lines',
            line=dict(
                color=colors[idx % len(colors)],  # Cycle through colors
                width=4
            ),
            name=f'{file_name}'  # Use the extracted file name
        ))

        # Customize layout for the 3D plot
        fig_3d.update_layout(
                title=dict(
                    text=f"State-Space Plot for {selected_folder}",
                    font=dict(family="Times New Roman, serif", size=20, weight="bold"),  # Adjust title font size
                    x=0.5,  # Center-align the title
                    y=0.925  # Position it higher above the plot
                ),
            font=dict(family="Times New Roman, serif", size=14),
            scene=dict(
                xaxis=dict(
                    title="Velocity (m/s)", 
                    range=[0, 300],
                    backgroundcolor="white",
                    gridcolor="lightgray",
                    zerolinecolor="gray"
                ),
                yaxis=dict(
                    title="Altitude (m)", 
                     range=[0,  2500],
                    backgroundcolor="white",
                    gridcolor="lightgray",
                    zerolinecolor="gray",
                    tickvals=[500, 1000, 1500, 2000, 2500],  # Custom tick positions
                ),
                zaxis=dict(
                    title="Ballistic Coefficient (kg/m²)", 
                    range=[1000, 4000],
                    backgroundcolor="white",
                    gridcolor="lightgray",
                    zerolinecolor="gray",
                    tickvals=[1500, 2000, 2500, 3000, 3500, 4000],  # Custom tick positions
                ),
                camera=dict(
                    eye=dict(x=-1.5, y=2, z=0.5)  # Set the camera position to zoom out
                )
            ),
            legend=dict(
                font=dict(size=14),
                orientation="h",  # Horizontal legend for better layout
                xanchor="center",
                x=0.5,
                y=0  # Position below the plot
            ),
            width=800,
            height=600,
            autosize=True,
            margin=dict(l=0, r=0, t=0, b=0)
        )

    # Show the 3D figure
    fig_3d.show()


    # Initialize plot for the 2D velocity-altitude plane
    fig_2d = go.Figure()

    # Add flight data line plots to the 2D figure
    for idx, file_path in enumerate(file_paths):
        df = pd.read_csv(file_path)

        # Extract flight data from CSV
        flight_data_altitude = df["True Z"].tolist() # Assuming "True Z" is the first column
        flight_data_velocity = df["True VZ"].tolist()  # Assuming "True VZ" is the fourth column
        
        # Extract the burnout altitude and velocity from the first file
        if idx == 0:
            max_velocity_index = np.argmax(flight_data_velocity)
            burnout_altitude = flight_data_altitude[max_velocity_index]
            burnout_velocity = flight_data_velocity[max_velocity_index]

            # Get the values for launch conditions
            min_altitude_index = np.argmin(flight_data_altitude)
            launch_altitude = flight_data_altitude[min_altitude_index]
            launch_velocity = flight_data_velocity[min_altitude_index]

            # Get the values for the apogee conditions
            max_altitude_index = np.argmax(flight_data_altitude)
            apogee_altitude = flight_data_altitude[max_altitude_index]
            apogee_velocity = flight_data_velocity[max_altitude_index]

        
        # Extract filename without extension for the legend
        file_name = os.path.basename(file_path).replace('.csv', '')

         # Add green diamond marker for launch point to the 2D figure
        fig_2d.add_trace(go.Scatter(
            x=[launch_velocity],  # Launch velocity
            y=[launch_altitude],  # Launch altitude
            mode='markers',
            marker=dict(size=24, color='green', symbol='diamond'),
            name='Launch' if idx == 0 else None,  # Add "Launch" to the legend only once
            showlegend=(idx == 0)  # Show legend entry only for the first dataset
        ))
        

        # Add marker for burnout point to the 2D figure
        fig_2d.add_trace(go.Scatter(
            x=[burnout_velocity],  # Max velocity
            y=[burnout_altitude],  # Altitude at max velocity
            mode='markers',
            marker=dict(size=20, color='orange', symbol='diamond'),
            name='Burnout' if idx == 0 else None,  # Add "Burnout" to the legend only once
            showlegend=(idx == 0)  # Show legend entry only for the first dataset
        ))

        # Add red diamond marker for apogee point to the 2D figure
        #fig_2d.add_trace(go.Scatter(
        #    x=[apogee_velocity],  # Apogee velocity
        #    y=[apogee_altitude],  # Apogee altitude
        #    mode='markers',
       #     marker=dict(size=20, color='red', symbol='diamond'),
        #    name='Apogee' if idx == 0 else None,  # Add "Apogee" to the legend only once
        #    showlegend=(idx == 0)  # Show legend entry only for the first dataset
        #))
    


        # Add line plot for flight data
        fig_2d.add_trace(go.Scatter(
            x=flight_data_velocity,  # Velocity on x-axis
            y=flight_data_altitude,  # Altitude on y-axis
            mode='lines',
            line=dict(width=2),
            name=f'{file_name}'  # Use the extracted file name
        ))

   
    # For the 2D plot, select a subset of C_b values to display
    selected_Cb_indices = np.linspace(0, num_lines - 1, num=5, dtype=int)  # Select indices for 10 lines
    selected_Cb_values = Cb_values[selected_Cb_indices]

    # Create lists to store selected values for plotting
    selected_X_vals = []
    selected_X_dot_vals = []

    # Generate selected lines for the 2D plot
    for idx in selected_Cb_indices:
        C_b = Cb_values[idx]
        y0 = [reference_apogee, x_dot0]  # Use the defined initial altitude
        sol = euler_solver(y0, C_b, t_span, dt)
        selected_X_vals.append(sol[:, 1])  # Velocity
        selected_X_dot_vals.append(sol[:, 0])  # Altitude

    # Convert lists to 2D arrays for plotting
    selected_X_vals = np.array(selected_X_vals)
    selected_X_dot_vals = np.array(selected_X_dot_vals)

    # Add the surface to the 2D figure with the selected C_b values
    # Generate and plot isolines with labels
    # Maximum velocity to show on the plot
    velocity_limit = 260  # Adjust as needed to keep labels on the plot

    # Generate and plot isolines with labels
    for i, C_b in enumerate(selected_Cb_values):
        # Solve for the current C_b value
        y0 = [reference_apogee, x_dot0]  # Initial conditions
        sol = euler_solver(y0, C_b, t_span, dt)  # Solve the system

        # Extract velocity (X) and altitude (X_dot) data
        X_vals = sol[:, 1]  # Velocity
        X_dot_vals = sol[:, 0]  # Altitude

        # Mask data where velocity exceeds the limit
        mask = X_vals <= velocity_limit
        clipped_X_vals = X_vals[mask]
        clipped_X_dot_vals = X_dot_vals[mask]

        # Ensure valid data remains after clipping
        if len(clipped_X_vals) == 0 or len(clipped_X_dot_vals) == 0:
            continue

        # Plot the isoline
        fig_2d.add_trace(go.Scatter(
            x=clipped_X_vals,
            y=clipped_X_dot_vals,
            mode='lines',
            line=dict(color='rgba(0, 0, 0, 0.25)', width=2, dash='dash'),
            showlegend=False
        ))

        # Find the position for the label (just before the velocity limit)
        label_index = np.argmax(clipped_X_vals)  # Index of the last point within the range

        # Add label at the chosen position
        fig_2d.add_trace(go.Scatter(
            x=[clipped_X_vals[label_index]],
            y=[clipped_X_dot_vals[label_index]],
            mode='text',
            text=[f'{round(C_b, -2):.0f} kg/m²'],  # Label directly from C_b
            textfont=dict(size=18, color='rgba(0, 0, 0, 0.75)'),
            textposition='top right',  # Position the text above and to the right
            showlegend=False
        ))





    # Customize layout for the 2D plot
    fig_2d.update_layout(
       title=dict(
            text=f"State-Space Plot with {selected_folder}",
            font=dict(family="Times New Roman, serif", size=30),  # Adjust title font size
            x=0.5,  # Center-align the title
            y=0.925  # Position it higher above the plot
                ),
        font=dict(family="Times New Roman, serif", size=18),
        plot_bgcolor='white',
        paper_bgcolor='white', 
         xaxis=dict(
                    title="Velocity (m/s)", 
                    range=[0, x_dotf],
                    gridcolor="lightgray",
                    zerolinecolor="gray"
                ),
         yaxis=dict(
                    title="Altitude (m)", 
                    range=[xf,  math.ceil(reference_apogee / 500.0) * 500],
                    gridcolor="lightgray",
                    zerolinecolor="gray"
                ),
        width=800,  # Increase width
        height=600,  # Decrease height
        autosize=True,
        margin=dict(l=100, r=100, t=100, b=100),
        legend=dict(
                font=dict(size=18),
                orientation="h",  # Horizontal legend for better layout
                xanchor="center",
                x=0.5,
                y=-0.15  # Position below the plot
            )
    )

    # Show the 2D figure
    #fig_2d.show()


    # Saving
    folder_path = f'analysis/output/{selected_folder}'
    file_path = os.path.join(folder_path, '3D_plot')
    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    fig_3d.write_image(f"{file_path}.png")
    fig_3d.write_image(f"{file_path}.eps")

    file_path = os.path.join(folder_path, '2D_plot')
    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    fig_2d.write_image(f"{file_path}.png")
    fig_2d.write_image(f"{file_path}.eps")

state_space_trajectory_plotting()
