import numpy as np
import matplotlib.pyplot as plt
import os

from utils.environmental_utils import get_air_density, get_gravity

def plot_true_states(times, true_states):
    # Define state names
    state_names = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'e0', 'e1', 'e2', 'e3', 'wx', 'wy', 'wz']
    
    # Define groups of states
    grouped_states = {
        'Displacements': [0, 1, 2],  # x, y, z
        'Velocities': [3, 4, 5],     # vx, vy, vz
        'Quaternions': [6, 7, 8, 9],  # e0, e1, e2, e3
        'Angular_Rates': [10, 11, 12] # wx, wy, wz
    }
    
    # Create the output directory if it doesn't exist
    output_dir = 'plots/true_states/'
    os.makedirs(output_dir, exist_ok=True)

    # Create grouped plots
    for group_name, indices in grouped_states.items():
        plt.figure(figsize=(12, 8))
        
        # Create subplots for each state in the group
        for i, index in enumerate(indices):
            plt.subplot(len(indices), 1, i + 1)  # Create a subplot for each state
            plt.plot(times, true_states[:, index], label=state_names[index], color="green")
            plt.title(f'{state_names[index]}')
            plt.xlabel('Time')
            plt.ylabel(state_names[index])
            plt.grid(True)
            plt.legend()
        
        plt.suptitle(group_name)  # Add a title for the group
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, f'{group_name.replace(" ", "_")}_plot.png'), dpi=300)  # Save as PNG file
        plt.close()  # Close the figure to free up memory

        z = true_states[:, 2]  # Extract the altitude (z) state
        z_max = np.max(z)  # Find the maximum altitude (apogee)

        # Define y-axis limits as ±10 meters around the maximum value
        y_min = z_max - 100
        y_max = z_max + 100

        plt.figure(figsize=(12, 6))
        plt.plot(times, z, label='Altitude (z)', color="blue")
        plt.title('Zoomed Altitude (z) - Apogee')
        plt.xlabel('Time')
        plt.ylabel('Altitude (m)')
        plt.grid(True)
        plt.legend()

        # Set y-axis limits to zoom in around the apogee
        plt.ylim(y_min, y_max)

        # Save the zoomed-in plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'Zoomed_Altitude_Apogee.png'), dpi=300)
        plt.close()  # Close the figure to free up memory



def plot_estimated_states(times, estimated_states):
    # Define state names
    state_names = ['x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot', 'x_ddot', 'y_ddot', 'z_ddot', 'q_w', 'q_x', 'q_y', 'q_z', 'omega_x', 'omega_y', 'omega_z', 'Cb_x', 'Cb_y', 'Cb_z']
    
    # Define groups of states
    grouped_states = {
        'Displacements': [0, 1, 2],  # x, y, z
        'Velocities': [3, 4, 5],     # vx, vy, vz
        'Accelerations': [6, 7, 8],   # ax, ay, az
        'Quaternions': [9, 10, 11, 12],  # e0, e1, e2, e3
        'Angular_Rates': [13, 14, 15], # wx, wy, wz
        'Ballistic_Coefficients': [16, 17, 18] #Cb_x, Cb_y, Cb_z
    }
    
    # Create the output directory if it doesn't exist
    output_dir = 'plots/estimated_states/'
    os.makedirs(output_dir, exist_ok=True)

    # Create grouped plots
    for group_name, indices in grouped_states.items():
        plt.figure(figsize=(12, 8))
        
        # Create subplots for each state in the group
        for i, index in enumerate(indices):
            plt.subplot(len(indices), 1, i + 1)  # Create a subplot for each state
            plt.plot(times, estimated_states[:, index], label=state_names[index], color="green")
            plt.title(f'{state_names[index]}')
            plt.xlabel('Time')
            plt.ylabel(state_names[index])
            plt.grid(True)
            plt.legend()
        
        plt.suptitle(group_name)  # Add a title for the group
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, f'{group_name.replace(" ", "_")}_plot.png'), dpi=300)  # Save as PNG file
        plt.close()  # Close the figure to free up memory


def plot_comparison_states(times, true_states, estimated_states):

    # Need to restructure estimated states such that it has the same structure as true states
    estimated_states[:,6:10] = estimated_states[:, 9:13] # Replace the acceleration with the quaternion
    estimated_states[:, 10:13] = estimated_states[:, 13:16] # Replace the other states with the angular rates

    # Define state names
    state_names_true = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'e0', 'e1', 'e2', 'e3', 'wx', 'wy', 'wz']    
    state_names_estimated = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'e0', 'e1', 'e2', 'e3', 'wx', 'wy', 'wz']
    
    # Define groups of states
    grouped_states = {
        'Displacements': [0, 1, 2],  # x, y, z
        'Velocities': [3, 4, 5],     # vx, vy, vz
        'Quaternions': [6, 7, 8, 9],  # e0, e1, e2, e3
        'Angular_Rates': [10, 11, 12] # wx, wy, wz
    }
    
    # Create the output directory if it doesn't exist
    output_dir = 'plots/comparison_states/'
    os.makedirs(output_dir, exist_ok=True)

    # Create grouped plots comparing true and estimated states
    for group_name, indices in grouped_states.items():
        plt.figure(figsize=(12, 8))
        
        # Create subplots for each state in the group
        for i, index in enumerate(indices):
            plt.subplot(len(indices), 1, i + 1)  # Create a subplot for each state
            
            # Plot true states
            plt.plot(times, true_states[:, index], label=f'True', color="black")
            
            # Plot estimated states
            plt.plot(times, estimated_states[:, index], label=f'Estimated',alpha = 0.5, color="blue")
            
            plt.title(f'{state_names_true[index]}')
            plt.xlabel('Time')
            plt.ylabel(state_names_true[index])
            plt.grid(True)
            if i == 0:
                plt.legend()
        
        plt.suptitle(f'Comparison: {group_name}')  # Add a title for the group
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
        
        # Save the plot
        plt.savefig(os.path.join(output_dir, f'{group_name.replace(" ", "_")}_comparison_plot.png'), dpi=300)  # Save as PNG file
        plt.close()  # Close the figure to free up memory





def plot_controller_states(times, controller_states):
    # Create the output directory if it doesn't exist
    output_dir = 'plots/controller_states/'
    os.makedirs(output_dir, exist_ok=True)

    # Define font size for better readability
    label_font_size = 14
    title_font_size = 16
    legend_font_size = 12

    # Plot error signal
    plt.figure(figsize=(12, 8))
    plt.plot(times, controller_states['error_signals'], label='Error Signal')
    plt.xlim([0, times[-1]])
    plt.xlabel('Time (s)', fontsize=label_font_size)
    plt.ylabel('Error Signal (kg/m^2)', fontsize=label_font_size)
    plt.title('Error Signal Over Time', fontsize=title_font_size)
    plt.legend(fontsize=legend_font_size)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')  # Add grid lines
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
    plt.savefig(os.path.join(output_dir, 'error_signal_plot.png'), dpi=300)
    plt.close()

    # Plot zoomed error signal
    plt.figure(figsize=(12, 8))
    plt.ylim([-100, 100])
    plt.plot(times, controller_states['error_signals'], label='Zoomed Error Signal')
    plt.xlabel('Time (s)', fontsize=label_font_size)
    plt.ylabel('Zoomed Error Signal', fontsize=label_font_size)
    plt.title('Zoomed Error Signal Over Time', fontsize=title_font_size)
    plt.legend(fontsize=legend_font_size)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')  # Add grid lines
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for suptitle
    plt.savefig(os.path.join(output_dir, 'zoomed_error_signal_plot.png'), dpi=300)
    plt.close()

    # Plot control output
    plt.figure(figsize=(12, 8))
    plt.plot(times, controller_states['control_outputs'], label='Control Output')
    plt.xlabel('Time (s)', fontsize=label_font_size)
    plt.ylabel('Control Output', fontsize=label_font_size)
    plt.title('Control Output Over Time', fontsize=title_font_size)
    plt.legend(fontsize=legend_font_size)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')  # Add grid lines
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'control_output_plot.png'), dpi=300)
    plt.close()

    # Plot airbrake deployment level
    plt.figure(figsize=(12, 8))
    plt.plot(times, controller_states['airbrake_deployment_levels'] * 90, label='Airbrake Deployment Level')
    plt.xlabel('Time (s)', fontsize=label_font_size)
    plt.ylabel('Airbrake Deployment Level (degrees)', fontsize=label_font_size)
    plt.title('Airbrake Deployment Level Over Time', fontsize=title_font_size)
    plt.ylim([0, 90])
    plt.legend(fontsize=legend_font_size)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')  # Add grid lines
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(output_dir, 'airbrake_deployment_plot.png'), dpi=300)
    plt.close()


def plot_vertical_states(times, estimated_states, true_states):
    # Define state names and groups (as in your original code)
    state_names = ['x', 'y', 'z', 'x_dot', 'y_dot', 'z_dot', 'x_ddot', 'y_ddot', 'z_ddot', 'q_w', 'q_x', 'q_y', 'q_z', 'omega_x', 'omega_y', 'omega_z', 'Cb_x', 'Cb_y', 'Cb_z']
    grouped_states = {
        'Displacements': [0, 1, 2],
        'Velocities': [3, 4, 5],
        'Accelerations': [6, 7, 8],
        'Quaternions': [9, 10, 11, 12],
        'Angular_Rates': [13, 14, 15],
        'Ballistic_Coefficients': [16, 17, 18]
    }

    true_altitude  = true_states[:, 2]
    estimated_altitude = estimated_states[:, 2, 0]

    true_vertical_velocity = true_states[:, 5]
    estimated_vertical_velocity = estimated_states[:, 5, 0]

    true_vertical_acceleration = np.append(np.diff(true_states[:, 5]) / np.diff(times), np.nan)
    estimated_vertical_acceleration = estimated_states[:, 8, 0]

    #true_ballistic_coefficient = - (get_air_density(true_altitude) * true_vertical_velocity**2) / (2 * (true_vertical_acceleration - get_gravity(true_altitude)))
    # Extract the estimated ballistic coefficient
    estimated_ballistic_coefficient = estimated_states[:, 18, 0]

    # Initialize true_ballistic_coefficient with NaN
    true_ballistic_coefficient = np.full_like(estimated_ballistic_coefficient, np.nan)

    # Define the window size
    window_size = 100

    # Compute the moving average looking ahead
    smoothed_values = np.convolve(
        estimated_ballistic_coefficient,
        np.ones(window_size)/window_size,
        mode='full'
    )[window_size-1:len(estimated_ballistic_coefficient)+window_size-1]  # Trim to align with input

    # Fill the result, keeping NaNs for the last `window_size-1` values
    true_ballistic_coefficient[:len(smoothed_values)] = smoothed_values

    # Detect apogee: velocity changes from positive to negative
    apogee_index = np.where((true_vertical_velocity[:-1] > 1) & (true_vertical_velocity[1:] <= 1))[0]
    if len(apogee_index) == 0:
        print("[ERROR] No apogee detected: check true_vertical_velocity data.")
        apogee_time = None
    else:
        apogee_time = times[apogee_index[0]]

    # Detect burnout: acceleration changes from positive to negative
    burnout_index = np.where((true_vertical_acceleration[:-1] > -9.81) & (true_vertical_acceleration[1:] <= -9.81))[0]
    if len(burnout_index) == 0:
        print("[ERROR] No burnout detected: check true_vertical_acceleration data.")
        burnout_time = None
    else:
        burnout_time = times[burnout_index[0]]

    # Create the output directory if it doesn't exist
    output_dir = 'plots/vertical_states/'
    os.makedirs(output_dir, exist_ok=True)

    # Set up the plot
    fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Vertical States Over Time', fontsize=16, fontweight='bold')

    # Subplot 1: Altitude
    axs[0].plot(times, estimated_altitude, color="#3da0d2", linewidth=1.5, label="Estimated States")
    axs[0].plot(times, true_altitude, color="black", linewidth=1.5, label="True States", linestyle='--')
    axs[0].set_ylabel("Altitude (m)", fontsize=12)
    axs[0].grid(True, linestyle='-', alpha=0.7)
    axs[0].set_ylim(0, 3000)

    # Subplot 2: Velocity
    axs[1].plot(times, estimated_vertical_velocity, color="#3da0d2", linewidth=1.5)
    axs[1].plot(times, true_vertical_velocity, color="black", linewidth=1.5, linestyle='--')
    axs[1].set_ylabel("Velocity (m/s)", fontsize=12)
    axs[1].grid(True, linestyle='-', alpha=0.7)
    axs[1].set_ylim(0, 275)

    # Subplot 3: Acceleration
    axs[2].plot(times, estimated_vertical_acceleration, color="#3da0d2", linewidth=1.5)
    axs[2].plot(times, true_vertical_acceleration, color="black", linewidth=1.5, linestyle='--')
    axs[2].set_ylabel("Acceleration (m/s²)", fontsize=12)
    axs[2].grid(True, linestyle='-', alpha=0.7)
    axs[2].set_ylim(-25, 100)

    # Subplot 4: Ballistic Coefficient
    axs[3].plot(times[burnout_index[0]:apogee_index[0]], estimated_ballistic_coefficient[burnout_index[0]:apogee_index[0]], color="#3da0d2", linewidth=1.5)
    axs[3].plot(times[burnout_index[0]:apogee_index[0]], true_ballistic_coefficient[burnout_index[0]:apogee_index[0]], color="black", linewidth=1.5, linestyle='--')
    axs[3].set_ylabel("Ballistic C. (kg/m²)", fontsize=12)
    axs[3].set_xlabel("Time (s)", fontsize=12)
    axs[3].grid(True, linestyle='-', alpha=0.7)
    axs[3].set_ylim(0, 5000)

    # Improve ticks on both axes
    for ax in axs:
        ax.tick_params(axis='both', which='minor', labelsize=12)

    # Add vertical lines for apogee and burnout
    for i, ax in enumerate(axs):
        ax.set_xlim(0, times[-1])
        if burnout_time is not None:
            ax.axvline(burnout_time, color='orange', linestyle='--', linewidth=1.5, label='Burnout' if i == 0 else None)
        if apogee_time is not None:
            ax.axvline(apogee_time, color='red', linestyle='--', linewidth=1.5, label='Apogee' if i == 0 else None)

    # Add legend to the first subplot only
    axs[0].legend(loc='lower right', fontsize=12)

    # Adjust spacing and layout
    plt.subplots_adjust(hspace=0.4)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the plot
    output_path = os.path.join(output_dir, 'vertical_states_plot.png')
    plt.savefig(output_path, dpi=600)

    eps_path = os.path.join(output_dir, 'vertical_states_plot.eps')
    plt.savefig(eps_path, format='eps', dpi=600)
    plt.close()

    # Compute the residuals for each state (True - Estimated)
    residual_altitude = true_altitude - estimated_altitude
    residual_vertical_velocity = true_vertical_velocity - estimated_vertical_velocity
    residual_vertical_acceleration = true_vertical_acceleration - estimated_vertical_acceleration
    residual_ballistic_coefficient = true_ballistic_coefficient - estimated_ballistic_coefficient

    # Set up the residuals plot
    fig_residuals, axs_residuals = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    fig_residuals.suptitle('Residuals of Vertical States Over Time', fontsize=16, fontweight='bold')

    # Subplot 1: Residuals for Altitude
    axs_residuals[0].plot(times, residual_altitude, color="black", linewidth=1.5, label="Residuals")
    axs_residuals[0].set_ylabel("Altitude (m)", fontsize=12)
    axs_residuals[0].grid(True, linestyle='-', alpha=0.7)

    # Subplot 2: Residuals for Vertical Velocity
    axs_residuals[1].plot(times, residual_vertical_velocity, color="black", linewidth=1.5)
    axs_residuals[1].set_ylabel("Velocity (m/s)", fontsize=12)
    axs_residuals[1].grid(True, linestyle='-', alpha=0.7)

    # Subplot 3: Residuals for Vertical Acceleration
    axs_residuals[2].plot(times, residual_vertical_acceleration, color="black", linewidth=1.5)
    axs_residuals[2].set_ylabel("Acceleration (m/s²)", fontsize=12)
    axs_residuals[2].grid(True, linestyle='-', alpha=0.7)

    # Subplot 4: Residuals for Ballistic Coefficient
    axs_residuals[3].plot(times[burnout_index[0]:apogee_index[0]], residual_ballistic_coefficient[burnout_index[0]:apogee_index[0]], color="black", linewidth=1.5)
    axs_residuals[3].set_ylabel("Ballistic C. (kg/m²)", fontsize=12)
    axs_residuals[3].set_xlabel("Time (s)", fontsize=12)
    axs_residuals[3].grid(True, linestyle='-', alpha=0.7)
    #axs_residuals[3].set_ylim(-1000, 1000)

    # Add legends and format
    for ax in axs_residuals:
        ax.set_xlim(0, times[-1])
        if burnout_time is not None:
            ax.axvline(burnout_time, color='orange', linestyle='--', linewidth=1.5, label='Burnout' if ax == axs_residuals[0] else None)
        if apogee_time is not None:
            ax.axvline(apogee_time, color='red', linestyle='--', linewidth=1.5, label='Apogee' if ax == axs_residuals[0] else None)

    # Add legend to the first subplot only
    axs_residuals[0].legend(loc='lower right', fontsize=12)
    # Adjust spacing and layout
    plt.subplots_adjust(hspace=0.4)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the residual plot in PNG format
    residual_png_path = os.path.join(output_dir, 'vertical_states_residuals_plot.png')
    plt.savefig(residual_png_path, dpi=600)

    # Save the residual plot in EPS format
    residual_eps_path = os.path.join(output_dir, 'vertical_states_residuals_plot.eps')
    plt.savefig(residual_eps_path, format='eps', dpi=600)

    # Close the figure after saving
    plt.close()
