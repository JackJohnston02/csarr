import numpy as np
import csv

def process_flight_data(flight):
    obs_vars = flight.get_controller_observed_variables()

    # Initialize lists to store results
    times = []
    true_states = []
    estimated_states = []
    controller_states = []
    gyro_measurements = []
    barometer_measurements = []
    accelerometer_measurements = []
    magnetometer_measurements = []
    gps_measurements = []

    # Lists to store individual controller state variables
    error_signals = []
    control_outputs = []
    airbrake_deployment_levels = []

    # Loop through observed variables and extract the necessary data
    for var in obs_vars:
        if isinstance(var, dict):
            time = var.get('time', None)
            true_state = var.get('true_state', None)
            estimated_state = var.get('estimated_state', None)
            measurements = var.get('measurements', None)
            controller_state = var.get('controller_state', None)

            # Append time and states
            times.append(time)
            true_states.append(true_state)
            estimated_states.append(estimated_state)
            controller_states.append(controller_state)

            # Extract individual controller state variables
            if controller_state:
                error_signals.append(controller_state.get('error_signal', 0))
                control_outputs.append(controller_state.get('control_output', 0))
                airbrake_deployment_levels.append(controller_state.get('airbrake_deployment_level', 0))

            # Append measurements
            if measurements:
                gyro_measurements.append(measurements['gyroscope'])
                barometer_measurements.append(measurements['barometer'])
                accelerometer_measurements.append(measurements['accelerometer'])
                magnetometer_measurements.append(measurements['magnetometer'])
                gps_measurements.append(measurements['gps'])

    # Convert lists to numpy arrays for further processing
    times = np.array(times)
    gyro_measurements = np.array(gyro_measurements)
    barometer_measurements = np.array(barometer_measurements)
    accelerometer_measurements = np.array(accelerometer_measurements)
    magnetometer_measurements = np.array(magnetometer_measurements)
    gps_measurements = np.array(gps_measurements)

    # Combine estimated and true state vectors into numpy arrays
    estimated_states = np.stack(estimated_states) if estimated_states else np.zeros((0, 3, 1))
    true_states = np.stack(true_states) if true_states else np.zeros((0, 13, 1))

    # Combine controller state variables into arrays
    error_signals = np.array(error_signals)
    control_outputs = np.array(control_outputs)
    airbrake_deployment_levels = np.array(airbrake_deployment_levels)

    return {
        'times': times,
        'true_states': true_states,
        'estimated_states': estimated_states,
        'controller_states': {
            'error_signals': error_signals,
            'control_outputs': control_outputs,
            'airbrake_deployment_levels': airbrake_deployment_levels
        },
        'accelerometer_measurements': accelerometer_measurements,
        'barometer_measurements': barometer_measurements,
        'gps_measurements': gps_measurements,
        'gyroscope_measurements': gyro_measurements,
        'magnetometer_measurements': magnetometer_measurements,
    }


def export_flight_data(data: dict, filename: str):
    """
    Save flight data to a CSV file, including true state, measured, and estimated sections.

    Args:
        data (dict): A dictionary containing 'times', 'true_states', 'accelerometer_measurements',
                     'barometer_measurements', 'gyroscope_measurements', 'magnetometer_measurements',
                     'gps_measurements', and 'estimated_states'.
        filename (str): The name of the CSV file to save the data to.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write headers for the true state, measured data, and estimated state, controller state
        writer.writerow([
            "Time", 
            "True X", "True Y", "True Z", "True VX", "True VY", "True VZ", "True qw", "True qx", "True qy", "True qz", "True wx", "True wy", "True wz", 
            "Barometer", "Accelerometer X", "Accelerometer Y", "Accelerometer Z", 
            "Gyroscope X", "Gyroscope Y", "Gyroscope Z", 
            "Magnetometer X", "Magnetometer Y", "Magnetometer Z", 
            "GPS Latitude", "GPS Longitude", "GPS Altitude", 
            "Estimated X", "Estimated Y", "Estimated Z", "Estimated VX", "Estimated VY", "Estimated VZ" , "Estimated aX", "Estimated aY", "Estimated aZ", "Estimated qw", "Estimated qx", "Estimated qy", "Estimated qz", "Estimated wx", "Estimated wy", "Estimated wz", "Estimated Cbx", "Estimated Cby", "Estimated Cbz",
            "Error Signal", "Controller Output", "Airbrake Deployment Level"
        ])

        # Iterate through the data and write each time step's data in a single row
        for i in range(len(data['times'])):
            t = data['times'][i]
            true_state = data['true_states'][i]
            estimated_state = data['estimated_states'][i]
            accelerometer = data['accelerometer_measurements'][i]
            barometer = data['barometer_measurements'][i]
            gyroscope = data['gyroscope_measurements'][i]
            magnetometer = data['magnetometer_measurements'][i]
            gps = data['gps_measurements'][i]

            error_signal = data['controller_states']['error_signals'][i]
            control_output = data['controller_states']['control_outputs'][i]
            airbrake_deployment_levels = data['controller_states']['airbrake_deployment_levels'][i]

            # Combine all data into a single row
            row = [
                t, 
                *true_state,  # True state data
                barometer, *accelerometer, *gyroscope, *magnetometer, *gps,  # Measured data
                *estimated_state,  # Estimated state data
                error_signal, control_output, airbrake_deployment_levels
            ]

            # Write the combined row
            writer.writerow(row)

    print(f"[INFO] Data saved to {filename}")