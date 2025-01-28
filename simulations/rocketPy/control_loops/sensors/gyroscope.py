import numpy as np
from utils.quaternion_utils import quaternion_conjugate, quaternion_multiply, quaternion_normalize

# TODO
    # Angular rates are in the body axis already...
    # Rocket’s angular velocity Omega 1 as a function of time. Direction 1 is in the rocket’s body axis and points perpendicular to the rocket’s axis of cylindrical symmetry.
    # Rocket’s angular velocity Omega 2 as a function of time. Direction 2 is in the rocket’s body axis and points perpendicular to the rocket’s axis of cylindrical symmetry and direction 1.
    # Rocket’s angular velocity Omega 3 as a function of time. Direction 3 is in the rocket’s body axis and points in the direction of cylindrical symmetry.

class Gyroscope:
    def __init__(self, sampling_rate: float, n_rms: float):
        self.previous_time = None  # To store the last reading time
        self.previous_quaternion = None  # To store the previous quaternion
        
        if sampling_rate == 0:
            self.sampling_interval = None
        else:
            self.sampling_interval = 1 / sampling_rate  # Time interval in seconds
        
        self.noise_std = ((2 * np.pi * n_rms)/360) * np.sqrt(sampling_rate)  # Standard deviation for the noise

    def read(self, state: np.ndarray, current_time: float) -> np.ndarray:
        """
        Read gyroscope measurement from the current state, respecting the sampling rate.
        
        Args:
            state (np.ndarray): The current state of the rocket.
            current_time (float): The current time for calculating the sampling interval.

        Returns:
            np.ndarray: The computed gyroscope measurement (3x1 array) or None if it's not time to read again.
        """
        if self.sampling_interval is None:
            return np.array([[np.nan], [np.nan], [np.nan]])

        # If it's time to read the gyroscope
        if self.previous_time is None or (current_time - self.previous_time >= self.sampling_interval):
            self.previous_time = current_time
            
            omega_body = state[10:13] + np.random.normal(0, self.noise_std, size=3) # Get euler rates from rocketpy's state vector
            omega_body_qaut = np.array([0] + list(omega_body)) # Add noise to measurements

            # Rotate from rocketpy body axis into sensor axis rotation of 90 degrees around y axis
            rotation_quat = np.array([0.70710678, 0, 0.70710678, 0])  # [cos(π/4), 0, sin(π/4), 0]
            omega_sensor_quat = quaternion_multiply(quaternion_multiply(rotation_quat, omega_body_qaut), quaternion_conjugate(rotation_quat))
            omega_sensor = omega_sensor_quat[1:]  # Discard the scalar part

            return omega_sensor.reshape(3,1) 

        return np.array([[np.nan], [np.nan], [np.nan]])