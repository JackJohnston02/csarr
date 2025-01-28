import numpy as np
from utils.environmental_utils import get_gravity
from utils.quaternion_utils import quaternion_conjugate, quaternion_multiply, quaternion_normalize

class Accelerometer:
    def __init__(self, sampling_rate: float, n_rms: float, bias_drift_std: float = 0.01):
        self.previous_time = None  # To store the last reading time
        self.sampling_interval = 1 / sampling_rate if sampling_rate > 0 else None  # Time interval in seconds
        self.noise_std = n_rms * np.sqrt(sampling_rate)  # Standard deviation for the noise
        self.bias = np.zeros(3)  # Bias for each axis
        self.bias_drift_std = bias_drift_std  # Standard deviation for bias drift
        self.last_bias_update_time = 0  # Last time the bias was updated

    def update_bias(self, current_time: float):
        """Update the bias with random walk (bias drift)"""
        time_elapsed = current_time - self.last_bias_update_time
        if time_elapsed > 0:
            # Random walk for bias drift
            bias_drift = np.random.normal(scale=self.bias_drift_std, size=3)
            self.bias += bias_drift * time_elapsed
            self.last_bias_update_time = current_time

    def read(self, state: np.ndarray, current_time: float) -> np.ma.MaskedArray:
        current_velocity = np.array([state[3], state[4], state[5]])

        # Sensor is turned off
        if self.sampling_interval is None:
            return np.ma.array([[np.nan], [np.nan], [np.nan]])  # Return masked array

        # Cannot calculate acceleration if the sensor is turned off
        if self.previous_time is None:
            self.previous_velocity = current_velocity
            self.previous_time = current_time
            return np.ma.array([[np.nan], [np.nan], [np.nan]])  # Masked first read

        time_interval = current_time - self.previous_time

        # Normal operation
        if current_time - self.previous_time >= self.sampling_interval:
            # Calculate acceleration
            acceleration = (current_velocity - self.previous_velocity) / time_interval
            
            acceleration[2] -= get_gravity(state[2])  # Add gravity to the z-axis (considering ENU)

            # TODO
                # This correction factor shouldnt be required
                # It is due to rocketpy using energy calculations, there is an issue with rocketpys calculation of acceleration
            acceleration[2] -= 0.05 # Correction factor because rocketpy is crap
            #print(f" Acceleration, {acceleration[2]}") 
            # Update bias before applying it
            self.update_bias(current_time)

            #print(f"current_velocity, {current_velocity}")
            self.previous_velocity = current_velocity
            self.previous_time = current_time

            # Add noise to the acceleration
            noise = np.random.normal(scale=self.noise_std, size=3)

            # Create a quaternion representation of the acceleration vector (0, ax, ay, az)
            acceleration_enu_quat = np.array([0] + list(acceleration))

            # Calculate the quaternion transformation from the rocketpy ENU frame to the sensor frame
            # Part1 is a rotation from ENU into the rocketpy frame
            q1 = quaternion_conjugate(state[6:10])  # Rotation from ENU into rocketpy body axis
            q2 = [0.70710678, 0, 0.70710678, 0]  # Rotation by 90 degrees around y
            q3 = [0.70710678, 0.70710678, 0, 0]  # Rotation by 90 degrees around x
            q = quaternion_multiply(quaternion_multiply(q3, q2), q1)

            # Transform the acceleration into the sensor frame
            acceleration_sensor_quat = quaternion_multiply(
                quaternion_multiply(q, acceleration_enu_quat), quaternion_conjugate(q)
            )


            # Extract the body acceleration from the quaternion result
            acceleration_sensor = acceleration_sensor_quat[1:]  # Discard the scalar part

            acceleration_sensor = acceleration_sensor + noise + self.bias

            # Return the acceleration in body frame reshaped
            return np.array(acceleration_sensor.reshape(3, 1))

        return np.ma.array([[np.nan], [np.nan], [np.nan]])  # Masked if not enough time has passed
