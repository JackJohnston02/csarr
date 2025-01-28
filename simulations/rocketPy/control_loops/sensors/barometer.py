import numpy as np

class Barometer:
    def __init__(self, sampling_rate: float, noise_std: float):
        self.previous_time = None  # To store the last reading time
        
        if sampling_rate == 0:
            self.sampling_interval = None
        else:
            self.sampling_interval = 1 / sampling_rate  # Time interval in seconds
        
        self.noise_std = noise_std  # Standard deviation for the noise

    def read(self, state: np.ndarray, current_time: float):
        """
        Read altitude from the current state, respecting the sampling rate.
        
        Args:
            state (np.ndarray): The current state of the rocket.
            current_time (float): The current time for calculating the sampling interval.

        Returns:
            float: The computed altitude measurement or None if it's not time to read again.
        """
        if self.sampling_interval is None:
            return np.array([np.nan])

        if self.previous_time is None or (current_time - self.previous_time >= self.sampling_interval):
            self.previous_time = current_time
            measurement = state[2] + np.random.normal(0, self.noise_std)
            return np.array([measurement])

        return np.array([np.nan])  # If it's not time to read again, return None
