import numpy as np
from utils.coordinate_transforms import geodetic_to_ecef, enu_to_ecef, ecef_to_geodetic

class GPS:
    def __init__(self, sampling_rate: float, noise_std: float, launch_coordinates: tuple):
        self.previous_time = None
        if sampling_rate == 0:
            self.sampling_interval = None
        else:
            self.sampling_interval = 1 / sampling_rate  # Time interval in seconds
        
        self.noise_std = noise_std  # Standard deviation for the noise
        self.launch_coordinates = launch_coordinates  # Lat, Lon, Alt
    
    def read(self, state: np.ndarray, current_time: float) -> np.ndarray:
        """
        Read the GPS measurements.
        :param state: The state array containing displacement in ENU coordinates.
        :param current_time: The current time in seconds.
        :return: Current GPS coordinates (latitude, longitude, altitude) in degrees and meters.
        """
        # Not time for a measurement
        if self.sampling_interval is None:
            return np.array([[np.nan], [np.nan], [np.nan]])

        # If it's time to read the GPS
        if self.previous_time is None or (current_time - self.previous_time >= self.sampling_interval):
            self.previous_time = current_time

            # 1. Get ENU displacement (make a copy to avoid modifying the true state)
            displacement_enu = np.copy(state[0:3]).astype(float)  # Cast to float

            # 2. Add Gaussian noise to the copy
            displacement_enu += np.random.normal(0, self.noise_std, 3)

            # 3. Convert launch geodetic coordinates to ECEF
            launch_ecef = geodetic_to_ecef(self.launch_coordinates[0], self.launch_coordinates[1], 0)

            # 4. Convert ENU displacement to ECEF displacement
            displacement_ecef = enu_to_ecef(displacement_enu, self.launch_coordinates[0], self.launch_coordinates[1])

            # 5. Add ECEF displacement to the initial ECEF coordinates
            new_ecef = launch_ecef + displacement_ecef

            # 6. Convert new ECEF coordinates back to geodetic (lat, lon, alt)
            current_coordinates = ecef_to_geodetic(new_ecef[0], new_ecef[1], new_ecef[2])

            # 7. Return the new geodetic coordinates (latitude, longitude, altitude)
            return np.array([[current_coordinates[0]], [current_coordinates[1]], [current_coordinates[2]]])
        
        # If not time for a new measurement, return NaN
        return np.array([[np.nan], [np.nan], [np.nan]])
