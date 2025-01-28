import numpy as np
from utils.quaternion_utils import quaternion_conjugate, quaternion_multiply
from utils.coordinate_transforms import geodetic_to_ecef, enu_to_ecef, ecef_to_geodetic

class Magnetometer:
    def __init__(self, sampling_rate: float, noise_std: float, launch_coordinates: tuple):
        self.previous_time = None
        if sampling_rate == 0:
            self.sampling_interval = None
        else:
            self.sampling_interval = 1 / sampling_rate  # Time interval in seconds
        
        self.noise_std = noise_std  # Standard deviation for the noise

        self.launch_coordinates = launch_coordinates

    def calculate_earth_magnetic_field(self, location: tuple) -> np.ndarray:
        """
        Calculate the Earth's magnetic field at the given ENU location
        TODO
            Use the eart magnetic model not this basic function

        """
        # Takes Lat, Lon, Alt

        # Returns magnetic vector in ENU really simple model asssumes is only vector pointing in the N direction
        return np.array([0, 1, 0])
    


    def read(self, state: np.ndarray, current_time: float) -> float:
        """
        Read the magnetometer measurements in the body frame.
        :param state: The state array containing orientation as a quaternion
        :return: Magnetic field vector in the body frame
        """

        # Not time for a measurement
        if self.sampling_interval is None:
            return np.array([[np.nan], [np.nan], [np.nan]])

        # If it's time to read the magnetometer
        if self.previous_time is None or (current_time - self.previous_time >= self.sampling_interval):
            self.previous_time = current_time


            # 1. Calculate where the rocket is (lat, lon, alt)
            displacement_enu = state[0:3] # displacement in m, (x,y,z), (enu)
            # We can set launch altitude to 0 as z state is ASL not AGL
            launch_ecef = geodetic_to_ecef(self.launch_coordinates[0], self.launch_coordinates[1], 0)

            # Convert ENU displacement to ECEF displacement
            displacement_ecef = enu_to_ecef(displacement_enu, self.launch_coordinates[0], self.launch_coordinates[1])

            # Apply the displacement in ECEF coordinates
            new_ecef = launch_ecef + displacement_ecef

            # Convert the new ECEF coordinates back to geodetic (lat, lon, alt)
            current_coordinates = ecef_to_geodetic(new_ecef[0], new_ecef[1], new_ecef[2])

            magnetic_field_enu = self.calculate_earth_magnetic_field(current_coordinates)

            # Add noise to the magnetic field
            magnetic_field_enu_quat = np.array([0] + list(magnetic_field_enu))

            # Part1 is a rotation from enu into the rocketpy frame
            q1 = quaternion_conjugate(state[6:10]) # Represents the rotation from enu into rocketpy body axis
            q2 = [0.70710678, 0, 0.70710678, 0] # Rotation by 90 degress around y
            q3 = [0.70710678, 0.70710678, 0, 0] # Rotation by -90 around x
            q = quaternion_multiply(quaternion_multiply(q3, q2), q1)

            magnetic_field_sensor_quat = quaternion_multiply(quaternion_multiply(q, magnetic_field_enu_quat), quaternion_conjugate(q))

            magnetic_field_sensor = magnetic_field_sensor_quat[1:]  # Discard the scalar part

            noise = np.random.normal(0, self.noise_std, 3)
            magnetic_field_sensor = magnetic_field_sensor + noise

            # Return the magnetic field in body frame reshaped
            return np.array(magnetic_field_sensor.reshape(3, 1))
    
        return np.array([[np.nan], [np.nan], [np.nan]])