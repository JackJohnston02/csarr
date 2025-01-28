# controllers.velocity_regulation_controller.py

# Controller that genenerates a trajectory before launch during initialisation based on constant ballistic coeficient assumption and target apogee

#TODO
    # Implement the sampling rate

import numpy as np
from utils.environmental_utils import get_air_density, get_gravity 
from scipy.interpolate import interp1d


class VelocityRegulationController:
    def __init__(self, reference_apogee, reference_ballistic_coefficient, kp, ki, kd, sampling_rate):
        self.reference_apogee = reference_apogee
        self.reference_ballistic_coefficient = reference_ballistic_coefficient
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.error_signal = 0
        self.integral = 0
        self.control_output = 0
        self.previous_error = 0
        self.previous_time = 0
        self.integral_error = 0  # Integral of the error
        self.sampling_interval = 1 / sampling_rate if sampling_rate > 0 else None

        self.reference_trajectory = self.get_reference_trajectory()

        


    def get_control_signal(self, estimated_state, current_time):
        """
        Calculate the control signal based on the current state.
        
        :param estimated_state: The current system state (e.g., current velocity).
        :param current_time: The current time (for calculating the derivative term).
        :return: Control signal to apply.
        """
        if self.sampling_interval is None:
            return self.control_output

        if self.previous_time is None or (current_time - self.previous_time >= self.sampling_interval):
            self.previous_time = current_time

            z = estimated_state[2, 0]  # Vertical position (altitude)
            estimated_velocity = estimated_state[5, 0]  # Vertical velocity

            # Create an interpolation function for the reference trajectory
            reference_trajectory_interp = interp1d(
                np.arange(len(self.reference_trajectory)), 
                self.reference_trajectory, 
                kind='linear', 
                fill_value="extrapolate"
            )

            # Interpolate to get the reference velocity at the current altitude
            reference_velocity = reference_trajectory_interp(z)
            #print(f"reference_velocity, {reference_velocity}")

            # Calculate the error signal
            # If the rocket is close to the target apogee, the error is zero
            if estimated_velocity < 25:
                error = 0
                self.integral = 0  # Reset integral term
                derivative = 0  # Reset derivative term
            else:
                error = reference_velocity - estimated_velocity
                # Calculate time difference
                dt = self.sampling_interval
                if dt > 0:  # To avoid division by zero in the derivative term
                    # Integral term (I)
                    self.integral += error * dt  # Accumulate the error over time

                    # Derivative term (D)
                    derivative = (error - self.previous_error) / dt  # Rate of change of error
                else:
                    derivative = 0.0  # If time hasn't changed, set derivative to zero
    
                self.error_signal = error

                # Calculate time difference
                dt = current_time - self.previous_time if self.previous_time is not None else 0.0
                if dt > 0:  # To avoid division by zero in the derivative term
                    # Integral term (I)
                    self.integral += error * dt  # Accumulate the error over time

                    # Derivative term (D)
                    derivative = (error - self.previous_error) / dt  # Rate of change of error
                else:
                    derivative = 0.0  # If time hasn't changed, set derivative to zero

                # Control signal calculation
                control_signal = (
                    -1 * self.kp * error + 
                    -1 * self.ki * self.integral + 
                    -1 * self.kd * derivative
                )

                # Update previous error and time
                self.previous_error = error
                self.previous_time = current_time

                self.control_output = control_signal

        if self.control_output > 0.4:
            self.control_output = 0.4
        if self.control_output < -0.4:
            self.control_output = -0.4

        return self.control_output


    
    def get_reference_trajectory(self):
        
        # Starting at apogee, back propagate until altitude is <= 0 using the constant ballistic assumption and the reference ballistic coefficient
        initial_position = self.reference_apogee
        initial_velocity = 0
        ballistic_coefficient = self.reference_ballistic_coefficient

        x = np.zeros([2,1])
        x[0,0] = initial_position
        x[1,0] = initial_velocity

        dt = -1e-4

        next_int = initial_position

        reference_trajectory = np.empty([initial_position + 1])

        reference_trajectory[:] = np.nan

        while x[0,0] >= 0:
            x[0, 0] = x[0,0] + dt *  x[1,0]
            x[1, 0] = x[1, 0] + dt*(get_gravity(x[0,0]) - (get_air_density(x[0,0]) * x[1, 0]**2) / (2 * ballistic_coefficient))

            if x[0, 0] < next_int:
                reference_trajectory[int(next_int)] = x[1, 0]
                next_int = np.rint(x[0, 0] - 1)

        return reference_trajectory
    

