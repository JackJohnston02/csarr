import numpy as np
from utils.environmental_utils import get_air_density, get_gravity

class ApogeePredictionController:

    def __init__(self, reference_apogee, kp, ki, kd, sampling_rate):
        self.reference_apogee = reference_apogee
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.sampling_rate = sampling_rate
        self.sampling_interval = 1 / sampling_rate if sampling_rate > 0 else None
        
        self.error_signal = np.nan
        self.control_output = 0
        self.control_output = 0
        self.previous_error = 0
        self.previous_time = None
        self.integral_error = 0


    def get_control_signal(self, estimated_state, current_time):
        if self.sampling_interval is None:
            return 0
        
        altitude = estimated_state[2, 0]
        velocity = estimated_state[5, 0]
        ballistic_coefficient = estimated_state[18, 0]

        predicted_apogee = self.get_apogee_prediction(altitude, velocity, ballistic_coefficient)

        # Impliment a PID controller 
        self.error_signal = self.reference_apogee - predicted_apogee

        dt = current_time - self.previous_time if self.previous_time is not None else 0.0
        if dt > 0:
            self.integral_error += self.error_signal * dt
            derivative = (self.error_signal - self.previous_error) / dt
        else:
            derivative = 0.0
        
        self.control_output = (
            -1 * self.kp * self.error_signal + 
            -1 * self.ki * self.integral_error + 
            -1 * self.kd * derivative
        )

        return self.control_output
    
    def get_apogee_prediction(self, altitude, velocity, ballistic_coefficient):
        dt = 0.05
        while velocity > 0:
            altitude = altitude + dt * velocity
            velocity = velocity + dt * (get_gravity(altitude) - (get_air_density(altitude_ASL=altitude) * velocity * velocity) / (2 * ballistic_coefficient))

        apogee_prediction = altitude
        return apogee_prediction 