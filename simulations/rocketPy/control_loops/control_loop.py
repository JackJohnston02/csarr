from control_loops.estimators.full_nav_EKF import FullNavEKF

from control_loops.controllers.sliding_mode_controller import SlidingModeController
from control_loops.controllers.velocity_regulation_controller import VelocityRegulationController
from control_loops.controllers.apogee_prediction_controller import ApogeePredictionController   

from utils.quaternion_utils import quaternion_conjugate, quaternion_multiply, quaternion_normalize

import numpy as np
from enum import Enum
from typing import Any, Dict
from control_loops.sensors import Barometer, Accelerometer, Gyroscope, Magnetometer, GPS  # Import sensor classes
from utils.environmental_utils import get_gravity

class FlightState(Enum):
    PAD = 0
    BURNING = 1
    COASTING = 2
    DESCENT = 3
    LANDED = 4


class RocketController:
    def __init__(self, launch_latitude: float, launch_longitude: float, launch_elevation: float):
        # Store environmental parameters
        self.launch_latitude = launch_latitude
        self.launch_longitude = launch_longitude
        self.launch_elevation = launch_elevation

        # Initialise the sensors (realistic noise)
        self.accelerometer = Accelerometer(sampling_rate=100, n_rms=np.array([120e-6, 120e-6, 120e-6]))  # 120 μg
        self.barometer = Barometer(sampling_rate=100, noise_std=0.1)  # 0.1 hPa
        self.gps = GPS(sampling_rate=10, noise_std=2.5, launch_coordinates=(launch_latitude, launch_longitude, launch_elevation))  # 2.5 m
        self.gyroscope = Gyroscope(sampling_rate=100, n_rms=0.004)  # 0.004 °/s
        self.magnetometer = Magnetometer(sampling_rate=100, noise_std=0.05, launch_coordinates=(launch_latitude, launch_longitude, launch_elevation))  # 0.05 μT



        self.flight_state = FlightState.PAD

        ## Define the estimators and choose one
        self.estimator = FullNavEKF(
            launch_coordinates=(launch_latitude, launch_longitude, launch_elevation),
            accelerometer_measurement_noise= self.accelerometer.noise_std,
            barometer_measurement_noise=self.barometer.noise_std,
            gps_measurement_noise=self.gps.noise_std,
            gyroscope_measurement_noise=self.gyroscope.noise_std,
            magnetometer_measurement_noise=self.magnetometer.noise_std
        )

        target_apogee = 2800
        
        apogee_prediction_controller = ApogeePredictionController(
            reference_apogee = target_apogee,
            kp = 0.0025,
            ki = 0.0008,
            kd = 0.0005,
            sampling_rate = 100
        )

        sliding_mode_controller = SlidingModeController(
            reference_apogee = target_apogee,
            sampling_rate = 100,
            mode = "super_twisting",
            neural_network_flag=False
        )

        reference_trajectory_controller = VelocityRegulationController(
            reference_apogee = target_apogee, 
            reference_ballistic_coefficient = 2000, 
            kp = 0.25,
            ki = 0.0104,
            kd = 0.0084,
            sampling_rate = 100
        )


        self.controller = reference_trajectory_controller



    def control_loop_function(
        self,
        time: Any,
        sampling_rate: Any,
        state: np.ndarray,
        state_history: Any,
        observed_variables: Any,
        air_brakes: Any
    ) -> Dict[str, Any]:
        
        # Print the time rounded to 2dp to keep track of the simulation, returning to start of line
        print(f"[INFO] Time: {time:.2f} s", end="\r")
        # ===================== Sensors =====================
        
        accelerometer_measurement = self.accelerometer.read(state, time)
        barometer_measurement = self.barometer.read(state, time)
        gps_measurement = self.gps.read(state, time)
        gyroscope_measurement = self.gyroscope.read(state, time)
        magnetometer_measurement = self.magnetometer.read(state, time)


        # ===================== Estimator =====================
        
        # Set the orienation before launch, this will be done by the Madgwick filter in reality 
        if self.estimator.set_orientation_flag == False:
            self.estimator.set_orientation(state)
            self.estimator.set_orientation_flag = True

        # Perform prediction step
        self.estimator.predict(u=0, current_time=time)

        # Update the ballistic coefficient estimate if in balistic flight
        if self.flight_state == FlightState.COASTING:
            self.estimator.update_Cb()         
    
        #print(time)
        # Update the Kalman filter with the accelerometer measurement if available
        if not np.any(np.isnan(accelerometer_measurement)):
            self.estimator.update_accelerometer_position(accelerometer_measurement)
            self.estimator.update_accelerometer_attitude(accelerometer_measurement)
            #print("acc")
            
        # Update the Kalman filter with the barometer measurement if available
        if not np.any(np.isnan(barometer_measurement)):
            self.estimator.update_barometer(barometer_measurement)
            #print("baro")

        # Update the Kalman filter with the gps measurement if available
        if not np.any(np.isnan(gps_measurement)):
            self.estimator.update_gps(gps_measurement)
            #print("gps")

        # Update the Kalman filter with the gyroscope measurement if available
        if not np.any(np.isnan(gyroscope_measurement)):
            self.estimator.update_gyroscope(gyroscope_measurement)
            #print("gyro")

        # Update the Kalman filter with the gyroscope measurement if available
        if not np.any(np.isnan(magnetometer_measurement)):
            self.estimator.update_magnetometer(magnetometer_measurement)
            #print("mag")

        
            
        # Get the updated state estimate
        estimated_state = self.estimator.get_state_estimate()

        #print(f" estimated acceleration {estimated_state[8]}")
        # ===================== State Machine ================
        match self.flight_state:
            case FlightState.PAD:
                if abs(estimated_state[8]) > 1:
                    self.flight_state = FlightState.BURNING
                    print("[INFO] PAD -> BURNING at", time, "s")

            case FlightState.BURNING:
                if estimated_state[8] < -9.81:
                    self.flight_state = FlightState.COASTING
                    print("[INFO] BURNING -> COASTING at", time, "s")

            case FlightState.COASTING:
                if estimated_state[5] < 0:
                    self.flight_state = FlightState.DESCENT
                    print("[INFO] COASTING -> DESCENT at", time, "s")

            case FlightState.DESCENT:
                if abs(estimated_state[5]) < 1 and abs(estimated_state[8]) < 1:
                    self.flight_state = FlightState.LANDED
                    print("[INFO] DESCENT -> LANDED at", time, "s")
                    print(f"velocity,acceleration {estimated_state[5], estimated_state[8]}")
            case FlightState.LANDED:
                ...
        


        # ===================== Controller =====================
        if self.flight_state == FlightState.COASTING:
            control_signal = self.controller.get_control_signal(estimated_state, time)
        else:
            control_signal = 0

        #print(f"control_signal = {control_signal}")
        new_deployment_level = air_brakes.deployment_level + control_signal / sampling_rate

        # ===================== Plant =====================

        # Limiting the spefed of the air_brakes to 0.2 per second
        max_change = 0.8 / sampling_rate
        lower_bound = air_brakes.deployment_level - max_change
        upper_bound = air_brakes.deployment_level + max_change
        # Clipping on airbrake speed
        new_deployment_level = np.clip(new_deployment_level, lower_bound, upper_bound)

        # Clipping on airbrake position such that it cannot go above 80 degrres which is 80/90
        new_deployment_level = np.clip(new_deployment_level, 0, 80/90)
        air_brakes.deployment_level = new_deployment_level

        # ===================== Logging =====================

        true_state = np.copy(state)

        # Now process true states such that they are in the same frame as the estimates
        # Add the additional components to the quaternion to encapsulate the full rotation from ENU to the sensor frame
        q1 = quaternion_conjugate(true_state[6:10]) # Represents the rotation from enu into rocketpy body axis
        q2 = [0.70710678, 0, 0.70710678, 0] # Rotation by 90 degress around y
        q3 = [0.70710678, 0.70710678, 0, 0] # Rotation by 90 around x
        q_conj = quaternion_multiply(quaternion_multiply(q3, q2), q1)
        q = quaternion_conjugate(q_conj)
        q = quaternion_normalize(q)
        true_state[6:10] = q 
        
        # Will need to rotate the body rates before plotting as well
        # Make the body rate a quaternion
        body_rate_qaut = np.array([0] + list(true_state[10:13]))
        q = [0.70710678, 0, 0.70710678, 0]
        sensor_rate_quat = quaternion_multiply(quaternion_multiply(q, body_rate_qaut), quaternion_conjugate(q))
        true_state[10:13] = sensor_rate_quat[1:]

        # Tare the altitude
        true_state[2] = true_state[2] - self.launch_elevation


        # Create a dictionary to hold measurements
        measurements = {
            "barometer": barometer_measurement,
            "accelerometer": accelerometer_measurement,
            "gps": gps_measurement,
            "gyroscope": gyroscope_measurement,
            "magnetometer": magnetometer_measurement
        }



        controller_state = {
            "airbrake_deployment_level": air_brakes.deployment_level,
            "error_signal": self.controller.error_signal,
            "control_output": self.controller.control_output
        }


        return {
            "time": time,
            "true_state": true_state, 
            "estimated_state": estimated_state,
            "measurements": measurements,
            "controller_state": controller_state,
        }
