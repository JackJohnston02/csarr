'''
FULL NAVIGATION EXTENDED KALMAN FILTER
- Includes methods for sensor fusion
- Includes methods for ballistic coefficient estimation
'''


import numpy as np
from control_loops.estimators.base_kalman_filter import BaseKalmanFilter
from utils.coordinate_transforms import geodetic_to_ecef, lat_lon_alt_to_enu
from utils.quaternion_utils import quaternion_conjugate, quaternion_multiply, quaternion_normalize, rotate_vector_by_quaternion
from utils.environmental_utils import get_gravity, get_air_density
import math

class FullNavEKF(BaseKalmanFilter):
    def __init__(self,
                 launch_coordinates,
                 accelerometer_measurement_noise,
                 barometer_measurement_noise,
                 gyroscope_measurement_noise,
                 gps_measurement_noise,
                 magnetometer_measurement_noise):
        
        super().__init__()  # Call the parent class's __init__

        # State vector [x, y, z, x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot, q_w, q_x, q_y, q_z, omega_x, omega_y, omega_z, Cb_x, Cb_y, Cb_z]
        # State vector [0, 1, 2, 3,     4,     5,     6,      7,      8,      9,   10,  11,  12,  13,      14,      15,      16,   17,   18]
        self.x = np.zeros((19, 1))  # State estimate vector
        
        # Initial quaternion from sensor axis, to enu (assumes rocket is vertical on the pad)
        # This is just a rotation around the body y axis
       # Set the initial quaternion (90 degrees rotation around y-axis)
        self.x[9, 0] = 0.7071
        self.x[10, 0] = 0
        self.x[11, 0] = 0.7071
        self.x[12, 0] = 0
        

        self.x[16,0] = self.x[17,0] = self.x[18,0] = 3500
    
        self.P = np.eye(19)         # Covariance matrix

        self.P[0,0] = self.P[1,1] = self.P[2,2] = 10 # Initial position covariance
        self.P[3,3] = self.P[4,4] = self.P[5,5] = 1 # Initial velocity covariance
        self.P[6,6] = self.P[7,7] = self.P[8,8] = 1 # Initial acceleration covariance
        self.P[9,9] = self.P[10,10] = self.P[11,11] = self.P[12,12] = 1e-10 # Initial orientation covariance
        self.P[13,13] = self.P[14,14] = self.P[15,15] = 0.00001 # Initial body rate covariance
        self.P[16,16] = self.P[17,17] = self.P[18,18] = 1e-5 # Initial Cb covariance
        self.P = 1 * self.P

        # Measurement noise covariance matrices
        self.R_accelerometer = accelerometer_measurement_noise**2 * np.eye(3,3)
        self.R_barometer = barometer_measurement_noise**2
        self.R_gps = gps_measurement_noise**2 * np.eye(3,3)
        self.R_gyroscope = gyroscope_measurement_noise**2 *  np.eye(3,3)
        self.R_magnetometer = magnetometer_measurement_noise**2 *  np.eye(3,3)

        # Ballistic Coefficient update covariance matrix
        self.R_Cb = 1 * np.eye(3,3)

        # State transition matrix (for predictions)
        self.F = np.eye(19)

        # Initialize time tracking
        self.last_time = None  # To track the last time predict was called

        # Starting coordinate, convert into ecef 
        self.launch_coordinates = launch_coordinates

        # Flags
        self.set_orientation_flag = False



    def set_orientation(self, state):
        q1 = quaternion_conjugate(state[6:10]) # Represents the rotation from enu into rocketpy body axis
        q2 = [0.70710678, 0, 0.70710678, 0] # Rotation by 90 degress around y
        q3 = [0.70710678, 0.70710678, 0, 0] # Rotation by 90 around x
        q_conj = quaternion_multiply(quaternion_multiply(q3, q2), q1)
        q = quaternion_conjugate(q_conj)
        q = quaternion_normalize(q)
        self.x[9:13] = q.reshape(4,1)
        return None
    

    def predict(self, u, current_time):
        """
        Predicts the next state based on the state transition model.
        u: Control input (optional)
        current_time: Current time for time tracking (optional)
        """
        # If current_time is provided, update last_time
        if current_time is not None:
            if self.last_time is not None:
                dt = current_time - self.last_time  # Time difference
            else:
                dt = 0  # First call, no time difference
            self.last_time = current_time  # Update last_time


        self.F = self.get_process_model(dt, self.x) # Calculate the new Jacobian
        self.Q = self.get_process_noise(dt) # Calculate the new process noise matrix

        # Predict the next state
        self.x = self.F @ self.x

        # Normalise the quaternion
        self.x[9:13, 0] = quaternion_normalize(self.x[9:13, 0])


        # Update covariance matrix
        self.P = self.F @ self.P @ self.F.T + self.Q


    def update_Cb(self):
        # Current Cb estimate is generated from current states, i.e. acceleration, velocity, etc.

        g = get_gravity(self.x[2, 0])
        rho = get_air_density(self.x[2, 0])

        # Calculate acceleration in the body frame
        m_acc = self.x[6:9] - np.array([[0], [0], [g]])  # Subtract gravity from the state vector

        # Calculate velocity in the body frame
        m_vel = self.x[3:6]  # We want the velocity in the body axis

        # This can be the new "measurement" for the ballistic coefficient in the body axis
        z = -(rho * m_vel * np.linalg.norm(m_vel)) / (2 * m_acc)

        # The previous state for the ballistic coefficient is already in ENU axis and therefore doesn't need to be rotated
        m = self.x[16:19]

        self.H_Cb = np.zeros((3, 19))
        self.H_Cb[0, 16] = 1
        self.H_Cb[1, 17] = 1
        self.H_Cb[2, 18] = 1

        # Kalman gain calculation
        S = self.H_Cb @ self.P @ self.H_Cb.T + self.R_Cb
        K = self.P @ self.H_Cb.T @ np.linalg.inv(S)

        # Update estimate with adjusted barometer measurement
        y = z - self.H_Cb @ self.x  # Residual (innovation)
        self.x = self.x + K @ y

        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H_Cb) @ self.P

        # Clip the ballistic coefficients between 0 and 5000
        self.x[16:19] = np.clip(self.x[16:19], 500, 4000)

        return None


        
        

    def update_accelerometer_attitude(self, z):
        """
            Method for updating the orientation based on the accelerometer measurement and the estimated gravity vector
            z: Accelerometer measurement vector
        """

        # Define the reference vector, this the expected measured acceleration in ENU
        g = get_gravity(self.x[2,0])
        r = self.x[6:9] -  np.array([[0] ,[0] ,[g]]) # Subtract gravity from the estimated accelerations, IMU can't measure gravity 

        # We want the conjugate, q represents body -> ENU, we want ENU -> body
        q = quaternion_conjugate(self.x[9:13]) # get the quaternion from the state vector

        # Convert the reference vector into a quat
        r_quat =  np.array([0, r[0,0], r[1,0], r[2,0]])

        # Rotate the reference vector from enu to sensor frame
        m_quat = quaternion_multiply(quaternion_multiply(q, r_quat), quaternion_conjugate(q))
        m = m_quat[1:].reshape(3,1) # Remove scalar part

        r0 = r[0]
        r1 = r[1]
        r2 = r[2]

        q0 = q[0,0]
        q1 = q[1,0]
        q2 = q[2,0]
        q3 = q[3,0]

        # Initialize the Jacobian with zeros
        self.H_accelerometer = np.zeros((3, 19))

        # Populate first row
        self.H_accelerometer[0, 6] = q0**2 + q1**2 - q2**2 - q3**2
        self.H_accelerometer[0, 7] = -2*q0*q3 + 2*q1*q2
        self.H_accelerometer[0, 8] = 2*q0*q2 + 2*q1*q3
        self.H_accelerometer[0, 9] = 2*q0*r0 + 2*q2*r2 - 2*q3*r1
        self.H_accelerometer[0, 10] = 2*q1*r0 + 2*q2*r1 + 2*q3*r2
        self.H_accelerometer[0, 11] = 2*q0*r2 + 2*q1*r1 - 2*q2*r0
        self.H_accelerometer[0, 12] = -2*q0*r1 + 2*q1*r2 - 2*q3*r0

        # Populate second row
        self.H_accelerometer[1, 6] = 2*q0*q3 + 2*q1*q2
        self.H_accelerometer[1, 7] = q0**2 - q1**2 + q2**2 - q3**2
        self.H_accelerometer[1, 8] = -2*q0*q1 + 2*q2*q3
        self.H_accelerometer[1, 9] = 2*q0*r1 - 2*q1*r2 + 2*q3*r0
        self.H_accelerometer[1, 10] = -2*q0*r2 - 2*q1*r1 + 2*q2*r0
        self.H_accelerometer[1, 11] = 2*q1*r0 + 2*q2*r1 + 2*q3*r2
        self.H_accelerometer[1, 12] = 2*q0*r0 + 2*q2*r2 - 2*q3*r1

        # Populate third row
        self.H_accelerometer[2, 6] = -2*q0*q2 + 2*q1*q3
        self.H_accelerometer[2, 7] = 2*q0*q1 + 2*q2*q3
        self.H_accelerometer[2, 8] = q0**2 - q1**2 - q2**2 + q3**2
        self.H_accelerometer[2, 9] = 2*q0*r2 + 2*q1*r1 - 2*q2*r0
        self.H_accelerometer[2, 10] = 2*q0*r1 - 2*q1*r2 + 2*q3*r0
        self.H_accelerometer[2, 11] = -2*q0*r0 - 2*q2*r2 + 2*q3*r1
        self.H_accelerometer[2, 12] = 2*q1*r0 + 2*q2*r1 + 2*q3*r2


        # Kalman update 
        S = self.H_accelerometer @ self.P @ self.H_accelerometer.T + self.R_accelerometer  # Innovation covariance
        K = self.P @ self.H_accelerometer.T @ np.linalg.inv(S)  # Kalman gain

        y = z - m  # Residual

        # Update the state estimate
        self.x = self.x + K @ y 

        # Renormalise the quaternion
        self.x[9:13] = quaternion_normalize(self.x[9:13])

        # Update the Error Covariance Matrix
        self.P = (np.eye(len(self.P)) - K @ self.H_accelerometer) @ self.P

        return None

    
    

    def update_accelerometer_position(self, z):
        """
        Updates the state estimate based on a new accelerometer measurement.
        z: Accelerometer measurement vector
        """

        # Generate predicted measurement m from states
        q = quaternion_conjugate(self.x[9:13]) # get the inverse quaternion from the state vector
        #q = self.x[9:13]
        qw = q[0,0]
        qx = q[1,0]
        qy = q[2,0]
        qz = q[3,0]
        
        g = get_gravity(self.x[2,0])
        m = self.x[6:9] -  np.array([[0] ,[0] ,[g]]) # Gravity measurement should be +ve
        m_quat = np.array([0, m[0,0], m[1,0], m[2,0]])
        m_quat = quaternion_multiply(quaternion_multiply(q, m_quat), quaternion_conjugate(q)) 
        m = m_quat[1:]

        H_sub = np.empty((3,3)) # measurement x states
        # rotation matrix from enu to sensor frame
        H_sub[0, 0] = 1 - 2 * (qy**2 + qz**2)
        H_sub[0, 1] = 2 * (qx * qy - qw * qz)
        H_sub[0, 2] = 2 * (qx * qz + qw * qy)

        H_sub[1, 0] = 2 * (qx * qy + qw * qz)
        H_sub[1, 1] = 1 - 2 * (qx**2 + qz**2)
        H_sub[1, 2] = 2 * (qy * qz - qw * qx)

        H_sub[2, 0] = 2 * (qx * qz - qw * qy)
        H_sub[2, 1] = 2 * (qy * qz + qw * qx)
        H_sub[2, 2] = 1 - 2 * (qx**2 + qy**2)

        self.H_accelerometer = np.zeros((3, 19))
        self.H_accelerometer[:, 6:9] = H_sub

        # Kalman gain calculation for accelerometer
        S = self.H_accelerometer @ self.P @ self.H_accelerometer.T + self.R_accelerometer
        K = self.P @ self.H_accelerometer.T @ np.linalg.inv(S)

        # Update estimate with adjusted accelerometer measurement
        y = z - m # Residual (innovation)
        self.x = self.x + K @ y

        # Update covariance matrix
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H_accelerometer) @ self.P

        return None

    def update_barometer(self, z):
        """
        Updates the state estimate based on a new barometer measurement.
        z: Barometer measurement vector
        """

        # Adjust the barometer reading based on the ground level
        z_adjusted = z - self.launch_coordinates[2]
        self.H_barometer = np.zeros((1,19))
        self.H_barometer[0,2] = 1

        # Kalman gain calculation for barometer
        S = self.H_barometer @ self.P @ self.H_barometer.T + self.R_barometer
        K = self.P @ self.H_barometer.T @ np.linalg.inv(S)

        # Update estimate with adjusted barometer measurement
        y = z_adjusted - self.H_barometer @ self.x  # Residual (innovation)
        self.x = self.x + K @ y

        # Update covariance matrix
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H_barometer) @ self.P


    def update_gps(self, z):
        """
        Updates the state estimate based on a new GPS measurement (latitude, longitude, altitude).
        z: GPS measurement vector [lat, lon, alt]
        """
        # Convert GPS geodetic coordinates to ENU relative to launch position
        # TODO
            # Probably need a measurement Jacobian for this stuff

        current_position = (z[0, 0], z[1, 0], z[2, 0])
        initial_position = (self.launch_coordinates[0], self.launch_coordinates[1], self.launch_coordinates[2])
        
        enu_displacement = lat_lon_alt_to_enu(*initial_position, *current_position)
        
        # Create the measurement vector in ENU
        z_enu = np.array([[enu_displacement[0]], [enu_displacement[1]], [enu_displacement[2]]])
        
        # Define the measurement matrix H for GPS (maps state to measurement space)
        H_gps = np.zeros((3, 19))
        H_gps[0, 0] = 1  # x (East)
        H_gps[1, 1] = 1  # y (North)
        H_gps[2, 2] = 1  # z (Up)
        
        # Calculate the Kalman gain
        S = H_gps @ self.P @ H_gps.T + self.R_gps  # Innovation covariance
        K = self.P @ H_gps.T @ np.linalg.inv(S)    # Kalman gain
        
        # Update the state estimate
        y = z_enu - H_gps @ self.x  # Innovation (residual)
        self.x = self.x + K @ y      # State update
        
        # Update the covariance matrix
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H_gps) @ self.P
    
       
    

    def update_gyroscope(self, z):

        # Measurement Jacobian
        self.H_gyroscope = np.zeros((3,19))
        self.H_gyroscope[0, 13] = 1
        self.H_gyroscope[1,14] = 1
        self.H_gyroscope[2,15] =  1

        # Kalman gain calculation
        S = self.H_gyroscope @ self.P @ self.H_gyroscope.T + self.R_gyroscope
        K = self.P @ self.H_gyroscope.T @ np.linalg.inv(S)

        # Update estimate
        y = z - self.H_gyroscope @ self.x # Residual
        self.x = self.x + K @ y

        # Updaet covariance matrix
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H_gyroscope) @ self.P
        return None


    def update_magnetometer(self, z):
        """
        Updates the state estimate based on a new magnetometer measurement.
        
        Parameters:
        z: Magnetometer measurement vector in the body frame.
        """ 
        # Normalise the sensors measurements
        z = z/(np.sqrt(z[0,0]**2 + z[1,0]**2 + z[2,0]**2))

        # Expected measurement vector in enu frame
        theta_dip_angle = 0
        r = (1/np.sqrt(np.cos(theta_dip_angle)**2 + np.sin(theta_dip_angle)**2)) * np.array([0, np.cos(theta_dip_angle), -np.sin(theta_dip_angle)])

        # Need to convert from the enu frame to the sensor frame

        q = self.x[9:13] # get the quaternion from the state vector

        # We want the conjugate, q represents body -> ENU, we want ENU -> body
        qw = q[0,0]
        qx = q[1,0]
        qy = q[2,0]
        qz = q[3,0]

        r_quat = np.array([0] + list(r))
        # Rotate the reference vector from enu to sensor frame
        rotation_quat = [self.x[9,0], self.x[10,0], self.x[11,0], self.x[12,0]]
        m_quat = quaternion_multiply(quaternion_multiply(quaternion_conjugate(rotation_quat), r_quat), rotation_quat)
        m = m_quat[1:].reshape(3,1) # Remove scalar part

        H_sub = np.empty((3,4)) # measurement x states

        rx = r[0]
        ry = r[1]
        rz = r[2]

        # Taken from, https://ahrs.readthedocs.io/en/latest/filters/ekf.html
        H_sub[0, 0] = rx*qw + ry*qz - rz * qy
        H_sub[0, 1] = rx * qx + ry * qy + rz * qz
        H_sub[0, 2] = -rx * qy + ry * qx - rz * qw
        H_sub[0, 3] = -rx * qz + ry * qw + rz * qx

        H_sub[1, 0] = -rx * qz + ry * qw + rz * qx
        H_sub[1, 1] = rx * qy - ry * qx+ rz * qw
        H_sub[1, 2] = rx * qx + ry * qy + rz * qz
        H_sub[1, 3] =  -rx * qw - ry*qz + rz * qy

        H_sub[2, 0] = rx * qy - ry*qx + rz * qw
        H_sub[2, 1] = rx * qz - ry * qw - rz * qx
        H_sub[2, 2] = rx * qw + ry * qz - rz * qy
        H_sub[2, 3] =  rx * qx + ry * qy + rz * qz

        H_sub = 2 * H_sub
        
        self.H_magnetometer = np.zeros((3, 19))
        self.H_magnetometer[:, 9:13] = H_sub

        # Calculate the Kalman Gain K
        S = self.H_magnetometer @ self.P @ self.H_magnetometer.T + self.R_magnetometer  # Innovation covariance
        K = self.P @ self.H_magnetometer.T @ np.linalg.inv(S)  # Kalman gain

        # Update estimate with adjusted magnetometer measurement
        y = z - m  # Residual

        # Update the state estimate
        self.x = self.x + K @ y 

        # Renormalise the quaternion
        self.x[9:13] = quaternion_normalize(self.x[9:13])

        # Update the Error Covariance Matrix
        self.P = (np.eye(len(self.P)) - K @ self.H_magnetometer) @ self.P

        return None

        

    def get_state_estimate(self):
        """
        Returns the current state estimate.
        """
        return self.x



    

    def get_process_model(self, dt, x):

        # State vector [x, y, z, x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot, q_w, q_x, q_y, q_z, omega_x, omega_y, omega_z, Cb_x, Cb_y, Cb_z]
        # State vector [0, 1, 2, 3,     4,     5,     6,      7,      8,      9,   10,  11,  12,  13,      14,      15,      16,   17,   18]
        
        F = np.eye(19, 19)

        # Position and velocity updates (already present)
        F[0, 3] = dt  # x_dot affects x
        F[1, 4] = dt  # y_dot affects y
        F[2, 5] = dt  # z_dot affects z
        F[3, 6] = dt  # x_ddot affects x_dot
        F[4, 7] = dt  # y_ddot affects y_dot
        F[5, 8] = dt  # z_ddot affects z_dot


        # TODO
            # Need the jacobian elements for the orientation update are not complete!
   
        # Extract quaternion components from state vector x
        q_w = x[9, 0]
        q_x = x[10, 0]
        q_y = x[11, 0]
        q_z = x[12, 0]

        # Extract angular velocities from state vector x
        omega_x = x[13, 0]
        omega_y = x[14, 0]
        omega_z = x[15, 0]

        F[9, 9] = 1  # q_w does not change with respect to q_w directly
        F[9, 10] = -dt/2 * omega_x  # q_w changes with respect to q_x
        F[9, 11] = -dt/2 * omega_y  # q_w changes with respect to q_y
        F[9, 12] = -dt/2 * omega_z  # q_w changes with respect to q_z
        #F[9, 13] = -dt/2 * q_x  # q_w changes with respect to omega_x
        #F[9, 14] = -dt/2 * q_y  # q_w changes with respect to omega_y
        #F[9, 15] = -dt/2 * q_z  # q_w changes with respect to omega_z

        F[10, 9] = dt/2 * omega_x  # q_x changes with respect to q_w
        F[10, 10] = 1  # q_x changes with respect to q_x directly
        F[10, 11] = dt/2 * omega_z  # q_x changes with respect to q_y
        F[10, 12] = -dt/2 * omega_y  # q_x changes with respect to q_z
        #F[10, 13] = dt/2 * q_w  # q_x changes with respect to omega_x
        #F[10, 14] = -dt/2 * q_z  # q_x changes with respect to omega_y
        #F[10, 15] = dt/2 * q_y  # q_x changes with respect to omega_z

        F[11, 9] = dt/2 * omega_y  # q_y changes with respect to q_w
        F[11, 10] = -dt/2 * omega_z  # q_y changes with respect to q_x
        F[11, 11] = 1  # q_y changes with respect to q_y directly
        F[11, 12] = dt/2 * omega_x  # q_y changes with respect to q_z
        #F[11, 13] = dt/2 * q_z  # q_y changes with respect to omega_x
        #F[11, 14] = dt/2 * q_w  # q_y changes with respect to omega_y
        #F[11, 15] = -dt/2 * q_x  # q_y changes with respect to omega_z

        F[12, 9] = dt/2 * omega_z  # q_z changes with respect to q_w
        F[12, 10] = dt/2 * omega_y  # q_z changes with respect to q_x
        F[12, 11] = -dt/2 * omega_x  # q_z changes with respect to q_y
        F[12, 12] = 1  # q_z changes with respect to q_z directly
        #F[12, 13] = -dt/2 * q_y  # q_z changes with respect to omega_x
        #F[12, 14] = dt/2 * q_x  # q_z changes with respect to omega_y
        #F[12, 15] = dt/2 * q_w  # q_z changes with respect to omega_z



        return F





    def get_process_noise(self, dt):
        """
        Get the process noise covariance matrix for the EKF.

        :param dt: Time elapsed since the last update.
        :return: Process noise covariance matrix Q.
        """

        # TODO
            # Tune properly
        process_noise_variances = {
        'sigma_x': 1e-1, 'sigma_y': 1e-1, 'sigma_z': 1e-1,
        'sigma_dot_x': 1e-5, 'sigma_dot_y': 1e-5, 'sigma_dot_z': 1e-5,
        'sigma_ddot_x': 1e-2, 'sigma_ddot_y': 1e-2, 'sigma_ddot_z': 1e-2,
        'sigma_qw': 1e-4, 'sigma_qx': 1e-4, 'sigma_qy': 1e-4, 'sigma_qz': 1e-4,
        'sigma_omega_x': 1e-3, 'sigma_omega_y': 1e-3, 'sigma_omega_z': 1e-3,
        'sigma_Cb_x': 20e-3, 'sigma_Cb_y': 20e-3, 'sigma_Cb_z': 20e-3 
        }



        # Initialize the 16x16 process noise covariance matrix
        Q = np.zeros((19, 19))

        # Position noise (x, y, z)
        Q[0, 0] = process_noise_variances['sigma_x']**2 * dt**4 / 4  # x position noise
        Q[0, 1] = process_noise_variances['sigma_x']**2 * dt**3 / 2  # Cross covariance x position with x velocity
        Q[0, 3] = process_noise_variances['sigma_x']**2 * dt**2 / 2  # Cross covariance x position with x acceleration
        
        Q[1, 1] = process_noise_variances['sigma_y']**2 * dt**4 / 4  # y position noise
        Q[1, 2] = process_noise_variances['sigma_y']**2 * dt**3 / 2  # Cross covariance y position with y velocity
        Q[1, 4] = process_noise_variances['sigma_y']**2 * dt**2 / 2  # Cross covariance y position with y acceleration
        
        Q[2, 2] = process_noise_variances['sigma_z']**2 * dt**4 / 4  # z position noise
        Q[2, 0] = process_noise_variances['sigma_z']**2 * dt**3 / 2  # Cross covariance z position with z velocity
        Q[2, 5] = process_noise_variances['sigma_z']**2 * dt**2 / 2  # Cross covariance z position with z acceleration

        # Velocity noise (\dot{x}, \dot{y}, \dot{z})
        Q[3, 3] = process_noise_variances['sigma_dot_x']**2 * dt**2  # x velocity noise
        Q[3, 0] = process_noise_variances['sigma_dot_x']**2 * dt    # Cross covariance x velocity with x position
        Q[3, 4] = process_noise_variances['sigma_dot_x']**2 * dt      # Cross covariance x velocity with y acceleration
        
        Q[4, 4] = process_noise_variances['sigma_dot_y']**2 * dt**2  # y velocity noise
        Q[4, 1] = process_noise_variances['sigma_dot_y']**2 * dt      # Cross covariance y velocity with y position
        Q[4, 5] = process_noise_variances['sigma_dot_y']**2 * dt      # Cross covariance y velocity with z acceleration
        
        Q[5, 5] = process_noise_variances['sigma_dot_z']**2 * dt**2  # z velocity noise
        Q[5, 2] = process_noise_variances['sigma_dot_z']**2 * dt      # Cross covariance z velocity with z position
        Q[5, 3] = process_noise_variances['sigma_dot_z']**2 * dt      # Cross covariance z velocity with x acceleration

        # Acceleration noise (\ddot{x}, \ddot{y}, \ddot{z})
        Q[6, 6] = process_noise_variances['sigma_ddot_x']**2  # x acceleration noise
        Q[6, 3] = process_noise_variances['sigma_ddot_x']**2 * dt  # Cross covariance x acceleration with x velocity
        Q[6, 4] = process_noise_variances['sigma_ddot_x']**2 * dt  # Cross covariance x acceleration with y acceleration
        Q[6, 5] = process_noise_variances['sigma_ddot_x']**2 * dt  # Cross covariance x acceleration with z acceleration
        
        Q[7, 7] = process_noise_variances['sigma_ddot_y']**2  # y acceleration noise
        Q[7, 3] = process_noise_variances['sigma_ddot_y']**2 * dt  # Cross covariance y acceleration with x velocity
        Q[7, 4] = process_noise_variances['sigma_ddot_y']**2 * dt  # Cross covariance y acceleration with y acceleration
        Q[7, 5] = process_noise_variances['sigma_ddot_y']**2 * dt  # Cross covariance y acceleration with z acceleration
        
        Q[8, 8] = process_noise_variances['sigma_ddot_z']**2  # z acceleration noise
        Q[8, 3] = process_noise_variances['sigma_ddot_z']**2 * dt  # Cross covariance z acceleration with x velocity
        Q[8, 4] = process_noise_variances['sigma_ddot_z']**2 * dt  # Cross covariance z acceleration with y acceleration
        Q[8, 5] = process_noise_variances['sigma_ddot_z']**2 * dt  # Cross covariance z acceleration with z acceleration

        # Quaternion noise (q_w, q_x, q_y, q_z)
        Q[9, 9] = process_noise_variances['sigma_qw']**2  # q_w noise
        Q[9, 10] = process_noise_variances['sigma_qw']**2 * dt  # Cross covariance q_w with q_x
        Q[9, 11] = process_noise_variances['sigma_qw']**2 * dt  # Cross covariance q_w with q_y
        Q[9, 12] = process_noise_variances['sigma_qw']**2 * dt  # Cross covariance q_w with q_z

        Q[10, 10] = process_noise_variances['sigma_qx']**2  # q_x noise
        Q[10, 9] = process_noise_variances['sigma_qx']**2 * dt  # Cross covariance q_x with q_w
        Q[10, 11] = process_noise_variances['sigma_qx']**2 * dt  # Cross covariance q_x with q_y
        Q[10, 12] = process_noise_variances['sigma_qx']**2 * dt  # Cross covariance q_x with q_z

        Q[11, 11] = process_noise_variances['sigma_qy']**2  # q_y noise
        Q[11, 9] = process_noise_variances['sigma_qy']**2 * dt  # Cross covariance q_y with q_w
        Q[11, 10] = process_noise_variances['sigma_qy']**2 * dt  # Cross covariance q_y with q_x
        Q[11, 12] = process_noise_variances['sigma_qy']**2 * dt  # Cross covariance q_y with q_z

        Q[12, 12] = process_noise_variances['sigma_qz']**2  # q_z noise
        Q[12, 9] = process_noise_variances['sigma_qz']**2 * dt  # Cross covariance q_z with q_w
        Q[12, 10] = process_noise_variances['sigma_qz']**2 * dt  # Cross covariance q_z with q_x
        Q[12, 11] = process_noise_variances['sigma_qz']**2 * dt  # Cross covariance q_z with q_y

        # Angular velocity noise (\omega_x, \omega_y, \omega_z)
        Q[13, 13] = process_noise_variances['sigma_omega_x']**2  # omega_x noise
        Q[13, 14] = process_noise_variances['sigma_omega_x']**2 * dt  # Cross covariance omega_x with omega_y
        Q[13, 15] = process_noise_variances['sigma_omega_x']**2 * dt  # Cross covariance omega_x with omega_z

        Q[14, 14] = process_noise_variances['sigma_omega_y']**2  # omega_y noise
        Q[14, 13] = process_noise_variances['sigma_omega_y']**2 * dt  # Cross covariance omega_y with omega_x
        Q[14, 15] = process_noise_variances['sigma_omega_y']**2 * dt  # Cross covariance omega_y with omega_z

        Q[15, 15] = process_noise_variances['sigma_omega_z']**2  # omega_z noise
        Q[15, 13] = process_noise_variances['sigma_omega_z']**2 * dt  # Cross covariance omega_z with omega_x
        Q[15, 14] = process_noise_variances['sigma_omega_z']**2 * dt  # Cross covariance omega_z with omega_y """

        # Ballistic Coefficient noise (Cb_x, Cb_y, Cb_z)
        Q[16, 16] = process_noise_variances['sigma_Cb_x']**2
        
        Q[17, 17] = process_noise_variances['sigma_Cb_y']**2

        Q[18, 18] = process_noise_variances['sigma_Cb_z']**2

        return Q
    
