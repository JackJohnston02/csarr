import numpy as np
from utils.environmental_utils import get_air_density, get_gravity
from utils.dynamic_models import ballistic_flight_model
from utils.solvers import rk4

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

class SlidingModeController:
    def __init__(self, reference_apogee: float, sampling_rate: float, mode: str = "None", neural_network_flag: bool = False):
        """
        Unified Sliding Mode Controller with configurable behavior.

        :param reference_apogee: Target/reference value (desired apogee).
        :param sampling_rate: Rate at which the controller runs (Hz).
        :param mode: Behavior of the controller ("no_anti_chattering", "deadband", "soft_switching", "super_twisting").
        """
        print(f"[INFO] Target apogee is {reference_apogee}m\n       Sampling Rate: {sampling_rate} Hz\n       Mode: {mode}\n       Neural Network Sliding Surface: {neural_network_flag}")
        if mode == "None":
            raise ValueError("[ERROR] Controller mode not specified. Please select one of the following modes: 'no_anti_chattering', 'deadband', 'soft_switching', 'super_twisting'.")


        self.reference_apogee = reference_apogee
        self.previous_time = None
        self.sampling_interval = 1 / sampling_rate if sampling_rate > 0 else None
        self.mode = mode
        
        # Flag to enable neural network approximation of the sliding surface
        self.neural_network_flag = neural_network_flag 

        if self.neural_network_flag:
            self.target_manifold = self.generate_target_manifold(reference_apogee, 300)
        else:
            self.target_manifold = None


        # Controller state variables
        self.error_signal = 0
        self.control_output = 0
        self.previous_error = 0
        self.integral_error = 0

        self.sigma = 2000
        self.upper_bound = 4000
        self.lower_bound = 0
        self.epsilon = 10e-6

        # No anti-chattering variables
        self.K_no_anti_chattering = 0.4 # At 10Hz 0.1 is good

        # Deadband variables
        self.K_deadband = 0.4 # Just set to the same as no_anti_chattering
        self.deadband_width = 100 # Really whatever you want, 100 works pretty well

        # Soft-switching variables
        self.K_soft_switching = 0.4 # Same as the other two
        self.soft_switching_width = 100 # Really whatever just soft clips the error signal

        # Super-twisting variables
        self.k_1 = 0.0125 # At 10Hz 0.005 is good, think about it in terms of proportional gain
        self.k_2 = 0.008 # At 10Hz 0.005 is good, think about it in terms of integral gain
        self.v = 0 # Integral term for super-twisting, should be 0

       

    def get_control_signal(self, estimated_state, current_time):
        """
        Calculate the control signal based on the current state.

        :param estimated_state: The current system state (e.g., current apogee).
        :param current_time: The current time (for calculating the derivative term).
        :return: Control signal to apply.
        """
        if self.sampling_interval is None:
            return self.control_output

        if self.previous_time is None or (current_time - self.previous_time >= self.sampling_interval):
            self.previous_time = current_time

            # State estimate vector, [altitude, velocity]^T
            x_hat = np.array([[estimated_state[2, 0]], [estimated_state[5, 0]]])

            # Evaluate the sliding surface at the current state estimates
            if self.neural_network_flag:
                self.sigma = self.evaluate_sliding_surface_neural_network(x_hat)
            else:
                self.sigma = self.evaluate_sliding_surface_discrete(x_hat)

            Cb_hat = estimated_state[18, 0]

            # Calculate the error signal
            # If the error is within 200m of the reference apogee, set the error signal to zero
            if x_hat[1] < 50:
                self.error_signal = 0
                self.v = 0
            else:    
                self.error_signal = self.sigma - Cb_hat

            # Apply control law based on the mode
            if self.mode == "no_anti_chattering":
                self.control_output = -self.K_no_anti_chattering * np.sign(self.error_signal)

            elif self.mode == "deadband":
                self.control_output = -self.K_deadband * np.sign(self.error_signal)
                if abs(self.error_signal) < self.deadband_width:
                    self.control_output = 0

            elif self.mode == "soft_switching":
                self.control_output = -self.K_soft_switching * np.tanh(self.error_signal / self.soft_switching_width)

            elif self.mode == "super_twisting":
                v_dot = -self.k_2 * np.sign(self.error_signal)
                self.v += v_dot * self.sampling_interval
                self.control_output = -self.k_1 * np.sqrt(np.abs(self.error_signal)) * np.sign(self.error_signal) + self.v

            self.previous_error = self.error_signal

        
        if self.control_output > 0.4:
            self.control_output = 0.4
        elif self.control_output < -0.4:
            self.control_output = -0.4


        return self.control_output



    def evaluate_sliding_surface_discrete(self, x0):
        """
        Estimate the Cb value that results in the desired apogee using the Secant Method.

        :param x0: Initial state vector (e.g., [altitude, velocity]).
        :return: Converged estimate of Cb.
        """
        dt = 0.05  # Time step for simulation
        max_iter = 20  # Maximum iterations for convergence
        position_tolerance = 0.5  # Tolerance level for apogee error

        self.upper_bound += 50
        self.lower_bound -= 50
        epsilon = self.epsilon if self.epsilon > 0 else 1e-6  # Avoid zero division

        # Initial guesses for Cb
        Cb_0 = (self.upper_bound + self.lower_bound) / 2 + epsilon
        Cb_1 = Cb_0 + 100
 
        def compute_apogee_error(Cb):
            # Initialize state variables
            x = np.zeros((3, 1))
            x[0, 0] = x0[0]  # Initial altitude
            x[1, 0] = x0[1]  # Initial velocity
            x[2, 0] = Cb  # Set current Cb estimate

            # Simulate until velocity is zero or less (apogee reached)
            while x[1, 0] > 0:
                x = rk4(ballistic_flight_model, x, dt)

            return self.reference_apogee - x[0, 0]

        for iteration in range(max_iter):
            error_0 = compute_apogee_error(Cb_0)
            error_1 = compute_apogee_error(Cb_1)

            if abs(error_1) < position_tolerance:
                return Cb_1

            Cb_new = Cb_1 - error_1 * (Cb_1 - Cb_0) / (error_1 - error_0)

            if Cb_new > 10000:
                print("[WARNING] Cb exceeded upper practical limit. Returning maximum allowed value.")
                return 10000
            if Cb_new < 100:
                print("[WARNING] Cb fell below lower practical limit. Returning minimum allowed value.")
                return 100

            Cb_0, Cb_1 = Cb_1, Cb_new

        return Cb_1
    
    def evaluate_sliding_surface_neural_network(self, x_hat):
        model = self.target_manifold
        x1 = float(x_hat[1, 0])
        x2 = float(x_hat[0, 0])

        x1_norm = (x1 - self.x1_min) / (self.x1_max - self.x1_min)
        x2_norm = (x2 - self.x2_min) / (self.x2_max - self.x2_min)

        x_point = torch.tensor([[x1_norm, x2_norm]], dtype=torch.float32)
        model.eval()
        with torch.no_grad():
            y_pred = model(x_point).detach().numpy()

        return y_pred[0][0]
    


    def generate_target_manifold(self, reference_apogee, velocity_bound):
        # Check if the model already exists for the given reference_apogee
        model_filename = f"data/target_manifolds/target_manifold_{reference_apogee:.2f}.pth"
        
        # Neural Network Definition
        class ImprovedNeuralNet(nn.Module):
            def __init__(self):
                super(ImprovedNeuralNet, self).__init__()
                self.fc1 = nn.Linear(2, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, 32)
                self.fc4 = nn.Linear(32, 1)

                self.batch_norm1 = nn.BatchNorm1d(128)
                self.batch_norm2 = nn.BatchNorm1d(64)
                self.batch_norm3 = nn.BatchNorm1d(32)
                self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)

            def forward(self, x):
                x = self.leaky_relu(self.batch_norm1(self.fc1(x)))
                x = self.leaky_relu(self.batch_norm2(self.fc2(x)))
                x = self.leaky_relu(self.batch_norm3(self.fc3(x)))
                return self.fc4(x)

        # If the model exists, load it and normalization parameters
        if os.path.exists(model_filename):
            torch.serialization.add_safe_globals([np.core.multiarray.scalar])
            print(f"[INFO] Model for reference_apogee {reference_apogee} already exists. Loading the model.")
            checkpoint = torch.load(model_filename, weights_only=False)
            model = ImprovedNeuralNet()
            
            # Check if the key 'model_state_dict' exists in the checkpoint
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                model.eval()

                # Load the normalization parameters
                self.x1_min = checkpoint['x1_min']
                self.x1_max = checkpoint['x1_max']
                self.x2_min = checkpoint['x2_min']
                self.x2_max = checkpoint['x2_max']
            else:
                raise KeyError("[ERROR] The checkpoint does not contain 'model_state_dict'")

            return model

        # Otherwise, we proceed with simulation and training
        def simulate(reference_apogee, velocity_bound, ballistic_coefficient):
            dt = -0.1
            altitude = reference_apogee
            velocity = 0

            x = np.zeros((3, 1))
            x[0, 0] = altitude
            x[1, 0] = velocity
            x[2, 0] = ballistic_coefficient

            altitudes, velocities, ballistic_coefficients = [], [], []

            while x[1, 0] < velocity_bound and x[0, 0] > 0:
                x = rk4(ballistic_flight_model, x, dt)
                altitudes.append(x[0, 0])
                velocities.append(x[1, 0])
                ballistic_coefficients.append(x[2, 0])

            return altitudes, velocities, ballistic_coefficients

        all_altitudes, all_velocities, all_ballistic_coefficients = [], [], []
        Cb_inputs = np.logspace(np.log10(1000), np.log10(5000), 20)

        i = 0
        for Cb in Cb_inputs:
            i += 1
            print(f"[INFO] Generating target manifold {i}/{len(Cb_inputs)}", end="\r")
            altitudes, velocities, ballistic_coefficients = simulate(reference_apogee, velocity_bound, Cb)
            
            # Get the indexes whee the velocity is greater than 25
            idx = np.where(np.array(velocities) < 25)[0]

            # Get the max of these indexes
            max_idx = np.max(idx)


            # Crop the initial part of the trajectory, removing the steep region of the manifold
            altitudes = altitudes[max_idx:]
            velocities = velocities[max_idx:]
            ballistic_coefficients = ballistic_coefficients[max_idx:]

            all_altitudes.extend(altitudes)
            all_velocities.extend(velocities)
            all_ballistic_coefficients.extend(ballistic_coefficients)
            plt.plot(velocities, altitudes)

        plt.show()


        # Normalize data
        x1 = np.array(all_velocities)
        x2 = np.array(all_altitudes)
        y = np.array(all_ballistic_coefficients)

        self.x1_min, self.x1_max = np.min(x1), np.max(x1)
        self.x2_min, self.x2_max = np.min(x2), np.max(x2)

        x1_norm = (x1 - self.x1_min) / (self.x1_max - self.x1_min)
        x2_norm = (x2 - self.x2_min) / (self.x2_max - self.x2_min)

        X = np.stack([x1_norm, x2_norm], axis=1)
        y = y.reshape(-1, 1)

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

       

        # Custom Weighted MSE Loss
        class WeightedMSELoss(nn.Module):
            def __init__(self, weights):
                super(WeightedMSELoss, self).__init__()
                self.weights = weights

            def forward(self, outputs, targets):
                loss = (outputs - targets) ** 2
                return (self.weights * loss.squeeze()).mean()

        # KDE for Weight Calculation
        kde = KernelDensity(kernel='gaussian', bandwidth=1).fit(X)
        log_density = kde.score_samples(X)
        density = np.exp(log_density)
        weights = 1 / density
        weights /= np.sum(weights)

        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(X_tensor.device)

        # Initialize and train the model
        model = ImprovedNeuralNet()
        criterion = WeightedMSELoss(weights_tensor)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        epochs = 15000

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 1000 == 0:
                print(f"[INFO] Training Neural Network Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}", end="\r")

        # Save the trained model
        # Save the model and normalization parameters
        torch.save({
            'model_state_dict': model.state_dict(),
            'x1_min': self.x1_min,
            'x1_max': self.x1_max,
            'x2_min': self.x2_min,
            'x2_max': self.x2_max
        }, model_filename)


        print(f"[INFO] Model for reference_apogee {reference_apogee} saved as {model_filename}.")

        return model

