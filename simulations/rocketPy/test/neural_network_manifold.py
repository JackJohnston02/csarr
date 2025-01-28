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

class NeuralNetworkManifold:
    def __init__(self):
        self.target_manifold = self.generate_target_manifold(2600, 300)
        self.x1_min, self.x1_max = 0, 5000
        self.x2_min, self.x2_max = 0, 5000
        self.epsilon = 1e-6
        self.reference_apogee = 2600
        self.upper_bound = 5000
        self.lower_bound = 100



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
            crop = 100
            for Cb in Cb_inputs:
                i += 1
                print(f"[INFO] Generating target manifold {i}/{len(Cb_inputs)}", end="\r")
                altitudes, velocities, ballistic_coefficients = simulate(reference_apogee, velocity_bound, Cb)
                
                # Crop the initial part of the trajectory, removing the steep region of the manifold
                altitudes = altitudes[crop:]
                velocities = velocities[crop:]
                ballistic_coefficients = ballistic_coefficients[crop:]

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
    

    def evaluate_sliding_surface_discrete(self, x0):
        """
        Estimate the Cb value that results in the desired apogee using the Secant Method.

        :param x0: Initial state vector (e.g., [altitude, velocity]).
        :return: Converged estimate of Cb.
        """
        dt = 0.1  # Time step for simulation
        max_iter = 10  # Maximum iterations for convergence
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

            Cb_0, Cb_1 = Cb_1, Cb_new

        return Cb_1


neural_network = NeuralNetworkManifold()


velocities = np.linspace(0, 300, 100)
altitudes = np.linspace(0, 2600, 100)

# Sample from the neural network over all the points in the grid, the inputs are the velocities and altitudes, the output is the ballistic coefficient
Cb = np.zeros((100, 100))
for i, v in enumerate(velocities):
    for j, h in enumerate(altitudes):
        Cb[j, i] = neural_network.evaluate_sliding_surface_neural_network(np.array([[h], [v]]))

# Plot the surface that is the evaluated neural network
plt.figure()
plt.contourf(velocities, altitudes, Cb, 100, cmap='jet')
plt.colorbar()
plt.xlabel("Velocity [m/s]")
plt.ylabel("Altitude [m]")
plt.title("Ballistic Coefficient")
plt.show()

# For the same grid of points lets evaluate the nonlinearity of the sliding surface, using evaluate_sliding_surface with an indicatino of the progress
Cb_discrete = np.zeros((100, 100))
for i, v in enumerate(velocities):
    for j, h in enumerate(altitudes):
        # Indicate the pgrogress of the evaluation
        print(f"[INFO] Evaluating point {i}/{len(velocities)}", end="\r")
        Cb_discrete[j, i] = neural_network.evaluate_sliding_surface_discrete(np.array([[h], [v]]))

# Plot the surface that is the evaluated neural network
plt.figure()
plt.contourf(velocities, altitudes, Cb_discrete, 100, cmap='jet')
plt.colorbar()
plt.xlabel("Velocity [m/s]")
plt.ylabel("Altitude [m]")
plt.title("Ballistic Coefficient")
plt.show()

# Calculate the difference between the two surfaces
diff = np.abs(Cb - Cb_discrete)
# PLot the difference
plt.figure()
plt.contourf(velocities, altitudes, diff, 100, cmap='jet')
plt.colorbar()
plt.xlabel("Velocity [m/s]")
plt.ylabel("Altitude [m]")
plt.title("Difference")
plt.show()
