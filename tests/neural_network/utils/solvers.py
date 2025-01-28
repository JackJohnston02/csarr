import numpy as np

def rk4(dynamics, x, dt):
    """
    Perform a single step of the Runge-Kutta 4th order integration.

    :param dynamics: The dynamics function (e.g., ballistic_flight_model).
    :param x: The current state vector.
    :param dt: The timestep.
    :return: The updated state vector.
    """
    # Compute the intermediate slopes k1, k2, k3, k4
    k1 = dt * dynamics(x)
    k2 = dt * dynamics(x + 0.5 * k1)
    k3 = dt * dynamics(x + 0.5 * k2)
    k4 = dt * dynamics(x + k3)
    
    # Compute the next state
    x_next = x + (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return x_next
