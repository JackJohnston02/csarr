from utils.environmental_utils import get_air_density, get_gravity
import numpy as np

def ballistic_flight_model(x):
    # Get gravitational acceleration and air density at current altitude (x[0] is altitude)
    g = get_gravity(x[0])
    rho = get_air_density(x[0])

    dx = np.zeros_like(x) 
    
    dx[0] = x[1]
    dx[1] = g - (rho * x[1]**2) / (2 * x[2])
    dx[2] = 0
    
    return dx
