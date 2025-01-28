import numpy as np

def get_gravity(altitude_ASL):
    """
    Returns gravitational acceleration as a function of altitude above sea level (ASL).
    This function approximates Earth's gravity assuming a perfect sphere.

    Parameters:
    altitude_ASL (float): Altitude above sea level in meters.

    Returns:
    float: Gravitational acceleration at the given altitude in m/s².
    """

    g_0 = -9.80665  # Standard gravity at sea level (m/s²)
    R_e = 6371000  # Earth's radius in meters

    # Gravity decreases with altitude according to the inverse-square law
    g = g_0 * (R_e / (R_e + altitude_ASL)) ** 2
    
    return g


def get_air_density(altitude_ASL):
    """
    Returns air density as a function of altitude above sea level (ASL).
    This is valid up to an altitude of 11 kilometers, and it uses the barometric formula
    assuming a constant temperature lapse rate in the troposphere.

    Parameters:
    altitude_ASL (float): Altitude above sea level in meters.

    Returns:
    float: Air density at the given altitude in kg/m³.
    """

    p_0 = 101325  # Standard sea level atmospheric pressure in Pascals
    M = 0.0289652  # Molar mass of dry air in kg/mol
    R = 8.31445  # Universal gas constant in J/(mol·K)
    T_0 = 288.15  # Standard sea level temperature in Kelvin
    L = 0.0065  # Temperature lapse rate in K/m

    # Get gravity at the specified altitude
    g = get_gravity(altitude_ASL)

    # Calculate the air density using the barometric formula
    rho = (p_0 * M) / (R * T_0) * (1 - (L * altitude_ASL) / T_0) ** (((-g * M) / (R * L)) - 1)

    return rho
