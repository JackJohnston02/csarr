import numpy as np

# Constants for the WGS84 ellipsoid
a = 6378137.0  # Semi-major axis (meters)
f = 1 / 298.257223563  # Flattening
b = a * (1 - f)  # Semi-minor axis (meters)
e2 = 1 - (b**2 / a**2)  # Square of eccentricity

def geodetic_to_ecef(lat, lon, alt):
    # Constants for WGS-84
    a = 6378137.0  # semi-major axis in meters
    e_sq = 6.69437999014e-3  # square of Earth's eccentricity

    # Convert latitude and longitude to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Calculate prime vertical radius of curvature
    N = a / np.sqrt(1 - e_sq * np.sin(lat_rad) ** 2)

    # Calculate ECEF coordinates
    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = ((1 - e_sq) * N + alt) * np.sin(lat_rad)

    return np.array([x, y, z])


def enu_to_ecef(d_enu, ref_lat, ref_lon):
    # Convert reference latitude and longitude to radians
    lat_rad = np.radians(ref_lat)
    lon_rad = np.radians(ref_lon)

    # Create the rotation matrix for ENU to ECEF
    R = np.array([
        [-np.sin(lon_rad), np.cos(lon_rad), 0],
        [-np.sin(lat_rad) * np.cos(lon_rad), -np.sin(lat_rad) * np.sin(lon_rad), np.cos(lat_rad)],
        [np.cos(lat_rad) * np.cos(lon_rad), np.cos(lat_rad) * np.sin(lon_rad), np.sin(lat_rad)]
    ])

    # Convert ENU displacement to ECEF displacement
    d_ecef = R.T @ d_enu  # Transpose to rotate from ENU to ECEF

    return d_ecef


def ecef_to_geodetic(x, y, z):
    # Constants for WGS-84
    a = 6378137.0  # semi-major axis
    e_sq = 6.69437999014e-3  # square of Earth's eccentricity
    b = np.sqrt(a**2 * (1 - e_sq))  # semi-minor axis

    # Calculate longitude
    lon = np.arctan2(y, x)

    # Iterative process to calculate latitude and altitude
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - e_sq))  # initial approximation of latitude
    alt = 0  # initial altitude

    # Iterate to improve latitude and altitude estimates
    for _ in range(5):  # Typically converges quickly in a few iterations
        N = a / np.sqrt(1 - e_sq * np.sin(lat) ** 2)
        alt = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e_sq * N / (N + alt)))

    # Convert results to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)

    return np.array([lat, lon, alt])



def lat_lon_alt_to_enu(lat_ref, lon_ref, alt_ref, lat, lon, alt):
    # Constants
    a = 6378137.0  # WGS-84 Earth semimajor axis (meters)
    f = 1 / 298.257223563  # WGS-84 flattening factor
    e_sq = f * (2 - f)  # Square of eccentricity
    
    # Convert reference and current positions to radians
    lat_ref_rad = np.deg2rad(lat_ref)
    lon_ref_rad = np.deg2rad(lon_ref)
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    
    # Convert geodetic coordinates to ECEF for reference position
    N_ref = a / np.sqrt(1 - e_sq * np.sin(lat_ref_rad)**2)
    x_ref = (N_ref + alt_ref) * np.cos(lat_ref_rad) * np.cos(lon_ref_rad)
    y_ref = (N_ref + alt_ref) * np.cos(lat_ref_rad) * np.sin(lon_ref_rad)
    z_ref = (N_ref * (1 - e_sq) + alt_ref) * np.sin(lat_ref_rad)
    
    # Convert geodetic coordinates to ECEF for current position
    N = a / np.sqrt(1 - e_sq * np.sin(lat_rad)**2)
    x = (N + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (N + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (N * (1 - e_sq) + alt) * np.sin(lat_rad)
    
    # Calculate the difference between the reference and current ECEF coordinates
    dx = x - x_ref
    dy = y - y_ref
    dz = z - z_ref
    
    # Rotation matrix from ECEF to ENU
    sin_lat_ref = np.sin(lat_ref_rad)
    cos_lat_ref = np.cos(lat_ref_rad)
    sin_lon_ref = np.sin(lon_ref_rad)
    cos_lon_ref = np.cos(lon_ref_rad)
    
    R = np.array([
        [-sin_lon_ref, cos_lon_ref, 0],
        [-cos_lon_ref * sin_lat_ref, -sin_lon_ref * sin_lat_ref, cos_lat_ref],
        [cos_lon_ref * cos_lat_ref, sin_lon_ref * cos_lat_ref, sin_lat_ref]
    ])
    
    # Compute ENU coordinates
    enu = R @ np.array([dx, dy, dz])
    
    return enu[0], enu[1], enu[2]  # East, North, Up




