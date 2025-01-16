"""This module contains the function to convert geocentric coordinates to ellipsoidal coordinates."""

import numpy as np  # type: ignore


# WGS DATUM CONSTANTS
A = 6378137.0  # semi-major axis in meters
F = 1 / 298.257223563  # flattening


def geocentric_to_ellipsoidal(
    x: np.float64, y: np.float64, z: np.float64, max_iter: np.int32 = 1000
) -> np.ndarray:
    """Convert geocentric coordinates to ellipsoidal coordinates.

    Args:
        x (np.float64): x-coordinate in meters
        y (np.float64): y-coordinate in meters
        z (np.float64): z-coordinate in meters
        max_iter (int): Maximum number of iterations. Defaults to 100.

    Returns:
        np.ndarray: Ellipsoidal coordinates
    """
    # WGS84 ellipsoid parameters
    e = np.sqrt(2 * F - F**2)

    # Calculate longitude and latitude
    lon = np.arctan2(y, x)
    p = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, (1 - e**2) * p)

    # Iteratively refine latitude using the iterative method
    tolerance = 1e-12
    counter = 0
    while counter < max_iter:
        previous_lat = lat
        N = A / np.sqrt(1 - e**2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - N
        lat = np.arctan2(z, p * (1 - e**2 * N / (N + h)))

        # Break the loop if the change in latitude is small enough
        if np.abs(lat - previous_lat) < tolerance:
            break
        # Increment the counter
        counter += 1

    # Convert latitude and longitude from radians to degrees
    lat = np.rad2deg(lat)
    lon = np.rad2deg(lon)

    # Normalize longitude to be between -180 and 180
    lon = (lon + 180) % 360 - 180

    # Normalize latitude to be between -90 and 90
    lat = (lat + 90) % 180 - 90

    return np.array([lat, lon, h])


def ellipsoidal_to_geocentric(
    lat: np.float64, lon: np.float64, alt: np.float64
) -> np.ndarray:
    """Convert geocentric coordinates to ellipsoidal coordinates.

    Args:
        lat (np.float64): Latitude in degrees
        lon (np.float64): Longitude in degrees
        alt (np.float64): Altitude in meters

    Returns:
        np.ndarray: Ellipsoidal coordinates
    """
    # WG84 ellipsoid parameters
    e2 = 2 * F - F**2
    # Convert degrees to radians
    lat = np.radians(lat)
    lon = np.radians(lon)
    N = A / np.sqrt(1 - e2 * np.sin(lat) ** 2)
    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e2) + alt) * np.sin(lat)
    return np.array([x, y, z])


def ellipsoidal_to_enu(
    lat: float, long: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert ellipsoidal coordinates to ENU coordinates.

    Args:
        lat (float): The latitude in ellipsoidal coordinates. Units: degrees
        long (float): The longitude in ellipsoidal coordinates. Units: degrees

    Returns:
        tuple: Unit vectors for the east, north, and up directions.
    """
    # Convert the latitude and longitude to radians
    lat_rad = np.deg2rad(lat)
    long_rad = np.deg2rad(long)

    e_hat = np.array([-np.sin(long_rad), np.cos(long_rad), 0])
    n_hat = np.array(
        [
            -np.sin(lat_rad) * np.cos(long_rad),
            -np.sin(lat_rad) * np.sin(long_rad),
            np.cos(lat_rad),
        ]
    )
    u_hat = np.array(
        [
            np.cos(lat_rad) * np.cos(long_rad),
            np.cos(lat_rad) * np.sin(long_rad),
            np.sin(lat_rad),
        ]
    )

    return e_hat, n_hat, u_hat


def geocentric_to_enu(
    x: float, y: float, z: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert geocentric coordinates to ENU coordinates.

    Args:
        x (float): The x-coordinate in geocentric coordinates.
        y (float): The y-coordinate in geocentric coordinates.
        z (float): The z-coordinate in geocentric coordinates.

    Returns:
        tuple: Unit vectors for the east, north, and up directions.
    """
    lat, long, _ = geocentric_to_ellipsoidal(x, y, z, max_iter=1000)
    return ellipsoidal_to_enu(lat, long)
