from .coordinate_transforms import geocentric_to_enu
import numpy as np

__all__ = ["project_in_enu", "enu_position_errors", "enu_velocity_error"]


def project_in_enu(error: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Project the error vector onto the unit vectors of the ENU coordinate system.

    Args:
        error: The error vector from the actual to predicted location. (3)
        actual: The actual location in geocentric coordinates.  (3)

    Returns:
        The error vector projected onto the unit vectors of the ENU coordinate system.
    """
    # Calculate the unit vectors e, n, u at the actual location
    e_hat, n_hat, u_hat = geocentric_to_enu(*actual)

    # Project the error vector onto the unit vectors
    E_error = np.dot(error, e_hat)
    N_error = np.dot(error, n_hat)
    U_error = np.dot(error, u_hat)

    return np.array([E_error, N_error, U_error])


def enu_position_errors(predicted: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Calculate the ENU position errors.

    Args:
        predicted: The predicted location in geocentric coordinates. (N, 3)
        actual: The actual location in geocentric coordinates.  (N, 3)

    Returns:
        np.ndarray: The ENU position errors.
    """
    # Calculate the error vector from the actual to predicted location
    error = predicted - actual

    # Calculate the ENU position errors
    return np.vstack([project_in_enu(e, a) for e, a in zip(error, actual)])


def enu_velocity_error(
    predicted_velocity: np.ndarray,
    actual_velocity: np.ndarray,
    actual_position: np.ndarray,
) -> np.ndarray:
    """Calculate the ENU velocity errors.

    Args:
        predicted_velocity: The predicted velocity in geocentric coordinates. (N, 3)
        actual_velocity: The actual velocity in geocentric coordinates.  (N, 3)
        actual_position: The actual location in geocentric coordinates.  (N, 3)

    Returns:
        np.ndarray: The ENU velocity errors.
    """
    # Calculate the error vector from the actual to predicted velocity
    error = predicted_velocity - actual_velocity

    # Calculate the ENU velocity errors
    return np.vstack([project_in_enu(e, a) for e, a in zip(error, actual_position)])
