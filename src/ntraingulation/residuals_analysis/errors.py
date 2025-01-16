"""Model Fit tests for the kalman filter."""

__all__ = ["MSE", "MSSE", "MAD"]

import pandas as pd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt


def MSE(innovation: pd.DataFrame) -> pd.DataFrame:
    """Mean Squared Error (MSE) for the innovation.

    Args:
        innovation (pd.DataFrame): The innovation dataframe.

    Returns:
        pd.DataFrame: The MSE dataframe with cumulative mean of innovation squared.
    """
    # Calculat the cumulative sum of the squared error
    return innovation.apply(np.square).expanding().mean()


def MAD(innovation: pd.DataFrame) -> pd.DataFrame:
    """Mean Absolute Deviation (MAD) for the innovation.

    Args:
        innovation (pd.DataFrame): The innovation dataframe.

    Returns:
        pd.DataFrame: The MAD dataframe with cumulative mean of absolute innovation.
    """
    # Calculate the cumulative sum of the absolute error
    return innovation.apply(np.abs).expanding().mean()


def MSSE(standerdized_innovation: pd.DataFrame) -> pd.DataFrame:
    """Mean of Squared Scaled Error (MSSE) for the standerdized innovation.

    Args:
        standerdized_innovation (pd.DataFrame): The standerdized innovation dataframe.

    Returns:
        pd.DataFrame: The MSSE dataframe with cumulative mean of standerdized innovation squared.
    """
    # Calculate the cumulative sum of the squared error
    return MSE(standerdized_innovation)


def is_positive_definite(matrix: np.ndarray) -> bool:
    """Check if the matrix is positive definite.

    Args:
        matrix (np.ndarray): The matrix.

    Returns:
        bool: True if the matrix is positive definite, False otherwise.
    """
    return np.all(np.linalg.eigvals(matrix) > 0)


def sqrtm(matrix: np.ndarray) -> np.ndarray:
    """Calculate the square root of a matrix based on the spectral decomposition.

    Args:
        matrix (np.ndarray): The matrix.

    Returns:
        np.ndarray: The square root of the matrix.
    """
    # Do the UDV decomposition
    U, D, V = np.linalg.svd(matrix)
    # Calculate the square root of the diagonal matrix
    D_sqrt = np.diag(np.sqrt(D))
    # Calculate the square root of the matrix
    return U @ D_sqrt @ V


def standerized_residuals(
    innovations: np.ndarray, covariances: np.ndarray
) -> np.ndarray:
    """Standerdize the innovations.

    Args:
        innovations (np.ndarray): The innovations. (T, dim_z)
        covariances (np.ndarray): The covariances. (T, dim_z, dim_z)

    Returns:
        np.ndarray: The standerdized innovations. (T, dim_z)
    """
    # Check that all the covariance matrices are positive definite
    if not np.all([is_positive_definite(cov) for cov in covariances]):
        raise ValueError("All covariance matrices must be positive definite.")

    # Calculate the spectral squareroot of the covariance matrix
    inv_sqrt_cov = np.real(np.stack([np.linalg.inv(sqrtm(cov)) for cov in covariances]))

    # Calculate the standerdized innovations
    return np.einsum("tij,tj->ti", inv_sqrt_cov, innovations)


def autocorrelation(
    innovation: pd.Series, confidence_interval: float = 0.95
) -> plt.Figure:
    """Autocorrelation plot for the innovation.

    Args:
        innovation (pd.Series): The innovation.
        confidence_interval (float, optional): The confidence interval. Defaults to 0.95.
    """
    return plot_acf(innovation, alpha=1 - confidence_interval)
