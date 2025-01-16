"""Multiple Error Models for Modelling Error in range measurements."""

import scipy.stats as stats
import numpy as np

__all__ = [
    "multivariate_normal_error",
    "multivariate_student_t_error",
    "multivariate_exponential_error",
    "multivariate_laplace_error",
]


def multivariate_normal_error(
    mean: np.ndarray, cov: np.ndarray, n: int = 1
) -> np.ndarray:
    """Generate multivariate normal error.

    Parameters
    ----------
    mean : np.ndarray
        Mean of the error distribution.
    cov : np.ndarray
        Covariance matrix of the error distribution.
    n : int
        Number of samples to generate. Default is 1.

    Returns
    -------
    np.ndarray
        Error samples.
    """
    return np.random.multivariate_normal(mean, cov, n)


def multivariate_student_t_error(
    mean: np.ndarray, cov: np.ndarray, n: int = 1, df: float = 1
) -> np.ndarray:
    """Generate multivariate student-t error.

    Parameters
    ----------
    mean : np.ndarray
        Mean of the error distribution.
    cov : np.ndarray
        Covariance matrix of the error distribution.
    n : int
        Number of samples to generate. Default is 1.
    df : float
        Degrees of freedom. Default is 1.

    Returns
    -------
    np.ndarray
        Error samples.
    """
    return stats.multivariate_t.rvs(loc=mean, shape=cov, df=df, size=n)


def multivariate_exponential_error(scale: np.ndarray, n: int = 1) -> np.ndarray:
    """Generate multivariate exponential error.

    Parameters
    ----------

    scale : np.ndarray
        Scale of the error distribution.

    n : int
        Number of samples to generate. Default is 1.

    Returns
    -------
    np.ndarray
        Error samples.
    """

    return np.array([np.random.exponential(scale[i], n) for i in range(len(scale))]).T


def multivariate_laplace_error(
    mean: np.ndarray, scale: np.ndarray, n: int = 1
) -> np.ndarray:
    """Generate multivariate laplace error.

    Parameters
    ----------
    mean : np.ndarray
        Mean of the error distribution.
    cov : np.ndarray
        Covariance matrix of the error distribution.
    n : int
        Number of samples to generate. Default is 1.

    Returns
    -------
    np.ndarray
        Error samples.
    """
    return np.array(
        [np.random.laplace(mean[i], scale[i], n) for i in range(len(mean))]
    ).T
