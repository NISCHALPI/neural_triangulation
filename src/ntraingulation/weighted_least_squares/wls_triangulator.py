"""Weighted Least Squares algorithms for triangulation problems."""

import numpy as np

from .wls import non_linear_weighted_least_squares

__all__ = ["wls_triangulation"]


def f(x: np.ndarray, sv_pos: np.ndarray) -> np.ndarray:
    """Computes the non-linear observation model for the triangulation problem.

    Args:
        x (np.ndarray): The estimated receiver position and clock bias (cdt) of shape (4,)
        sv_pos (np.ndarray): The satellite positions of shape (num_svs, 3).

    Returns:
        np.ndarray: The non-linear observation model of shape (num_svs, 1).
    """
    return np.linalg.norm(x[:3] - sv_pos, axis=1) + x[3]


def HJacobian_at(x: np.ndarray, sv_pos: np.ndarray) -> np.ndarray:
    """Computes the Jacobian of the non-linear observation model for the triangulation problem.

    Args:
        x (np.ndarray): The estimated receiver position and clock bias (cdt) of shape (4,)
        sv_pos (np.ndarray): The satellite positions of shape (num_svs, 3).

    Returns:
        np.ndarray: The Jacobian of the non-linear observation model of shape (num_svs, 4).
    """
    los = (x[:3] - sv_pos) / np.linalg.norm(x[:3] - sv_pos, axis=1)[:, None]

    return np.hstack((los, np.ones((los.shape[0], 1))))  # (num_svs, 4)


def wls_triangulation(
    pseudorange: np.ndarray,
    sv_pos: np.ndarray,
    W: np.ndarray | None = None,
    x0: np.ndarray = np.zeros(4),
    max_iter: int = 10000,
    eps: float = 1e-7,
) -> dict[str, np.ndarray | float]:
    """Estimates the receiver position and clock bias using the non-linear weighted least squares method given noisy pseudorange measurements and satellite positions.

    Args:
        pseudorange (np.ndarray): The pseudorange measurements of shape (num_svs).
        sv_pos (np.ndarray): The satellite positions of shape (num_svs, 3).
        x0 (np.ndarray): The initial guess for the receiver position and clock bias of shape (4,).
        W (np.ndarray, optional): The weight matrix of shape (num_svs, num_svs). Defaults to None.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        eps (float, optional): The convergence threshold. Defaults to 1e-5.

    Returns:
        dict[str, np.ndarray | float]: The estimated receiver position and clock bias, the error covariance matrix of the residuals, and the DOPs values.
    """
    # Check if the weight matrix is provided
    if W is None:
        W = np.eye(len(pseudorange), dtype=x0.dtype)

    sol, covar = non_linear_weighted_least_squares(
        y=pseudorange,
        f=f,
        HJacobian=HJacobian_at,
        W=W,
        x0=x0,
        eps=eps,
        max_iter=max_iter,
        f_args=(sv_pos,),
        HJacobian_args=(sv_pos,),
    )
    # Calculate the DOPS for satellites
    H = HJacobian_at(sol, sv_pos)

    # Calculate the covariance matrix for dops calculation
    Q = np.linalg.inv(H.T @ W @ H)

    # Calculate the error covariance matrix of the residuals
    return {"solution": sol, "error_covariance": covar, "state_covar": Q}
