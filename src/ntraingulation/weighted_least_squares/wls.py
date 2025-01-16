"""Implements the least squares method for linear and non-linear observation models."""

import numpy as np

__all__ = ["weighted_least_square", "non_linear_weighted_least_squares"]


def weighted_least_square(y: np.ndarray, H: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Estimates the constant vector x using the weighted least squares method given noisy measurements y and a linear observation model H.

    Args:
        y (np.ndarray): The measurements of shape (M,).
        H (np.ndarray): The observation matrix of shape (M, N).
        W (np.ndarray): The weight matrix of shape (M, M).

    Returns:
        np.ndarray: The estimated constant vector x of shape (N,).
    """
    # Check if the number of measurements is greater than or equal to the number of states
    if y.shape[0] < H.shape[1]:
        raise ValueError(
            f"Number of measurements ({y.shape[0]}) must be greater than the number of states ({H.shape[1]})."
        )
    return np.linalg.inv(H.T @ W @ H) @ H.T @ W @ y


def non_linear_weighted_least_squares(
    y: np.ndarray,
    f: callable,
    HJacobian: callable,
    W: np.ndarray,
    x0: np.ndarray,
    eps: float = 1e-5,
    max_iter: int = 1000,
    f_args: tuple = (),
    HJacobian_args: tuple = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Estimates the constant vector x using the non-linear weighted least squares method given noisy measurements y and a non-linear observation model f.

    Args:
        y (np.ndarray): The measurements of shape (M,)
        f (callable): The non-linear observation model.
        HJacobian (callable): The Jacobian of the observation model.
        W (np.ndarray): The weight matrix of shape (M, M).
        x0 (np.ndarray): The initial guess for x of shape (N,).
        eps (float, optional): The convergence threshold. Defaults to 1e-5.
        max_iter (int, optional): The maximum number of iterations. Defaults to 1000.
        f_args (tuple, optional): Additional arguments to pass to the observation model. Defaults to ().
        HJacobian_args (tuple, optional): Additional arguments to pass to the Jacobian. Defaults to ().

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - The estimated constant vector x of shape (N,).
            - The final change in x (dx) of shape (N,) indicating convergence status.
    """
    x_prev = x0
    for _ in range(max_iter):
        # Linearize the observation model
        y_0 = f(x_prev, *f_args)

        # Get the Jacobian
        H = HJacobian(x_prev, *HJacobian_args)

        # Compute the residual
        dx = weighted_least_square(y - y_0, H, W)

        # Check convergence
        if np.linalg.norm(dx) < eps:
            break

        # Update the guess
        x_prev = x_prev + dx

    # Compute the error covariance matrix of the residuals
    residuals = y - f(x_prev, *f_args)
    Q = np.outer(residuals, residuals)

    return x_prev, Q
