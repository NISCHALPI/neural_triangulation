"""Run monte carlo sim for the simulated experiment."""

import tqdm
import torch
import pandas as pd
from ..transforms.enu_error import enu_position_errors
from ..error_models.errors import (
    multivariate_normal_error,
    multivariate_student_t_error,
)

__all__ = ["run_monte_carlo_sim"]


def apply_gaussian_noise(
    measurement: torch.Tensor, mean: torch.Tensor, covarience: torch.Tensor
) -> torch.Tensor:
    """Applies the gaussian noise to the measurement.

    Args:
        measurement: Measurement to apply the noise. (N, DIM_Z)
        mean:  Mean of the gaussian noise. (DIM_Z,)
        covarience: Covariance of the gaussian noise. (DIM_Z, DIM_Z)

    Returns:
        Measurement with the gaussian noise.
    """
    # Sample the noise
    noise = multivariate_normal_error(mean=mean, cov=covarience, n=measurement.shape[0])
    return measurement + torch.tensor(
        noise, dtype=measurement.dtype, device=measurement.device
    )


def get_enu_position_errors(
    predicted: pd.DataFrame,
    true: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate the ENU position errors.

    Args:
        predicted: The predicted location in geocentric coordinates.
        actual: The actual location in geocentric coordinates.

    Returns:
        np.ndarray: The ENU position errors.
    """
    return pd.DataFrame(
        enu_position_errors(
            predicted=predicted[["x", "y", "z"]].values,
            actual=true[["x", "y", "z"]].values,
        ),
        columns=["E", "N", "U"],
    )


def run_monte_carlo_sim(
    z_true: torch.Tensor,
    n_mc: int,
    true_position: pd.DataFrame,
    mu_noise: torch.Tensor,
    P_noise: torch.Tensor,
    estimator: callable,
    is_sim : bool = True,
    reduce: bool = True,
    **kwargs
) -> pd.DataFrame:
    """Run the monte carlo simulation and get the ENU error for each run.

    Args:
        z_true: The true sv measurements. (N, DIM_Z)
        n_mc: The number of monte carlo runs. (int)
        true_position: The true position of the receiver. (pd.DataFrame)
        mu_noise: The mean of the noise. (DIM_Z,)
        P_noise: The standard deviation of the noise. (DIM_Z, DIM_Z)
        estimator: The estimator to use to infer the coordinates.
        is_sim: Weather to add the noise or not.
        reduce: Weather to reduce the ENU errors or not.
        **kwargs: Additional arguments to pass to the estimator.

    Returns:
        pd.DataFrame: The ENU errors for each monte carlo run.

    Note:
        - The estimator signature should be estimator(z_noisy, **kwargs) -> pd.DataFrame that returns the estimated position i.e x, y, z.
    """

    # Initialize the ENU errors
    enu_errors = []

    # Run the monte carlo simulation
    with torch.no_grad():
        with tqdm.tqdm(total=n_mc, desc="Monte Carlo Simulation") as pbar:
            for _ in range(n_mc):
                # Generate the noisy measurements
                if is_sim:
                    z_noisy = apply_gaussian_noise(z_true, mu_noise, P_noise)
                else:
                    z_noisy = z_true
                
                z_noisy = apply_gaussian_noise(z_true, mu_noise, P_noise)

                # Estimate the position
                estimated_position = estimator(z_noisy, **kwargs)

                # Calculate the ENU errors
                enu_error = get_enu_position_errors(
                    predicted=estimated_position, true=true_position[["x", "y", "z"]]
                )
                
                # Append the ENU errors
                if reduce:
                    enu_errors.append(enu_error.mean(axis=0).values)
                else:
                    enu_errors = enu_error
                    
                # Update the progress bar
                pbar.update(1)

    return pd.DataFrame(enu_errors, columns=["E", "N", "U"])
