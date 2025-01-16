"""Estimators for the Monte Carlo runs. Following estimators are available:

- Weighted Least Squares Estimator
- Extended Kalman Filter Estimator
- Unscented Kalman Filter Estimator
- Adaptive Extended Kalman Filter Estimator

Each returns the estimated position in geocentric coordinates.
"""

import torch
import pandas as pd
import torch.nn as nn
from ..adaptive_kalman_filter.aekf import AdaptiveExtendedKalmanFilter
from diffkalman import DiffrentiableKalmanFilter
from filterpy.kalman import UnscentedKalmanFilter
import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Saver
from ..weighted_least_squares.wls_triangulator import wls_triangulation


__all__ = [
    "wls_estimator",
    "ekf_estimator",
    "ukf_estimator",
    "aekf_estimator",
]


def aekf_estimator(
    measurements: torch.Tensor,
    x0: torch.Tensor,
    P0: torch.Tensor,
    Q: torch.Tensor,
    R: torch.Tensor,
    sv_seq: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    transistion_model: nn.Module,
    measurement_model: nn.Module,
) -> pd.DataFrame:
    """Estimate the position using the Adaptive Extended Kalman Filter.

    Args:
        measurements: The noisy measurements. (N, DIM_Z)
        x0: The initial state. (DIM_X,)
        P0: The initial covariance. (DIM_X, DIM_X)
        Q: The process noise covariance. (DIM_X, DIM_X)
        R: The measurement noise covariance. (DIM_Z, DIM_Z)
        sv_seq: The sequence of sv's coordinates. (N,3)
        device: The device to use.
        dtype: The datatype to use.

    Args:
        pd.DataFrame: The estimated position in geocentric coordinates.
    """

    # Initialize the filter
    aekf = AdaptiveExtendedKalmanFilter(
        dim_x=8,
        dim_z=measurements.shape[1],
        f=transistion_model,
        h=measurement_model,
    ).to(device=device, dtype=dtype)
    # Take the measurements and other parameters to the device and dtype
    measurements = measurements.to(device=device, dtype=dtype)
    x0 = x0.to(device=device, dtype=dtype)
    P0 = P0.to(device=device, dtype=dtype)
    Q = Q.to(device=device, dtype=dtype)
    R = R.to(device=device, dtype=dtype)
    sv_seq = sv_seq.to(device=device, dtype=dtype)

    # Get the output
    outs = aekf.batch_filtering(
        z=measurements,
        x0=x0,
        P0=P0,
        Q=Q,
        R=R,
        h_args=(sv_seq,),
    )

    # Get the estimated position
    position = outs["x_posterior"][:, [0, 2, 4]].cpu().numpy()

    return pd.DataFrame(position, columns=["x", "y", "z"])


def ekf_estimator(
    measurements: torch.Tensor,
    x0: torch.Tensor,
    P0: torch.Tensor,
    Q: torch.Tensor,
    R: torch.Tensor,
    sv_seq: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    transistion_model: nn.Module,
    measurement_model: nn.Module,
) -> pd.DataFrame:
    """Estimate the position using the Extended Kalman Filter.

    Args:
        measurements: The noisy measurements. (N, DIM_Z)
        x0: The initial state. (DIM_X,)
        P0: The initial covariance. (DIM_X, DIM_X)
        Q: The process noise covariance. (DIM_X, DIM_X)
        R: The measurement noise covariance. (DIM_Z, DIM_Z)
        sv_seq: The sequence of sv's coordinates. (N,3)
        device: The device to use.
        dtype: The datatype to use.

    Args:
        pd.DataFrame: The estimated position in geocentric coordinates.
    """

    # Initialize the filter
    ekf = DiffrentiableKalmanFilter(
        dim_x=8,
        dim_z=measurements.shape[1],
        f=transistion_model,
        h=measurement_model,
    ).to(device=device, dtype=dtype)

    # Take the measurements and other parameters to the device and dtype
    measurements = measurements.to(device=device, dtype=dtype)
    x0 = x0.to(device=device, dtype=dtype)
    P0 = P0.to(device=device, dtype=dtype)
    Q = Q.to(device=device, dtype=dtype)
    R = R.to(device=device, dtype=dtype)
    sv_seq = sv_seq.to(device=device, dtype=dtype)

    # Get the output
    outs = ekf.sequence_filter(
        z_seq=measurements,
        x0=x0,
        P0=P0,
        Q=Q.repeat(measurements.shape[0], 1, 1),
        R=R.repeat(measurements.shape[0], 1, 1),
        h_args=(sv_seq,),
    )

    # Get the estimated position
    position = outs["x_post"][:, [0, 2, 4]].cpu().numpy()

    return pd.DataFrame(position, columns=["x", "y", "z"])


def get_filter_and_saver(
    dt: float,
    x0: np.ndarray,
    P0: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    h: callable,
    f: callable,
) -> tuple:
    """Returns the UKF filter and saver.

    Args:
        points (MerweScaledSigmaPoints): Sigma points.
        x0 (np.ndarray): Initial state.
        P0 (np.ndarray): Initial covariance.
        Q (np.ndarray): Process noise.
        R (np.ndarray): Measurement noise.

    Returns:
        tuple: UKF filter and saver.
    """
    kf = UnscentedKalmanFilter(
        dim_x=x0.shape[0],
        dim_z=R.shape[0],
        dt=dt,
        hx=h,
        fx=f,
        points=MerweScaledSigmaPoints(
            n=x0.shape[0], alpha=0.1, beta=2.0, kappa=3 - x0.shape[0]
        ),
    )
    saver = Saver(kf)

    # Set the initial state and noise matrices
    kf.x = x0
    kf.P = P0
    kf.Q = Q
    kf.R = R

    return kf, saver


def ukf_estimator(
    measurements: torch.Tensor,
    dt: float,
    x0: torch.Tensor,
    P0: torch.Tensor,
    Q: torch.Tensor,
    R: torch.Tensor,
    sv_seq: torch.Tensor,
    transistion_function: callable,
    measurement_function: callable,
) -> pd.DataFrame:
    """Estimate the position using the Unscented Kalman Filter.

    Args:
        dt: The time step. (float)
        measurements: The noisy measurements. (N, DIM_Z)
        x0: The initial state. (DIM_X,)
        P0: The initial covariance. (DIM_X, DIM_X)
        Q: The process noise covariance. (DIM_X, DIM_X)
        R: The measurement noise covariance. (DIM_Z, DIM_Z)
        sv_seq: The sequence of sv's coordinates. (N,3)

    Args:
        pd.DataFrame: The estimated position in geocentric coordinates.
    """

    # Initialize the filter
    ukf, saver = get_filter_and_saver(
        dt=dt,
        x0=x0.cpu().numpy().astype(np.float64),
        P0=P0.cpu().numpy().astype(np.float64),
        Q=Q.cpu().numpy().astype(np.float64),
        R=R.cpu().numpy().astype(np.float64),
        h=measurement_function,
        f=transistion_function,
    )

    # Take the measurements and other parameters to the device and dtype
    z = measurements.cpu().numpy().astype(np.float64)
    sv_seq = sv_seq.cpu().numpy().astype(np.float64)

    # Run the UKF
    for i in range(len(z)):
        ukf.predict()
        ukf.update(z[i], sv_pos=sv_seq[i])
        saver.save()

    x_post = np.array(saver.x_post)

    # Get the estimated position
    position = x_post[:, [0, 2, 4]]

    return pd.DataFrame(position, columns=["x", "y", "z"])


def get_wls_estimates(
    range_measurements: np.ndarray,
    satellite_postions: np.ndarray,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Get the Weighted Least Squares estimates of the position.

    Args:
        range_measurements (np.ndarray): Range measurements from the satellites.(N,  DIM_Z)
        satellite_postions (np.ndarray): Satellite positions in the ECEF frame. (N, 3)
        weights (np.ndarray): Weights for the range measurements. (DIM_Z, DIM_Z)

    Returns:
        pd.DataFrame: DataFrame containing the estimates of the position.

    """
    x0 = np.zeros(4, dtype=np.float64)
    if weights is None:
        weights = np.eye(range_measurements.shape[1])
    range_measurements = range_measurements.astype(np.float64)
    satellite_postions = satellite_postions.astype(np.float64)
    positions = []
    for i in range(len(range_measurements)):
        solutions = wls_triangulation(
            pseudorange=range_measurements[i],
            sv_pos=satellite_postions[i],
            W=weights,
            x0=x0 if i == 0 else positions[-1],
        )
        positions.append(solutions["solution"].copy())

    return pd.DataFrame(positions, columns=["x", "y", "z", "cdt"])


def wsl_estimator(
    z_noisy: torch.Tensor,
    sv_seq: torch.Tensor,
) -> pd.DataFrame:
    """Estimate the position using the Weighted Least Squares Estimator.

    Args:
        z_noisy: The noisy measurements. (N, DIM_Z)
        sv_seq: The sequence of sv's coordinates. (N,3)

    Returns:
        pd.DataFrame: The estimated position in geocentric coordinates.
    """
    # Get the estimated position
    position = get_wls_estimates(
        range_measurements=z_noisy.cpu().numpy().astype(np.float64),
        satellite_postions=sv_seq.cpu().numpy().astype(np.float64),
    )

    return position[["x", "y", "z"]]
