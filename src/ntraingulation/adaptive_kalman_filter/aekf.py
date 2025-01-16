"""Adaptive Extended Kalman Filter (AEKF) implementation.

Src:
- https://www.sciencedirect.com/science/article/pii/S0360544219303469
"""

import torch
import torch.nn as nn
from diffkalman.joint_jacobian_transform import joint_jacobian_transform

__all__ = ["AdaptiveExtendedKalmanFilter"]


class AdaptiveExtendedKalmanFilter(nn.Module):
    """
    Implementation of the Adaptive Extended Kalman Filter (EKF) algorithm for state estimation.

    Args:
        dim_x (int): Dimension of the state vector.
        dim_z (int): Dimension of the measurement vector.
        f (nn.Module): State transition function.
        h (nn.Module): Measurement function.

    Attributes:
        dim_x (int): Dimension of the state vector.
        dim_z (int): Dimension of the measurement vector.
        f (nn.Module): State transition function.
        h (nn.Module): Measurement function.
        I (torch.Tensor): Identity matrix of size (dim_x, dim_x).
        _f (Callable): State transition function with Jacobian computation.
        _h (Callable): Measurement function with Jacobian computation.

    Methods:
        predict: Predicts the state of the system.
        update: Updates the state estimate based on the measurement.
        predict_update: Runs the predict-update loop.
        batch_filtering: Processes the sequence of measurements.
        fixed_interval_smoothing: Performs fixed-interval smoothing on the state estimates.
        batch_smoothing: Processes the sequence of measurements to form a Maximum Likelihood Estimation (MLE) loss.
        autocorreleation: Computes the autocorrelation of the innovation residuals sequence.
        forward: Processes the sequence of measurements to form a Maximum Likelihood Estimation (MLE) loss.
    """

    TERMS = {
        "PriorEstimate": "x_prior",
        "PriorCovariance": "P_prior",
        "StateJacobian": "State_Jacobian",
        "PosteriorEstimate": "x_posterior",
        "PosteriorCovariance": "P_posterior",
        "InnovationResidual": "innovation_residual",
        "InnovationCovariance": "innovation_covariance",
        "KalmanGain": "Kalman_gain",
        "MeasurementJacobian": "Measurement_Jacobian",
        "PosteriorReisdual": "residual",
        "AdaptedProcessNoise": "Q_adapted",
        "AdaptedMeasurementNoise": "R_adapted",
    }

    def __init__(
        self,
        dim_x: int,
        dim_z: int,
        f: nn.Module,
        h: nn.Module,
        alpha: float = 0.98,
        q_adaption_threshold: float = 1e3,
        r_adaption_threshold: float = 1e3,
    ) -> None:
        """
        Initializes the PEKF algorithm.

        Args:
            dim_x (int): Dimension of the state vector.
            dim_z (int): Dimension of the measurement vector.
            f (nn.Module): State transition function.
            h (nn.Module): Measurement function.
            alpha (float, optional): Forgetting or Adaptive factor. Defaults to 0.98.
            q_adaption_threshold (float, optional): Threshold for adapting the process noise covariance matrix. Defaults to 1e3.
            r_adaption_threshold (float, optional): Threshold for adapting the measurement noise covariance matrix. Defaults to 1e3.

        Note:
            - The state transition function f signature is f(x: torch.Tensor, *args) -> torch.Tensor
            - The measurement function h signature is h(x: torch.Tensor, *args) -> torch.Tensor
            - Any argument must scale with the batch dimension.
            - The adaption thresholds are used to determine if the process and measurement noise covariance matrices should be adapted based on the norm of the residuals.
        """
        super().__init__()
        # Store the dimensions of the matrices
        self.dim_x = dim_x
        self.dim_z = dim_z

        # Store the state transition and measurement functions
        self.f = f
        self.h = h

        # Register the jacobian functions
        self._f = joint_jacobian_transform(f)
        self._h = joint_jacobian_transform(h)

        # Store the adaptive factor
        self.adaptive_factor = alpha

        # Store the adaption thresholds
        self.q_adaption_threshold = q_adaption_threshold
        self.r_adaption_threshold = r_adaption_threshold

    def predict(
        self,
        x_posterior: torch.Tensor,
        P_posterior: torch.Tensor,
        Q: torch.Tensor | nn.Parameter,
        f_args: list = [],
    ) -> dict[str, torch.Tensor]:
        """
        Predicts the state of the system.

        Args:
            x_posterior (torch.Tensor): Posterior state estimate. (dim_x, )
            P_posterior (torch.Tensor): Posterior state error covariance. (dim_x, dim_x)
            Q (torch.Tensor | nn.Parameter): Process noise covariance. (dim_x, dim_x)
            f_args (list, optional): Additional arguments for the state transition function. Defaults to ().

        Returns:
            dict[str, torch.Tensor]: Dictionary containing the predicted state estimate, state error covariance, and the state transition matrix.

        Note:
            - f_args must scale with the batch dimension.
        """
        # Return the predicted state and state error covariance
        F, x_prior = self._f(x_posterior, *f_args)

        P_prior = F @ P_posterior @ F.T + Q

        return {
            self.TERMS["PriorEstimate"]: x_prior,
            self.TERMS["PriorCovariance"]: P_prior,
            self.TERMS["StateJacobian"]: F,
        }

    def update(
        self,
        z: torch.Tensor,
        x_prior: torch.Tensor,
        P_prior: torch.Tensor,
        R: torch.Tensor | nn.Parameter,
        h_args: list = [],
    ) -> dict[str, torch.Tensor]:
        """
        Updates the state estimate based on the measurement.

        Args:
            x_prior (torch.Tensor): Prior state estimate. (dim_x, )
            P_prior (torch.Tensor): Prior state error covariance. (dim_x, dim_x)
            z (torch.Tensor): Measurement vector. (dim_z, )
            R (torch.Tensor | nn.Parameter): Measurement noise covariance. (dim_z, dim_z)
            h_args (list, optional): Additional arguments for the measurement function. Defaults to ().

        Returns:
            dict[str, torch.Tensor]: Dictionary containing the updated state estimate, state error covariance, and the innovation.

        Note:
            - h_args must scale with the batch dimension.
        """
        # Compute the predicted measurement and the Jacobian
        H, z_pred = self._h(x_prior, *h_args)

        # Compute the innovation
        y = z - z_pred
        # Compute the innovation covariance matrix
        S = H @ P_prior @ H.T + R
        # Compute the Kalman gain
        K = P_prior @ H.T @ torch.linalg.inv(S)

        # Update the state vector
        x_post = x_prior + K @ y
        # Update the state covariance matrix using joseph form since
        # EKF is not guaranteed to be optimal
        factor = torch.eye(self.dim_x, device=x_post.device, dtype=x_post.dtype) - K @ H
        P_post = factor @ P_prior @ factor.T + K @ R @ K.T

        # Caluclate the posterior residual
        _, z_post = self._h(x_post, *h_args)
        residual = z - z_post

        return {
            self.TERMS["PosteriorEstimate"]: x_post,
            self.TERMS["PosteriorCovariance"]: P_post,
            self.TERMS["InnovationResidual"]: y,
            self.TERMS["InnovationCovariance"]: S,
            self.TERMS["KalmanGain"]: K,
            self.TERMS["MeasurementJacobian"]: H,
            self.TERMS["PosteriorReisdual"]: residual,
        }

    def correction_q(
        self,
        innovation_residuals: torch.Tensor,
        K: torch.Tensor,
        P_prior: torch.Tensor,
    ) -> torch.Tensor:
        """Adapts the process noise covariance matrix.

        Args:
            innovation_residuals (torch.Tensor): Innovation residuals. (T, dim_z)
            K (torch.Tensor): Kalman gain. (dim_x, dim_z)
            P_prior (torch.Tensor): Prior state error covariance. (dim_x, dim_x)

        Returns:
            torch.Tensor: Adapted process noise covariance matrix.
        """
        # Compute the outer product of the innovation residuals
        return K @ torch.outer(innovation_residuals, innovation_residuals) @ K.T

    def correction_r(
        self,
        posterior_residuals: torch.Tensor,
        H: torch.Tensor,
        P_prior: torch.Tensor,
    ) -> torch.Tensor:
        """Adapts the measurement noise covariance matrix.

        Args:
            posterior_residuals (torch.Tensor): Posterior residuals. (T, dim_z)
            H (torch.Tensor): Measurement Jacobian. (dim_z, dim_x)
            P_prior (torch.Tensor): Prior state error covariance. (dim_x, dim_x)

        Returns:
            torch.Tensor: Adapted measurement noise covariance matrix.
        """
        # Compute the outer product of the posterior residuals
        return torch.outer(posterior_residuals, posterior_residuals) + H @ P_prior @ H.T

    def predict_update(
        self,
        x_posterior: torch.Tensor,
        P_posterior: torch.Tensor,
        z: torch.Tensor,
        Q: torch.Tensor | nn.Parameter,
        R: torch.Tensor | nn.Parameter,
        f_args: list = [],
        h_args: list = [],
    ) -> dict[str, torch.Tensor]:
        """
        Runs the predict-update loop.

        Args:
            x_posterior (torch.Tensor): Posterior state estimate. (dim_x, )
            P_posterior (torch.Tensor): Posterior state error covariance. (dim_x, dim_x)
            z (torch.Tensor): Measurement vector. (dim_z, )
            Q (torch.Tensor | nn.Parameter): Process noise covariance. (dim_x, dim_x)
            R (torch.Tensor | nn.Parameter): Measurement noise covariance. (dim_z, dim_z)
            f_args (list, optional): Additional arguments for the state transition function. Defaults to ().
            h_args (list, optional): Additional arguments for the measurement function. Defaults to ().

        Returns:
            dict[str, torch.Tensor]: Dictionary containing the state estimates and state error covariances.

        Note:
            - h_args and f_args must scale with the batch dimension.
        """
        # Predict the state
        prediction = self.predict(x_posterior, P_posterior, Q, f_args)

        # Update the state
        update = self.update(
            z=z,
            x_prior=prediction[self.TERMS["PriorEstimate"]],
            P_prior=prediction[self.TERMS["PriorCovariance"]],
            R=R,
            h_args=h_args,
        )

        # Adapt the measurement noise matrix
        delta_r = self.correction_r(
            posterior_residuals=update[self.TERMS["PosteriorReisdual"]],
            H=update[self.TERMS["MeasurementJacobian"]],
            P_prior=prediction[self.TERMS["PriorCovariance"]],
        )
        delta_q = self.correction_q(
            innovation_residuals=update[self.TERMS["InnovationResidual"]],
            K=update[self.TERMS["KalmanGain"]],
            P_prior=prediction[self.TERMS["PriorCovariance"]],
        )

        # Adapt the process noise matrix
        if self.q_adaption_logic(delta_q):
            Q_k = self.adaptive_factor * Q + (1 - self.adaptive_factor) * delta_q
        else:
            Q_k = Q

        # Adapt the measurement noise matrix
        if self.r_adaption_logic(delta_r):
            R_k = self.adaptive_factor * R + (1 - self.adaptive_factor) * delta_r
        else:
            R_k = R

        return {
            **prediction,
            **update,
            self.TERMS["AdaptedProcessNoise"]: Q_k,
            self.TERMS["AdaptedMeasurementNoise"]: R_k,
        }

    def q_adaption_logic(
        self,
        delta_q: torch.Tensor,
    ) -> bool:
        """Adaptive logic for the process noise covariance matrix.

        Args:
            delta_r (torch.Tensor): Adapted measurement noise covariance matrix.

        Returns:
            bool: True if the process noise covariance matrix should be adapted.
        """
        return torch.norm(delta_q) <= self.q_adaption_threshold

    def r_adaption_logic(
        self,
        delta_r: torch.Tensor,
    ) -> bool:
        """Adaptive logic for the measurement noise covariance matrix.

        Args:
            delta_r (torch.Tensor): Adapted measurement noise covariance matrix.

        Returns:
            bool: True if the measurement noise covariance matrix should be adapted.
        """
        return torch.norm(delta_r) <= self.r_adaption_threshold

    def batch_filtering(
        self,
        z: torch.Tensor,
        x0: torch.Tensor,
        P0: torch.Tensor,
        Q: torch.Tensor | nn.Parameter,
        R: torch.Tensor | nn.Parameter,
        f_args: list = [],
        h_args: list = [],
    ) -> dict[str, torch.Tensor]:
        """
        Processes the sequence of measurements.

        Args:
            z (torch.Tensor): Measurement sequence. (num_timesteps, dim_z)
            x0 (torch.Tensor): Initial state estimate. (dim_x, )
            P0 (torch.Tensor): Initial state error covariance. (dim_x, dim_x)
            Q (torch.Tensor | nn.Parameter): Process noise covariance. (dim_x, dim_x)
            R (torch.Tensor | nn.Parameter): Measurement noise covariance. (dim_z, dim_z)
            f_args (list, optional): Additional arguments for the state transition function. Defaults to ().
            h_args (list, optional): Additional arguments for the measurement function. Defaults to ().

        Returns:
            dict[str, list[torch.Tensor]]: Dictionary containing lists of state estimates and state error covariances at each time step.

        Note:
            - h_args and f_args must scale with the batch dimension.
        """
        # Sequence length
        T = z.shape[0]
        # Initialize the intermediate variables
        output = {
            self.TERMS[key]: []
            for key in self.TERMS.keys()
            if not key.startswith("Smoothed")
        }

        # Run the filtering algorithm
        for t in range(T):
            # Perform the predict-update loop
            results = self.predict_update(
                x_posterior=(
                    x0 if t == 0 else output[self.TERMS["PosteriorEstimate"]][-1]
                ),
                P_posterior=(
                    P0 if t == 0 else output[self.TERMS["PosteriorCovariance"]][-1]
                ),
                z=z[t],
                Q=(Q if t == 0 else output[self.TERMS["AdaptedProcessNoise"]][-1]),
                R=(R if t == 0 else output[self.TERMS["AdaptedMeasurementNoise"]][-1]),
                f_args=[args[t] for args in f_args],
                h_args=[args[t] for args in h_args],
            )

            # Update the output
            for term in output:
                output[term].append(results[term])

        # Stack the results
        for term in output:
            output[term] = torch.stack(output[term])

        return output

    @staticmethod
    def autocorreleation(
        innovation_residuals: torch.Tensor,
        lag: int = 0,
    ) -> torch.Tensor:
        """Computes the autocorrelation of the innovation residuals sequence.

        Args:
            innovation_residuals (torch.Tensor): Innovation residuals. (T, dim_z)
            lag (int, optional): Lag. Defaults to 1.

        Returns:
            torch.Tensor: Autocorrelation.
        """
        # If the T dimension is less than the lag, return 0
        if innovation_residuals.shape[0] < lag:
            return 0

        # Center the residuals
        residuals = innovation_residuals - torch.mean(innovation_residuals, dim=0)

        # Compute the outer product expectation of the residuals
        outer_product = 0
        for i in range(len(residuals) - lag):
            outer_product += torch.outer(residuals[i], residuals[i + lag])

        # Compute the autocorrelation
        return outer_product / (len(residuals) - lag)
