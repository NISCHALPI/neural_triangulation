"""
Implements a constant velocity dynamics model for Kalman filter used in triangulation.

In the context of GNSS (Global Navigation Satellite System) triangulation, the dynamics model 
predicts the future state estimates of the target. The state vector is defined as:

x = [x, x_dot, y, y_dot, z, z_dot, cdt, cdt_dot]

where:
- x, y, z are the position coordinates of the target in the ECEF (Earth-Centered, Earth-Fixed) frame.
- x_dot, y_dot, z_dot are the velocities along the respective axes in the ECEF frame.
- cdt is the clock drift of the target.
- cdt_dot is the rate of change of clock drift.

Functions:
- G(dt: float) -> torch.Tensor:
  Returns the state transition matrix for the constant velocity model.

- h(x: torch.Tensor, sv_pos: torch.Tensor) -> torch.Tensor:
  Returns the measurement matrix for the constant velocity model.

- HJacobian(x: torch.Tensor, sv_pos: torch.Tensor) -> torch.Tensor:
  Returns the Jacobian matrix of the measurement matrix for the constant velocity model.

- Q(dt: float, Q_0: torch.Tensor) -> torch.Tensor:
  Returns the process noise matrix for the constant velocity model.

Classes:
- ObservationModel(nn.Module):
  A PyTorch module that implements the observation model for the constant velocity model.

- TransitionModel(nn.Module):
  A PyTorch module that implements the transition model for the constant velocity model.

- SymmetricPositiveDefiniteMatrix(nn.Module):
  A PyTorch module that ensures the input matrix remains symmetric and positive definite.

Usage:
>>> from constant_velocity_triangulation_model import G, h, HJacobian, Q, ObservationModel, TransitionModel, SymmetricPositiveDefiniteMatrix
>>> import torch

# Example usage of the functions
>>> dt = 0.1
>>> A = G(dt)
>>> x = torch.zeros(8)
>>> sv_pos = torch.randn(5, 3)
>>> measurement = h(x, sv_pos)
>>> jacobian = HJacobian(x, sv_pos)
>>> process_noise = Q(dt, torch.eye(8))

# Example usage of the classes
>>> observation_model = ObservationModel()
>>> transition_model = TransitionModel(dt=0.1)
>>> Q_matrix = SymmetricPositiveDefiniteMatrix(Q_0=torch.eye(8))
>>> Q_matrix_output = Q_matrix()

"""

import torch
import torch.nn as nn


__all__ = [
    "G",
    "h",
    "HJacobian",
    "Q",
    "ObservationModel",
    "TransitionModel",
    "SymmetricPositiveDefiniteMatrix",
]


def G(dt: float) -> torch.Tensor:
    """Returns the state transition matrix for the constant velocity model.

    Args:
        dt (float): Time step.

    Returns:
        torch.Tensor: State transition matrix.
    """
    A = torch.Tensor(
        [
            [1, dt, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, dt, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, dt, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, dt],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )

    return A


def h(x: torch.Tensor, sv_pos: torch.Tensor) -> torch.Tensor:
    """Returns the measurement matrix for the constant velocity model.

    Args:
        x (torch.Tensor): State vector. (8,)
        sv_pos (torch.Tensor): Satellite position. (n, 3)

    Returns:
        torch.Tensor: Measurement matrix.
    """
    pos = x[[0, 2, 4]]
    return torch.linalg.norm(pos - sv_pos, dim=1) + x[6]


def HJacobian(x: torch.Tensor, sv_pos: torch.Tensor) -> torch.Tensor:
    """Returns the Jacobian of the measurement matrix for the constant velocity model.

    Args:
        x (torch.Tensor): State vector. (8,)
        sv_pos (torch.Tensor): Satellite position. (n, 3)

    Returns:
        torch.Tensor: Jacobian of the measurement matrix.
    """
    pos = x[[0, 2, 4]]
    diff = pos - sv_pos
    norm = torch.linalg.norm(diff, dim=1)

    # Initialize the Jacobian matrix
    HJ = torch.zeros((sv_pos.shape[0], 8))

    # Add the derivative of the measurement matrix with respect to position
    HJ[:, [0, 2, 4]] = diff / norm[:, None]

    # Add the derivative of the measurement matrix with respect to clock drift
    HJ[:, 6] = 1

    return HJ


def Q(dt: float, Q_0: torch.Tensor) -> torch.Tensor:
    """Returns the process noise matrix for the constant velocity model.

    Args:
        dt (float): Time step.
        Q_0 (torch.Tensor): Process noise power spectral density matrix.

    Returns:
        torch.Tensor: Process noise matrix.
    """
    G_ = G(dt)

    return G_ @ Q_0 @ G_.T


class ObservationModel(nn.Module):
    """The observation model for the GNSS triangulation."""

    def __init__(self, trainable: bool = False, dim_measurement: int = 8):
        super().__init__()

        if trainable:
            self.W = nn.Sequential(
                nn.Linear(8, dim_measurement),
                nn.ReLU(),
                nn.Linear(dim_measurement, dim_measurement),
                nn.ReLU(),
                nn.Linear(dim_measurement, dim_measurement),
            )

            # Initialzie the parameters to be zero
            for param in self.W.parameters():
                # Initialize the bias to zero
                if param.dim() == 1:
                    param.data.zero_()

                # Initialize the weights to kaming normal
                else:
                    nn.init.kaiming_normal_(param.data)

                # Make the weights very small to the transition behaves like a constant velocity model at the start of training
                param.data *= 1e-30

    def forward(self, x: torch.Tensor, sv_pos: torch.Tensor) -> torch.Tensor:
        """Forward pass of the observation model.

        Args:
            x (torch.Tensor): State vector. (8,)
            sv_pos (torch.Tensor): Satellite position. (n, 3)

        Returns:
            torch.Tensor: Predicted measurements.
        """
        return h(x, sv_pos) + self.W(x) if hasattr(self, "W") else h(x, sv_pos)


class BiasedObservationModel(nn.Module):
    """The observation model for the GNSS triangulation."""

    def __init__(self, trainable: bool = False, dim_measurement: int = 8):
        super().__init__()

        if trainable:
            self.register_buffer("U", torch.ones(dim_measurement))
            self.alpha = nn.Parameter(torch.ones(1) * 1e-30, requires_grad=True)

    def forward(self, x: torch.Tensor, sv_pos: torch.Tensor) -> torch.Tensor:
        """Forward pass of the observation model.

        Args:
            x (torch.Tensor): State vector. (8,)
            sv_pos (torch.Tensor): Satellite position. (n, 3)

        Returns:
            torch.Tensor: Predicted measurements.
        """
        if hasattr(self, "U"):
            return h(x, sv_pos) + self.U * self.alpha

        return h(x, sv_pos)


class TransitionModel(nn.Module):
    """The transition model for the GNSS triangulation.

    Note:
        The transistion model can be parametrized by a neural network to learn the dynamics of the system.
    """

    def __init__(self, dt: float, learnable: bool = False):
        super().__init__()
        self.dt = dt
        self.register_buffer("F", G(dt))

        # Learnable Linear Layer
        if learnable:
            self.W = nn.Sequential(
                nn.Linear(8, 8),
                nn.GELU(),
                nn.Linear(8, 8),
                nn.GELU(),
                nn.Linear(8, 8),
            )

            # Initialzie the parameters to be zero
            for param in self.W.parameters():
                # Initialize the bias to zero
                if param.dim() == 1:
                    param.data.zero_()

                # Initialize the weights to kaming normal
                else:
                    nn.init.kaiming_normal_(param.data)

                # Make the weights very small to the transition behaves like a constant velocity model at the start of training
                param.data *= 1e-30

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the transition model.

        Args:
            x (torch.Tensor): State vector. (8,)

        Returns:
            torch.Tensor: Predicted next state vector.
        """
        return self.F @ x + self.W(x) if hasattr(self, "W") else self.F @ x
