import torch
import torch.nn as nn

class MomentumSSM(nn.Module):
    """
    Continuous-time State-space model for NHL game momentum.
    Uses gradient descent (Kalman filter forward pass) to optimize parameters.
    """
    def __init__(self, n_events: int = 9, n_strength: int = 3):
        super().__init__()
        
        # Decay rate (a0), bounded (0, 1) through sigmoid
        self.log_a0 = nn.Parameter(torch.tensor(-0.04))
        
        # Event impact weights B (e.g. hit_for, shot_for, ...)
        self.B = nn.Parameter(torch.zeros(n_events))
        
        # Observation loading C (maps latent momentum to proxy like zone share)
        self.log_C = nn.Parameter(torch.tensor(0.0))
        
        # Team strength observation baselines (D)
        self.D = nn.Parameter(torch.zeros(n_strength))
        
        # Initial state strength differential weights
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        
        # Log-variances (Process Noise Q, Observation Noise R)
        self.log_Q = nn.Parameter(torch.tensor(-1.0))
        self.log_R = nn.Parameter(torch.tensor(-1.0))
        
        self.log_P0 = nn.Parameter(torch.tensor(0.0))

    def forward(self, U_pad: torch.Tensor, Y_pad: torch.Tensor, Z_pad: torch.Tensor, dt_pad: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Runs the Kalman filter forward pass for a padded batch of games.
        Returns the scalar sum of the Negative Log Likelihood (NLL) across all valid events.
        
        Args:
            U_pad: (B, T, 13)
            Y_pad: (B, T)
            Z_pad: (B, T, 4)
            dt_pad: (B, T)
            mask: (B, T) 1.0 for valid events, 0.0 for padding
        """
        a0 = torch.sigmoid(self.log_a0)
        C = torch.exp(self.log_C)
        Q = torch.exp(self.log_Q)
        R = torch.exp(self.log_R)
        P0 = torch.exp(self.log_P0)
        
        # Initial state based on PyExp/process metrics (Z_pad[:, 0, :])
        # Z_pad[..., 0] is is_home, Z_pad[..., 1] is delta_pythagorean
        mu0 = self.alpha * Z_pad[:, 0, 0] + self.beta * Z_pad[:, 0, 1]
        x = mu0  # shape: (B,)
        P = P0 * torch.ones_like(x) # shape: (B,)
        
        nll = torch.tensor(0.0, device=U_pad.device)
        max_T = U_pad.shape[1]
        
        for t in range(max_T):
            A_t = a0 ** dt_pad[:, t]
            
            # Predict
            x_pred = A_t * x + torch.matmul(U_pad[:, t, :], self.B)
            P_pred = (A_t ** 2) * P + Q
            
            # Update
            y_hat = C * x_pred + torch.matmul(Z_pad[:, t, :], self.D)
            innov = Y_pad[:, t] - y_hat
            S = (C ** 2) * P_pred + R
            K = (P_pred * C) / S
            
            x = x_pred + K * innov
            P = (1 - K * C) * P_pred
            
            # NLL contribution masked for valid events only
            step_nll = 0.5 * (torch.log(S) + (innov ** 2) / S)
            nll = nll + torch.sum(step_nll * mask[:, t])
            
        return nll
