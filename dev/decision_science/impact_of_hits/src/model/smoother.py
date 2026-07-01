import torch
import numpy as np

def extract_smoothed_states(model, games):
    """
    Implements a Rauch-Tung-Striebel (RTS) backward smoother to extract
    the smoothed latent momentum E[x_t | y_{1:T}] for each game.
    """
    model.eval()
    smoothed_results = []
    
    a0 = torch.sigmoid(model.log_a0).item()
    C  = torch.exp(model.log_C).item()
    Q  = torch.exp(model.log_Q).item()
    R  = torch.exp(model.log_R).item()
    P0 = torch.exp(model.log_P0).item()
    B  = model.B.detach().numpy()
    D  = model.D.detach().numpy()
    alpha = model.alpha.item()
    beta  = model.beta.item()

    for game in games:
        U = game["U"].numpy()
        Y = game["Y"].numpy()
        Z = game["Z"].numpy()
        dt = game["dt"].numpy()
        T = len(U)
        
        # Forward pass (Filter)
        x_filt = np.zeros(T)
        P_filt = np.zeros(T)
        x_pred = np.zeros(T)
        P_pred = np.zeros(T)
        
        # Initial state
        x = alpha * Z[0, 0] + beta * Z[0, 1]
        P = P0
        
        for t in range(T):
            A_t = a0 ** dt[t]
            
            # Predict
            x_p = A_t * x + B @ U[t]
            P_p = A_t**2 * P + Q
            x_pred[t] = x_p
            P_pred[t] = P_p
            
            # Update
            y_hat = C * x_p + D @ Z[t]
            innov = Y[t] - y_hat
            S = C**2 * P_p + R
            K = P_p * C / S
            
            x = x_p + K * innov
            P = (1 - K * C) * P_p
            
            x_filt[t] = x
            P_filt[t] = P
            
        # Backward pass (RTS Smoother)
        x_smooth = np.zeros(T)
        P_smooth = np.zeros(T)
        x_smooth[-1] = x_filt[-1]
        P_smooth[-1] = P_filt[-1]
        
        for t in range(T - 2, -1, -1):
            A_t = a0 ** dt[t+1]
            
            # Smoother gain
            J = P_filt[t] * A_t / P_pred[t+1]
            
            x_smooth[t] = x_filt[t] + J * (x_smooth[t+1] - x_pred[t+1])
            P_smooth[t] = P_filt[t] + J**2 * (P_smooth[t+1] - P_pred[t+1])
            
        smoothed_results.append({
            "game_id": game["game_id"],
            "x_smooth": x_smooth,
            "Y": Y,
            "Z": Z
        })
        
    return smoothed_results
