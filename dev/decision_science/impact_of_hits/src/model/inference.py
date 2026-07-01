import numpy as np

def compute_half_life(a0_val: float) -> float:
    """
    Returns half-life (in seconds) of an event impulse strictly from the model's a0_val decay form.
    """
    if a0_val <= 0.0 or a0_val >= 1.0:
        return float('inf')
    return -np.log(2.0) / np.log(a0_val)

def impulse_response(a0_val: float, b_val: float, max_sec: int = 180) -> np.ndarray:
    """
    Returns deterministic discrete trajectory of the latent momentum state.
    """
    t_range = np.arange(1, max_sec + 1)
    trajectory = b_val * (a0_val ** t_range)
    return trajectory

def integrate_total_momentum(a0_val: float, b_val: float) -> float:
    """
    Evaluates exact expected cumulative shift area mathematically.
    """
    if a0_val >= 1.0:
        return float('inf')
    return b_val / (1.0 - a0_val)
