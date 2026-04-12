# Implied probability calculations, vig removal including Shin method.

from typing import Iterable, List, Tuple
import math

from .shinodds import Odds

def _to_decimal_list(odds: Iterable, kind: str) -> List[float]:
    return [Odds.format_to_decimal(o, kind) for o in odds]

def _raw_probabilities_from_decimals(decimals: List[float]) -> List[float]:
    return [1.0 / d for d in decimals]

def normalize_basic(p_star: List[float]) -> Tuple[List[float], float]:
    """
    Basic normalization: divide each implied probability by the sum.
    Returns (adjusted_probs, vig) where vig = sum(p_star) - 1
    """
    s = sum(p_star)
    if s == 0:
        raise ValueError("Sum of raw probabilities is zero.")
    vig = s - 1.0
    adjusted = [p / s for p in p_star]
    return adjusted, vig

def _shin_adjusted_probabilities(p_star: List[float], tol: float = 1e-12, max_iter: int = 200) -> Tuple[List[float], float]:
    """
    Shin (1993) algorithm adapted: find z in [0, 1) so that adjusted probabilities sum to 1.
    The closed-form transform for each outcome given z is:
      p_i(z) = (1 / (2*(1 - z))) * ( sqrt( (z/m)^2 + 4*(1 - z)*(p_star_i)/m ) - z/m )
    Solve for z such that sum_i p_i(z) = 1.
    Returns (adjusted_probs, vig) where vig = sum(p_star) - 1
    """
    m = len(p_star)
    if m == 0:
        return [], 0.0
    s = sum(p_star)
    vig = max(0.0, s - 1.0)

    # If no vig (or extremely small), just normalize
    if s <= 1.0 + 1e-15:
        adjusted = [p / s for p in p_star] if s != 0 else [0.0] * m
        return adjusted, vig

    def sum_adjusted(z: float) -> float:
        # returns sum_i p_i(z)
        total = 0.0
        z_over_m = z / m
        denom = 2.0 * (1.0 - z)
        for a in p_star:
            inside = (z_over_m * z_over_m) + 4.0 * (1.0 - z) * (a / m)
            total += (math.sqrt(inside) - z_over_m) / denom
        return total

    # We need to find z in [0, 1 - eps) such that sum_adjusted(z) == 1
    lo = 0.0
    hi = 1.0 - 1e-12

    # Ensure the function has opposite signs at endpoints:
    # At z = 0, sum_adjusted(0) = sum(p_star) == s (>1)
    # As z -> 1-, sum_adjusted(z) -> 0
    f_lo = sum_adjusted(lo) - 1.0  # positive
    f_hi = sum_adjusted(hi) - 1.0  # negative (expected)
    if f_lo <= 0:
        # no need to solve: sum already <= 1
        adjusted = [p / s for p in p_star]
        return adjusted, vig

    # Bisection
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        f_mid = sum_adjusted(mid) - 1.0
        if abs(f_mid) < tol:
            z = mid
            break
        if f_mid > 0:
            lo = mid
        else:
            hi = mid
    else:
        z = (lo + hi) / 2.0

    # compute adjusted probs with found z
    z_over_m = z / m
    denom = 2.0 * (1.0 - z)
    adjusted = []
    for a in p_star:
        inside = (z_over_m * z_over_m) + 4.0 * (1.0 - z) * (a / m)
        adjusted.append((math.sqrt(inside) - z_over_m) / denom)
    # final normalizing safeguard
    sum_adj = sum(adjusted)
    if sum_adj <= 0:
        adjusted = [1.0 / m] * m
    else:
        adjusted = [p / sum_adj for p in adjusted]
    return adjusted, vig

def implied_probabilities(odds: Iterable,
                          odds_kind: str = "decimal",
                          remove_vig: str = "basic") -> Tuple[List[float], float]:
    """
    Calculate implied probabilities from a list of odds.

    Parameters:
    - odds: iterable of odds values (strings for fractional, numbers for decimal/american).
    - odds_kind: 'decimal', 'fractional', or 'american'
    - remove_vig: 'basic' (normalization) or 'shin' (Shin method). 'none' returns raw implied probs.

    Returns: (probabilities_list, vig_estimate)
    """
    decimals = _to_decimal_list(odds, odds_kind)
    p_star = _raw_probabilities_from_decimals(decimals)
    if remove_vig is None or remove_vig == "none":
        vig = max(0.0, sum(p_star) - 1.0)
        return p_star, vig
    elif remove_vig == "basic" or remove_vig == "proportional":
        return normalize_basic(p_star)
    elif remove_vig == "shin":
        return _shin_adjusted_probabilities(p_star)
    else:
        raise ValueError("remove_vig must be one of: None/'none', 'basic', 'proportional', 'shin'")

def implied_odds_from_probabilities(probs: Iterable,
                                    target_format: str = "decimal"):
    """
    Convert true probabilities to odds. target_format = 'decimal', 'fractional', 'american'
    """
    probs = list(probs)
    if any(p <= 0 for p in probs):
        raise ValueError("Probabilities must be > 0 to form odds.")
    if target_format == "decimal":
        return [1.0 / p for p in probs]
    elif target_format == "american":
        return [Odds.decimal_to_american(1.0 / p) for p in probs]
    elif target_format == "fractional":
        return [Odds.decimal_to_fractional(1.0 / p) for p in probs]
    else:
        raise ValueError("Unknown target_format")