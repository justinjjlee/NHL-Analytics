# Odds format conversion utilities
# Converted from https://github.com/opisthokonta/implied/blob/master/R/implied_odds.R
from typing import Union, Tuple
import re

Number = Union[int, float]

class Odds:
    """Utilities for converting between odds formats and probabilities."""

    @staticmethod
    def fractional_to_decimal(frac: Union[str, Tuple[Number, Number]]) -> float:
        """
        Convert fractional odds (e.g. '3/1' or (3,1)) to decimal odds.
        Fractional odds a/b -> decimal = 1 + a/b
        """
        if isinstance(frac, str):
            m = re.match(r'^\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*$', frac)
            if not m:
                raise ValueError(f"Invalid fractional string: {frac}")
            a, b = float(m.group(1)), float(m.group(2))
        else:
            a, b = float(frac[0]), float(frac[1])
        if b == 0:
            raise ValueError("Denominator in fractional odds cannot be zero.")
        return 1.0 + (a / b)

    @staticmethod
    def american_to_decimal(american: Number) -> float:
        """
        Convert American odds to decimal odds.
        Positive: +150 -> 1 + 150/100
        Negative: -150 -> 1 + 100/150
        """
        o = float(american)
        if o > 0:
            return 1.0 + (o / 100.0)
        elif o < 0:
            return 1.0 + (100.0 / abs(o))
        else:
            raise ValueError("American odds cannot be zero.")

    @staticmethod
    def decimal_to_probability(decimal: Number) -> float:
        d = float(decimal)
        if d <= 0:
            raise ValueError("Decimal odds must be > 0")
        return 1.0 / d

    @staticmethod
    def format_to_decimal(value: Union[str, Number, Tuple[Number, Number]], kind: str = "decimal") -> float:
        """
        Normalize a single odds value to decimal form.
        kind: 'decimal', 'fractional', 'american'
        """
        kind = kind.lower()
        if kind == "decimal":
            return float(value)
        elif kind == "fractional":
            return Odds.fractional_to_decimal(value)
        elif kind == "american":
            return Odds.american_to_decimal(value)
        else:
            raise ValueError(f"Unknown odds kind: {kind}")

    @staticmethod
    def decimal_to_american(d: Number) -> int:
        d = float(d)
        if d <= 1:
            raise ValueError("Decimal odds must be > 1 to convert to American.")
        if d >= 2.0:
            return int(round((d - 1.0) * 100.0))
        else:
            return int(round(-100.0 / (d - 1.0)))

    @staticmethod
    def decimal_to_fractional(d: Number) -> Tuple[int, int]:
        d = float(d)
        if d <= 1:
            raise ValueError("Decimal odds must be > 1 to convert to fractional.")
        frac = d - 1.0
        # Return a simple rational approximation using continued fractions or fixed denominator
        # For simplicity return numerator/denominator with denominator 10000 reduced.
        denom = 10000
        numer = int(round(frac * denom))
        import math
        g = math.gcd(numer, denom)
        return (numer // g, denom // g)