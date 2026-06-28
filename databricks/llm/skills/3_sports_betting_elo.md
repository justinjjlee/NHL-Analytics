# Skillset 3: Sports Betting & ELO Calibration

This skillset covers the methods for converting raw sports betting lines, removing the bookmaker's commission (overround or vig) using Shin's Method, and calibrating rolling team ELO models to check betting market efficiency.

---

## 1. Scope & Target Measurement Level
* **Measurement Level:** Sports Betting Level, Game Level.
* **Objective:** Establish true win probabilities from biased bookmaker odds to build profitable prediction models and validate ELO ratings.

---

## 2. Mathematical Formulations

### 2.1 American Odds to Implied Probability
Moneyline odds ($L$) represent the payout ratio. They must be converted to implied probabilities ($p^*$).
* **Negative Moneyline ($L < 0$, e.g., $-150$, Home Favored):**
  $$p^*_{\text{home}} = \frac{|L_{\text{home}}|}{|L_{\text{home}}| + 100}$$
* **Positive Moneyline ($L > 0$, e.g., $+130$, Away Underdog):**
  $$p^*_{\text{away}} = \frac{100}{L_{\text{away}} + 100}$$
* **Overround (Vig):**
  $$\text{Overround} = \sum_{i=1}^{m} p^*_i - 1.0$$
  * For two-way moneylines ($m=2$: home and away), the sum $s = p^*_{\text{home}} + p^*_{\text{away}}$ typically ranges from $1.03$ to $1.07$. The excess $s - 1.0$ is the bookmaker's margin (vig).

### 2.2 Shin's Method for Vig Removal
Shin (1993) models betting odds as a market with three agents: the bookmaker, uninformed bettors, and informed bettors. Informed bettors know the true outcome with certainty.
* **True Probability $p_i$ given Informed Trader Share $z \in [0, 1)$:**
  The relationship between true probability $p_i$ and bookmaker-implied probability $p^*_i$ is:
  $$p_i(z) = \frac{1}{2(1 - z)} \left( \sqrt{\left(\frac{z}{m}\right)^2 + \frac{4(1 - z) p^*_i}{\sum_{k=1}^m p^*_k}} - \frac{z}{m} \right)$$
  * Where $m$ is the number of outcomes ($m=2$ for NHL moneyline).
* **Bisection Search:** We solve for $z$ numerically by finding the root of:
  $$f(z) = \sum_{i=1}^{m} p_i(z) - 1.0 = 0$$
  Once $z$ is found, we evaluate $p_i(z)$ to obtain the true, vig-free probabilities.

### 2.3 ELO Rating and Logit Calibration
Let $R_h$ and $R_a$ be the ELO ratings of the home and away teams. The ELO-implied home win probability is:
$$P(\text{Home Win}) = \sigma(\beta_0 + \beta_1 \cdot \Delta R)$$
$$\Delta R = R_h - R_a + \text{HomeIceAdvantage}$$
Where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function.
* **Calibration Model (PyMC / Bayesian Logit):**
  $$y_g \sim \text{Bernoulli}(p_g)$$
  $$\text{logit}(p_g) = \beta_0 + \beta_1 \cdot (p_{\text{Shin, home}} - 0.5)$$
  If the market is efficient and ELO is unbiased, the intercept $\beta_0 \approx 0$ and the slope $\beta_1 \approx 1$.

---

## 3. Odds and ELO Calibration Workflow

```mermaid
flowchart TD
    A[Raw Betting Odds\nDK / Pinnacle lines] --> B[Convert American Lines\nto Implied Probs p*]
    B --> C[Compute Overround\nSum p* - 1.0]
    C --> D[Shin Method Solver\nBisection search for z]
    D --> E[Extract Vig-Free\nTrue Probabilities p]
    F[Pre-game Team ELOs] --> G[Compute ELO Differential\nDelta R = R_h - R_a + HomeIce]
    E --> H[Logit Calibration\nPyTorch / PyMC Regression]
    G --> H
    H --> I[Estimate coefficients\nBeta_0 (Bias) & Beta_1 (Scale)]
    I --> J[Evaluate Market Efficiency\nCalculate Brier Scores]
```

---

## 4. Input & Output Schemas

### 4.1 Input Schema (Raw Betting CSV / Volume Table)
* `gameid` (int): Game ID.
* `odds_description` (str): `"Moneyline"`.
* `home_odds_value` (float): American moneyline for home team (e.g. `-150`).
* `away_odds_value` (float): American moneyline for away team (e.g. `130`).
* `bettingPartner` (str): Bookmaker name (e.g. `"DraftKings"`).

### 4.2 Output Schema (Processed Probabilities)
* `gameid` (int): Game ID.
* `odds_home` (float): Implied probability for home team (with vig).
* `odds_away` (float): Implied probability for away team (with vig).
* `true_prob_home` (float): Vig-free true probability for home team (Shin adjusted).
* `true_prob_away` (float): Vig-free true probability for away team (Shin adjusted).
* `z_informed` (float): Solved informed-trader fraction $z$.

---

## 5. Generalized Python Implementation

```python
import math
import numpy as np
import pandas as pd

def american_to_implied(line: float) -> float:
    """Converts American moneyline to implied probability."""
    if line < 0:
        return abs(line) / (abs(line) + 100)
    else:
        return 100 / (line + 100)

def shin_vig_removal(p_star: list, tol: float = 1e-12, max_iter: int = 100) -> tuple:
    """
    Removes overround (vig) using Shin's method for a list of raw implied probabilities.
    Returns (vig_free_probabilities, z_value)
    """
    m = len(p_star)
    s = sum(p_star)
    if s <= 1.0 + 1e-15:
        # No vig to remove
        return [p / s for p in p_star], 0.0

    def sum_adjusted(z: float) -> float:
        total = 0.0
        z_over_m = z / m
        denom = 2.0 * (1.0 - z)
        for p in p_star:
            # normalized raw probabilities inside
            inside = (z_over_m**2) + 4.0 * (1.0 - z) * (p / s)
            total += (math.sqrt(inside) - z_over_m) / denom
        return total

    # Bisection search for z in [0, 1 - eps]
    lo = 0.0
    hi = 1.0 - 1e-10
    
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

    # Calculate final adjusted probabilities
    z_over_m = z / m
    denom = 2.0 * (1.0 - z)
    adjusted = []
    for p in p_star:
        inside = (z_over_m**2) + 4.0 * (1.0 - z) * (p / s)
        adjusted.append((math.sqrt(inside) - z_over_m) / denom)
        
    # Safeguard normalization
    sum_adj = sum(adjusted)
    adjusted = [p / sum_adj for p in adjusted]
    
    return adjusted, z

def process_game_odds(df_odds: pd.DataFrame) -> pd.DataFrame:
    """
    Processes American betting odds to extract true probabilities.
    """
    results = []
    for idx, row in df_odds.iterrows():
        p_home = american_to_implied(row['home_odds_value'])
        p_away = american_to_implied(row['away_odds_value'])
        
        p_star = [p_home, p_away]
        true_probs, z = shin_vig_removal(p_star)
        
        results.append({
            "gameid": row['gameid'],
            "odds_home": p_home,
            "odds_away": p_away,
            "true_prob_home": true_probs[0],
            "true_prob_away": true_probs[1],
            "z_informed": z
        })
        
    return pd.DataFrame(results)
```

---

## 6. Weaknesses, Caveats & Considerations
1. **Three-Way Betting Markets (Tie/Regulation Draw):** In standard regular-season games, moneylines include overtime and shootout results (two-way). However, in European books or 3-way regulation lines (Home/Draw/Away), $m=3$. While the Shin formulation holds for $m=3$, the bisection search limits must be monitored carefully to ensure convergence, as the sum of raw probabilities including regulation draws can have wider overrounds ($1.08$ to $1.12$).
2. **Favorite-Longshot Bias:** Sports betting markets exhibit favorite-longshot bias, where longshots (heavy underdogs) are systematically overvalued by raw implied probabilities. Shin's method corrects this better than simple normalization (proportional adjustment), but it assumes bookmakers set odds *strictly* to balance risk against informed traders. In reality, bookmakers also adjust lines to exploit public sentiment (e.g. popular teams like Toronto or New York tend to have shorter odds than their mathematical win probability warrants).
3. **Pre-game ELO vs. Live Odds Information Gap:** ELO ratings are updated game-by-game. They do not capture late-breaking pre-game adjustments like starting goalie swaps (which severely impact win chances) or player injuries. Therefore, live betting line movements will often diverge from ELO projections.
