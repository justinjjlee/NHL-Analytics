# Skillset 5: Decision Science & Latent Momentum SSM

This skillset covers play-by-play decision science: evaluating conditional scoring probabilities of faceoffs and shot blocks, and formulating a continuous-time Kalman State-Space Model (SSM) to estimate latent team momentum.

---

## 1. Scope & Target Measurement Level
* **Measurement Level:** Play-by-Play Level (ticks, micro-events), Game Level.
* **Objective:** Extract structural value from specific tactical choices (faceoffs, blocks) and estimate unobserved, time-varying game momentum.

---

## 2. Mathematical Formulations

### 2.1 Conditional Scoring Probabilities (Faceoffs)
Let $FO \in \{-1, 1\}$ be the faceoff outcome ($1$ if home team wins, $-1$ if lost), and $Z_0 \in \{\text{Offense}, \text{Defense}\}$ be the ice zone posture. Let $G \in \{-1, 0, 1\}$ be the goal outcome of the subsequent play sequence.
* **Conditional Goal Probability:**
  $$P(G = g \mid FO = f,\, Z_0 = z)$$
* **Bootstrap Validation:** Because goals are rare events, we resample game sequences $B = 10,000$ times to obtain robust confidence intervals for the probability differentials:
  $$\Delta P_g = P(G=1 \mid FO=1,\, z) - P(G=1 \mid FO=-1,\, z)$$

### 2.2 Empirical CDF & Kolmogorov-Smirnov Test (Blocked Shots)
We test if a blocked shot alters the subsequent goal-scoring probability compared to standard missed shots or shots-on-goal.
* **Empirical CDF ($F_n$):**
  $$F_n(x) = \frac{1}{n} \sum_{i=1}^n \mathbb{I}(X_i \le x)$$
* **Two-Sample Kolmogorov-Smirnov Test:**
  $$D = \sup_x |F_{1, n_1}(x) - F_{2, n_2}(x)|$$
  * *Hypothesis:* $H_0: F_{\text{blocked}} = F_{\text{unblocked}}$ vs $H_1: F_{\text{blocked}} \neq F_{\text{unblocked}}$.
* **Bayesian Parametric Density (Maxwell-Boltzmann):**
  $$f(x \mid \mu, \sigma) = \sqrt{\frac{2}{\pi}} \frac{(x - \mu)^2}{\sigma^3} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right) \quad \text{for } x \ge \mu$$

### 2.3 Latent Momentum State-Space Model (SSM)
We estimate an unobserved momentum state $x_t \in \mathbb{R}$ for a focal team at event tick $t$.
* **State Transition Equation (Continuous-Time Decay):**
  $$x_t = A_t \cdot x_{t-1} + \mathbf{B} \cdot \mathbf{u}_t + w_t, \qquad w_t \sim \mathcal{N}(0,\, Q)$$
  * $A_t = a_0^{\Delta t_t}$: Time-varying decay, where $a_0 \in (0,1)$ is the per-second decay rate, and $\Delta t_t$ is the elapsed seconds between event $t$ and $t-1$.
  * $\mathbf{u}_t \in \mathbb{R}^9$: Binary event vector at tick $t$ (hits for/against, shots for/against, takeaways, giveaways, blocks for/against, faceoff win).
  * $\mathbf{B}$: Vector of event-impact weights to be estimated.
* **Observation Equation (Team Strength Adjusted):**
  $$y_t = C \cdot x_t + \mathbf{D} \cdot \mathbf{z}_t + v_t, \qquad v_t \sim \mathcal{N}(0,\, R)$$
  * $y_t$: Observed proxy (rolling 90-second offensive zone time share).
  * $\mathbf{z}_t$: Pre-game team strength covariates: $[\Delta\text{ELO},\, \Delta\text{PyExp},\, \text{score\_state}_t]^T$.
  * $\mathbf{D}$: Baselines weights adjusting for team quality and score state.
* **Initial State Prior:**
  $$x_0 \sim \mathcal{N}(\mu_0,\, P_0) \qquad \text{where } \mu_0 = \alpha \cdot \Delta\text{ELO} + \beta \cdot \Delta\text{PyExp}$$

---

## 3. Kalman SSM Recursion Workflow

```mermaid
flowchart TD
    A[Start Game Event Loop] --> B[Initialize x_0 = mu_0, P_0]
    B --> C[Fetch event t\nGet delta_t, binary vector u_t]
    C --> D[Predict State\nx_pred = a_0^delta_t * x + B * u_t\nP_pred = a_0^2*delta_t * P + Q]
    D --> E[Observation Update\ny_hat = C * x_pred + D * z_t\ninnov = y_t - y_hat]
    E --> F[Kalman Gain\nS = C^2 * P_pred + R\nK = P_pred * C / S]
    F --> G[Update State\nx = x_pred + K * innov\nP = (1 - K * C) * P_pred]
    G --> H[Accumulate Negative Log-Likelihood\nNLL = NLL + 0.5 * log S + innov^2 / 2S]
    H --> I{Next Event?}
    I -->|Yes| C
    I -->|No| J[Gradient Step\nOptimize theta via backprop\nin PyTorch]
```

---

## 4. Input & Output Schemas

### 4.1 Input Schema (Master Event Parquet)
* `gameid` (int): Game ID.
* `elapsed_sec` (int): Seconds elapsed in game.
* `delta_t` (float): Time gap since last event in seconds.
* `u_t` (tensor, shape `[9]`): Binary event vector.
* `y_t` (float): Rolling 90s offensive zone share (value in $[0, 1]$).
* `delta_elo` (float): Pre-game ELO difference.
* `delta_pyexp` (float): Pre-game Pythagorean differential.
* `score_state` (int): Goal differential, clipped to $[-2, 2]$.

### 4.2 Output Schema (Estimated Parameters)
* `a_0` (float): Learned per-second momentum decay rate.
* `B` (array, shape `[9]`): Event impact coefficients.
* `C` (float): Latent state loading.
* `D` (array, shape `[3]`): Strength adjustments.
* `x_hat` (array): Smoothed latent momentum path for each event.

---

## 5. Generalized PyTorch Implementation

```torch
import torch
import torch.nn as nn

class MomentumSSM(nn.Module):
    """
    Pytorch implementation of the hockey momentum State-Space Model.
    Learns parameters directly from sequential event tensors via backprop.
    """
    def __init__(self, n_events: int = 9, n_strength: int = 3):
        super().__init__()
        # Continuous decay parameter (learned in log-space)
        self.log_a0 = nn.Parameter(torch.tensor(-0.04)) # initial a0 ~ 0.96
        
        # Event weights (B)
        self.B = nn.Parameter(torch.zeros(n_events))
        
        # Observation loading (C)
        self.log_C = nn.Parameter(torch.tensor(0.0))
        
        # Strength baseline adjustments (D)
        self.D = nn.Parameter(torch.zeros(n_strength))
        
        # Prior parameters (alpha, beta)
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.beta = nn.Parameter(torch.tensor(0.0))
        
        # Noise variances (Q, R) and initial variance (P0)
        self.log_Q = nn.Parameter(torch.tensor(-1.0))
        self.log_R = nn.Parameter(torch.tensor(-1.0))
        self.log_P0 = nn.Parameter(torch.tensor(0.0))

    def forward(self, U: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        Runs the Kalman filter forward pass.
        Returns the negative log-likelihood (NLL) for gradient optimization.
        
        U: [T, 9] event vector
        Y: [T] observed zone share
        Z: [T, 3] ELO, PyExp, score_state
        dt: [T] time gaps (sec)
        """
        a0 = torch.sigmoid(self.log_a0)
        C = torch.exp(self.log_C)
        Q = torch.exp(self.log_Q)
        R = torch.exp(self.log_R)
        P0 = torch.exp(self.log_P0)
        
        # Compute prior mean from ELO & PyExp
        mu0 = self.alpha * Z[0, 0] + self.beta * Z[0, 1]
        x = mu0
        P = P0
        nll = torch.tensor(0.0)
        
        for t in range(len(U)):
            # Continuous decay coefficient
            A_t = a0 ** dt[t]
            
            # Predict step
            x_pred = A_t * x + torch.dot(self.B, U[t])
            P_pred = (A_t ** 2) * P + Q
            
            # Update step
            y_hat = C * x_pred + torch.dot(self.D, Z[t])
            innov = Y[t] - y_hat
            S = (C ** 2) * P_pred + R # Innovation variance
            K = P_pred * C / S        # Kalman Gain
            
            x = x_pred + K * innov
            P = (1.0 - K * C) * P_pred
            
            # Accumulate likelihood
            nll += 0.5 * (torch.log(S) + (innov ** 2) / S)
            
        return nll
```

---

## 6. Weaknesses, Caveats & Considerations
1. **Collinearity of Actions:** Hits, takeaways, and blocked shots frequently occur in quick succession during defensive scrambles. Estimating disaggregated weights ($\mathbf{B}$) requires massive game samples to avoid high parameter variance and ensure identification.
2. **Rink-specific Tracking Bias:** Play-by-play event counts (especially hits and giveaways) are recorded by home-arena statisticians. It is a documented NHL phenomenon that certain rinks (e.g. Montreal, Long Island) systematically over-report hits by up to 30% compared to others. This introduces measurement error into $\mathbf{u}_t$ that can inflate the estimated hit parameter.
3. **Score-Effect Confounding (H4):** Teams that are trailing hit more because they lack possession and are chasing the puck. Conversely, leading teams dump and chase, conceding possession metrics. Stratifying model parameters by `score_state` is mandatory to avoid the spurious conclusion that "hits cause teams to lose."
