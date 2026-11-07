# Executive Summary: 26-Year NHL Franchise Rebuild & Regime-Switching Study (2000–2026)

> [!IMPORTANT]
> ### Epistemological Notice: Post-Hoc Retrospective Historical Evaluation
> This report presents a **post-hoc, retrospective historical investigation** of 764 franchise-seasons and 97 rebuild/retool episodes across the 2000-01 through 2025-26 NHL seasons. Grounded in time-series **Markov-Switching State-Space & Dynamic Equilibrium Econometrics** (Hamilton, 1989; Kim, 1994), every franchise-season is classified retrospectively into one of **Three Macro Equilibrium Regimes (`Inaction`, `Rebuild`, `Peak`)**.
>
> **Not a Future Forecast**: Because regime classification looks back across the completed historical window to observe whether structural changes ultimately culminated in `Peak` contention or stagnated in `Inaction`, **these findings are in no way deterministic forecasts or guarantees of future franchise behaviors or trajectories**. However, the estimated transition parameters reveal general economic regularities—quantifying why specific front-office decisions have historically proven **effective or tragically ineffective** in moving a franchise from the `Inaction` and `Rebuild` regimes into the `Peak` regime.

---

## 1. The Three Macro Equilibrium Regimes (`Inaction`, `Rebuild`, `Peak`) — 26-Year Census

Across all **764 franchise-seasons** measured between 2000-01 through 2025-26:
1. **`Peak` Regime (45.8% of NHL team-seasons)**: Successful competitive seasons (`R0_CONTENDER` and sustained `R4_EMERGENCE`) where a franchise wins playoff series, posts elite regular-season points percentages ($\text{PTS}\% \ge .605$), or sustains a $\ge 4$-year playoff window with an intact Core Player Group ($\text{CCM} \ge 0.60$).
2. **`Rebuild` Regime (28.8% of NHL team-seasons)**: Unsuccessful or transitional seasons (`R2_TEARDOWN`, `R3_INCUBATION`, and initial `R4_EMERGENCE`) where management is **actively executing or incubating major structural changes across years**—liquidating veteran assets ($\text{ALI} \ge 0.49$) and elevating young/retool talent ($\text{YIS} \ge 0.54$) hoping to transition into a future `Peak` regime.
3. **`Inaction` Regime (25.4% of NHL team-seasons)**: Unsuccessful seasons where a franchise fails to contend and **either makes no major structural changes to improve (and has nothing to show for it)** (`R1_SLUMP` mediocrity / first-round treadmill) or stagnates in chronic **Rebuild Purgatory** (`R5_PURGATORY`, $\ge 5$ consecutive lottery seasons or $\ge 7$ years without a series win).

---

## 2. Historical Actuarial Odds of Moving from `Rebuild` to `Peak`

Evaluating all **82 completed rebuild/retool cycles** (out of 97 total episodes):
- **Tier 1 Success (Sustained Playoff Viability $\ge 3$ yrs, Core Intact)**: **70.7%** (58 episodes)
- **Tier 2 Success (Conference Finals / Final Four Contention)**: **54.9%** (45 episodes)
- **Tier 3 Success (Stanley Cup Championship)**: **14.6%** (12 episodes)
- **Tragically Ineffective Collapse into `Inaction` / Purgatory ($\ge 5$ yrs stagnant)**: **29.3%** (24 episodes)
- **Median Rebuild Duration**: **4.0 seasons** from initial teardown/trough to regime resolution.

---

## 3. Why Rebuilds Succeed vs. Why They Tragically Fail (Econometric Synthesis)

| Strategic Dimension | Effective Action (`Rebuild` $\to$ `Peak`) | Tragically Ineffective Action (`Rebuild` $\to$ `Inaction`) | Empirical Parameter / Effect Size |
|:---|:---|:---|:---|
| **1. Core Positional Spine** | Anchor reconstruction around a **#1 Center and Top-Pair Defenseman** ($\ge 65\%$ of core TOI) | Centering rebuild around wingers or volatile goaltending without blueline spine | $\beta_C + \beta_D = 0.730^{***}$ ($73\%$ of contention lift) |
| **2. Sourcing & Cap Window** | **Hybrid Draft + Prime Trade Arbitrage** (acquiring suppressed 23–26 yr skaters via $\psi_2 > 0$) | Pure multi-year draft tanking ($>4$ yrs) where early ELC stars hit \$9M+ RFA extensions before depth arrives | $\hat{\gamma}_{\text{trade}} = +0.432 > \hat{\gamma}_{\text{draft}}^{(3)} = +0.246$ |
| **3. Age & Experience Balance** | Insulate `0–3 yr` ELC youth ($s_{B_1} \le 28\%$) with `7–9 yr` peak veterans ($\text{VIR} \ge 0.80$) | **Youth Overload Trap**: Deploying $>30\%$ TOI to uninsulated rookies ($\text{VIR} < 0.40$) | $-\lambda_{\text{youth\_overload}} = -0.342^{***}$, $\vartheta_{B_1, B_3} = +0.268^{***}$ |
| **4. Playing Style & Hit Synergy** | Pair high-danger skill ($\text{xGF}\%$) with **proactive O-zone forecheck hits** ($\Pi_{\text{rec}}(O)=54.15\%$) | Accumulating **reactive D-zone chasing hits** ($\Pi_{\text{rec}}(D)=39.98\%$, $-17.71$ net SOG/100 hits) | $\hat{B}_{\text{hit\_O}} = +0.3765$, $z_{4, \text{syn}}^{\text{Playoffs}} = +0.384^{***}$ |
| **5. Management Posture** | Decisive asset reallocation or disciplined compositional retool | **Passive `Inaction`** (refusing to restructure a mediocre aging roster) or mid-rebuild GM panic trades | $\zeta_{\text{Inact}\to\text{Inact}} = +1.340^{***}$ (high stagnation inertia) |
