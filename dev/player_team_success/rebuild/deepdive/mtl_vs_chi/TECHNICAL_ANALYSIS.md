# Technical Companion & Econometric Decomposition: Montréal Canadiens vs. Chicago Blackhawks Since 2019 (`deepdive/mtl_vs_chi`)

> [!IMPORTANT]
> **Post-Hoc Retrospective Study Notice**: Word count of [`BLOG_POST.md`](file:///Users/jjlee/Documents/github/nhl-analytics/dev/player_team_success/rebuild/deepdive/mtl_vs_chi/BLOG_POST.md) is **564 words** (strictly within the $\le 600$-word ceiling). All parameter estimates are derived from our longitudinal NHL panel ($N = 764$ team-seasons, 2000–2026).

---

## 1. Structural Regression Specification & Parameter Estimates ($N = 764$)

To isolate how much of the post-2019 divergence between Montréal (`MTL`) and Chicago (`CHI`) is attributable to observable structural roster and style decisions versus idiosyncratic variance ($\hat{\varepsilon}_{i,t}$), we estimate the structural performance equation via OLS with White HC1 heteroskedasticity-robust standard errors:

$$\text{PTS}\%_{i,t} = \beta_0 + \beta_1 (s_{B_2,i,t} + s_{B_3,i,t}) + \beta_2 \max(0, s_{B_1,i,t} - 0.30)^2 + \beta_3 \left[s_{B_1,i,t} \ln(1 + \text{VIR}_{i,t})\right] + \beta_4 \text{CCM}_{i,t} + \beta_5 \text{xGF}\%_{i,t} + \beta_6 \text{SPSI}_{i,t} + \varepsilon_{i,t}$$

Source table: [`tables/structural_model_parameters.csv`](file:///Users/jjlee/Documents/github/nhl-analytics/dev/player_team_success/rebuild/deepdive/mtl_vs_chi/tables/structural_model_parameters.csv).

| Parameter | Variable Name | Estimate ($\hat{\beta}$) | Robust SE (HC1) | $t$-Statistic | Economic Interpretation |
|:---|:---|:---:|:---:|:---:|:---|
| $\beta_0$ | `Intercept` | `-0.0369` | `0.0149` | `-2.47` | Baseline intercept |
| $\beta_1$ | `prime_bridge_b2_b3` ($s_{B_2} + s_{B_3}$) | **`+0.0290`** | `0.0114` | **`+2.54`** | Prime Bridge & Peak Core (`4–9 yrs`) direct lift |
| $\beta_2$ | `youth_overload_sq` ($\max(0, s_{B_1}-0.30)^2$) | **`-0.0119`** | `0.0142` | `-0.84` | Non-linear convex penalty when uninsulated ELC TOI $>30\%$ |
| $\beta_3$ | `insulation_interaction` ($s_{B_1} \ln(1+\text{VIR})$) | **`+0.0352`** | `0.0138` | **`+2.55`** | Positive return to ELC youth *when buffered by veteran insulation* |
| $\beta_4$ | `ccm_to_prior_year` ($\text{CCM}_{i,t}$) | **`+0.0044`** | `0.0108` | `+0.41` | Core continuity stabilization effect |
| $\beta_5$ | `xg_share_5v5` ($\text{xGF}\%_{i,t}$) | **`+1.0869`** | `0.0175` | **`+62.23`** | Primary on-ice territorial possession & shot-quality driver |
| $\beta_6$ | `spsi_style_synergy` ($\text{SPSI}_{i,t}$) | **`+0.0132`** | `0.0048` | **`+2.75`** | Complementary boost from proactive O-zone forecheck hitting $\times$ skill |

---

## 2. Complete Season-by-Season Parameter Ledger: `MTL` vs. `CHI` (2019–2025)

Source table: [`tables/mtl_vs_chi_parameter_comparison_2019_2025.csv`](file:///Users/jjlee/Documents/github/nhl-analytics/dev/player_team_success/rebuild/deepdive/mtl_vs_chi/tables/mtl_vs_chi_parameter_comparison_2019_2025.csv).

| Team | Season | Regime State | Actual $\text{PTS}\%$ | Model $\widehat{\text{PTS}}\%$ | Residual $\hat{\varepsilon}$ | $s_{B_1}$ (0–3y) | $s_{B_2}$ (4–6y) | $s_{B_3}$ (7–9y) | $s_{B_4}$ (10+y) | Entropy $H^{\text{exp}}$ | $\text{VIR}$ | $\text{CCM}$ | Draft Core $\%$ | Trade Core $\%$ | 5v5 $\text{xGF}\%$ | O-Zone Hit $\%$ | $\text{SPSI}$ | Surplus (\$M) |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **MTL** | 2019 | `R1_SLUMP` | `.5000` | `.5131` | `-.0131` | `16.8%` | `38.1%` | `18.7%` | `26.5%` | `1.332` | `2.69` | `1.000` | `0.0%` | `100.0%` | `48.9%` | `39.5%` | `.387` | `-4.25` |
| **MTL** | 2020 | `R0_CONTENDER` | `.5268` | `.5265` | `+.0003` | `21.0%` | `12.0%` | `38.5%` | `28.5%` | `1.308` | `3.19` | `0.850` | `0.0%` | `100.0%` | `49.9%` | `43.8%` | `.604` | `+0.42` |
| **MTL** | 2021 | `R2_TEARDOWN` | `.3354` | `.3916` | `-.0562` | `35.9%` | `14.7%` | `24.0%` | `25.5%` | `1.340` | `1.38` | `0.827` | `33.1%` | `66.9%` | `38.0%` | `38.8%` | `.171` | `+10.10` |
| **MTL** | 2022 | `R2_TEARDOWN` | `.4146` | `.3981` | `+.0165` | `31.7%` | `22.4%` | `31.0%` | `14.8%` | `1.345` | `1.44` | `0.622` | `9.6%` | `90.4%` | `38.7%` | `37.1%` | `.171` | `+23.60` |
| **MTL** | 2023 | `R3_INCUBATION` | `.4634` | `.4482` | `+.0152` | `34.8%` | `28.0%` | `11.5%` | `25.8%` | `1.322` | `1.07` | `1.000` | `54.1%` | `45.9%` | `43.1%` | `41.6%` | `.252` | `+16.74` |
| **MTL** | 2024 | **`R4_EMERGENCE`** | **`.5549`** | **`.5341`** | **`+.0208`** | **`28.1%`** | **`29.9%`** | **`13.2%`** | **`28.9%`** | **`1.343`** | **`1.50`** | **`1.000`** | **`56.4%`** | **`43.6%`** | **`50.7%`** | **`47.9%`** | **`.436`** | **`+14.17`** |
| **CHI** | 2019 | `R1_SLUMP` | `.5143` | `.5214` | `-.0071` | `24.4%` | `20.8%` | `24.2%` | `30.7%` | `1.376` | `2.25` | `1.000` | `71.7%` | `28.3%` | `50.0%` | `37.1%` | `.097` | `+10.64` |
| **CHI** | 2020 | `R3_INCUBATION` | `.4911` | `.4767` | `+.0144` | `34.7%` | `20.5%` | `24.7%` | `20.1%` | `1.360` | `1.29` | `0.823` | `64.4%` | `35.6%` | `45.9%` | `36.0%` | `.158` | `+13.84` |
| **CHI** | 2021 | `R2_TEARDOWN` | `.4146` | `.3972` | `+.0174` | `30.5%` | `26.7%` | `22.5%` | `20.3%` | `1.374` | `1.40` | `0.772` | `56.7%` | `43.3%` | `38.6%` | `39.1%` | `.121` | `-1.75` |
| **CHI** | 2022 | `R2_TEARDOWN` | `.3598` | `.3901` | `-.0303` | `25.0%` | `15.9%` | `22.1%` | `37.0%` | `1.340` | `2.36` | `0.639` | `11.6%` | `88.4%` | `38.0%` | `36.0%` | `.099` | `+15.54` |
| **CHI** | 2023 | `R2_TEARDOWN` | `.3171` | `.3904` | `-.0733` | `39.3%` | `19.5%` | `18.0%` | `23.2%` | `1.333` | `1.05` | `0.767` | `50.1%` | `49.9%` | `38.0%` | `37.9%` | `.099` | `+11.10` |
| **CHI** | 2024 | **`R2_TEARDOWN`** | **`.3720`** | **`.3912`** | **`-.0192`** | **`29.8%`** | **`16.0%`** | **`12.3%`** | **`41.9%`** | **`1.276`** | **`1.82`** | **`1.000`** | **`55.3%`** | **`44.7%`** | **`38.0%`** | **`39.9%`** | **`.038`** | **`+19.15`** |

### Continuation into 2025–26 (`latest/team/season/2025_team_season.csv`)
- **Montréal (`MTL`, 2025–26)**: Win Percentage rose further to **`.564`** (`57` wins across regular + postseason tracking), Rolling Pythagorean Expectation (`rpe`) reached **`.538`**, and Rating Percentage Index (`rpi`) hit **`.537`**.
- **Chicago (`CHI`, 2025–26)**: Win Percentage remained depressed at **`.354`** (`29` wins in `82` games), with Rolling Pythagorean Expectation (`rpe`) stuck at **`.375`** and Fenwick differential at **`-85.0`**.

---

## 3. Player-Level Core Audit (2024–25): Why Montréal’s Core Balanced While Chicago’s Hollowed Out

Source table: [`tables/mtl_vs_chi_core_players_2024.csv`](file:///Users/jjlee/Documents/github/nhl-analytics/dev/player_team_success/rebuild/deepdive/mtl_vs_chi/tables/mtl_vs_chi_core_players_2024.csv).

| Franchise | Core Player | Position | NHL Tenure (Yrs) | Experience Cluster | Sourcing Pathway | Draft Overall (Team) | GP | Points | TOI / GP | Est. Cap Hit (\$M) | Est. MRP (\$M) | Net Surplus (\$M) |
|:---:|:---|:---:|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **MTL** | **Mike Matheson** | D | 10 | `B4_10_PLUS` | `ACQUIRED_PRIME` (Trade) | #23 (`FLA`) | 80 | 31 | `25.1` | `$5.62M` | `$7.42M` | **`+$1.80M`** |
| **MTL** | **Lane Hutson** | D | 2 | `B1_0_3` | `DRAFT_TARGET` (Homegrown) | #62 (`MTL`) | 82 | **66** | `22.7` | `$0.92M` | `$10.32M` | **`+$9.40M`** |
| **MTL** | **Nick Suzuki** | F | 6 | `B2_4_6` | `ACQUIRED_PRIME` (Trade) | #13 (`VGK`) | 82 | **89** | `20.1` | `$7.88M` | `$11.37M` | **`+$3.49M`** |
| **MTL** | **Cole Caufield** | F | 5 | `B2_4_6` | `DRAFT_TARGET` (Homegrown) | #15 (`MTL`) | 82 | **70** | `18.1` | `$7.85M` | `$10.17M` | **`+$2.32M`** |
| **MTL** | **Juraj Slafkovský** | F | 3 | `B1_0_3` | `DRAFT_TARGET` (Homegrown) | #1 (`MTL`) | 79 | **51** | `17.0` | `$0.95M` | `$5.65M` | **`+$4.70M`** |
| **CHI** | **Alex Vlasic** | D | 4 | `B2_4_6` | `DRAFT_TARGET` (Homegrown) | #43 (`CHI`) | 82 | 30 | `23.2` | `$4.60M` | `$6.52M` | **`+$1.92M`** |
| **CHI** | **Connor Bedard** | F | 2 | `B1_0_3` | `DRAFT_TARGET` (Homegrown) | #1 (`CHI`) | 82 | **67** | `20.3` | `$0.95M` | `$7.74M` | **`+$6.79M`** |
| **CHI** | **Connor Murphy** | D | 12 | `B4_10_PLUS` | `ACQUIRED_PRIME` (Trade) | #20 (`ARI`) | 68 | 19 | `20.4` | `$4.40M` | `$4.64M` | `+$0.24M` |
| **CHI** | **Teuvo Teräväinen** | F | 12 | `B4_10_PLUS` | `DRAFT_TARGET` (Returned UFA) | #18 (`CHI`) | 82 | 58 | `16.8` | `$5.40M` | `$6.37M` | `+$0.97M` |
| **CHI** | **Tyler Bertuzzi** | F | 9 | `B3_7_9` | `ACQUIRED_PRIME` (UFA) | #58 (`DET`) | 82 | 46 | `15.5` | `$5.50M` | `$5.74M` | `+$0.24M` |
| **CHI** | **Ryan Donato** | F | 8 | `B3_7_9` | `ACQUIRED_PRIME` (UFA) | #56 (`BOS`) | 80 | 62 | `15.3` | `$2.00M` | `$3.11M` | `+$1.11M` |
