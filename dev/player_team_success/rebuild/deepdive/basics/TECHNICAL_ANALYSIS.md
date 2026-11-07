# Technical Companion & Ad-Hoc Econometric Analysis: Quantifying Generational Struggle vs. Active Rebuilds (`deepdive/basics`)

> [!IMPORTANT]
> **Post-Hoc Retrospective Study Notice**: All regime classifications (`Peak`, `Rebuild`, `Inaction` and granular sub-states `R0`–`R5`) are retrospective econometric measurements over the 2000–2026 NHL sample ($N = 764$ team-seasons). Word count of [`BLOG_POST.md`](file:///Users/jjlee/Documents/github/nhl-analytics/dev/player_team_success/rebuild/deepdive/basics/BLOG_POST.md) is **538 words** (strictly within the $\le 600$-word ceiling).

---

## 1. Mathematical Identification of Macro-Regimes and Granular Sub-States

In single-season hockey analytics, observed standings points percentage ($\text{PTS}\%_{i,t}$) is a noisy signal contaminated by shooting percentage and goaltending variance:

$$\text{PTS}\%_{i,t} = \mu(Z_{i,t}) + \varepsilon_{i,t}, \qquad Z_{i,t} \in \{\text{Inaction}, \text{Rebuild}, \text{Peak}\}$$

To prevent conflating a one-year injury slump with a genuine structural rebuild—or mistaking chronic front-office stagnation for "patient rebuilding"—our state-space framework identifies the latent regime $Z_{i,t}$ by jointly observing **competitive output** ($\text{PTS}\%_{i,t}$, Playoff Rounds Won, 5v5 $\text{xGF}\%$) and **management structural intervention** ($\text{ALI}_{i,t}$, $\text{YIS}_{i,t}$, $\text{RCI}_{i,t}$, and spell duration $d_{i,t}$):

```mermaid
flowchart LR
    subgraph Inputs["Observed Franchise Telemetry (Season t)"]
        A["Competitive Output\nPTS%, Playoff Wins, xGF%"]
        B["Management Action\nALI (Liquidation), RCI (Churn)"]
        C["Demographic Shift\nYIS (Youth Influx), ELC TOI%"]
    end

    subgraph SubStates["Granular 6-State Filter (R0–R5)"]
        R0["R0: Contender\n(PTS% >= .550 or Series Win)"]
        R4["R4: Emergence\n(Post-Rebuild Breakout)"]
        R2["R2: Teardown\n(High ALI + High Churn)"]
        R3["R3: Incubation\n(High ELC/Draft Maturation)"]
        R1["R1: Slump\n(Sub-.550, Low Action, Yr 1-2)"]
        R5["R5: Rebuild Purgatory\n(>=5 Yrs Stagnant / Failed Exit)"]
    end

    subgraph Macro["3 Macro Equilibrium Regimes"]
        P["PEAK REGIME\n(46.0% of Team-Seasons)"]
        RB["REBUILD REGIME\n(28.8% of Team-Seasons)"]
        IN["INACTION REGIME\n(25.1% of Team-Seasons)"]
    end

    Inputs --> SubStates
    R0 --> P
    R4 --> P
    R4 --> RB
    R2 --> RB
    R3 --> RB
    R1 --> IN
    R5 --> IN
```

---

## 2. Season-by-Season Telemetry: Montréal (`MTL`) vs. Chicago (`CHI`), 2018–2025

Source table: [`tables/mtl_chi_regime_identification_2019_2025.csv`](file:///Users/jjlee/Documents/github/nhl-analytics/dev/player_team_success/rebuild/deepdive/basics/tables/mtl_chi_regime_identification_2019_2025.csv).

| Team | Season | $\text{PTS}\%$ | Pyth Win $\%$ | 5v5 $\text{xGF}\%$ | Playoff Rds | Macro Regime | Granular State | $\text{ALI}$ (Liquidation) | $\text{YIS}$ (Youth Influx) | $\text{RCI}$ (Churn) | ELC ($B_1$) TOI $\%$ | Bridge ($B_2$) TOI $\%$ | O-Zone Hit $\%$ | Style Synergy ($\text{SPSI}$) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **MTL** | 2018–19 | `.585` | `.568` | `53.2%` | 0 | **`Peak`** | `R0_CONTENDER` | `0.165` | `0.760` | `0.408` | `17.9%` | `26.9%` | `40.4%` | `0.579` |
| **MTL** | 2019–20 | `.500` | `.465` | `48.9%` | 0 | **`Inaction`** | `R1_SLUMP` | `0.089` | `0.625` | `0.470` | `16.8%` | `38.1%` | `39.5%` | `0.387` |
| **MTL** | 2020–21 | `.527` | `.481` | `49.9%` | **3** | **`Peak`** | `R0_CONTENDER` | `0.194` | `0.667` | `0.379` | `21.0%` | `12.0%` | `43.8%` | `0.604` |
| **MTL** | 2021–22 | `.335` | `.266` | `38.0%` | 0 | **`Rebuild`** | `R2_TEARDOWN` | `0.333` | `0.750` | `0.480` | `35.9%` | `14.7%` | `38.8%` | `0.171` |
| **MTL** | 2022–23 | `.415` | `.342` | `38.7%` | 0 | **`Rebuild`** | `R2_TEARDOWN` | `0.365` | `0.963` | `0.468` | `31.7%` | `22.4%` | `37.1%` | `0.171` |
| **MTL** | 2023–24 | `.463` | `.389` | `43.1%` | 0 | **`Rebuild`** | `R3_INCUBATION` | `0.263` | `0.987` | `0.366` | `34.8%` | `28.0%` | `41.6%` | `0.252` |
| **MTL** | 2024–25 | **`.555`** | **`.471`** | **`50.7%`** | 0 | **`Rebuild`** | **`R4_EMERGENCE`** | **`0.114`** | **`1.000`** | **`0.366`** | **`28.1%`** | **`29.9%`** | **`47.9%`** | **`0.436`** |
| **CHI** | 2018–19 | `.512` | `.467` | `47.2%` | 0 | **`Rebuild`** | `R3_INCUBATION` | `0.156` | `0.818` | `0.471` | `26.5%` | `22.2%` | `32.5%` | `0.171` |
| **CHI** | 2019–20 | `.514` | `.487` | `50.0%` | 0 | **`Inaction`** | `R1_SLUMP` | `0.113` | `0.715` | `0.461` | `24.4%` | `20.8%` | `37.1%` | `0.097` |
| **CHI** | 2020–21 | `.491` | `.407` | `45.9%` | 0 | **`Rebuild`** | `R3_INCUBATION` | `0.378` | `0.874` | `0.461` | `34.7%` | `20.5%` | `36.0%` | `0.158` |
| **CHI** | 2021–22 | `.415` | `.317` | `38.6%` | 0 | **`Rebuild`** | `R2_TEARDOWN` | `0.329` | `0.780` | `0.462` | `30.5%` | `26.7%` | `39.1%` | `0.121` |
| **CHI** | 2022–23 | `.360` | `.266` | `38.0%` | 0 | **`Rebuild`** | `R2_TEARDOWN` | `0.387` | `0.598` | `0.578` | `25.0%` | `15.9%` | `36.0%` | `0.099` |
| **CHI** | 2023–24 | `.317` | `.212` | `38.0%` | 0 | **`Rebuild`** | `R2_TEARDOWN` | `0.486` | `0.934` | `0.608` | `39.3%` | `19.5%` | `37.9%` | `0.099` |
| **CHI** | 2024–25 | **`.372`** | **`.336`** | **`38.0%`** | 0 | **`Rebuild`** | **`R2_TEARDOWN`** | **`0.307`** | **`0.998`** | **`0.516`** | **`29.8%`** | **`16.0%`** | **`39.9%`** | **`0.038`** |

---

## 3. Ad-Hoc Comparison Across the 2020s Rebuilding Cohort (9 Franchises, 2019–2025)

Source table: [`tables/cohort_2020s_rebuilds_comparison.csv`](file:///Users/jjlee/Documents/github/nhl-analytics/dev/player_team_success/rebuild/deepdive/basics/tables/cohort_2020s_rebuilds_comparison.csv).

| Franchise | 2019 $\text{PTS}\%$ | Trough $\text{PTS}\%$ (Season) | 2024 $\text{PTS}\%$ | Rebound From Trough | Net Change (2019–24) | Mean $\text{ALI}$ | Mean $\text{YIS}$ | Mean $B_1$ (ELC) $\%$ | Mean $B_2$ (Bridge) $\%$ | `Inaction` Seasons | `Rebuild` Seasons | 2024–25 State |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **DET** | `.275` | `.275` (2019) | `.524` | `+.250` | `+.250` | `0.274` | `0.849` | `24.8%` | `27.3%` | 3 | 3 | `R5_PURGATORY` (`Inaction`) |
| **MTL** | `.500` | `.335` (2021) | **`.555`** | **`+.219`** | **`+.055`** | `0.226` | `0.832` | `28.0%` | `24.2%` | 1 | 4 | **`R4_EMERGENCE` (`Rebuild`)** |
| **UTA (ex-ARI)** | `.529` | `.348` (2021) | `.543` | `+.195` | `+.014` | `0.277` | `0.825` | `21.2%` | `29.5%` | 3 | 3 | `R5_PURGATORY` (`Inaction`) |
| **CBJ** | `.579` | `.360` (2022) | `.543` | `+.183` | `-.036` | `0.243` | `0.768` | `28.5%` | `25.9%` | 1 | 4 | `R5_PURGATORY` (`Inaction`) |
| **OTT** | `.437` | `.437` (2019) | **`.592`** | `+.155` | `+.155` | `0.327` | `0.852` | `24.1%` | `41.8%` | 3 | 2 | **`R4_EMERGENCE` (`Peak`)** |
| **BUF** | `.493` | `.330` (2020) | `.482` | `+.151` | `-.011` | `0.219` | `0.928` | `25.1%` | `38.3%` | **6** | **0** | `R5_PURGATORY` (`Inaction`) |
| **ANA** | `.472` | `.354` (2022) | `.488` | `+.134` | `+.016` | `0.243` | `0.830` | `25.9%` | `21.1%` | 1 | 5 | `R5_PURGATORY` (`Inaction`) |
| **CHI** | `.514` | `.317` (2023) | **`.372`** | **`+.055`** | **`-.142`** | **`0.333`** | `0.816` | **`30.6%`** | **`19.9%`** | 1 | 5 | **`R2_TEARDOWN` (`Rebuild`)** |
| **SJS** | `.450` | `.287` (2023) | `.317` | `+.030` | `-.133` | `0.301` | `0.689` | `22.6%` | `29.0%` | 2 | 4 | `R2_TEARDOWN` (`Rebuild`) |

### Key Analytical Takeaways
1. **Why Only 2 of 9 Rebuilding Teams Reached `R4_EMERGENCE` (`MTL` and `OTT`)**: Both Montréal (`.555`) and Ottawa (`.592`) transitioned out of asset liquidation (`ALI` dropping below `0.150` by 2024–25) while expanding their **4–6 year NHL experience bridge cohort ($B_2 \ge 29.9\%$)** and elevating their proactive forecheck hitting above **$45\%$**.
2. **Why Detroit, Buffalo, Anaheim, and Columbus Slid into `R5_PURGATORY` (`Inaction`)**: Once a rebuild exceeds 4–5 seasons without crossing `.550` Points % or winning a playoff round, incremental youth stops adding marginal wins and the franchise becomes trapped in a stationary sub-contender equilibrium.
3. **Why Chicago (`CHI`) and San Jose (`SJS`) Remained in `R2_TEARDOWN`**: Chicago maintained the highest mean Asset Liquidation Index ($\text{ALI} = 0.333$) and highest ELC youth share ($30.6\%$), coupled with the lowest Prime Bridge share ($19.9\%$), preventing the roster from transitioning into `R3_INCUBATION`.
