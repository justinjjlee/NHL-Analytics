# Technical Companion & Era-Shift Econometrics: Why Hybrid Trade & Cap-Arbitrage Rebuilds Outperform Pure Tanking (`deepdive/modern_shift`)

> [!IMPORTANT]
> **Post-Hoc Retrospective Study Notice**: Word count of [`BLOG_POST.md`](file:///Users/jjlee/Documents/github/nhl-analytics/dev/player_team_success/rebuild/deepdive/modern_shift/BLOG_POST.md) is **561 words** (strictly within the $\le 600$-word ceiling). All archetype comparisons and era shifts are computed across $N = 82$ completed NHL rebuild episodes and $N = 764$ team-seasons (2000–2026).

---

## 1. Econometric Comparison of the Three Rebuild Strategy Archetypes (2000–2025)

In our framework, each rebuild episode $e$ is classified into one of three strategic archetypes based on its peak Asset Liquidation Index ($\text{ALI}$), Expected Draft Value invested ($\text{EDV}$), and Core Sourcing ratio ($\text{Pathway A: Homegrown Draft}$ vs. $\text{Pathway B: External Trade/UFA}$):

1. **`SCORCHED_EARTH_DRAFT`**: Deep multi-year liquidation ($\text{EDV} \ge 600$, multiple top-5/top-10 lottery picks, $\ge 60\%$ homegrown draft core target).
2. **`HYBRID_REBUILD`**: Targeted 3–4 year reset combining high-surplus ELC draft hits with aggressive external trades for 22–26 year-old prime bridge players ($\approx 46\%$ Draft / $\approx 54\%$ Trade core share).
3. **`COMPOSITIONAL_RETOOL`**: Rapid 1–3 year on-the-fly core re-architecture without bottoming out into a multi-year lottery teardown ($\text{EDV} \approx 257$, preserving veteran insulation $\text{VIR} > 2.2$).

Source table: [`tables/era_shift_rebuild_outcomes_2000_2025.csv`](file:///Users/jjlee/Documents/github/nhl-analytics/dev/player_team_success/rebuild/deepdive/modern_shift/tables/era_shift_rebuild_outcomes_2000_2025.csv).

| Strategy Archetype | Completed Episodes ($N$) | Mean Duration (Yrs) | Median Duration (Yrs) | Pathway A: Draft Core $\%$ | Pathway B: Trade/UFA Core $\%$ | Tier 1: Playoff Core Rate | Tier 2: Final Four Rate | Tier 3: Stanley Cup Rate | Rebuild Purgatory Failure Rate | Mean Draft Capital ($\text{EDV}$) | Mean Net Surplus (\$M) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`SCORCHED_EARTH_DRAFT`** | 40 | **`5.38`** | `5.0` | `67.1%` | `31.3%` | `45.0%` | `37.5%` | `17.5%` | **`55.0%`** | `821.4` | `$82.2M` |
| **`HYBRID_REBUILD`** | 12 | **`4.17`** | `4.0` | `46.3%` | **`53.7%`** | **`100.0%`** | **`75.0%`** | `8.3%` | **`0.0%`** | `603.3` | `$52.5M` |
| **`COMPOSITIONAL_RETOOL`** | 30 | **`2.10`** | `2.0` | `65.6%` | `34.0%` | **`93.3%`** | **`70.0%`** | `13.3%` | **`6.7%`** | `256.9` | `$29.2M` |

---

## 2. Structural Evolution Across the Four NHL Cap Eras (2000–2025)

Source table: [`tables/era_evolution_summary.csv`](file:///Users/jjlee/Documents/github/nhl-analytics/dev/player_team_success/rebuild/deepdive/modern_shift/tables/era_evolution_summary.csv).

| NHL Structural Era | Completed Episodes ($N$) | Mean Duration (Yrs) | Pathway A: Draft Core $\%$ | Pathway B: Trade Core $\%$ | Playoff Core Success Rate | Purgatory Failure Rate | Structural Mechanism |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **Era 1: Pre-Cap (`2000–04`)** | 19 | `4.05` | `71.1%` | `24.7%` | `84.2%` | `15.8%` | Uncapped payrolls allowed rapid veteran purchases |
| **Era 2: Early Cap (`2005–13`)** | 29 | `4.31` | `60.8%` | `39.2%` | `72.4%` | `27.6%` | ELC rules created the original PIT/CHI/LAK tank window |
| **Era 3: Speed & Compression (`2013–20`)** | 25 | `3.56` | `63.5%` | `36.5%` | `72.0%` | `28.0%` | Flattened lottery odds + expensive RFA bridge/2nd contracts |
| **Era 4: Flat-Cap & Arbitrage (`2020–25`)** | 9 | `4.11` | `56.3%` | `43.7%` | `33.3%` | `66.7%` | Cap-space arbitrage trades (FLA, VGK, CAR, MTL) dominate |

---

## 3. Modern Franchise Comparison Matrix (2021–2025): Contenders vs. Rebuilders

Source table: [`tables/modern_rebuilds_trade_vs_draft_matrix.csv`](file:///Users/jjlee/Documents/github/nhl-analytics/dev/player_team_success/rebuild/deepdive/modern_shift/tables/modern_rebuilds_trade_vs_draft_matrix.csv).

| Franchise | 2021–25 Mean $\text{PTS}\%$ | 2024–25 $\text{PTS}\%$ | Total Playoff Series Won (2021–25) | Pathway A: Homegrown Draft Core $\%$ | Pathway B: External Trade/UFA Core $\%$ | Prime Bridge ($B_2$: 4–6y) Share | ELC Youth ($B_1$: 0–3y) Share | Avg Annual Net Surplus (\$M) | Style Synergy ($\text{SPSI}$) | Strategic Classification |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **FLA** | `.643` | `.598` | **12** (2 Cups) | `32.5%` | **`67.5%`** | `29.5%` | `10.2%` | `+$5.20M` | `0.738` | Elite Trade/Arbitrage Contender |
| **VGK** | `.630` | `.671` | **5** (1 Cup) | `3.8%` | **`96.2%`** | `20.7%` | `8.7%` | `+$3.91M` | `0.647` | Aggressive Trade Acquisition Engine |
| **CAR** | `.669` | `.604` | **6** | `55.2%` | **`44.8%`** | `17.6%` | `8.5%` | `+$6.47M` | `0.939` | Balanced Hybrid + Forecheck System |
| **DAL** | `.648` | `.646` | **6** | `65.3%` | `34.7%` | `26.0%` | `19.1%` | `+$13.18M` | `0.421` | Multi-Wave Draft + Veteran Insulation |
| **COL** | `.666` | `.622` | **5** (1 Cup) | `55.9%` | **`44.1%`** | `26.8%` | `9.5%` | `+$10.60M` | `0.539` | Elite Draft Spine + Targeted Pro Trades |
| **MTL** | `.442` | **`.555`** | 0 | `38.3%` | **`61.7%`** | `23.7%` (`29.9%` in '24) | `32.6%` (`28.1%` in '24) | **`+$16.15M`** | `0.258` (`0.436` in '24) | **Emerging Hybrid Trade + ELC Breakout** |
| **OTT** | `.509` | **`.592`** | 0 | `67.8%` | `32.2%` | `41.2%` | `22.7%` | `+$6.14M` | `0.543` | Matured Bridge Core Emergence |
| **DET** | `.505` | `.524` | 0 | `62.8%` | `37.2%` | `23.9%` | `24.8%` | `+$13.66M` | `0.324` | Stalled in `R5_PURGATORY` (Yr 8) |
| **BUF** | `.502` | `.482` | 0 | `54.9%` | `45.1%` | `43.6%` | `24.3%` | `+$11.51M` | `0.123` | Chronic `Inaction` / Low Style Synergy |
| **ANA** | `.416` | `.488` | 0 | `62.4%` | `37.6%` | `21.6%` | `25.3%` | `+$5.90M` | `0.169` | Prolonged Draft Incubation (`R5_PURGATORY`) |
| **CHI** | `.366` | **`.372`** | 0 | `43.4%` | `56.6%` (35yo UFAs) | **`19.5%`** (`16.0%` in '24) | `31.2%` | `+$11.01M` | **`0.089`** | **Prolonged `R2_TEARDOWN` (Dumbbell Roster)** |
| **SJS** | `.360` | `.317` | 0 | `58.4%` | `41.6%` | `29.1%` | `25.0%` | `+$10.54M` | `0.227` | Early-Stage `R2_TEARDOWN` |

### Why "Trade Core Share" Means Two Very Different Things in Montréal (`MTL`) vs. Chicago (`CHI`)
Notice that both Montréal (`61.7%`) and Chicago (`56.6%`) had high external acquisition shares during `2021–2025`, yet produced opposite trajectories:
- **Montréal used external trades to acquire $B_2$ Prime Bridge stars and top-pair insulators**: Nick Suzuki (acquired at age 19, now 25yo $B_2$ 1C with `89` pts), Kirby Dach (acquired at age 21 from Chicago!), Alex Newhook (acquired at age 22 from Colorado), and Mike Matheson (`25.1` min/gm #1 defenseman).
- **Chicago liquidated its $B_2$ Prime Bridge stars (trading away Hagel, DeBrincat, and Dach) and filled the vacated TOI with $B_4$ (`10+ yr`) short-term UFA stopgaps** (Foligno, Martinez, Brodie, Maroon, Smith, Bertuzzi, Teräväinen), leaving Connor Bedard without a prime-age supporting cast.
