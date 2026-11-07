# Technical Companion & Competing-Risks Survival Tables: Benchmarking the Rebuild Clock (`deepdive/timeline`)

> [!IMPORTANT]
> **Post-Hoc Retrospective Study Notice**: Word count of [`BLOG_POST.md`](file:///Users/jjlee/Documents/github/nhl-analytics/dev/player_team_success/rebuild/deepdive/timeline/BLOG_POST.md) is **559 words** (strictly within the $\le 600$-word ceiling). All competing-risks cumulative incidence functions (CIF) are estimated across $N = 97$ total NHL rebuild episodes ($82$ completed, $15$ right-censored active in 2025–26).

---

## 1. Econometric Competing-Risks Formulation (Fine & Gray, 1999)

Let $T_e \in \{1, 2, \dots, 10\}$ denote the duration (in seasons) of rebuild episode $e$ from entry year $t_0$, and let $J_e \in \{1, 2\}$ denote the mutually exclusive absorbing exit state:
- $J_e = 1$: **Competitive Graduation (`Peak` / Tier 1–3 Contender Exit)** — achieving sustained playoff contention ($\text{PTS}\% \ge .550$ or winning a playoff series).
- $J_e = 2$: **Purgatory Failure (`Inaction` / `R5_PURGATORY` Collapse)** — stagnating for $\ge 5$ consecutive seasons without playoff advancement or structural progression.

The cause-specific discrete hazard rate at elapsed duration $\tau$ is:

$$h_j(\tau \mid \mathbf{X}_e) = P\left(T_e = \tau, J_e = j \;\middle|\; T_e \ge \tau, \mathbf{X}_e\right)$$

and the **Cumulative Incidence Function (CIF)** for outcome $j$ by year $\tau$ is:

$$\text{CIF}_j(\tau \mid \mathbf{X}_e) = \sum_{u=1}^{\tau} h_j(u \mid \mathbf{X}_e) \cdot S(u - 1 \mid \mathbf{X}_e)$$

where $S(u - 1) = P(T_e \ge u)$ is the overall survival probability of remaining unresolved in the rebuild state through year $u - 1$.

---

## 2. Complete Actuarial Rebuild Clock ($t = 1 \dots 10$ Years)

Source table: [`tables/actuarial_rebuild_clock_by_year.csv`](file:///Users/jjlee/Documents/github/nhl-analytics/dev/player_team_success/rebuild/deepdive/timeline/tables/actuarial_rebuild_clock_by_year.csv).

| Elapsed Rebuild Year ($\tau$) | Cumulative Tier 1: Playoff Core ($\text{CIF}_1$) | Cumulative Tier 2: Final Four | Cumulative Tier 3: Stanley Cup | Cumulative Purgatory Collapse ($\text{CIF}_2$) | Probability Still Unresolved ($S(\tau)$) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Year 1** | `17.07%` | `15.85%` | `3.66%` | `0.00%` | `82.93%` |
| **Year 2** | `32.93%` | `29.27%` | `7.32%` | `0.00%` | `67.07%` |
| **Year 3** | `46.34%` | `39.02%` | `8.54%` | `0.00%` | `53.66%` |
| **Year 4** | **`57.32%`** | **`46.34%`** | **`12.20%`** | **`3.66%`** | **`39.02%`** |
| **Year 5** | `60.98%` | `47.56%` | `13.41%` | **`13.41%`** | `25.61%` |
| **Year 6** | `65.85%` | `50.00%` | `14.63%` | `14.63%` | `19.51%` |
| **Year 7** | `68.29%` | `52.44%` | `14.63%` | `19.51%` | `12.20%` |
| **Year 8+** | **`70.73%`** | **`54.88%`** | **`14.63%`** | **`29.27%`** | **`0.00%`** |

---

## 3. Impact of GM/Coach/Core Continuity ($\text{CCM}$) Across Rebuild Archetypes

Source table: [`tables/episode_benchmark_comparison.csv`](file:///Users/jjlee/Documents/github/nhl-analytics/dev/player_team_success/rebuild/deepdive/timeline/tables/episode_benchmark_comparison.csv).

| Rebuild Strategy Archetype | Core Continuity Cohort ($\text{CCM}$) | $N$ Episodes | Median Duration (Yrs) | Mean Duration (Yrs) | Tier 1 Playoff Success Rate | Tier 2 Final Four Rate | Tier 3 Stanley Cup Rate | Purgatory Failure Rate | Mean $\text{CCM}$ | Mean $\text{VIR}$ |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`COMPOSITIONAL_RETOOL`** | High Stability ($\text{CCM} \ge 0.87$) | 15 | `1.0` | `1.80` | **`93.3%`** | `73.3%` | `13.3%` | **`6.7%`** | `0.937` | `2.80` |
| **`COMPOSITIONAL_RETOOL`** | High Churn ($\text{CCM} < 0.87$) | 15 | `2.0` | `2.40` | `93.3%` | `66.7%` | `13.3%` | `6.7%` | `0.841` | `1.79` |
| **`HYBRID_REBUILD`** | High Stability ($\text{CCM} \ge 0.87$) | 9 | **`4.0`** | **`4.00`** | **`100.0%`** | `66.7%` | `11.1%` | **`0.0%`** | `0.916` | `1.14` |
| **`HYBRID_REBUILD`** | High Churn ($\text{CCM} < 0.87$) | 3 | `3.0` | `4.67` | `100.0%` | `100.0%` | `0.0%` | `0.0%` | `0.820` | `1.60` |
| **`SCORCHED_EARTH_DRAFT`** | High Stability ($\text{CCM} \ge 0.87$) | 17 | `6.0` | `5.65` | **`52.9%`** | `41.2%` | `17.6%` | `47.1%` | `0.911` | `1.27` |
| **`SCORCHED_EARTH_DRAFT`** | High Churn ($\text{CCM} < 0.87$) | 23 | `5.0` | `5.17` | **`39.1%`** | `34.8%` | `17.4%` | **`60.9%`** | `0.810` | `1.39` |

### Key Analytical Takeaways for Montréal (`MTL`) vs. Chicago (`CHI`)
1. **Montréal’s 4-Year Graduation (`EP-MTL-2021`)**: Entered teardown in `2021–22` (`.335` Points %), stabilized its core spine ($\text{CCM} = 1.000$ in `2023–24` and `2024–25`, episode mean $\text{CCM} = 0.862$), and reached `.555` Points % (`R4_EMERGENCE`) in `2024–25`—hitting the exact **4.0-year median sweet spot** where cumulative playoff graduation crosses `57.3%`.
2. **Chicago’s Multi-Wave Teardown (`EP-CHI-2017` / Post-2020 Reset)**: Because Chicago underwent a secondary mid-stream demolition in `2022–23` ($\text{CCM}$ dropping to `0.639` as Hagel, DeBrincat, Dach, and Kane were liquidated), it reset its developmental clock and remained in `R2_TEARDOWN` (`.372` Points %) in `2024–25`.
