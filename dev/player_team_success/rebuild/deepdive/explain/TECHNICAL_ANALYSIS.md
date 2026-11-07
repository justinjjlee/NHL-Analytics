# Technical Companion & Parameter Dictionary: Evaluating Player Composition, Production Clusters, and State-Space Regimes (`deepdive/explain`)

> [!IMPORTANT]
> **Post-Hoc Retrospective Study Notice**: Word count of [`BLOG_POST.md`](file:///Users/jjlee/Documents/github/nhl-analytics/dev/player_team_success/rebuild/deepdive/explain/BLOG_POST.md) is **552 words** (strictly within the $\le 600$-word ceiling). All parameter centroids are estimated across $N = 764$ NHL team-seasons (2000–2026).

---

## 1. Mathematical Definitions of the Four Telemetry Pillars

### 1.1 Pillar 1: Front-Office Structural Action ($\text{ALI}_{i,t}$, $\text{YIS}_{i,t}$, $\text{RCI}_{i,t}$)
- **Asset Liquidation Index ($\text{ALI}_{i,t}$)**: Standardized composite of departing veteran TOI share and rolling 3-year expected draft value ($\text{EDV}_{i,t}$) accumulated above the league baseline.
- **Youth Influx Score ($\text{YIS}_{i,t}$)**: Standardized composite of under-24 TOI share ($s_{i,t}^{\text{U24}}$), ELC tenure share ($s_{B_1,i,t}$), and top-10 draft capital deployed on the active roster.
- **Roster Churn Index ($\text{RCI}_{i,t} = 1 - \text{TOI}_{\text{retained},i,t}$)**: Fraction of all-situations team TOI turned over from season $t-1$ to season $t$.

### 1.2 Pillar 2: Roster Experience Clusters ($B_1 \dots B_4$), Shannon Entropy ($H_{i,t}^{\text{exp}}$), and Insulation ($\text{VIR}_{i,t}$)
Every skater and goaltender $p$ on team $i$ in season $t$ is assigned to one of four mutually exclusive NHL experience cohorts based on `year_inNHL`:
- **$B_1$ (`0–3 yrs`, ELC / Developing Youth)**: Share $s_{B_1,i,t} = \sum_{p \in B_1} \text{TOI}_{p,i,t} / \text{TOI}_{i,t}$.
- **$B_2$ (`4–6 yrs`, Prime Bridge Core)**: Share $s_{B_2,i,t}$.
- **$B_3$ (`7–9 yrs`, Peak Veteran Core)**: Share $s_{B_3,i,t}$.
- **$B_4$ (`10+ yrs`, Greybeard Stabilizers)**: Share $s_{B_4,i,t}$.

From this 4-simplex ($\sum_{k=1}^4 s_{B_k,i,t} = 1$), we compute three structural parameters:
1. **Shannon Experience Entropy**:
   $$H_{i,t}^{\text{exp}} = -\sum_{k=1}^{4} s_{B_k,i,t} \ln(s_{B_k,i,t} + \epsilon)$$
2. **Non-Linear Youth Overload Penalty**:
   $$\text{Overload}_{i,t} = \max\left(0, s_{B_1,i,t} - 0.30\right)^2$$
3. **Veteran Insulation Ratio ($\text{VIR}_{i,t}$)**:
   $$\text{VIR}_{i,t} = \frac{s_{B_2,i,t} + s_{B_3,i,t} + s_{B_4,i,t}}{\max(s_{B_1,i,t}, 0.05)}$$

### 1.3 Pillar 3: Core Continuity ($\text{CCM}_{i,t}$) & Acquisition Sourcing ($\text{Pathway A vs. B}$)
Restricting to the team's **Core Roster** ($\mathcal{C}_{i,t}$: top-6 forwards by TOI, top-4 defensemen by TOI, top-1 goaltender):
- **Core Continuity Metric ($\text{CCM}_{i,t}$)**: Cosine similarity of core player TOI weights between seasons $t-1$ and $t$.
- **Pathway A (`pathway_a_draft_core_share`)**: Fraction of core TOI played by athletes originally drafted by franchise $i$.
- **Pathway B (`pathway_b_trade_core_share`)**: Fraction of core TOI played by athletes acquired via external trade, UFA signing, or waiver claim.

### 1.4 Pillar 4: Expected Goals & Proactive Physicality Synergy ($\text{SPSI}_{i,t}$)
Integrating our repository's hit-sequence tracking (`latest/team/team_season_hit_kpi.csv`):
- **Offensive-Zone Hit Share ($\text{O-Zone Hit}\%_{i,t}$)**: Percentage of team hits delivered in the offensive zone (forecheck pressure to force turnovers) rather than defensive-zone chasing.
- **Proactive Forecheck Share ($\text{PFHS}_{i,t}$)**: Composite of O-Zone Hit Share and post-hit puck win rate (`hit_puck_win_pct`).
- **Style-Production Synergy Index ($\text{SPSI}_{i,t}$)**: Multiplicative interaction between 5v5 Expected Goals Share ($\text{xGF}\%_{i,t}$) and Proactive Forecheck Share ($\text{PFHS}_{i,t}$), normalized on $[0, 1]$.

---

## 2. Empirical Parameter Centroids Across the 3 Macro-Regimes and 6 Sub-States ($N = 764$)

Source tables: [`tables/model_parameter_estimates_and_clusters.csv`](file:///Users/jjlee/Documents/github/nhl-analytics/dev/player_team_success/rebuild/deepdive/explain/tables/model_parameter_estimates_and_clusters.csv) and [`tables/granular_substate_centroids.csv`](file:///Users/jjlee/Documents/github/nhl-analytics/dev/player_team_success/rebuild/deepdive/explain/tables/granular_substate_centroids.csv).

| Regime / Sub-State | $N$ | $\text{PTS}\%$ | $\text{ALI}$ | $\text{YIS}$ | $\text{RCI}$ | $\text{CCM}$ | $s_{B_1}$ (0–3y) | $s_{B_2}$ (4–6y) | $s_{B_3}$ (7–9y) | $s_{B_4}$ (10+y) | $\text{VIR}$ | Draft Core $\%$ | Trade Core $\%$ | 5v5 $\text{xGF}\%$ | O-Zone Hit $\%$ | $\text{SPSI}$ |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **`Peak` (Macro)** | **337** | **`.631`** | `.137` | `.688` | `.367` | `.879` | `36.6%` | `28.6%` | `18.8%` | `16.1%` | `2.71` | `66.5%` | `31.4%` | `59.7%` | `44.4%` | **`.685`** |
| **`Rebuild` (Macro)** | **211** | **`.468`** | **`.263`** | **`.841`** | **`.448`** | `.862` | `46.5%` | `25.8%` | `14.5%` | `13.2%` | `1.59` | `54.8%` | `43.8%` | `45.5%` | `40.4%` | `.297` |
| **`Inaction` (Macro)** | **184** | **`.496`** | `.160` | `.722` | `.353` | `.888` | `32.9%` | `29.7%` | `20.4%` | `17.0%` | `2.58` | `55.3%` | `40.8%` | `47.5%` | `41.5%` | `.335` |
| `R0_CONTENDER` | 295 | `.633` | `.130` | `.666` | `.362` | `.880` | `37.1%` | `28.1%` | `18.6%` | `16.2%` | `2.75` | `67.1%` | `30.8%` | `60.2%` | `44.4%` | `.695` |
| `R4_EMERGENCE` | 81 | `.585` | `.176` | `.848` | `.415` | `.864` | `38.1%` | `30.4%` | `17.5%` | `14.0%` | `2.21` | `60.8%` | `37.4%` | `53.6%` | `43.8%` | `.546` |
| `R3_INCUBATION` | 98 | `.488` | `.202` | `.821` | `.407` | `.882` | `48.6%` | `26.1%` | `13.8%` | `11.5%` | `1.44` | `57.2%` | `41.2%` | `47.0%` | `40.4%` | `.310` |
| `R2_TEARDOWN` | 74 | `.399` | **`.380`** | **`.862`** | **`.517`** | `.837` | `48.1%` | `23.5%` | `13.6%` | `14.8%` | `1.46` | `48.9%` | `49.9%` | `39.9%` | `38.8%` | `.163` |
| `R1_SLUMP` | 94 | `.505` | `.116` | `.663` | `.326` | `.904` | `34.9%` | `28.5%` | `20.2%` | `16.4%` | `2.70` | `57.9%` | `36.3%` | `48.6%` | `41.5%` | `.366` |
| `R5_PURGATORY` | 90 | `.486` | `.206` | `.784` | `.380` | `.871` | `30.8%` | `31.0%` | `20.6%` | `17.7%` | `2.46` | `52.5%` | `45.5%` | `46.2%` | `41.6%` | `.303` |

---

## 3. Direct Telemetry Comparison: Montréal (`MTL`) vs. Chicago (`CHI`) in 2024–25

Source table: [`tables/mtl_chi_component_decomposition.csv`](file:///Users/jjlee/Documents/github/nhl-analytics/dev/player_team_success/rebuild/deepdive/explain/tables/mtl_chi_component_decomposition.csv).

- **Why Montréal Triggered `R4_EMERGENCE`**:
  1. $\text{ALI}$ dropped to **`0.114`** (ending the sell-off phase) while $\text{YIS}$ reached **`1.000`**.
  2. $B_2$ Prime Bridge TOI reached **`29.9%`** (matching the `R4_EMERGENCE` centroid of `30.4%`), while $B_1$ ELC TOI stayed controlled at **`28.1%`** (below the `30%` overload threshold).
  3. 5v5 $\text{xGF}\%$ crossed break-even to **`50.7%`** and O-Zone Hit Share surged to **`47.9%`** (exceeding even the historical `Peak` average of `44.4%`), lifting $\text{SPSI}$ to **`0.436`**.
- **Why Chicago Remained Locked in `R2_TEARDOWN`**:
  1. $\text{ALI}$ remained elevated at **`0.307`** with high roster churn ($\text{RCI} = 0.516$, matching the `R2_TEARDOWN` centroid of `0.517`).
  2. $B_2$ Prime Bridge TOI collapsed to **`16.0%`** ($-14.4$ percentage points below the `R4_EMERGENCE` benchmark), while $B_4$ ($10+$ yr veterans) swelled to **`41.9%`**.
  3. 5v5 $\text{xGF}\%$ remained at the league floor (**`38.0%`**) and O-Zone Hit Share sat at **`39.9%`**, yielding an $\text{SPSI}$ of just **`0.038`**.
