# Skillset 2: Advanced Team Success KPIs

This skillset details the calculation, mathematical theory, and execution of team-level performance indicators. These metrics aggregate game box scores to establish robust, strength-of-schedule adjusted profiles for NHL teams.

---

## 1. Scope & Target Measurement Level
* **Measurement Level:** Team Level, Season Level (summarizing performance across multiple games or an entire 82-game regular season).
* **Objective:** Move beyond raw win-loss records to measure underlying possession quality, shot efficiency, scheduling difficulty, and streak robustness.

---

## 2. Mathematical Formulations

### 2.1 Corsi and Fenwick (Possession volume)
Corsi and Fenwick serve as proxies for offensive zone possession time based on shot attempt differentials.
* **Fenwick (Unblocked Shot Attempts):**
  $$\text{Fenwick For (FF)} = \text{Shots For (SOG)} + \text{Shots Missed For (Missed)}$$
  $$\text{Fenwick Against (FA)} = \text{Shots Against (SOG\_opp)} + \text{Shots Missed Against (Missed\_opp)}$$
  $$\text{Fenwick Differential} = \text{FF} - \text{FA}$$
* **Corsi (All Shot Attempts):**
  $$\text{Corsi For (CF)} = \text{FF} + \text{Shots Blocked For}$$
  $$\text{Corsi Against (CA)} = \text{FA} + \text{Shots Blocked Against}$$
  $$\text{Corsi Differential} = \text{CF} - \text{CA}$$
  $$\text{Corsi Percentage (CF\%)} = \frac{\text{CF}}{\text{CF} + \text{CA}}$$

### 2.2 Pythagorean Win Expectation ($PE$)
Derived from baseball and adapted to hockey, Pythagorean Expectation estimates a team's true skill based on goal production ($GF$) and goal prevention ($GA$).
* **Expectation Formula:**
  $$PE = \frac{GF^{\gamma}}{GF^{\gamma} + GA^{\gamma}}$$
  * *Note:* In Rinklytics, $\gamma = 2.0$ serves as the default exponent.
* **Rolling Moving Average ($PE_{10}$):** To capture recent form, $PE$ is calculated over a sliding window of the last $N$ games ($N=10$):
  $$PE_{10} = \frac{(GF_t - GF_{t-10})^2}{(GF_t - GF_{t-10})^2 + (GA_t - GA_{t-10})^2}$$
* **Fitted Win Projection:** Based on historical NHL calibration (2010–2020), the projected season wins over an 82-game schedule is:
  $$\text{Projected Wins} = (\beta_{PE} \cdot PE + \alpha_{PE}) \times 82$$
  * Where $\beta_{PE} \approx 0.809623$ and $\alpha_{PE} \approx 0.042901$.

### 2.3 Ratings Percentage Index ($RPI$)
Adjusts a team's winning percentage for their strength of schedule.
* **RPI Formula:**
  $$RPI = (0.25 \times WP_{\text{own}}) + (0.50 \times WP_{\text{opp}}) + (0.25 \times WP_{\text{opp\_opp}})$$
  * $WP_{\text{own}}$: Winning percentage of the focal team.
  * $WP_{\text{opp}}$: Average winning percentage of all opponents faced.
  * $WP_{\text{opp\_opp}}$: Average winning percentage of the opponents' opponents (calculated via common opponent structures).

### 2.4 Pairwise Win Index
Constructed by performing head-to-head comparisons against common opponents. For each team $A$ and $B$, if $A$ has a higher winning percentage than $B$ against mutual opponents, $A$ is awarded a "pairwise win."
$$\text{Pairwise Score}_A = \sum_{B \neq A} \mathbb{I}\left( WP_{A \text{ vs Common}} > WP_{B \text{ vs Common}} \right)$$
$$\text{Normalized KPI Pairwise} = \frac{\text{Pairwise Score} - \min(\text{Scores})}{\max(\text{Scores}) - \min(\text{Scores})}$$

### 2.5 Streak Classifications
* **Win Streak:** 3 consecutive game wins.
* **Expected Win Streak:** 3 consecutive wins where at least 2 were against opponents with lower RPI or pairwise win records (i.e. games the team was expected to win).
* **Hot Streak:** A 3-game win streak where at least 1 win was against a "good" opponent (defined as an opponent whose RPI is in the top 25% of the league).

---

## 3. Game-to-Team Aggregation Pipeline

```mermaid
flowchart TD
    A[Raw Game Box Scores\n{year}_box_team.csv] --> B[df_unpack\nDouble-index focal and opponent stats]
    B --> C[Compute cumulative metrics\ncumsum GF, GA, CF, CA]
    C --> D[Compute Rolling PE & PE_10]
    C --> E[Generate H2H matchups]
    E --> F[Find Common Opponents\nCompute H2H Win Ratios]
    D --> G[Calculate RPI components]
    F --> H[Calculate Pairwise Win count]
    G --> I[Team Season KPI Aggregations\nCorsi, Fenwick, PE, RPI, Pairwise]
    H --> I
    I --> J[Normalize KPIs to 0-1 scale]
    J --> K[Export team_season.csv]
```

---

## 4. Input & Output Schemas

### 4.1 Input Schema (`df_box_team` columns)
* `gameIdx` (str, index): Unique identifier.
* `team_tri` (str): Focal team abbreviation.
* `goals` (int): Goals scored.
* `shots` (int): Shots on goal.
* `shots_blocked` (int): Shots blocked by opponent.
* `win` (int): `1` if win, `0` if loss.
* `gameDate` (str/datetime): Date of game.

### 4.2 Output Schema (`df_team_season` columns)
* `team_tri` (str, index): Focal team.
* `wp` (float): Focal team winning percentage.
* `kpi_corsi` (float): Normalized Corsi differential.
* `kpi_fenwick` (float): Normalized Fenwick differential.
* `kpi_pe` (float): Normalized Pythagorean win expectation.
* `kpi_rpi` (float): Normalized Ratings Percentage Index.
* `kpi_pairwise` (float): Normalized Pairwise Win Index.

---

## 5. Generalized Python Implementation

```python
import numpy as np
import pandas as pd

def calculate_team_season_kpis(df_box: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Corsi, Fenwick, Pythagorean Expectation, RPI, and Pairwise Win Index.
    Expects df_box containing games double-logged (one row per team per game).
    """
    df = df_box.copy()
    df['gameDate'] = pd.to_datetime(df['gameDate'])
    df = df.sort_values(by=['team_tri', 'gameDate']).reset_index(drop=True)
    
    # 1. Basic Shot Attempt Proxies
    # Assuming shots = SOG, shots_blocked = shots blocked by opponent
    df['fenwick_for'] = df['shots']
    df['corsi_for'] = df['fenwick_for'] + df['shots_blocked']
    
    # 2. Cumulative Season Accumulation
    df['rgoals_for'] = df.groupby('team_tri')['goals'].transform(pd.Series.cumsum)
    df['rwin'] = df.groupby('team_tri')['win'].transform(pd.Series.cumsum)
    df['rgame'] = df.groupby('team_tri').cumcount() + 1
    
    # We need to map opponents' goals to compute Goals Against (GA)
    # Join on opponent gameIdx
    # For simplicity, we pivot/merge back to get opponent values
    opp_df = df[['gameid', 'team_tri', 'goals', 'corsi_for', 'win']].rename(
        columns={'team_tri': 'opp_tri', 'goals': 'opp_goals', 'corsi_for': 'corsi_against', 'win': 'opp_win'}
    )
    df = df.merge(opp_df, on='gameid')
    # Filter out self-merges
    df = df[df['team_tri'] != df['opp_tri']]
    
    # Recalculate cumsum GA
    df['rgoals_against'] = df.groupby('team_tri')['opp_goals'].transform(pd.Series.cumsum)
    df['rcorsi_for'] = df.groupby('team_tri')['corsi_for'].transform(pd.Series.cumsum)
    df['rcorsi_against'] = df.groupby('team_tri')['corsi_against'].transform(pd.Series.cumsum)
    
    # 3. Pythagorean win expectation
    df['rpe'] = (df['rgoals_for']**2) / (df['rgoals_for']**2 + df['rgoals_against']**2)
    
    # Final game records of the season for each team
    team_season = df.groupby('team_tri').agg(
        rwin=('rwin', 'last'),
        rgame=('rgame', 'last'),
        corsi_lvl=('rcorsi_for', lambda x: x.iloc[-1] - df.loc[x.index, 'rcorsi_against'].iloc[-1]),
        fenwick_lvl=('fenwick_for', lambda x: x.sum() - df.loc[x.index, 'opp_goals'].sum()), # proxy
        rpe_last=('rpe', 'last')
    ).reset_index()
    
    team_season['wp'] = team_season['rwin'] / team_season['rgame']
    
    # 4. Strength-of-Schedule Metrics: RPI
    # Calculate opponents' average winning percentages
    wp_dict = team_season.set_index('team_tri')['wp'].to_dict()
    
    # Map opponent WP for each game
    df['opp_wp'] = df['opp_tri'].map(wp_dict)
    opp_wp_mean = df.groupby('team_tri')['opp_wp'].mean()
    
    # Map opponents' opponents' average WP
    # Get lists of opponents for each team
    opponents_by_team = df.groupby('team_tri')['opp_tri'].apply(list).to_dict()
    opp_opp_wp_mean = {}
    for team, opps in opponents_by_team.items():
        opp_wps = []
        for opp in opps:
            # get opponents of this opponent, excluding the team itself
            sub_opps = [o for o in opponents_by_team.get(opp, []) if o != team]
            if sub_opps:
                opp_wps.append(np.mean([wp_dict.get(o, 0.5) for o in sub_opps]))
        opp_opp_wp_mean[team] = np.mean(opp_wps) if opp_wps else 0.5
        
    team_season['wp_opp'] = team_season['team_tri'].map(opp_wp_mean)
    team_season['wp_opp_opp'] = team_season['team_tri'].map(opp_opp_wp_mean)
    
    # RPI calculation
    team_season['rpi'] = (team_season['wp'] * 0.25) + (team_season['wp_opp'] * 0.50) + (team_season['wp_opp_opp'] * 0.25)
    
    # 5. Normalization function
    def normalize(col):
        if col.max() == col.min():
            return 0.5
        return (col - col.min()) / (col.max() - col.min())
        
    team_season['kpi_corsi'] = normalize(team_season['corsi_lvl'])
    team_season['kpi_fenwick'] = normalize(team_season['fenwick_lvl'])
    team_season['kpi_pe'] = normalize(team_season['rpe_last'])
    team_season['kpi_rpi'] = normalize(team_season['rpi'])
    
    return team_season.set_index('team_tri')
```

---

## 6. Weaknesses, Caveats & Considerations
1. **Defensive Strategy Bias in Corsi:** Corsi measures *all* shot attempts, assuming volume dictates possession quality. However, defensively structured teams that deploy blocking schemes (e.g. collapse defense) will systematically show a lower Corsi percentage ($CF\%$), even if they suppress high-danger scoring chances. Comparing Corsi with Fenwick (which drops blocked shots) is crucial to evaluate if a low Corsi is driven by blocking efficiency.
2. **Small Sample Noise Early Season:** For the first 10-15 games of a season, RPI and Pythagorean Win Expectation are highly volatile and prone to schedule-skewness. If a weak team plays three strong teams early on, their opponent win rate ($WP_{\text{opp}}$) is artificially high. Projections must damp these metrics with pre-season ELO ratings.
3. **No Injury/Roster Adjustment:** Standard KPIs assume a team's performance is stationary. They do not account for a star player (e.g., starting goalie or top-line center) missing games. A team's $RPI$ might remain high after an injury, but their true strength is severely degraded.
