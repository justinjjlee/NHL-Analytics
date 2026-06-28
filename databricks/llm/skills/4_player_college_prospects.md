# Skillset 4: NCAA-NHL Player Prospects

This skillset covers the techniques used to link amateur college hockey career records (NCAA) with professional NHL player profiles, and to analyze how draft choices and college tenures influence career performance and goal production.

---

## 1. Scope & Target Measurement Level
* **Measurement Level:** Player Level, College/NCAA Level, Season Level.
* **Objective:** Cleanly match records across distinct, uncoordinated datasets and evaluate the predictive value of college parameters (tenure, draft position) on NHL success metrics (e.g. goals, points, time on ice).

---

## 2. Mathematical Formulations

### 2.1 Fuzzy String Matching
Because player names are recorded by different entities, exact string equality fails due to spelling differences, nicknaming, and accents (e.g., "Alex" vs "Alexander", "Mitch" vs "Mitchell").
* **Levenshtein Distance ($D_L$):**
  Measures the minimum single-character edits (insertions, deletions, substitutions) required to change string $a$ into $b$.
* **Token Sort Ratio ($R_{TS}$):**
  To handle name transpositions (e.g. "McDavid, Connor" vs "Connor McDavid"), tokens are sorted alphabetically before computing distance:
  $$R_{TS}(a, b) = 100 \times \frac{|a| + |b| - D_L(\text{sort}(a),\, \text{sort}(b))}{|a| + |b|}$$
  * A threshold of $R_{TS} \ge 85$ is established as a positive match.

### 2.2 tenure impact and Rookie decay
Let $T_i \in \{1, 2, 3, 4\}$ be the NCAA college tenure (years in college) of player $i$, and $Y_i \in \mathbb{N}$ be the draft round (or overall pick position). Let $TOI_i$ be the NHL Time on Ice per game, and $G_i$ be goals scored.
* **Rookie Debut Performance Model:**
  $$TOI_{i, \text{Rookie}} = \beta_0 + \beta_1 \cdot T_i + \beta_2 \cdot Y_i + \epsilon_i$$
  * *Finding:* $\beta_1 < 0$ and $\beta_2 < 0$. Highly touted prospects leave college early ($T_i \le 2$), thus shorter college tenure correlates with higher rookie-year NHL ice time and point production.
* **Career Maturity Model (Non-Rookie Seasons):**
  $$TOI_{i, \text{Vet}} = \delta_0 + \delta_1 \cdot T_i + \delta_2 \cdot Y_i + \eta_i$$
  * *Finding:* $\delta_1 \approx 0$ and $\delta_2 \approx 0$. As players mature, the initial effects of college tenure and draft order decay, indicating that roster filters and on-ice efficiency dominate long-term success.

---

## 3. Record Linkage & Analysis Workflow

```mermaid
flowchart TD
    A[Scrape College Alums\nCollege Hockey Inc] --> B[Parse Names\nname_last + name_first = fullName]
    C[Clean NHL Player Stats\n{year}_02_player.csv] --> D[Aggregate Player stats\nfullName + team_triCode + season]
    B --> E[Step 1: Exact Match\nJoin on fullName & team_tri]
    D --> E
    E --> F{Matched?}
    F -->|Yes| G[Output Linked Database]
    F -->|No| H[Step 2: Fuzzy Matching\nCross-join unmatched names]
    H --> I[Compute Token Sort Ratio]
    I --> J{Ratio >= 85?}
    J -->|Yes| G
    J -->|No| K[Manual Review / Discard]
    G --> L[Regression Analysis\nTenure & Draft vs. TOI/Goals]
```

---

## 4. Input & Output Schemas

### 4.1 Input Schema 1: NCAA Alumni Sheet (`nhl_college_alums.xlsx` columns)
* `name_last` (str): Last name.
* `name_first` (str): First name.
* `college` (str): NCAA university name (e.g. `"Boston Univ."`, `"Michigan"`).
* `nhl` (str): Team abbreviations split by slashes (e.g. `"BUF/MIN"`).
* `college_yrs` (int): Number of years spent in college (NCAA tenure, $1$-$4$).
* `season` (int): Year ending of college season (e.g. `2022`).

### 4.2 Input Schema 2: NHL Player Boxscore Stats (`{year}_02_player.csv` columns)
* `id_player` (int): NHL player ID.
* `fullName` (str): Full name.
* `team_triCode` (str): Team tri-code.
* `rookie` (bool): Rookie status flag.
* `goals` (int): Game goals.
* `timeOnIce` (str): MM:SS format time on ice.

### 4.3 Output Schema: Linked Prospect Analytics Database
* `fullName` (str): Player name.
* `college` (str): College attended.
* `college_yrs` (int): College tenure.
* `draft` (str/int): Draft pick index.
* `rookie_avg_toi` (float): Average rookie-season TOI in minutes.
* `career_avg_toi` (float): Average career-season TOI in minutes.

---

## 5. Generalized Python Implementation

```python
import glob
import pandas as pd
from fuzzywuzzy import fuzz

def convert_toi_to_minutes(toi_str: str) -> float:
    """Converts MM:SS time-on-ice string to float minutes."""
    if pd.isna(toi_str):
        return 0.0
    try:
        parts = toi_str.split(':')
        return float(parts[0]) + float(parts[1]) / 60.0
    except:
        return 0.0

def fuzzy_join_college_nhl(df_college: pd.DataFrame, df_nhl: pd.DataFrame, threshold: int = 85) -> pd.DataFrame:
    """
    Performs exact and fuzzy record linkage between NCAA college alums and NHL player stats.
    """
    # Clean NCAA team list and explode
    df_col = df_college.copy()
    df_col['fullName'] = df_col['name_first'] + " " + df_col['name_last']
    df_col['team_triCode'] = df_col['nhl'].str.split('/')
    df_col = df_col.explode('team_triCode')
    df_col['team_triCode'] = df_col['team_triCode'].str.strip()
    
    # Process NHL Player stats
    df_pro = df_nhl.copy()
    df_pro['toi_min'] = df_pro['timeOnIce'].apply(convert_toi_to_minutes)
    
    # Aggregate NHL statistics
    nhl_agg = df_pro.groupby(['fullName', 'team_triCode']).agg(
        id_player=('id_player', 'first'),
        avg_toi=('toi_min', 'mean'),
        total_goals=('goals', 'sum'),
        rookie=('rookie', 'min') # 1 if they ever played as rookie this season
    ).reset_index()
    
    # Pass 1: Exact join
    exact_match = df_col.merge(nhl_agg, on=['fullName', 'team_triCode'], how='inner')
    print(f"Exact matched records: {len(exact_match)}")
    
    # Pass 2: Fuzzy join for unmatched records
    matched_names = set(exact_match['fullName'])
    unmatched_col = df_col[~df_col['fullName'].isin(matched_names)]
    unmatched_nhl = nhl_agg[~nhl_agg['fullName'].isin(matched_names)]
    
    fuzzy_records = []
    
    for _, col_row in unmatched_col.iterrows():
        # Compare with all unmatched NHL players in the same team
        candidates = unmatched_nhl[unmatched_nhl['team_triCode'] == col_row['team_triCode']]
        best_ratio = 0
        best_match = None
        
        for _, nhl_row in candidates.iterrows():
            ratio = fuzz.token_sort_ratio(col_row['fullName'], nhl_row['fullName'])
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = nhl_row
                
        if best_ratio >= threshold and best_match is not None:
            combined = {**col_row.to_dict(), **best_match.to_dict()}
            combined['fuzzy_ratio'] = best_ratio
            fuzzy_records.append(combined)
            
    fuzzy_match = pd.DataFrame(fuzzy_records)
    print(f"Fuzzy matched records: {len(fuzzy_match)}")
    
    # Combine passes
    master_matched = pd.concat([exact_match, fuzzy_match], ignore_index=True)
    return master_matched
```

---

## 6. Weaknesses, Caveats & Considerations
1. **Name Accents and Unicode Variations:** NHL source files often retain French or Swedish accents (e.g. `ë`, `ö`, `é`), whereas NCAA files frequently omit them. Pre-processing must strip accents and normalize strings to ASCII standard before running distance algorithms.
2. **Survival Bias:** Evaluating "NCAA tenure vs. NHL longevity" introduces a substantial survival bias. Players who complete 4 years of college are often weaker prospects who needed development time, whereas elite players leave after 1 or 2 years. Comparing these groups directly without controlling for draft round ($Y_i$) will result in the spurious conclusion that "college degrades player talent."
3. **Draft Pick Inflation:** Over time, the number of NHL draft rounds and teams has changed. A 3rd round pick in 1990 represents a different talent tier than a 3rd round pick in 2024. Pick values must be normalized to overall pick numbers ($1$ to $224$) rather than round numbers.
