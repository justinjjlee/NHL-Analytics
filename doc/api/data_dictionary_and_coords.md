# NHL API Data Dictionary, Rink Geometry & Coordinates

This guide outlines the mathematical conventions, coordinate systems, game identifiers, and taxonomy schemas used by the NHL API and standardized within the **NHL-Analytics** data warehouse.

---

## 1. NHL Ice Rink Coordinate Geometry

Official NHL rinks are standard North American dimensions: **200 feet in length by 85 feet in width**, with a 28-foot corner radius.

```
                  Defending Zone            Neutral Zone           Attacking Zone
       [-100]          [-89]                    [-25]      [0]     [+25]                     [+89]          [+100]
    +42.5 +--------------|------------------------|---------|--------|-------------------------|---------------+ +42.5
          |              |                        |         |        |                         |               |
          |  (Goal)      | (Goal Line)            | (Blue)  | (Red)  | (Blue)                  | (Goal Line)   |
      0.0 |  [x=-89]     |                        |         |(0,0)   |                         | [x=+89]       | 0.0
          |              |                        |         |        |                         |               |
    -42.5 +--------------|------------------------|---------|--------|-------------------------|---------------+ -42.5
       [-100]          [-89]                    [-25]      [0]     [+25]                     [+89]          [+100]
```

### Coordinate Space Specifications
* **Origin $(0, 0)$:** Exact center ice at the center red line faceoff dot.
* **$x$-Axis:** Runs longitudinally along the center line from end board to end board:
  * Range: $[-100.0, +100.0]$ feet.
  * $x = -89.0$: Left goal line.
  * $x = -25.0$: Left blue line.
  * $x = 0.0$: Center red line.
  * $x = +25.0$: Right blue line.
  * $x = +89.0$: Right goal line.
* **$y$-Axis:** Runs laterally from side board to side board:
  * Range: $[-42.5, +42.5]$ feet.
  * Center of rink: $y = 0.0$.

---

## 2. Spatial Normalization (Offensive Attack Zone Transform)

In live games, teams switch sides between periods (Period 1 $\rightarrow$ Period 2 $\rightarrow$ Period 3 $\rightarrow$ Overtime). The API provides `homeTeamDefendingSide` (`"left"` or `"right"`) for each period.

To train predictive models (such as Expected Goals $xG$, shot density heatmaps, or hit momentum impact), all coordinates must be standardized so that **the attacking team is always shooting toward positive $x$ ($+89.0$)**:

$$\begin{aligned}
x_{\text{normalized}} &= \begin{cases} -x & \text{if attacking towards the negative } x \text{ net} \\ +x & \text{if attacking towards the positive } x \text{ net} \end{cases} \\
y_{\text{normalized}} &= \begin{cases} -y & \text{if } x \text{ was flipped} \\ +y & \text{otherwise} \end{cases}
\end{aligned}$$

After transformation:
* $x \in [25, 89]$ represents the offensive attacking zone.
* Distance to Net: $d = \sqrt{(89 - x_{\text{norm}})^2 + (y_{\text{norm}})^2}$.
* Shot Angle: $\theta = \arctan\left(\frac{|y_{\text{norm}}|}{89 - x_{\text{norm}}}\right)$.

---

## 3. Game Identifier (Game ID) Architecture

Every NHL game is identified by an exact **10-digit integer** formatted as `YYYYTTSSSS`:

$$\underbrace{2024}_{\text{Season Starting Year}}\quad \underbrace{02}_{\text{Game Type}}\quad \underbrace{0045}_{\text{Sequence Number}}$$

### Component Breakdown
1. **Season (`YYYY` - 4 digits):**
   * The calendar year in which the season begins (e.g., `2024` for the 2024–25 season).
2. **Game Type (`TT` - 2 digits):**
   * `01`: Pre-season.
   * `02`: Regular season.
   * `03`: Stanley Cup Playoffs.
   * `04`: All-Star Game.
3. **Sequence Number (`SSSS` - 4 digits):**
   * **Regular Season (`02`):** Sequentially numbered from `0001` to `1312` (for a 32-team, 82-game schedule: $32 \times 82 / 2 = 1312$).
   * **Playoffs (`03`):** Encoded as `0RGM`:
     * `R` (Round): `1` (First Round), `2` (Second Round), `3` (Conference Finals), `4` (Stanley Cup Final).
     * `M` (Matchup): `1` through `8` (identifies the series matchup).
     * `G` (Game in Series): `1` through `7`.
     * *Example:* `2023030417` = 2023–24 Season (`2023`), Playoffs (`03`), Round 4 Stanley Cup Final (`4`), Matchup 1 (`1`), Game 7 (`7`).

---

## 4. Situation Code Taxonomy (`situationCode`)

In the play-by-play feed, the on-ice manpower configuration is encoded as a **4-digit numeric string** representing the exact goalie and skater counts:

$$\text{Format: } [\text{Away Goalie}][\text{Away Skaters}][\text{Home Skaters}][\text{Home Goalie}]$$

| Code | Away Goalie | Away Skaters | Home Skaters | Home Goalie | Game State Description |
| :---: | :---: | :---: | :---: | :---: | :--- |
| `1551` | 1 | 5 | 5 | 1 | Standard Even Strength (5v5) |
| `1541` | 1 | 5 | 4 | 1 | Away Power Play (5v4) / Home Penalty Kill |
| `1451` | 1 | 4 | 5 | 1 | Home Power Play (5v4) / Away Penalty Kill |
| `1441` | 1 | 4 | 4 | 1 | 4-on-4 Even Strength |
| `1331` | 1 | 3 | 3 | 1 | 3-on-3 Overtime |
| `0651` | 0 | 6 | 5 | 1 | Away Empty Net (Goalie Pulled, 6v5) |
| `1560` | 1 | 5 | 6 | 0 | Home Empty Net (Goalie Pulled, 6v5) |
| `1531` | 1 | 5 | 3 | 1 | Away 5-on-3 Two-Man Advantage |

---

## 5. Event Taxonomy (`typeDescKey`)

| Event Type Key | Category | Description | Primary Detail Fields Available |
| :--- | :--- | :--- | :--- |
| `shot-on-goal` | Shot Attempt | Puck directed toward goal, saved by goalie or resulting in goal. | `xCoord`, `yCoord`, `shotType`, `shootingPlayerId`, `goalieInNetId` |
| `missed-shot` | Shot Attempt | Shot attempt that missed the goal frame or hit post/crossbar. | `xCoord`, `yCoord`, `shotType`, `shootingPlayerId`, `reason` |
| `blocked-shot` | Shot Attempt | Shot attempt stopped by a defending skater before reaching goal. | `xCoord`, `yCoord`, `shootingPlayerId`, `blockingPlayerId` |
| `goal` | Scoring | Goal scored. | `xCoord`, `yCoord`, `shotType`, `scoringPlayerId`, `assist1PlayerId`, `assist2PlayerId` |
| `hit` | Physicality | Body check delivered to opponent. | `xCoord`, `yCoord`, `hittingPlayerId`, `hitteePlayerId` |
| `faceoff` | Possession | Puck drop following a stoppage. | `xCoord`, `yCoord`, `winningPlayerId`, `losingPlayerId`, `zoneCode` |
| `giveaway` | Turnover | Unforced loss of puck control to the opponent. | `xCoord`, `yCoord`, `playerId` |
| `takeaway` | Turnover | Forced takeaway of puck control by defending skater. | `xCoord`, `yCoord`, `playerId` |
| `penalty` | Infraction | Penalty assessed to a player or bench. | `committedByPlayerId`, `drawnByPlayerId`, `descKey`, `duration` |
| `stoppage` | Clock | Whistle blown (icing, offside, goalie freeze, puck out of play). | `reason` |
| `period-start` / `period-end` | Clock | Boundary marks for regulation and overtime periods. | `periodDescriptor` |
