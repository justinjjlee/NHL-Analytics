# ELO Ratings & Bookmaker Odds
This analysis constructs and apply ELO-ratings, following ['The Betting Odds Rating System: Using soccer forecasts to forecast soccer'](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0198668) applying the NHL game-level outcomes and betting odds data.

The debut of the code and analysis will use 2024-2025 season data.

## Data
ELO ratings and cumulative team/win loss will use game outcome data and sportsbook betting data (`DraftKing`) available through `apinhle` as of 2025. Odds data prior to 2023 is from [Sportsbook Review](https://www.sportsbookreviewsonline.com/scoresoddsarchives/nhl/nhloddsarchives.htm).

All odds data are based on 2-way odds line.

I processed and unified data to be in one place and saved in `Unity Catalog` of my `Databricks` environment. 

### Team Win/Loss

For classifying 

|       | Win   | Tie   | Loss  |
|-------|-------|-------|-------|
| **Win**  | ✅    |       | ❌    |
| **Tie**  |       | ✅    |       |
| **Loss** | ❌    |       | ✅    |

### Bookmaker Bet
The spread data is used to compute the statistics identified for each game
 * Win/loss projected
 * Spread of the bet (converted to standard deviation of the season)

Note that the betting odds change frequently and over time until the game starts. To the best of my (automation) ability, the data is pulled at the same day 4 hours prior to the game. There were days where the data might not have been pulled or pulled at different hours. For the purpose of this analysis, the assumption we will make is that the projected win/loss outcome would not have changed around the time of the data pull (highly unlikely given that the betting odds are [set far in advance with existing knowledge built-in](https://medium.com/@thinkingjustin/better-than-puck-flips-money-line-and-market-expectation-in-nhl-91786fcf6f01)). The spread of the odds would vary over time, but given that the most of the odds recorded are pulled around the same time of day, we would anticipate that the information captured on average would be consistent estimator if we standardize the average spread (standard deviation of the difference). We would also need to assume the consistency of the odds estimator and information reflected fully across games played in different time zone (e.g. games played on EST vs. PST). 

The analysis would not account for those time-varying changes in information at the time of the data pull (e.g., player injury news breaks after the odds data are pulled.). The new information would significantly impact the outcome of the game, but the information would not be reflected on the odds data recorded.

#### Implied Probability of Win Odds
Implied probability of winning can be calculated using 2-way money *line*,

* If underdog (line value is positive) $$\frac{100}{1 + line}$$

* If favored (line value is negative) $$\frac{|line|}{|line| + 100}$$

