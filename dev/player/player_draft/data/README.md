# Data Pull & Process Stage
This is a brief high-level steps to download, process and merge draft & NHL player career data. All data are sourced from `NHL - API`

 1. Pull NHL draft history (Analyses mostly focus on the first-round draft players)
 2. Pull season statistics from NHL players, which would include player statistics of the season as well as NHL-normalized `firstName.default` and `lastName.default`. This is important since player id is not available in that draft history data. This data has limited stats view + only include their NHL career (not their amateur careers)
 3. Merge the first two data sources based on `firstName.default` and `lastName.default`.
 4. Pull player history over seasons (including their amateur career). This gives us player production changes over time in one place.

Note that not all drafted players will have NHL records, since some never manage to play in the league.

Given this data is not frequently updated nor used, I will not be putting this in production `Github Action`.