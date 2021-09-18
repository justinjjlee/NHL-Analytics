## Data collection and clean up

Below list current work examles of codes published in the repository
- [x] Data scrap - using [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) & [Selenium](https://selenium-python.readthedocs.io/) using Python
  - [x] Team - By season
  - [x] Team - By season, by game
  - [x] Skater - By season
  - [x] Skater - By season, by game
  - [x] Goalie - By season
  - [x] Goalie - By season, by game
- [x] Data scrape - Leveraging [http://statsapi.web.nhl.com](http://statsapi.web.nhl.com) (see [documentation](https://gitlab.com/dword4/nhlapi/-/blob/master/records-api.md))
  - [x] Live game data: Player (Goalie & Skater) by action
  - [x] [Example game-level data pull](/collection/nhl_game_data_pull_eg.md)
- [ ] Data join - merge relevant data
  - [ ] Player salary and salary cap from [hockey-reference.com](https://www.hockey-reference.com/friv/current_nhl_salaries.cgi), (annual)
  - [ ] (Private) NHL stadium foot traffic behaviors and demographics data (mobile / foot traffic data)
  - [ ] University of Michigan - Men's Ice Hockey [player statistics](https://statsarchive.ath.umich.edu/VS-IceHockey-M/ihockeystart.php)
- [ ] Data compilation & preliminary validations - using [Online Algorithm / Statistics](https://github.com/joshday/OnlineStats.jl) using julia and [PySpark](https://spark.apache.org/docs/latest/api/python/) using Python/Spark. to handle large / big data
  - [ ] Team abbreviation match and identifications

Web scraping portion of the code requires a careful consideration of followings,
* Define the web browser of your choice - the default one used by the code in this repository is Firefox
* Depending on the response time of the host website, users need to adjust explicit wait time for JavaScript table to load

## Example visualization 
1. Pythagorean Expectation [in Tableau Public](https://public.tableau.com/app/profile/justin.l.1253/viz/NHL-PythagoreanExpectation/Dashboard2)

## Other resources - Exercises
- Salary Cap: [Tracker](https://puckpedia.com/#salary-cap)
- NHL Hockey Reference: [Insights](https://www.hockeyzoneplus.com/)
- Articles - Christian Lee's [extensive analyses](https://medium.com/hockey-stats)
  - [R version of pulling data](https://medium.com/hockey-stats/how-to-scrape-nhl-com-dynamic-data-in-r-using-rvest-and-rselenium-ba3b5d87c728)
- See an interesting visualization published in [Toward Data Science by Kyle Pastor](https://towardsdatascience.com/nhl-analytics-with-python-6390c5d3206d)

[back to the main page](../index.md)