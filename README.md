# Sports Analytics - National Hockey League (NHL)

This repository contains data science toolkit codes to analyze National Hockey League (NHL) statistics. 

This is a selection of many ways to collect, compile, clean, analyze, model, and predict team and player (skaters and goalies) performances statistics. This repository does not claim any ownership of data and represents points of views of organizations or representations mentioned. All original codes (including generic and model algorithms) can be used freely with a proper citation to this repository.

## Data collection, clean-up, and exploratory analysis
- [ ] Data scrap - using [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) & [Selenium](https://selenium-python.readthedocs.io/) using Python
  - [x] Team - By season
  - [ ] Team - By season, by game
  - [ ] Skater - By season
  - [ ] Skater - By season, by game
  - [ ] Goalie - By season
  - [ ] Goalie - By season, by game
- [ ] Data compilation & preliminary validations - using [Online Algorithm / Statistics](https://github.com/joshday/OnlineStats.jl) using julia to handle large / big data
- [ ] Exploratory Data Analysis (EDA) and sample data visualization

Web scraping portion of the code requires a careful consideration of followings,
* Define the web browser of your choice - the default one used by the code in this repository is Firefox
* Depending on the response time of the host website, users need to adjust explicit wait time for JavaScript table to load
