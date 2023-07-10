[back to the main page](../index.md)

# Data Collection for NHL - Workbook example of scrapping summary statistics

Resources:
* [Selenium - for using the chrome driver: download](https://sites.google.com/a/chromium.org/chromedriver/downloads).
* [Another]((https://stackoverflow.com/questions/43020352/python-downloading-a-file-from-a-webpage-by-clicking-on-a-link)) good resource.
* Make sure that pages are fully loaded before scrapping - tables takes some time to get loaded.


```python
!pip install selenium
!pip install webdriver-manager
```


```python
import requests
import time
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By

# Enforce incognito mode
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument("--incognito")

from selenium.webdriver.common.keys import Keys
from webdriver_manager.firefox import GeckoDriverManager


import numpy as np
from numpy import array
import pandas as pd
import csv
from datetime import date, datetime
```


```python
int_year = 2022; # last year of season to pull.
int_year_start = int_year - 1; # season starts.
str_season_last = [str(int_year_start) + str(int_year)]

# For data strucutre pages that has more than 10,000 records - need to break things down by month
str_date_start = [iter.strftime('%Y-%m-%d') for iter in pd.date_range(start=f"{int_year_start}-08-10", end=f"{int_year}-08-01", freq='MS')]
str_date_end = [iter.strftime('%Y-%m-%d') for iter in pd.date_range(start=f"{int_year_start}-09-10", end=f"{int_year}-09-01", freq='M')]
```


```python
# define basics
int_year = 2021; # last year of season to pull.
int_year_start = int_year - 1; # season starts.
vec_season_start = list(map(str, list(range(1917, (int_year_start + 1)))))
vec_season_end = list(map(str, list(range(1918, (int_year + 1)))))

# Looking at each season, so no need to worry about, with optional pulling data in reverse order
str_season = [vec_season_start[i] + vec_season_end[i] for i, val in enumerate(vec_season_start)][::-1]; 

# For data strucutre pages that has more than 10,000 records - need to break things down by month
str_date_start = [iter.strftime('%Y-%m-%d') for iter in pd.date_range(start="1917-11-10", end=f"{int_year}-06-01", freq='MS')]
str_date_end = [iter.strftime('%Y-%m-%d') for iter in pd.date_range(start="1917-12-10", end=f"{int_year}-07-01", freq='M')]

str_data_type = ['teams', 'skaters', 'goalies'];
#str_aggregate = ['', 'aggregate=0&']; # In case the aggregation parameter becomes a problem in future.
str_report_type = ['season', 'game'];
str_datetype = ['season', 'date'];

# Other data, other than summary
str_page_team = ['faceoffpercentages', 'faceoffwins', 
                 'goalsagainstbystrength', 'goalsbyperiod','goalsforbystrength','leadingtrailing', 'outshootoutshotby',
                 'realtime', 
                 'penalties', 'penaltykill','penaltykilltime','powerplay','powerplaytime',
                 'summaryshooting', 'percentages', 'scoretrailfirst', 'shootout', 'shottype', 'goalgames'];
str_page_skater = ['bios', 'faceoffpercentages', 'faceoffwins', 'goalsForAgainst',
                   'realtime', 
                   'penalties', 'penaltykill', #'penaltyShots', # For some reason, the penalty shot page is throwing errors nto able to grab table. 
                   'powerplay', 'puckPossessions',
                   'summaryshooting', 'percentages', 'scoringRates', 'scoringpergame', 'shootout', 'shottype', 'timeonice'];
str_page_goalie = ['advanced', 'bios', 'daysrest', 'penaltyShots', 'savesByStrength', 'shootout', 'startedVsRelieved'];
```


```python
driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())
driver.delete_all_cookies(); # delete all cookies
driver.implicitly_wait(5) # secon.abs
```


```python
driver = webdriver.Chrome(executable_path = "chromedriver.exe", chrome_options=chrome_options)

driver.delete_all_cookies(); # delete all cookies
driver.implicitly_wait(3) # seconds
```

## Import functions to be used (in .py scripts)

```python
# PULLING BY SEASON
def nhl_pull(str_url):
    driver.get(str_url); # get to the url

    try: # Wait until the table appears - JavaScript table may appear slower than other page elements
        element = WebDriverWait(driver, 50).until(
            EC.presence_of_element_located((By.CLASS_NAME, "rt-table"))
        )
    finally:
        None
    time.sleep(2); #Just in case
    # Pull from information
    html = driver.page_source # Pull the script information
    soup = BeautifulSoup(html) # Soupify

    # Get table header
    rtheader = soup.find_all("div", {"class": "rt-table"})

    n_pagecount = int(soup.find_all("span", {"class": "-totalPages"})[0].text) - 1; # number of pages to scrape
    # NOTE: page numbers are zero indexed. be careful - using index, number of pages to pull
    # Inside a function - this is throwing an error

    tableheader = soup.find_all("div", {"class": "tableHeaderDiv"})[0].find_all("div", {"class": "rt-header-cell"})

    str_titles = ["idx_row"]#['season start', 'season end']
    for temp_str in tableheader:
        temp_str_extract = temp_str.get('title');
        if temp_str_extract == None:
            temp_str_extract
        else:
            str_titles.append(temp_str_extract)

    n_title = len(str_titles);

    # Pulling the data.
    table_data = soup.find_all("div", {"class": "rt-tbody"})[0].find_all("div", {"class" : "rt-tr-group"})

    ary_data = [];
    for idx_count, iter_row in enumerate(table_data):
        each_row = iter_row.find_all("div", {"class" : "rt-td"})
        temp_vec = [];
        for iter_col in each_row:
            temp_vec.append(iter_col.text) # save the data in order
        if idx_count == 0: #start creating the array
            ary_data = np.array(temp_vec)
        else: # Do horizontal stack
            ary_data = np.vstack((ary_data, np.array(temp_vec)))

    # Convert to data frame
    #     Note: converting to array just in case it becomes a one row list.
    df_data = pd.DataFrame(np.reshape(ary_data, (-1, len(str_titles))), columns = str_titles)
    
    # Pull total record count
    n_recordcount = int(soup.find_all("span", {"class": "-totalInfo"})[0].text.split()[0]);

    return {'df': df_data, 'n_pagecount': n_pagecount, 'n_title': n_title, "n_recordcount" : n_recordcount} # Return the dataframe of data & pagecount for multiple pages to pull

def strip_right(df, suffix):
    df.columns =  df.columns.str.rstrip(suffix)
    return df

# Pull URL of the team 
def url_team_pull(idx_data_type, idx_report, idx_report_type, iter_date_start, iter_date_end, str_gametype, i_npage, idx_datetype):
    URL_team_summary = (f"http://www.nhl.com/stats/"
                        f"{idx_data_type}?aggregate=0&{idx_report}reportType={idx_report_type}&"
                        f"{idx_datetype}From={iter_date_start}&{idx_datetype}To={iter_date_end}&"
                        f"gameType={str_gametype}&filter=gamesPlayed,gte,1&page={i_npage}&pageSize=100")
    # Note that in previous iteration idx_aggregate == 'aggregate=0&' - no need because the workflow is pulled by season.
    return URL_team_summary

def nhl_pull_loop(str_date_start, str_date_end, str_page, idx_data_type, idx_report_type, idx_datetype):
    for idx, iter_date_start in enumerate(str_date_start):
        iter_date_end = str_date_end[idx];
        df_fin = [];
        for idx_game, iter_game in enumerate(["regular", "playoff"]):
            # In-loop-specific initial settings
            str_gametype = idx_game + 2; # start with regular season
            i_npage = 0; # start with page 1
            idx_report = ''; # start with the summary page

            # temporary data frame save
            temp_df = [];
            
            URL_team_summary = url_team_pull(idx_data_type, idx_report, idx_report_type, iter_date_start, iter_date_end, str_gametype, i_npage, idx_datetype);

            temp_pulled = nhl_pull(URL_team_summary)
            temp_df = temp_pulled['df']; # Initialize
            npage = temp_pulled['n_pagecount'];
            nrecord = temp_pulled['n_recordcount'];

            if nrecord == 0:
                continue # break out from the loop.
            else: # Continue pulling the data for having a record
                # For more than one record
                if npage != 0:
                    for i_npage in range(1, npage + 1): # Python range, need to add one.
                        URL_team_summary = url_team_pull(idx_data_type, idx_report, idx_report_type, iter_date_start, iter_date_end, str_gametype, i_npage, idx_datetype);
                        temp_pulled = nhl_pull(URL_team_summary)

                        temp_df = temp_df.append(temp_pulled['df']);
                else:
                    None
                # All summary data pulled, remove empty rows
                temp_df = temp_df.loc[(temp_df.idx_row != '\xa0'),:];

                # Summary stats, just to check the right count of data.
                #temp_df.to_csv(f'df_{idx_data_type}_{idx_report_type}_{iter_season}_summaryOnly.csv',index = False)

                # Pull other data - more specific statistics,
                for temp_idx in str_page:
                    # Set specific parameters for different categories - pages
                    idx_report = "report=" + temp_idx + "&";
                    i_npage = 0; # start with page 1, Reset

                    URL_team_summary = url_team_pull(idx_data_type, idx_report, idx_report_type, iter_date_start, iter_date_end, str_gametype, i_npage, idx_datetype);

                    # Pull date
                    temp_pulled = nhl_pull(URL_team_summary)

                    # Because this is different categories - neeed to make master partial file
                    temp_df_partial = temp_pulled['df'];

                    # Need to join the data frame
                    npage = temp_pulled['n_pagecount']
                    if npage != 0: # Pull data from multiple pages
                        for i_npage in range(1, npage + 1): # Python range, need to add one.
                            URL_team_summary = url_team_pull(idx_data_type, idx_report, idx_report_type, iter_date_start, iter_date_end, str_gametype, i_npage, idx_datetype);
                            temp_pulled = nhl_pull(URL_team_summary); # Pull additional data
                            temp_df_partial = temp_df_partial.append(temp_pulled['df']); # stack multiple pages
                    else:
                        None
                    
                    # Save the data
                    #   First, must clean up the empty rows, just to make sure not to join empty-empty
                    temp_df_partial = temp_df_partial.loc[(temp_df_partial.idx_row != '\xa0'),:];

                    if (temp_pulled['df'].size != 0): # If the page has at least one entry 
                        if idx_data_type == 'teams': # For merging team statistics
                            if idx_report_type == 'season':
                                temp_df = pd.merge(temp_df, temp_df_partial, how = 'left', on = "Team", suffixes=('_x', '_y'))
                            elif idx_report_type == 'game':
                                temp_df = pd.merge(temp_df, temp_df_partial, how = 'left', on = ["Team", "Game"], suffixes=('_x', '_y'))
                            else:
                                None
                        else: ## For skaters and goalies
                            if idx_report_type == 'season':
                                if temp_idx == 'bios':
                                    if idx_data_type == 'skaters': # To match with unique player identity, in case there are players with same name in each period
                                        temp_df = pd.merge(temp_df, temp_df_partial, how = 'left', on = ['Player Name', 'Player Position', 'Games Played'], suffixes=('', '_y'))
                                    else: # For goalies
                                        temp_df = pd.merge(temp_df, temp_df_partial, how = 'left', on = ['Player Name', 'Goalie Catches'], suffixes=('', '_y'))
                                else:
                                    temp_df = pd.merge(temp_df, temp_df_partial, how = 'left', on = ["Player Name", "Teams Played For"], suffixes=('', '_y'))
                            elif idx_report_type == 'game':
                                if temp_idx == 'bios':
                                    # There is no common and reliable identification - instead of risk of many merges, we rely summary bio information in summary datafile.
                                    None
                                else:
                                    temp_df = pd.merge(temp_df, temp_df_partial, how = 'left', on = ["Player Name", 'Game'], suffixes=('', '_y'))
                            else:
                                None
                    else:
                        None
                    # End of appending relevant data to the summary table ====================================================================================================
                # End of pulling all data of the season/playoff of the year: specific table ==================================================================================

                # Remove redundant columns - the duplicate check and remove works from left to right (first ones appeard are kept, others are discarded)
                #   given the nature of left-join, the duplicate clear in order left to right works
                temp_df = strip_right(temp_df, '_x');
                temp_df = strip_right(temp_df, '_y');
                temp_df = temp_df.loc[:, ~temp_df.columns.duplicated()];

                # Save the game group type,
                temp_df['season_type'] = iter_game

                # For the first ieration, need to build the file
                #if (iter_game == "regular"): # pull from regular
                #    df_fin = temp_df;
                #else:
                #    df_fin = pd.concat([df_fin, temp_df]);
                try:
                    df_fin = pd.concat([df_fin, temp_df]);
                except: # If this is the first series pull
                    df_fin = temp_df;
            # End of pulling all data for the time period sought =============================================================================================================
        # Save the data for the time period - if there was any data points.
        try:
            df_fin.size # if no data was pulled, df_fin would be the list, and the loop should continue without saving any data

            df_fin.to_csv(f'df_{idx_data_type}_{idx_report_type}_{iter_date_start}.csv',index = False)
        except: # Do nothing, don't save the data
            None
        # End of pulling all data for the specific time sought ===============================================================================================================
    # End of pulling all data for the time sought ===========================================================================================================================
```


```python
str_data_type = ['teams', 'skaters', 'goalies'];
str_aggregate = ['', 'aggregate=0&'];
str_report_type = ['season', 'game'];
str_datetype = ['season', 'date'];

# Data block to pull
idx_datetype = str_datetype[1]; #Default to season, only use date for player-game stats.
# Test with skaters, per game
idx_data_type = str_data_type[1];
#idx_aggregate = str_aggregate[1];
idx_report_type = str_report_type[1];
str_page = str_page_skater;

nhl_pull_loop(str_date_start[697:], str_date_end[697:], str_page, idx_data_type, idx_report_type, idx_datetype)
```

## Pull Team data, by season and by game


```python
# Pull by team, by season
idx_datetype = str_datetype[0]; # Except pulling game-level player data, this should be 0 always
# If pulling game-level player data - change the first two entries of the function to str_date_start  and str_date_end, respectively
idx_data_type = str_data_type[0]; # team, skater, or goalie
idx_report_type = str_report_type[0]; # pull by season or by game level data 
str_page = str_page_team; # adjust based on what data type (team, skater, or goalie) you are pulling

nhl_pull_loop(str_season, str_season, str_page, idx_data_type, idx_report_type, idx_datetype)

# Pull by team, by game
idx_datetype = str_datetype[0]; # Except pulling game-level player data, this should be 0 always
# If pulling game-level player data - change the first two entries of the function to str_date_start  and str_date_end, respectively
idx_data_type = str_data_type[0]; # team, skater, or goalie
idx_report_type = str_report_type[1]; # pull by season or by game level data 
str_page = str_page_team; # adjust based on what data type (team, skater, or goalie) you are pulling

nhl_pull_loop(str_season, str_season, str_page, idx_data_type, idx_report_type, idx_datetype)
```

## Pull Skater data, by season and by game


```python
# Pull by skater, by season
idx_datetype = str_datetype[0]; # Except pulling game-level player data, this should be 0 always
# If pulling game-level player data - change the first two entries of the function to str_date_start  and str_date_end, respectively
idx_data_type = str_data_type[1]; # team, skater, or goalie
idx_report_type = str_report_type[0]; # pull by season or by game level data 
str_page = str_page_skater; # adjust based on what data type (team, skater, or goalie) you are pulling

nhl_pull_loop(str_season, str_season, str_page, idx_data_type, idx_report_type, idx_datetype)

# Pull by skater, by game
idx_datetype = str_datetype[1]; # Except pulling game-level player data, this should be 0 always
# If pulling game-level player data - change the first two entries of the function to str_date_start  and str_date_end, respectively
idx_data_type = str_data_type[1]; # team, skater, or goalie
idx_report_type = str_report_type[1]; # pull by season or by game level data 
str_page = str_page_skater; # adjust based on what data type (team, skater, or goalie) you are pulling

nhl_pull_loop(str_date_start, str_date_end, str_page, idx_data_type, idx_report_type, idx_datetype)
```

## Pull goalie data, by season and by game


```python
# Pull by goalie, by season
idx_datetype = str_datetype[0]; # Except pulling game-level player data, this should be 0 always
# If pulling game-level player data - change the first two entries of the function to str_date_start  and str_date_end, respectively
idx_data_type = str_data_type[2]; # team, skater, or goalie
idx_report_type = str_report_type[0]; # pull by season or by game level data 
str_page = str_page_goalie; # adjust based on what data type (team, skater, or goalie) you are pulling

nhl_pull_loop(str_season, str_season, str_page, idx_data_type, idx_report_type, idx_datetype)

# Pull by goalie, by game
idx_datetype = str_datetype[0]; # Except pulling game-level player data, this should be 0 always
# If pulling game-level player data - change the first two entries of the function to str_date_start  and str_date_end, respectively
idx_data_type = str_data_type[2]; # team, skater, or goalie
idx_report_type = str_report_type[1]; # pull by season or by game level data 
str_page = str_page_goalie; # adjust based on what data type (team, skater, or goalie) you are pulling

nhl_pull_loop(str_season, str_season, str_page, idx_data_type, idx_report_type, idx_datetype)
```

## Goalie Statistics


```python
df_examin = pd.read_csv('df_skaters_season_20202021.csv')
df_examin.groupby(['Player Name', 'Teams Played For']).first().reset_index()
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player Name</th>
      <th>Teams Played For</th>
      <th>idx_row</th>
      <th>Season</th>
      <th>Skater Shoots</th>
      <th>Player Position</th>
      <th>Games Played</th>
      <th>Goals</th>
      <th>Assists</th>
      <th>Points</th>
      <th>...</th>
      <th>Even Strength Time On Ice</th>
      <th>Even Strength Time On Ice Per Game Played</th>
      <th>Power Play Time On Ice Per Game Played</th>
      <th>Shorthanded Time On Ice Per Game Played</th>
      <th>Overtime Time on Ice (since 2009-10)</th>
      <th>Overtime Time on Ice Per Overtime Game Played (since 2009-10)</th>
      <th>Shifts</th>
      <th>Time On Ice Per Shift</th>
      <th>Shifts Per Game Played (since 1997-98)</th>
      <th>season_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A.J. Greer</td>
      <td>NJD</td>
      <td>847</td>
      <td>2020-21</td>
      <td>L</td>
      <td>L</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>8:33</td>
      <td>8:33</td>
      <td>0:00</td>
      <td>0:00</td>
      <td>0:00</td>
      <td>--</td>
      <td>12</td>
      <td>0:43</td>
      <td>12.0</td>
      <td>regular</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Aaron Ness</td>
      <td>ARI</td>
      <td>807</td>
      <td>2020-21</td>
      <td>L</td>
      <td>D</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>12:50</td>
      <td>12:50</td>
      <td>0:00</td>
      <td>0:00</td>
      <td>0:00</td>
      <td>0:00</td>
      <td>18</td>
      <td>0:43</td>
      <td>18.0</td>
      <td>regular</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adam Boqvist</td>
      <td>CHI</td>
      <td>332</td>
      <td>2020-21</td>
      <td>R</td>
      <td>D</td>
      <td>35</td>
      <td>2</td>
      <td>14</td>
      <td>16</td>
      <td>...</td>
      <td>474:10</td>
      <td>13:33</td>
      <td>3:23</td>
      <td>0:04</td>
      <td>4:00</td>
      <td>0:40</td>
      <td>689</td>
      <td>0:52</td>
      <td>19.7</td>
      <td>regular</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Adam Brooks</td>
      <td>TOR</td>
      <td>584</td>
      <td>2020-21</td>
      <td>L</td>
      <td>C</td>
      <td>11</td>
      <td>4</td>
      <td>1</td>
      <td>5</td>
      <td>...</td>
      <td>111:15</td>
      <td>10:07</td>
      <td>0:08</td>
      <td>0:26</td>
      <td>0:00</td>
      <td>0:00</td>
      <td>171</td>
      <td>0:41</td>
      <td>15.5</td>
      <td>regular</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alec Regula</td>
      <td>CHI</td>
      <td>889</td>
      <td>2020-21</td>
      <td>R</td>
      <td>D</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>35:31</td>
      <td>11:50</td>
      <td>0:32</td>
      <td>0:00</td>
      <td>0:00</td>
      <td>0:00</td>
      <td>43</td>
      <td>0:52</td>
      <td>14.3</td>
      <td>regular</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>432</th>
      <td>Zach Sanford</td>
      <td>STL</td>
      <td>322</td>
      <td>2020-21</td>
      <td>L</td>
      <td>L</td>
      <td>52</td>
      <td>10</td>
      <td>6</td>
      <td>16</td>
      <td>...</td>
      <td>643:08</td>
      <td>12:22</td>
      <td>0:57</td>
      <td>1:35</td>
      <td>0:00</td>
      <td>0:00</td>
      <td>1,018</td>
      <td>0:46</td>
      <td>19.6</td>
      <td>regular</td>
    </tr>
    <tr>
      <th>433</th>
      <td>Zach Senyshyn</td>
      <td>BOS</td>
      <td>852</td>
      <td>2020-21</td>
      <td>R</td>
      <td>R</td>
      <td>8</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>80:34</td>
      <td>10:04</td>
      <td>0:01</td>
      <td>0:02</td>
      <td>0:00</td>
      <td>0:00</td>
      <td>127</td>
      <td>0:38</td>
      <td>15.9</td>
      <td>regular</td>
    </tr>
    <tr>
      <th>434</th>
      <td>Zach Werenski</td>
      <td>CBJ</td>
      <td>264</td>
      <td>2020-21</td>
      <td>L</td>
      <td>D</td>
      <td>35</td>
      <td>7</td>
      <td>13</td>
      <td>20</td>
      <td>...</td>
      <td>711:14</td>
      <td>20:19</td>
      <td>2:10</td>
      <td>1:52</td>
      <td>15:12</td>
      <td>1:23</td>
      <td>987</td>
      <td>0:52</td>
      <td>28.2</td>
      <td>regular</td>
    </tr>
    <tr>
      <th>435</th>
      <td>Zach Whitecloud</td>
      <td>VGK</td>
      <td>429</td>
      <td>2020-21</td>
      <td>R</td>
      <td>D</td>
      <td>51</td>
      <td>2</td>
      <td>10</td>
      <td>12</td>
      <td>...</td>
      <td>807:55</td>
      <td>15:50</td>
      <td>0:03</td>
      <td>1:56</td>
      <td>0:00</td>
      <td>0:00</td>
      <td>1,115</td>
      <td>0:49</td>
      <td>21.9</td>
      <td>regular</td>
    </tr>
    <tr>
      <th>436</th>
      <td>Zack Kassian</td>
      <td>EDM</td>
      <td>595</td>
      <td>2020-21</td>
      <td>R</td>
      <td>R</td>
      <td>27</td>
      <td>2</td>
      <td>3</td>
      <td>5</td>
      <td>...</td>
      <td>314:55</td>
      <td>11:40</td>
      <td>0:11</td>
      <td>0:00</td>
      <td>0:00</td>
      <td>0:00</td>
      <td>400</td>
      <td>0:48</td>
      <td>14.8</td>
      <td>regular</td>
    </tr>
  </tbody>
</table>
<p>437 rows × 255 columns</p>
</div>


[back to the main page](../index.md)
