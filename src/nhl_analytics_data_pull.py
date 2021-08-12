import requests
import time
from datetime import date, datetime
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

from nhl_data_pull_function import *

# Data setting for pulling data
int_year = 2022; # last year of season to pull.
int_year_start = int_year - 1; # season starts.

# Your choice of browser to pull data
# Using Firefox
driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())
driver.delete_all_cookies(); # delete all cookies
driver.implicitly_wait(5) # secon.abs

# Using Chrome - you need chromedriver installed for this
#driver = webdriver.Chrome(executable_path = "YOUR CHROMEDRIVER FILE LOCATION", chrome_options=chrome_options)
#driver.delete_all_cookies(); # delete all cookies
#driver.implicitly_wait(3) # seconds

### No need to update below ===================================================================================================================
# Setting time and season vector
str_season = [str(int_year_start) + str(int_year)]
# For data strucutre pages that has more than 10,000 records - need to break things down by month
str_date_start = [iter.strftime('%Y-%m-%d') for iter in pd.date_range(start=f"{int_year_start}-08-10", end=f"{int_year}-08-01", freq='MS')];
str_date_end = [iter.strftime('%Y-%m-%d') for iter in pd.date_range(start=f"{int_year_start}-09-10", end=f"{int_year}-09-01", freq='M')];


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

# For those needing to pull all data, not just the latest - uncomment and run below
'''
# define basics
vec_season_start = list(map(str, list(range(1917, (int_year_start + 1)))))
vec_season_end = list(map(str, list(range(1918, (int_year + 1)))))

# Looking at each season, so no need to worry about, with optional pulling data in reverse order
str_season = [vec_season_start[i] + vec_season_end[i] for i, val in enumerate(vec_season_start)][::-1]; 

# For data strucutre pages that has more than 10,000 records - need to break things down by month
str_date_start = [iter.strftime('%Y-%m-%d') for iter in pd.date_range(start="1917-11-10", end=f"{int_year}-08-01", freq='MS')]
str_date_end = [iter.strftime('%Y-%m-%d') for iter in pd.date_range(start="1917-12-10", end=f"{int_year}-09-01", freq='M')]
'''

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

# Pull by skater, by season
idx_datetype = str_datetype[0]; # Except pulling game-level player data, this should be 0 always
# If pulling game-level player data - change the first two entries of the function to str_date_start  and str_date_end, respectively
idx_data_type = str_data_type[1]; # team, skater, or goalie
idx_report_type = str_report_type[0]; # pull by season or by game level data 
str_page = str_page_skater; # adjust based on what data type (team, skater, or goalie) you are pulling

nhl_pull_loop(str_season, str_season, str_page, idx_data_type, idx_report_type, idx_datetype)
'''
# Pull by skater, by game
idx_datetype = str_datetype[1]; # Except pulling game-level player data, this should be 0 always
# If pulling game-level player data - change the first two entries of the function to str_date_start  and str_date_end, respectively
idx_data_type = str_data_type[1]; # team, skater, or goalie
idx_report_type = str_report_type[1]; # pull by season or by game level data 
str_page = str_page_skater; # adjust based on what data type (team, skater, or goalie) you are pulling

nhl_pull_loop(str_date_start, str_date_end, str_page, idx_data_type, idx_report_type, idx_datetype)
'''
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