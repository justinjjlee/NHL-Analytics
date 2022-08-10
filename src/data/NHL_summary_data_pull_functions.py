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