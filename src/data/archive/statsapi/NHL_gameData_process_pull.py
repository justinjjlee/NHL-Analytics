# See
## Using NHL API for player in game level statistics
#https://towardsdatascience.com/nhl-analytics-with-python-6390c5d3206d
#[NHL API pull via python](https://gitlab.com/dword4/nhlapi)

#!pip3 install pickle5
import pickle5 as pickle
import requests
import numpy as np 
import pandas as pd 

# Set up the API call variables
year_last = '2020'
season_type = '02' 
max_game_ID = 3000; # just to be conservative in case there are data not accounted

yr = list(map(str, list(range(1917, year_last))))[::-1]; #reverse the list order

# Loop over the counter and format the API call

for year in yr:
    for season_type in ssn:
        game_data = []
        for i in range(0,max_game_ID):
            r = requests.get(url='https://api-web.nhle.com/v1/'
                + year + season_type +str(i).zfill(4)+'/feed/live')
            data = r.json()
            game_data.append(data)
        # Save as pickle file
        with open('/content/drive/My Drive/Learning/sports/nhl/'+year+'_'+ season_type + 'FullDataset.pkl', 'wb') as f:
            pickle.dump(game_data, f, pickle.HIGHEST_PROTOCOL)
        # Data pull done, let me know.
        print('Done! - year ' + year + ' and season type: ' + season_type)