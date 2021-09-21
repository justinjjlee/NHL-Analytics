# Using NHL API for player in game level statistics from NHL.com
Description of the API pull is discussed in [NHL API pull via Python](https://gitlab.com/dword4/nhlapi)


```python
!pip3 install pickle5
```

Import relevant packages


```python
import pickle5 as pickle
import requests
import numpy as np 
import pandas as pd 
```

 In case using Google CoLab like me,


```python
# Connect to my driver from google CoLab
from google.colab import drive
drive.mount('/content/drive')
```

    Mounted at /content/drive
    


```python
# Set up the API call variables
year_last = 2021
ssn = ['02', '03']; # Regular season and playoff, respectively
# just to be conservative in case there are data not accounted
max_game_ID = 3000; 
#reverse the list order
yr = list(map(str, list(range(1960, year_last))))[::-1]; 
```

Pull all relevant seasons

```python
# Loop over the counter and format the API call

for year in yr:
    for season_type in ssn:
        game_data = []
        for i in range(0,max_game_ID):
            r = requests.get(url='http://statsapi.web.nhl.com/api/v1/game/'
                + year + season_type +str(i).zfill(4)+'/feed/live')
            data = r.json()
            game_data.append(data)
        # Save as pickle file
        with open('/content/drive/My Drive/Learning/sports/nhl/' 
                  + year + '_' + season_type + 'FullDataset.pkl', 'wb'
                  ) as f:
            pickle.dump(game_data, f, pickle.HIGHEST_PROTOCOL)
        # Data pull done, let me know.
        print('Done! - year ' + year + ' and season type: ' + season_type)
```

    Done! - year 2020 and season type: 02
    Done! - year 2020 and season type: 03
    Done! - year 2019 and season type: 02
    Done! - year 2019 and season type: 03
    ...

[back](/collection_index.md)
