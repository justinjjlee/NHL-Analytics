import pandas as pd
import numpy as np
import requests
import os
import time

data_dir = os.path.dirname(os.path.abspath(__file__))
merged_file = os.path.join(data_dir, 'merged_draft_player_stats.csv')
chronicle_file = os.path.join(data_dir, 'all_drafted_players_chronicle.csv')

print("Loading data files...")
merged = pd.read_csv(merged_file)
if os.path.exists(chronicle_file):
    chronicle = pd.read_csv(chronicle_file)
    chronicle_ids = set(chronicle['id'].dropna().astype(int).unique())
else:
    chronicle = pd.DataFrame()
    chronicle_ids = set()

merged_ids = set(merged['id'].dropna().astype(int).unique())
missing_ids = sorted(list(merged_ids - chronicle_ids))

print(f"Total unique IDs in merged: {len(merged_ids)}")
print(f"IDs already in chronicle: {len(chronicle_ids)}")
print(f"IDs to fetch: {len(missing_ids)}")

if not missing_ids:
    print("All player chronicles already fetched!")
    exit(0)

# Sequential fetch function with retries and sleep to avoid rate limits
def fetch_player_chronicle(player_id):
    url = f"https://api-web.nhle.com/v1/player/{player_id}/landing"
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                totals = data.get('seasonTotals', [])
                if totals:
                    df = pd.json_normalize(totals)
                    df['id'] = player_id
                    return df
                return None
            elif response.status_code == 429:
                print(f"Rate limited on ID {player_id}, sleeping 5 seconds...")
                time.sleep(5)
            else:
                return None
        except Exception as e:
            print(f"Attempt {attempt+1} failed for ID {player_id}: {e}")
            time.sleep(2)
    return None

results = []
completed = 0
total_to_fetch = len(missing_ids)

print("Starting rate-limit-friendly sequential fetch...")
for pid in missing_ids:
    res_df = fetch_player_chronicle(pid)
    if res_df is not None:
        results.append(res_df)
    completed += 1
    if completed % 50 == 0:
        print(f"Completed {completed}/{total_to_fetch} fetches...")
    time.sleep(0.2)  # 200ms delay between requests to be nice to the API

if results:
    new_chronicles = pd.concat(results, ignore_index=True)
    drop_cols = [
        'teamCommonName.cs', 'teamCommonName.de', 'teamCommonName.sk',
        'teamCommonName.sv', 'teamName.cs', 'teamName.de', 'teamName.fi',
        'teamName.sk', 'teamName.sv', 'teamName.fr', 'teamPlaceNameWithPreposition.fr',
        'teamCommonName.default', 'teamCommonName.es',
        'teamCommonName.fi', 'teamCommonName.fr',
        'teamPlaceNameWithPreposition.default'
    ]
    for col in drop_cols:
        if col in new_chronicles.columns:
            new_chronicles = new_chronicles.drop(columns=[col])

    if not chronicle.empty:
        all_cols = list(set(chronicle.columns) | set(new_chronicles.columns))
        for col in all_cols:
            if col not in chronicle.columns:
                chronicle[col] = np.nan
            if col not in new_chronicles.columns:
                new_chronicles[col] = np.nan
        combined = pd.concat([chronicle[all_cols], new_chronicles[all_cols]], ignore_index=True)
    else:
        combined = new_chronicles

    combined.to_csv(chronicle_file, index=False)
    print(f"Successfully added {len(results)} player chronicles. Total records: {len(combined)}")
else:
    print("No new chronicles fetched.")
