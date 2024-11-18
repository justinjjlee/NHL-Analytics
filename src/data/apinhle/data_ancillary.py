# %% Data download logo images
import os
import requests

# List of teams (replace with actual team abbreviations)
teams = ["ANA", "BOS", "BUF", "CGY", "CAR", "CHI", "COL", "CBJ", "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SJS", "SEA", "STL", "TBL", "TOR", "UTA", "VAN", "VGK", "WSH", "WPG"]

# URL base for the images
url_base = "https://assets.nhle.com/logos/nhl/svg/{home_team}_light.svg"

# Directory to save images
#   Change as needed
save_dir = "logo"

# Create directory if it doesn't exist
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Download each image
for team in teams:
    url = url_base.format(home_team=team)
    try:
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful
        
        # Save the image
        image_path = os.path.join(save_dir, f"{team}_logo.svg")
        with open(image_path, "wb") as file:
            file.write(response.content)
        print(f"Downloaded {team}_logo.svg")
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {team}_logo.svg: {e}")

# %%
