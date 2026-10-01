import os
import datetime

# Check if running in Databricks
IS_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ

if IS_DATABRICKS:
    # Databricks Unity Catalog Volume Paths
    BOX_DIR = "/Volumes/nhl-databricks/data/box"
    TEAM_DIR = "/Volumes/nhl-databricks/data/team"
    PLAY_DIR = "/Volumes/nhl-databricks/data/play"
    BETTING_DIR = "/Volumes/nhl-databricks/data/betting"
    PLAYER_DIR = "/Volumes/nhl-databricks/data/player"
else:
    # Local GitHub Actions Paths (relative to project root)
    BOX_DIR = "./latest/box"
    TEAM_DIR = "./latest/team"
    PLAY_DIR = "./latest/play"
    BETTING_DIR = "./latest/box" # betting originally saved to box locally
    PLAYER_DIR = "./latest/player"

# Function to get the path config so we don't repeat logic
def get_box_dir(): return BOX_DIR
def get_team_dir(): return TEAM_DIR
def get_play_dir(): return PLAY_DIR
def get_betting_dir(): return BETTING_DIR
def get_player_dir(): return PLAYER_DIR

def get_current_season_year():
    """
    Get the starting calendar year of the current NHL season.
    NHL season schedule starts around August/September and concludes in June.
    - Months 8-12 (Aug-Dec): current calendar year (e.g. Oct 2026 -> 2026 for season 20262027)
    - Months 1-7 (Jan-Jul): previous calendar year (e.g. Feb 2027 -> 2026 for season 20262027)
    """
    today = datetime.datetime.today()
    if today.month < 8:
        return today.year - 1
    return today.year

# Only check this here as a fallback; locally we should run from root
if not IS_DATABRICKS:
    for d in [BOX_DIR, TEAM_DIR, os.path.join(TEAM_DIR, 'season'), PLAY_DIR, PLAYER_DIR]:
        os.makedirs(d, exist_ok=True)
