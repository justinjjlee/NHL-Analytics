import os

# Check if running in Databricks
IS_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ

if IS_DATABRICKS:
    # Databricks Unity Catalog Volume Paths
    BOX_DIR = "/Volumes/nhl-databricks/data/box"
    TEAM_DIR = "/Volumes/nhl-databricks/data/team"
    PLAY_DIR = "/Volumes/nhl-databricks/data/play"
    BETTING_DIR = "/Volumes/nhl-databricks/data/betting"
else:
    # Local GitHub Actions Paths (relative to project root)
    BOX_DIR = "./latest/box"
    TEAM_DIR = "./latest/team"
    PLAY_DIR = "./latest/play"
    BETTING_DIR = "./latest/box" # betting originally saved to box locally

# Function to get the path config so we don't repeat logic
def get_box_dir(): return BOX_DIR
def get_team_dir(): return TEAM_DIR
def get_play_dir(): return PLAY_DIR
def get_betting_dir(): return BETTING_DIR

# Only check this here as a fallback; locally we should run from root
if not IS_DATABRICKS:
    for d in [BOX_DIR, TEAM_DIR, os.path.join(TEAM_DIR, 'season'), PLAY_DIR]:
        os.makedirs(d, exist_ok=True)
