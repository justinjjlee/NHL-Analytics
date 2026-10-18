import os
import datetime

# Check if running in Databricks
IS_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ

if IS_DATABRICKS:
    # Databricks Unity Catalog Volume Paths
    BASE_DATA_DIR = "/Volumes/nhl-databricks/data"
    BOX_DIR = "/Volumes/nhl-databricks/data/box"
    TEAM_DIR = "/Volumes/nhl-databricks/data/team"
    PLAY_DIR = "/Volumes/nhl-databricks/data/play"
    BETTING_DIR = "/Volumes/nhl-databricks/data/betting"
    # Player metrics: Use dedicated volume if present, otherwise team volume
    PLAYER_DIR = "/Volumes/nhl-databricks/data/player" if os.path.exists("/Volumes/nhl-databricks/data/player") else TEAM_DIR
    # Hits metrics: Use dedicated action volume if present, otherwise fallback to play volume
    if os.path.exists("/Volumes/nhl-databricks/data/action"):
        ACTION_HITS_DIR = "/Volumes/nhl-databricks/data/action/hits"
        ACTION_SHIFTS_DIR = "/Volumes/nhl-databricks/data/action/shifts"
    elif os.path.exists("/Volumes/nhl-databricks/data/play"):
        ACTION_HITS_DIR = "/Volumes/nhl-databricks/data/play/action_hits"
        ACTION_SHIFTS_DIR = "/Volumes/nhl-databricks/data/play/action_shifts"
    else:
        ACTION_HITS_DIR = "/Volumes/nhl-databricks/data/action/hits"
        ACTION_SHIFTS_DIR = "/Volumes/nhl-databricks/data/action/shifts"
else:
    # Local GitHub Actions Paths (relative to project root)
    BASE_DATA_DIR = "./latest"
    BOX_DIR = "./latest/box"
    TEAM_DIR = "./latest/team"
    PLAY_DIR = "./latest/play"
    BETTING_DIR = "./latest/box" # betting originally saved to box locally
    PLAYER_DIR = "./latest/player"
    ACTION_HITS_DIR = "./latest/action/hits"
    ACTION_SHIFTS_DIR = "./latest/action/shifts"

# Function to get the path config so we don't repeat logic
def get_base_data_dir(): return BASE_DATA_DIR
def get_box_dir(): return BOX_DIR
def get_team_dir(): return TEAM_DIR
def get_play_dir(): return PLAY_DIR
def get_betting_dir(): return BETTING_DIR
def get_action_hits_dir():
    if IS_DATABRICKS:
        if os.path.exists("/Volumes/nhl-databricks/data/action"):
            path = "/Volumes/nhl-databricks/data/action/hits"
        elif os.path.exists("/Volumes/nhl-databricks/data/play"):
            path = "/Volumes/nhl-databricks/data/play/action_hits"
        else:
            path = "/Volumes/nhl-databricks/data/action/hits"
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            pass
        return path
    return ACTION_HITS_DIR

def get_action_shifts_dir():
    if IS_DATABRICKS:
        if os.path.exists("/Volumes/nhl-databricks/data/action"):
            path = "/Volumes/nhl-databricks/data/action/shifts"
        elif os.path.exists("/Volumes/nhl-databricks/data/play"):
            path = "/Volumes/nhl-databricks/data/play/action_shifts"
        else:
            path = "/Volumes/nhl-databricks/data/action/shifts"
        try:
            os.makedirs(path, exist_ok=True)
        except Exception:
            pass
        return path
    return ACTION_SHIFTS_DIR
def get_player_dir():
    if IS_DATABRICKS:
        if os.path.exists("/Volumes/nhl-databricks/data/player"):
            return "/Volumes/nhl-databricks/data/player"
        return TEAM_DIR
    return PLAYER_DIR

def get_teamlist_path():
    """
    Resolve path to teamlist.csv across local and Databricks environments.
    If running in Databricks and missing from TEAM_DIR, auto-seeds from the repository clone.
    """
    primary = os.path.join(TEAM_DIR, "teamlist.csv")
    if os.path.exists(primary):
        return primary

    base_dir = os.path.dirname(__file__)
    candidates = [
        os.path.abspath(os.path.join(base_dir, "../../../latest/team/teamlist.csv")),
        os.path.abspath(os.path.join(base_dir, "../../../app/data/teamlist.csv")),
        "./latest/team/teamlist.csv",
        "./app/data/teamlist.csv",
    ]
    for cand in candidates:
        if os.path.exists(cand):
            if IS_DATABRICKS and os.path.exists(TEAM_DIR):
                try:
                    import shutil
                    shutil.copy2(cand, primary)
                    return primary
                except Exception:
                    pass
            return cand
    return primary

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

SEASON_YEARS = [get_current_season_year()]
ALL_SEASON_YEARS = list(range(2013, get_current_season_year() + 1))

# Directory initialization
if not IS_DATABRICKS:
    for d in [BOX_DIR, TEAM_DIR, os.path.join(TEAM_DIR, 'season'), PLAY_DIR, PLAYER_DIR, ACTION_HITS_DIR]:
        os.makedirs(d, exist_ok=True)
else:
    if os.path.exists(TEAM_DIR):
        os.makedirs(os.path.join(TEAM_DIR, 'season'), exist_ok=True)
    try:
        if os.path.exists("/Volumes/nhl-databricks/data/action"):
            os.makedirs("/Volumes/nhl-databricks/data/action/hits", exist_ok=True)
        elif os.path.exists("/Volumes/nhl-databricks/data/play"):
            os.makedirs("/Volumes/nhl-databricks/data/play/action_hits", exist_ok=True)
    except Exception:
        pass

