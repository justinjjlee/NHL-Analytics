"""
player_stats.py
Data engine for skater and goalie stats pages.
"""
import streamlit as st
import pandas as pd
import numpy as np
import glob
import os
import re


def get_data_path(rel_path):
    """Resolve absolute path relative to project root."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    return os.path.join(project_root, rel_path)


# ─────────────────────────────────────────────────────────────────────────────
# Season-level data
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_player_season_data():
    """
    Load all latest/player/{YYYY}_player.csv files.
    Returns (skaters_df, goalies_df) with a 'season_label' column.
    """
    player_dir = get_data_path('latest/player')
    files = sorted(glob.glob(os.path.join(player_dir, '*_player.csv')))
    if not files:
        return None, None

    frames = []
    for f in files:
        year = os.path.basename(f).split('_')[0]
        try:
            df = pd.read_csv(f)
            df['season_label'] = f"{year}-{str(int(year)+1)[2:]}"
            df['season_year'] = int(year)
            frames.append(df)
        except Exception:
            pass

    if not frames:
        return None, None

    all_df = pd.concat(frames, ignore_index=True)

    all_df.rename(columns={
        'firstName.default': 'firstName',
        'lastName.default': 'lastName',
    }, inplace=True)

    # ── Correct names with accents from box score files + manual overrides ──
    box_dir = get_data_path('latest/box')
    box_files = sorted(glob.glob(os.path.join(box_dir, '*_box_player.csv')))
    id_to_accent = {}
    for f in box_files:
        try:
            df_b = pd.read_csv(f, low_memory=False)
            cols = ['name.cs', 'name.sk', 'name.fi', 'name.sv', 'name.de', 'name.es', 'name.fr']
            avail_cols = [c for c in cols if c in df_b.columns]
            if not avail_cols:
                continue
            subset = df_b.dropna(subset=['playerId']).copy()
            subset['playerId'] = subset['playerId'].astype(int)
            # Find the last non-null column for each player
            for col in avail_cols:
                rows = subset[subset[col].notna()]
                for _, row in rows.iterrows():
                    id_to_accent[int(row['playerId'])] = row[col]
        except Exception:
            pass

    all_df['fullName'] = all_df['firstName'].fillna('') + ' ' + all_df['lastName'].fillna('')
    for idx, row in all_df.iterrows():
        pid = row['playerId']
        if pd.isna(pid):
            continue
        pid = int(pid)
        # Manual overrides for names missing or incorrect in data (e.g. Jakub Dobeš)
        manual_overrides = {
            8482487: 'Jakub Dobeš'
        }
        if pid in manual_overrides:
            all_df.at[idx, 'fullName'] = manual_overrides[pid]
            # also split it
            all_df.at[idx, 'lastName'] = manual_overrides[pid].split()[-1]
            continue
        if pid in id_to_accent:
            box_name = id_to_accent[pid]
            box_parts = box_name.strip().split()
            if box_parts:
                box_last = box_parts[-1]
                # Compare stripped names to see if they are phonetically identical
                def clean(s):
                    import unicodedata
                    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').lower()
                if clean(box_last) == clean(str(row['lastName'])):
                    all_df.at[idx, 'lastName'] = box_last
                    all_df.at[idx, 'fullName'] = f"{row['firstName']} {box_last}"

    all_df['fullName'] = all_df['fullName'].str.strip()

    # Regular season only
    all_df = all_df[all_df['idx_season_type'].fillna(2) == 2].copy()

    numeric_cols = [
        'gamesPlayed', 'goals', 'assists', 'points', 'plusMinus',
        'shots', 'shootingPctg', 'avgTimeOnIcePerGame', 'faceoffWinPctg',
        'penaltyMinutes', 'powerPlayGoals', 'shorthandedGoals',
        'gameWinningGoals', 'overtimeGoals', 'avgShiftsPerGame',
        'gamesStarted', 'wins', 'losses', 'overtimeLosses',
        'goalsAgainstAverage', 'savePercentage', 'shotsAgainst',
        'saves', 'goalsAgainst', 'shutouts',
    ]
    for col in numeric_cols:
        if col in all_df.columns:
            all_df[col] = pd.to_numeric(all_df[col], errors='coerce')

    goalies = all_df[all_df['positionCode'] == 'G'].copy()
    skaters = all_df[all_df['positionCode'].isin(['C', 'L', 'R', 'D'])].copy()

    # Derived skater stats
    skaters['points_per_game'] = skaters['points'] / skaters['gamesPlayed'].replace(0, np.nan)
    skaters['goals_per_game'] = skaters['goals'] / skaters['gamesPlayed'].replace(0, np.nan)
    skaters['assists_per_game'] = skaters['assists'] / skaters['gamesPlayed'].replace(0, np.nan)
    skaters['shots_per_game'] = skaters['shots'] / skaters['gamesPlayed'].replace(0, np.nan)
    skaters['toi_per_game_min'] = skaters['avgTimeOnIcePerGame'].apply(_sec_to_mmss)

    # Derived goalie stats
    goalies['saves_per_game'] = goalies['saves'] / goalies['gamesStarted'].replace(0, np.nan)
    goalies['win_pct'] = goalies['wins'] / goalies['gamesPlayed'].replace(0, np.nan)

    return skaters, goalies



# ─────────────────────────────────────────────────────────────────────────────
# Game-level data
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_player_game_data():
    """
    Load all latest/box/{YYYY}_box_player.csv files.
    Returns (skater_games_df, goalie_games_df).
    """
    box_dir = get_data_path('latest/box')
    files = sorted(glob.glob(os.path.join(box_dir, '*_box_player.csv')))
    if not files:
        return None, None

    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            frames.append(df)
        except Exception:
            pass

    if not frames:
        return None, None

    all_df = pd.concat(frames, ignore_index=True)

    # Resolve best player name: pick last non-null locale variant (native spelling)
    # Priority: cs/sk (used for Slavic + Nordic diacritics) > fi > de > default
    locale_cols = ['name.cs', 'name.sk', 'name.fi', 'name.de', 'name.es']
    all_df['playerName'] = all_df.get('name.default', pd.Series(dtype=str))
    for col in locale_cols:
        if col in all_df.columns:
            all_df['playerName'] = all_df['playerName'].where(
                all_df[col].isna(), all_df[col]
            )

    all_df['gameDate'] = pd.to_datetime(all_df['gameDate'], errors='coerce')

    # Derive season_year from gameDate: Oct-Dec → that year; Jan-Sep → previous year
    # e.g., 2024-10-04 → 2024 (the "2024-25" season); 2025-03-01 → 2024
    all_df['season_year'] = all_df['gameDate'].apply(
        lambda d: d.year if pd.notna(d) and d.month >= 10 else (d.year - 1 if pd.notna(d) else None)
    )

    num_cols = [
        'goals', 'assists', 'points', 'plusMinus', 'pim', 'hits',
        'powerPlayGoals', 'sog', 'faceoffWinningPctg', 'blockedShots',
        'shifts', 'giveaways', 'takeaways',
        'savePctg', 'goalsAgainst', 'shotsAgainst', 'saves',
        'evenStrengthGoalsAgainst', 'powerPlayGoalsAgainst',
    ]
    for col in num_cols:
        if col in all_df.columns:
            all_df[col] = pd.to_numeric(all_df[col], errors='coerce')

    # Derive opponent team: abbrev is only populated for the first player per team per game.
    # Build gameid → {home: abbrev, away: abbrev} lookup, then broadcast to all rows.
    team_abbrev_rows = all_df[all_df['abbrev'].notna()][['gameid', 'teamloc', 'abbrev']].drop_duplicates()
    home_map = team_abbrev_rows[team_abbrev_rows['teamloc'] == 'home'].set_index('gameid')['abbrev'].to_dict()
    away_map = team_abbrev_rows[team_abbrev_rows['teamloc'] == 'away'].set_index('gameid')['abbrev'].to_dict()

    all_df['own_team'] = all_df.apply(
        lambda r: home_map.get(r['gameid']) if r['teamloc'] == 'home' else away_map.get(r['gameid']), axis=1
    )
    all_df['opp_team'] = all_df.apply(
        lambda r: away_map.get(r['gameid']) if r['teamloc'] == 'home' else home_map.get(r['gameid']), axis=1
    )

    skater_games = all_df[all_df['position'].isin(['C', 'L', 'R', 'D'])].copy()
    goalie_games = all_df[all_df['position'] == 'G'].copy()

    return skater_games, goalie_games


# ─────────────────────────────────────────────────────────────────────────────
# Player Performance Index (PPI)
# ─────────────────────────────────────────────────────────────────────────────

def compute_skater_ppi(skaters_df, season_year=None, min_games=20):
    """
    PPI = mean z-score across: pts/GP, G/GP, A/GP, +/-, SOG/GP, Sh%, TOI/GP(s)
    """
    df = skaters_df.copy()
    if season_year:
        df = df[df['season_year'] == season_year]
    df = df[df['gamesPlayed'] >= min_games].copy()
    if len(df) < 5:
        return pd.DataFrame()

    ppi_cols = [
        'points_per_game', 'goals_per_game', 'assists_per_game',
        'plusMinus', 'shots_per_game', 'shootingPctg', 'avgTimeOnIcePerGame',
    ]
    valid_cols = [c for c in ppi_cols if c in df.columns and df[c].notna().sum() > 0]

    z_scores = pd.DataFrame(index=df.index)
    for col in valid_cols:
        mu, sigma = df[col].mean(), df[col].std()
        z_scores[col] = (df[col] - mu) / sigma if sigma > 0 else 0.0

    df['PPI'] = z_scores[valid_cols].mean(axis=1).round(3)

    display_cols = [
        'fullName', 'team_tri', 'positionCode', 'gamesPlayed',
        'goals', 'assists', 'points', 'points_per_game',
        'plusMinus', 'shots', 'shootingPctg', 'toi_per_game_min',
        'powerPlayGoals', 'shorthandedGoals', 'gameWinningGoals',
        'penaltyMinutes', 'faceoffWinPctg', 'PPI'
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    return df[display_cols].sort_values('PPI', ascending=False).reset_index(drop=True)


def compute_goalie_ppi(goalies_df, season_year=None, min_games=15):
    """
    PPI = mean z-score of: SV%, -GAA, wins, shutouts, saves/GP
    """
    df = goalies_df.copy()
    if season_year:
        df = df[df['season_year'] == season_year]
    df = df[df['gamesPlayed'] >= min_games].copy()
    if len(df) < 5:
        return pd.DataFrame()

    if 'goalsAgainstAverage' in df.columns:
        df['gaa_inv'] = -df['goalsAgainstAverage']

    ppi_cols = ['savePercentage', 'gaa_inv', 'wins', 'shutouts', 'saves_per_game']
    valid_cols = [c for c in ppi_cols if c in df.columns and df[c].notna().sum() > 0]

    z_scores = pd.DataFrame(index=df.index)
    for col in valid_cols:
        mu, sigma = df[col].mean(), df[col].std()
        z_scores[col] = (df[col] - mu) / sigma if sigma > 0 else 0.0

    df['PPI'] = z_scores[valid_cols].mean(axis=1).round(3)

    display_cols = [
        'fullName', 'team_tri', 'gamesPlayed', 'gamesStarted',
        'wins', 'losses', 'overtimeLosses',
        'goalsAgainstAverage', 'savePercentage', 'shutouts',
        'saves', 'shotsAgainst', 'win_pct', 'PPI'
    ]
    display_cols = [c for c in display_cols if c in df.columns]
    return df[display_cols].sort_values('PPI', ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sec_to_mmss(secs):
    try:
        secs = float(secs)
        m = int(secs // 60)
        s = int(secs % 60)
        return f"{m}:{s:02d}"
    except Exception:
        return ""


def get_available_seasons(df):
    if df is None or 'season_year' not in df.columns:
        return []
    pairs = df[['season_year', 'season_label']].drop_duplicates().dropna()
    return sorted(pairs.itertuples(index=False, name=None), key=lambda x: x[0])


def filter_player_by_name(df, search_term):
    if not search_term:
        return df
    mask = df['fullName'].str.contains(search_term, case=False, na=False)
    return df[mask]


def get_player_list(df):
    """
    Return sorted list of unique full player names from season-level data.
    Uses fullName which already carries native Unicode spelling.
    """
    if df is None or 'fullName' not in df.columns:
        return []
    return sorted(df['fullName'].dropna().unique().tolist())


def _strip_accents(text):
    """Normalize unicode string, removing diacritical marks for fuzzy matching."""
    import unicodedata
    if not isinstance(text, str):
        return ''
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    ).lower()


def resolve_display_name_df(df, lang='FR'):
    """
    Dynamically update 'playerName' column in df based on current active language.
    If FR, check name.fr first. Otherwise/fallback, check other locales from right to left (last available).
    """
    if df is None or df.empty:
        return df
    df = df.copy()
    
    # Locales in order of priority (non-FR, last available from box files columns)
    locales_non_fr = ['name.es', 'name.de', 'name.sv', 'name.sk', 'name.fi', 'name.cs']
    
    def resolve_row(row):
        # 1. French language selection check
        if lang == 'FR' and 'name.fr' in row and pd.notna(row['name.fr']) and str(row['name.fr']).strip():
            return row['name.fr']
            
        # 2. Check other locales (last available)
        for loc in locales_non_fr:
            if loc in row and pd.notna(row[loc]) and str(row[loc]).strip():
                return row[loc]
                
        # 3. Fallback to default
        return row.get('name.default', row.get('playerName', ''))
        
    df['playerName'] = df.apply(resolve_row, axis=1)
    return df


def resolve_game_player_name(game_df, player_id, lang='FR'):
    """
    Given a player_id from season data, find matching rows in game_df.
    Dynamically resolves playerName to the correct language locale.
    """
    if pd.isna(player_id) or game_df is None or game_df.empty:
        return pd.DataFrame()
        
    # Match using numeric playerId
    result = game_df[game_df['playerId'].astype(float) == float(player_id)].copy()
    
    # Resolve correct language name format
    return resolve_display_name_df(result, lang=lang)



def aggregate_weekly_skater(game_df):
    """
    Aggregate skater game log to weekly frequency.
    Returns df with week_start (Monday), games played that week, and summed stats.
    Opponent shown as comma-separated list of teams faced that week.
    """
    if game_df is None or game_df.empty:
        return pd.DataFrame()
    df = game_df.copy()
    df['week_start'] = df['gameDate'] - pd.to_timedelta(df['gameDate'].dt.dayofweek, unit='D')
    df['week_start'] = df['week_start'].dt.normalize()

    agg_cols = {
        'goals': 'sum', 'assists': 'sum', 'points': 'sum', 'plusMinus': 'sum',
        'pim': 'sum', 'hits': 'sum', 'powerPlayGoals': 'sum',
        'sog': 'sum', 'blockedShots': 'sum', 'giveaways': 'sum', 'takeaways': 'sum',
        'shifts': 'sum', 'gameDate': 'count',  # count = GP that week
    }
    agg_cols_avail = {k: v for k, v in agg_cols.items() if k in df.columns}

    grouped = df.groupby('week_start').agg(agg_cols_avail).reset_index()
    grouped.rename(columns={'gameDate': 'gp_week'}, inplace=True)

    # Weekly TOI: parse "MM:SS" → seconds, sum, convert back
    if 'toi' in df.columns:
        def toi_to_sec(t):
            try:
                parts = str(t).split(':')
                return int(parts[0]) * 60 + int(parts[1])
            except Exception:
                return 0
        df['toi_sec'] = df['toi'].apply(toi_to_sec)
        toi_sum = df.groupby('week_start')['toi_sec'].sum().reset_index()
        toi_sum['toi_week'] = toi_sum['toi_sec'].apply(
            lambda s: f"{int(s)//60}:{int(s)%60:02d}"
        )
        grouped = grouped.merge(toi_sum[['week_start', 'toi_week']], on='week_start', how='left')

    # Face-off pct: weighted average
    if 'faceoffWinningPctg' in df.columns:
        fo = df.groupby('week_start')['faceoffWinningPctg'].mean().reset_index()
        fo.rename(columns={'faceoffWinningPctg': 'fo_pct_week'}, inplace=True)
        grouped = grouped.merge(fo, on='week_start', how='left')

    # Opponents faced
    if 'opp_team' in df.columns:
        opp = df.groupby('week_start')['opp_team'].apply(
            lambda x: ', '.join(sorted(x.dropna().unique()))
        ).reset_index()
        opp.rename(columns={'opp_team': 'opponents'}, inplace=True)
        grouped = grouped.merge(opp, on='week_start', how='left')

    # own team
    if 'own_team' in df.columns:
        own = df.groupby('week_start')['own_team'].first().reset_index()
        own.rename(columns={'own_team': 'team'}, inplace=True)
        grouped = grouped.merge(own, on='week_start', how='left')

    return grouped.sort_values('week_start', ascending=False).reset_index(drop=True)


def aggregate_weekly_goalie(game_df):
    """
    Aggregate goalie game log to weekly frequency.
    Returns df with week_start, games started, wins, goals against, SV%, GAA.
    """
    if game_df is None or game_df.empty:
        return pd.DataFrame()
    df = game_df.copy()
    df['week_start'] = df['gameDate'] - pd.to_timedelta(df['gameDate'].dt.dayofweek, unit='D')
    df['week_start'] = df['week_start'].dt.normalize()

    # Parse saves and shotsAgainst from saveShotsAgainst string "saves/total"
    if 'saveShotsAgainst' in df.columns:
        splits = df['saveShotsAgainst'].str.extract(r'(\d+)/(\d+)')
        df['saves_n'] = pd.to_numeric(splits[0], errors='coerce')
        df['sa_n'] = pd.to_numeric(splits[1], errors='coerce')
    else:
        df['saves_n'] = df.get('saves', np.nan)
        df['sa_n'] = df.get('shotsAgainst', np.nan)

    agg = df.groupby('week_start').agg(
        gp_week=('gameDate', 'count'),
        saves_week=('saves_n', 'sum'),
        sa_week=('sa_n', 'sum'),
        ga_week=('goalsAgainst', 'sum'),
        wins_week=('decision', lambda x: (x == 'W').sum()),
        losses_week=('decision', lambda x: (x == 'L').sum()),
    ).reset_index()

    agg['sv_pct_week'] = (agg['saves_week'] / agg['sa_week'].replace(0, np.nan)).round(3)
    agg['gaa_week'] = (agg['ga_week'] / agg['gp_week'].replace(0, np.nan)).round(2)

    # Opponents
    if 'opp_team' in df.columns:
        opp = df.groupby('week_start')['opp_team'].apply(
            lambda x: ', '.join(sorted(x.dropna().unique()))
        ).reset_index()
        opp.rename(columns={'opp_team': 'opponents'}, inplace=True)
        agg = agg.merge(opp, on='week_start', how='left')

    if 'own_team' in df.columns:
        own = df.groupby('week_start')['own_team'].first().reset_index()
        own.rename(columns={'own_team': 'team'}, inplace=True)
        agg = agg.merge(own, on='week_start', how='left')

    return agg.sort_values('week_start', ascending=False).reset_index(drop=True)
