# NHL API Endpoints Reference Catalog

This document provides a comprehensive technical reference for the NHL REST API endpoints used across the **NHL-Analytics (Rinklytics)** platform.

---

## 1. Game & Schedule Endpoints

### 1.1. Club Schedule (Season)
Retrieves the complete schedule and game outcomes for a specific franchise and season.

* **URL:** `GET https://api-web.nhle.com/v1/club-schedule-season/{team}/{season}`
* **Parameters:**
  * `team` *(string)*: Team tricode (e.g., `TOR`, `EDM`, `BOS`, `NYR`).
  * `season` *(string)*: 8-digit season identifier (e.g., `20242025`, `20252026`).
* **Pipeline Script:** [`src/data/apinhle/data_pull_box.py`](../../src/data/apinhle/data_pull_box.py#L60)
* **Key Response Fields:**
  ```json
  {
    "previousOwner": "...",
    "games": [
      {
        "id": 2024020001,
        "season": 20242025,
        "gameType": 2,
        "gameDate": "2024-10-04",
        "venue": {"default": "O2 Arena"},
        "startTimeUTC": "2024-10-04T17:00:00Z",
        "gameState": "OFF",
        "awayTeam": {"id": 1, "abbrev": "NJD", "score": 4},
        "homeTeam": {"id": 4, "abbrev": "BUF", "score": 1},
        "periodDescriptor": {"number": 3, "periodType": "REG"},
        "gameOutcome": {"lastPeriodType": "REG"}
      }
    ]
  }
  ```
* **Game States (`gameState`):**
  * `FUT`: Future scheduled game.
  * `PRE`: Pre-game warmup.
  * `LIVE`: In-progress.
  * `FINAL`: Regulation/OT/SO ended, awaiting official sign-off.
  * `OFF`: Official, final statistics verified and audited.

---

### 1.2. Gamecenter Boxscore
Provides detailed team and individual player statistics for a completed or live game.

* **URL:** `GET https://api-web.nhle.com/v1/gamecenter/{gameId}/boxscore`
* **Parameters:**
  * `gameId` *(integer/string)*: 10-digit game identifier (e.g., `2024020001`).
* **Pipeline Script:** [`src/data/apinhle/data_pull_box.py`](../../src/data/apinhle/data_pull_box.py#L183)
* **Key Response Fields:**
  ```json
  {
    "id": 2024020001,
    "season": 20242025,
    "gameType": 2,
    "gameDate": "2024-10-04",
    "awayTeam": {
      "id": 1,
      "abbrev": "NJD",
      "score": 4,
      "sog": 32,
      "faceoffWinningPctg": 0.528,
      "powerPlayConversion": "1/3",
      "pim": 6,
      "hits": 24,
      "blocks": 14
    },
    "homeTeam": { ... },
    "playerByGameStats": {
      "awayTeam": {
        "forwards": [
          {
            "playerId": 8481559,
            "name": {"default": "Jack Hughes"},
            "sweaterNumber": 86,
            "position": "C",
            "goals": 0,
            "assists": 2,
            "points": 2,
            "plusMinus": 1,
            "pim": 0,
            "hits": 1,
            "blockedShots": 0,
            "powerPlayToi": "03:12",
            "shorthandedToi": "00:00",
            "toi": "18:45"
          }
        ],
        "defense": [ ... ],
        "goalies": [
          {
            "playerId": 8478406,
            "name": {"default": "Jacob Markstrom"},
            "savePctg": "0.967",
            "saves": 29,
            "shots": 30,
            "goalsAgainst": 1,
            "toi": "60:00",
            "decision": "W"
          }
        ]
      }
    }
  }
  ```

---

## 2. Event-Level Telemetry & Spatial Analytics

### 2.1. Play-by-Play & Event Coordinates
Returns chronological on-ice events with exact ice coordinates and participating player IDs.

* **URL:** `GET https://api-web.nhle.com/v1/gamecenter/{gameId}/play-by-play`
* **Parameters:**
  * `gameId` *(integer/string)*: 10-digit game identifier (e.g., `2024020001`).
* **Pipeline Script:** [`src/data/apinhle/data_pull_plays.py`](../../src/data/apinhle/data_pull_plays.py#L83)
* **Processor Function:** [`proc_playbyplay_clean` in `src/data/apinhle/function/procs_playbyplay.py`](../../src/data/apinhle/function/procs_playbyplay.py#L6)
* **Key Response Fields:**
  ```json
  {
    "id": 2024020001,
    "season": 20242025,
    "gameType": 2,
    "homeTeam": {"id": 4, "abbrev": "BUF"},
    "awayTeam": {"id": 1, "abbrev": "NJD"},
    "plays": [
      {
        "eventId": 52,
        "periodDescriptor": {"number": 1, "periodType": "REG"},
        "timeInPeriod": "08:31",
        "timeRemaining": "11:29",
        "situationCode": "1551",
        "homeTeamDefendingSide": "left",
        "typeDescKey": "shot-on-goal",
        "details": {
          "xCoord": 68.0,
          "yCoord": -12.0,
          "zoneCode": "O",
          "shotType": "wrist",
          "shootingPlayerId": 8481559,
          "goalieInNetId": 8480382,
          "eventOwnerTeamId": 1,
          "awaySOG": 4,
          "homeSOG": 2
        }
      }
    ],
    "rosterSpots": [
      {
        "teamId": 1,
        "playerId": 8481559,
        "firstName": {"default": "Jack"},
        "lastName": {"default": "Hughes"},
        "sweaterNumber": 86,
        "positionCode": "C",
        "headshot": "https://assets.nhle.com/mugs/nhl/latest/8481559.png"
      }
    ]
  }
  ```

---

### 2.2. Shift Charts & Time-on-Ice Intervals
Queries the Cayenne-backed NHL Stats REST API for shift-by-shift time segments.

* **URL:** `GET https://api.nhle.com/stats/rest/en/shiftcharts?cayenneExp=gameId={gameId}`
* **Parameters:**
  * `gameId` *(integer/string)*: 10-digit game identifier (e.g., `2024020001`).
* **Pipeline Script:** [`src/data/apinhle/data_pull_lines.py`](../../src/data/apinhle/data_pull_lines.py#L56)
* **Key Response Fields:**
  ```json
  {
    "data": [
      {
        "id": 1234567,
        "gameId": 2024020001,
        "playerId": 8481559,
        "teamId": 1,
        "period": 1,
        "shiftNumber": 1,
        "startTime": "00:00",
        "endTime": "00:48",
        "duration": "00:48",
        "typeCode": 505
      }
    ],
    "total": 742
  }
  ```

---

## 3. Real-Time Betting Odds

### 3.1. Partner Game Odds (Now)
Provides live sportsbook odds (Moneyline 2-way, 3-way, Puck Line, and Over/Under totals).

* **URL:** `GET https://api-web.nhle.com/v1/partner-game/{country}/now`
* **Parameters:**
  * `country` *(string)*: Geographical market code (`US` or `CA`).
* **Pipeline Script:** [`src/data/apinhle/data_pull_box_odds.py`](../../src/data/apinhle/data_pull_box_odds.py#L40)
* **Key Response Fields:**
  ```json
  {
    "currentOddsDate": "2026-10-06",
    "lastUpdatedUTC": "2026-10-06T20:00:38Z",
    "bettingPartner": {"name": "DraftKings", "countryUrl": "..."},
    "games": [
      {
        "gameId": 2026020001,
        "gameType": 2,
        "startTimeUTC": "2026-10-07T21:00:00Z",
        "homeTeam": {
          "id": 13,
          "name": {"default": "Florida Panthers"},
          "abbrev": "FLA",
          "odds": [
            {"description": "MONEY_LINE_2_WAY", "value": -290.0},
            {"description": "PUCK_LINE", "value": -108.0},
            {"description": "OVER_UNDER", "value": -118.0}
          ]
        },
        "awayTeam": {
          "id": 16,
          "abbrev": "CHI",
          "odds": [
            {"description": "MONEY_LINE_2_WAY", "value": 235.0},
            {"description": "PUCK_LINE", "value": -112.0},
            {"description": "OVER_UNDER", "value": -102.0}
          ]
        }
      }
    ]
  }
  ```

---

## 4. Player & Club Roster Statistics

### 4.1. Club Stats (Season & Type)
Aggregated season statistics for all skaters and goalies rostered on a team.

* **URL:** `GET https://api-web.nhle.com/v1/club-stats/{team}/{season}/{seasonType}`
* **Parameters:**
  * `team` *(string)*: Team tricode (e.g., `EDM`).
  * `season` *(string)*: 8-digit season (e.g., `20242025`).
  * `seasonType` *(integer)*: `2` (Regular Season), `3` (Playoffs).
* **Pipeline Script:** [`src/data/apinhle/data_pull_player.py`](../../src/data/apinhle/data_pull_player.py#L40)

---

### 4.2. Player Profile & Career Landing
Historical career timeline, draft details, awards, and season totals across all professional and amateur leagues.

* **URL:** `GET https://api-web.nhle.com/v1/player/{playerId}/landing`
* **Parameters:**
  * `playerId` *(integer)*: Unique player identifier (e.g., Connor McDavid: `8478402`, Auston Matthews: `8479318`).
* **Dev Scripts:** [`dev/player/player_draft/data/dataproc.py`](../../dev/player/player_draft/data/dataproc.py#L293)
* **Key Response Fields:**
  ```json
  {
    "playerId": 8478402,
    "isActive": true,
    "currentTeamId": 22,
    "currentTeamAbbrev": "EDM",
    "position": "C",
    "shootsCatches": "L",
    "heightInInches": 73,
    "weightInPounds": 193,
    "birthDate": "1997-01-13",
    "birthCountry": "CAN",
    "draftDetails": {
      "year": 2015,
      "teamAbbrev": "EDM",
      "round": 1,
      "overallPick": 1
    },
    "seasonTotals": [
      {
        "season": 20232024,
        "gameTypeId": 2,
        "leagueAbbrev": "NHL",
        "teamName": {"default": "Edmonton Oilers"},
        "gamesPlayed": 76,
        "goals": 32,
        "assists": 100,
        "points": 132,
        "plusMinus": 35
      }
    ]
  }
  ```

---

### 4.3. Skater Stats Leaders
League-wide leaderboards for points, goals, assists, and goalie metrics.

* **URL:** `GET https://api-web.nhle.com/v1/skater-stats-leaders/{season}/{gameType}?categories={category}&limit={limit}`
* **Parameters:**
  * `season` *(string)*: 8-digit season (e.g., `20242025`).
  * `gameType` *(integer)*: `2` (Regular Season), `3` (Playoffs).
  * `categories` *(string)*: `points`, `goals`, `assists`, `shootingPctg`.
  * `limit` *(integer)*: Number of leaders or `-1` for all qualified skaters.

---

## 5. Entry Draft Archives

### 5.1. Draft Picks by Year
Complete amateur entry draft selections by round, overall pick, and junior/collegiate affiliate.

* **URL:** `GET https://api-web.nhle.com/v1/draft/picks/{year}/all`
* **Parameters:**
  * `year` *(integer)*: 4-digit draft year (e.g., `1963` through `2026`).
* **Dev Script:** [`dev/player/player_draft/data/dataproc.py`](../../dev/player/player_draft/data/dataproc.py#L66)
* **Key Response Fields:**
  ```json
  {
    "year": 2023,
    "picks": [
      {
        "round": 1,
        "pickInRound": 1,
        "overallPick": 1,
        "teamAbbrev": "CHI",
        "playerId": 8484144,
        "firstName": {"default": "Connor"},
        "lastName": {"default": "Bedard"},
        "positionCode": "C",
        "countryCode": "CAN",
        "amateurLeague": "WHL",
        "amateurClubName": "Regina Pats"
      }
    ]
  }
  ```

---

## 6. Static Media & CDN Assets

* **Team Light SVG Logo:**
  `https://assets.nhle.com/logos/nhl/svg/{TRICODE}_light.svg`
  *(e.g., `https://assets.nhle.com/logos/nhl/svg/EDM_light.svg`)*
* **Team Dark SVG Logo:**
  `https://assets.nhle.com/logos/nhl/svg/{TRICODE}_dark.svg`
* **Player Mugshot / Headshot:**
  `https://assets.nhle.com/mugs/nhl/latest/{playerId}.png`
  *(e.g., `https://assets.nhle.com/mugs/nhl/latest/8478402.png`)*
