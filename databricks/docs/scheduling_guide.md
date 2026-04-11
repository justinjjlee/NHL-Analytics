# Databricks Scheduling & Setup Guide

This document outlines the steps required to implement, schedule, and maintain the automated NHL data ingestion pipeline within Databricks. It assumes you are using Unity Catalog.

## 1. Volume & Path Creation
Ensure the following Unity Catalog schema and volumes exist prior to running the notebook for the first time:
- **Catalog**: `nhl-databricks`
- **Database/Schema**: `data`
- **Volumes**:
  - `box`: For game lists and box-score level statistics.
  - `play`: For play-by-play and shift event datasets.
  - `team`: For team-level aggregations and player-season metrics.
  - `betting`: For bookies' odds.

You also need to copy the static `teamlist.csv` reference file into `/Volumes/nhl-databricks/data/team/` manually, as the scripts rely on this file to map tricodes to API identifiers.

## 2. Importing the Repository
1. Navigate to **Workspace > Repos** (or Workspace UI) in your Databricks environment.
2. Select **Add Repo** and clone this GitHub repository (`NHL-Analytics`).
3. Ensure you have properly set up your Git credentials in Databricks to allow for automatic fetching.

## 3. Creating & Scheduling the Job
1. Go to **Workflows > Jobs** and click **Create Job**.
2. **Task Configuration**:
   - **Task Name**: `NHL_Daily_Ingestion`
   - **Type**: `Notebook`
   - **Source**: `Workspace` (select the file under your cloned repo: `databricks/jobs/nhl_daily_ingestion.py`).
   - **Cluster**: Recommend an **interactive, single-node cluster**. The operations natively rely on the `requests` and `pandas` libraries which do not leverage multi-node distributed spark environments. A lightweight cluster is cost-efficient and sufficient here.
3. **Schedule Configuration**:
   - Create a trigger to run automatically daily at **20:00 UTC** (matching your old GitHub Actions schedule).
   - Only activate the trigger during active hockey months, or leave it running year-round (the scripts safely evaluate `datetime` offsets and off-season logic).
4. **Permissions**: Make sure the Service Principal or User identity driving the job has `WRITE VOLUME` and `READ VOLUME` permissions on `nhl-databricks.data`.

## 4. Backfilling Historical Data
- The scripts dynamically evaluate backfilling. If the current season's files (`{year}_box.csv`, `{year}_playbyplay.csv`) *do not exist* in the volumes, the code will automatically span backward to the **2011** season to build the history.
- The first run will thus take significantly longer as it pages through over a decade of NHL API routes. Note that API rate-limit errors (429) could occur dynamically, but standard `time.sleep(1)` has been retained to alleviate this. 
- You can bypass the automatic 13-year pull by manually uploading your current localized `latest/*` history into the volumes prior to executing the run.

## 5. Next Steps: Delta Tables
Currently, the pipeline dumps raw `.csv` and `.json` data into the Volumes for analytics usage. To leverage Databricks and Delta Engine capabilities fully in the future, it's recommended to create a downstream **Delta Live Tables (DLT) Pipeline**:
- Create an Auto Loader (`cloud_files`) stream reading incoming updates from `/Volumes/nhl-databricks/data/*/*.csv`. 
- Transform these streams into curated, structured Delta Tables (`box_scores`, `team_seasons`, `plays`, etc.) for direct BI analytics and low-latency querying.
