# %% Geospatial Attributes
'''
    Workspace to 
        * Calculate distance between the locations
        * Create relevant features to calculate attributes used 

    Objectives:
        Acquire exact lat/lon location of stadiums that each teams play
            Calculate geo-distance between the stadium: Capture the proxy for travel distance

    Notes:
        For now, neurtral site locations are not collected (e.g., Winter Classics)
            The Overture data can be used to acquire the locations, if those data are available in the nhlapie
        At the time of the creation of this workflow, some locations need to be manually pulled
            CAR's Stadium name was still recorded as 'PNC Arena' not Lenovo Center
            Overture data had some record issue with BOS's TD Garden and had to manually pick the record
    References:
        Overture - Places: https://docs.overturemaps.org/schema/reference/places/place/
        DuckDB - Spatial: https://duckdb.org/docs/extensions/spatial/functions
'''

# %% Set up
# Import packages
import pandas as pd
import json
import duckdb

import itertools

db = duckdb.connect()
db.execute("INSTALL spatial")
db.execute("INSTALL httpfs")
db.execute("""
LOAD spatial;
LOAD httpfs;
SET s3_region='us-west-2';
""")

# Overture version to use
ver_overture = '2024-10-23.0'

# =============================================================================================================
# =============================================================================================================
# %% (A) Pull Data
# Load the team tricode
team_df = pd.read_csv('./latest/team/teamlist.csv')

overture_stadium_loc = []

for iter, vals in team_df.iterrows():
    print(vals['tricode'] + ' - Processing')

    # Pull the data and save the data with value specific with abbreviation
    result = db.execute(f"""
        SELECT
            names.primary AS name,
            JSON(addresses) AS addresses,
            confidence AS confidence,
            ST_AsGeoJSON(ST_GeomFromWKB(geometry)) as geometry
            -- Note that Overture prints out lon/lat order.
        FROM read_parquet('s3://overturemaps-us-west-2/release/{ver_overture}/theme=places/type=*/*', filename=true, hive_partitioning=1)
        WHERE
            names.primary == '{vals['arena']}'
            AND addresses[1].region == '{vals['region']}'
            --AND categories.primary == 'stadium_arena'
        --LIMIT 10
    """).df()

    # Add additional informations
    result['tricode'] = vals['tricode']

    overture_stadium_loc.append(result)

# %%
# Raw file aggregation
overture_raw = pd.concat(overture_stadium_loc)
# Select records
idx_record = overture_raw.groupby(['tricode'])['confidence'].transform('max') == overture_raw['confidence']
overture_clean = overture_raw.loc[idx_record, :]

# Some ad hoc fixes
# Boston TD garden, the true lat lon: 42.3662° N, 71.0621° W
#   Currently (11/10/2024), there are three TD Garden records with all same confidence records 
overture_clean = overture_clean.loc[
    ~(
        (overture_clean.tricode == 'BOS') &
        ((overture_clean.index == 0) | (overture_clean.index == 2))
    )
]

# Join the records
joindf = team_df.copy(deep=True)

joindf.set_index('tricode', inplace=True)
overture_clean.set_index('tricode', inplace=True)

joindf = joindf.join(overture_clean, on='tricode')

# Clean up
joindf.drop(columns=['name'], inplace=True)

# Save the data
joindf.to_csv('./latest/team/teamlist_locations.csv')

# =============================================================================================================
# =============================================================================================================
# %% (B) Calculate distance between the team

joindf = pd.read_csv('./latest/team/teamlist_locations.csv')
distancedf = joindf[['tricode', 'geometry']]

# %%
# Generate permutations of teams
#   I am just using permutation here (order does matter - hence duplicated combinations)
#       For the sake of future convenience, some measures like distance between the two (no direction) may be duplicated
permutations = list(itertools.permutations(distancedf['tricode'], 2))

new_data = []

# Loop through each permutation and create a corresponding list of geometries
for perm in permutations:
    tricode_home = perm[0]
    tricode_away = perm[1]
    
    # Get the geometry for the home and away tricode
    geometry_home = distancedf[distancedf['tricode'] == tricode_home]['geometry'].values[0]
    geometry_away = distancedf[distancedf['tricode'] == tricode_away]['geometry'].values[0]
    
    # Append the required data to the new data list
    new_data.append({
        'tricode_home': tricode_home,
        'tricode_away': tricode_away,
        'geometry_home': geometry_home,
        'geometry_away': geometry_away
    })

# Create a new DataFrame from the list of dictionaries
permutation_df = pd.DataFrame(new_data)

# %% object calculations
# Into DuckDB
fc = duckdb.connect()
fc.execute("INSTALL spatial")
fc.execute("INSTALL httpfs")
fc.execute("""
LOAD spatial;
LOAD httpfs;
SET s3_region='us-west-2';
""")
# Register the DataFrame as a DuckDB table
fc.register("locations", permutation_df)

# Query the DataFrame using SQL
#   Note that the order of coordinate has to be lat/lon. 
#       If not, use DuckDB function to flip coordinates
#   Pull both approach in calculating the spherical distance
#       Haversine & ellipsoidal model
#       For function details https://duckdb.org/docs/extensions/spatial/functions.html#st_distance_sphere
distancedf = fc.execute(
    """
        SELECT *,
            ST_Distance_Sphere(
                ST_FlipCoordinates(ST_GeomFromGeoJSON(geometry_home)), 
                ST_FlipCoordinates(ST_GeomFromGeoJSON(geometry_away))
            ) AS geo_distance_haversine,
            ST_Distance_Spheroid(
                ST_FlipCoordinates(ST_GeomFromGeoJSON(geometry_home)), 
                ST_FlipCoordinates(ST_GeomFromGeoJSON(geometry_away))
            ) AS geo_distance_ellipsoidal
        FROM locations
    """
).df()

# Save the data
distancedf.to_csv('./latest/team/teamlist_locations_distance_meters.csv', index=False)

# %% EDA: Who's furthest away from everyone?
eda_furthest = distancedf.groupby('tricode_home').geo_distance_haversine.mean()
eda_furthest.sort_values(ascending=False)

# For more robust and relevant calculation, the followings must be considered
#   * Actual travel distance, such as airport to stadiums
#   * Actual playing schedule (SJS travels to Western Conference teams multiple times, 
#       while travel to BOS once a season)