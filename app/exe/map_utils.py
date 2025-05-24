import os
import folium
from folium import DivIcon
import json
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium


def get_data_path(rel_path):
    """Helper function to get correct data paths regardless of run location"""
    # Start with the current file location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to the project root
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    # Construct the absolute path
    return os.path.join(project_root, 'app', 'data', rel_path)

def load_map_data():
    """Load the necessary data for the map visualization with proper path handling"""
    try:
        # Use absolute paths for data files
        teamlist_locations_path = get_data_path('teamlist_locations_distance_meters.csv')
        teamlist_path = get_data_path('teamlist.csv')
        
        # Check if files exist
        if not os.path.exists(teamlist_locations_path):
            st.error(f"File not found: {teamlist_locations_path}")
            return None
            
        if not os.path.exists(teamlist_path):
            st.error(f"File not found: {teamlist_path}")
            return None
        
        # Load the data
        df = pd.read_csv(teamlist_locations_path)
        df_team = pd.read_csv(teamlist_path)

        # Join the two
        df = df.merge(
            df_team, left_on="tricode_home", right_on="tricode"
        ).rename(columns={"team": "home_team", "city": "home_team_city"}).drop("tricode", axis=1)

        df = df.merge(
            df_team, left_on="tricode_away", right_on="tricode"
        ).rename(columns={"team": "away_team", "city": "away_team_city"}).drop("tricode", axis=1)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading map data: {e}")
        return None

def render_team_map(df, home_team="", away_team="", refresh_button=False, travel_mode=False):
    """Render the team locations map with various selection options"""
    # Check if data is available
    if df is None:
        st.error("Map data is not available. Please check the data files.")
        return None
        
    # Create a base map centered on North America
    m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

    # Create a row with two columns for the refresh button and the distance string
    col_text = st.empty()

    # Show all team logos if no teams are selected
    if (home_team == "" and away_team == "") or refresh_button:
        # Show all team logos
        tempdf = df.drop_duplicates(subset="tricode_home")
        tempdf.sort_values(by="tricode_home", ascending=True, inplace=True)
        tempdf.reset_index(drop=True, inplace=True)
        
        for index, row in tempdf.iterrows():
            home_geo = json.loads(row["geometry_home"])  # Convert GeoJSON to dict
            home_coords = home_geo["coordinates"]

            # Add the home team marker with the SVG image as the icon (larger)
            home_marker = folium.Marker(
                location=[home_coords[1], home_coords[0]],  # lat, lon order
                zIndexOffset=index,
                icon=DivIcon(icon_size=(50, 50), html=f'''
                    <div style="position: relative; z-index: {index};">
                        <img src="https://assets.nhle.com/logos/nhl/svg/{row["tricode_home"]}_light.svg" 
                             style="width: 50px; height: 50px;">
                    </div>
                ''')
            ).add_to(m)

        # Adjust zoom and bounds to fit tightly around all logos
        m.fit_bounds([[45, -115], [35, -80]], padding=(50, 50))

    # If only one team is selected (either home or away), show that team's logo
    elif home_team != "" and away_team == "":
        # Filter for home team only
        home_team_data = df[df["home_team"] == home_team]
        if not home_team_data.empty:
            home_geo = json.loads(home_team_data.iloc[0]["geometry_home"])  # Convert GeoJSON to dict
            home_coords = home_geo["coordinates"]
            # Add the home team marker with the SVG image as the icon (larger)
            home_marker = folium.Marker(
                location=[home_coords[1], home_coords[0]],  # lat, lon order
                icon=DivIcon(icon_size=(100, 100), html=f'<img src="https://assets.nhle.com/logos/nhl/svg/{home_team_data.iloc[0]["tricode_home"]}_light.svg" style="width: 100px; height: 100px;">')
            ).add_to(m)

        # Adjust zoom and bounds for a tighter fit around the home team
        m.fit_bounds([[45, -115], [35, -80]], padding=(50, 50))

    elif home_team == "" and away_team != "":
        # Filter for away team only
        away_team_data = df[df["away_team"] == away_team]
        if not away_team_data.empty:
            away_geo = json.loads(away_team_data.iloc[0]["geometry_away"])  # Convert GeoJSON to dict
            away_coords = away_geo["coordinates"]
            # Add the away team marker with the SVG image as the icon (larger)
            away_marker = folium.Marker(
                location=[away_coords[1], away_coords[0]],  # lat, lon order
                icon=DivIcon(icon_size=(100, 100), html=f'<img src="https://assets.nhle.com/logos/nhl/svg/{away_team_data.iloc[0]["tricode_away"]}_light.svg" style="width: 100px; height: 100px;">')
            ).add_to(m)

        # Adjust zoom and bounds for a tighter fit around the away team
        m.fit_bounds([[45, -115], [35, -80]], padding=(50, 50))

    # If both teams are selected, show both logos and draw the distance line
    else:
        # Filter the data based on selected teams
        filtered_data = df[(df["home_team"] == home_team) & (df["away_team"] == away_team)]

        if not filtered_data.empty:
            # Extract coordinates and distance (from GeoJSON in geometry columns)
            home_geo = json.loads(filtered_data.iloc[0]["geometry_home"])  # Convert GeoJSON to dict
            away_geo = json.loads(filtered_data.iloc[0]["geometry_away"])  # Convert GeoJSON to dict

            home_coords = home_geo["coordinates"]  # Get coordinates as [lon, lat]
            away_coords = away_geo["coordinates"]  # Get coordinates as [lon, lat]

            distance = filtered_data.iloc[0]["geo_distance_haversine"] / 1000  # Convert to kilometers

            # Add the home team marker with the SVG image as the icon (larger)
            home_marker = folium.Marker(
                location=[home_coords[1], home_coords[0]],  # lat, lon order
                icon=DivIcon(icon_size=(100, 100), html=f'<img src="https://assets.nhle.com/logos/nhl/svg/{filtered_data.iloc[0]["tricode_home"]}_light.svg" style="width: 100px; height: 100px;">')
            ).add_to(m)

            # Add the away team marker with the SVG image as the icon (larger)
            away_marker = folium.Marker(
                location=[away_coords[1], away_coords[0]],  # lat, lon order
                icon=DivIcon(icon_size=(100, 100), html=f'<img src="https://assets.nhle.com/logos/nhl/svg/{filtered_data.iloc[0]["tricode_away"]}_light.svg" style="width: 100px; height: 100px;">')
            ).add_to(m)

            # Draw a line between the two teams
            folium.PolyLine(
                locations=[(home_coords[1], home_coords[0]),  # lat, lon order
                           (away_coords[1], away_coords[0])],  # lat, lon order
                color='blue', weight=2.5, opacity=1
            ).add_to(m)
            
            # Add travel info if travel mode is enabled
            if travel_mode:
                # Calculate travel time (very approximate)
                avg_flight_speed = 800  # km/h for commercial aircraft
                travel_time_hours = distance / avg_flight_speed
                
                # Add travel info popup
                travel_html = f"""
                <div style="font-family: Arial; width: 200px;">
                    <h4 style="margin-bottom: 5px;">Travel Information</h4>
                    <hr style="margin: 5px 0;">
                    <p><b>Flight Distance:</b> {distance:.1f} km</p>
                    <p><b>Estimated Flight Time:</b> {travel_time_hours:.1f} hours</p>
                    <p><b>Time Zone Difference:</b> Varies by season</p>
                </div>
                """
                
                # Add popup at midpoint
                mid_lat = (home_coords[1] + away_coords[1]) / 2
                mid_lon = (home_coords[0] + away_coords[0]) / 2
                
                folium.Popup(travel_html, max_width=300).add_to(
                    folium.Marker(
                        location=[mid_lat, mid_lon],
                        icon=folium.Icon(icon="info-sign", color="blue")
                    ).add_to(m)
                )

            # Zoom to fit both markers with a little margin to ensure markers don't overlap the map edges
            bounds = [[home_coords[1], home_coords[0]], [away_coords[1], away_coords[0]]]
            m.fit_bounds(bounds, padding=(50, 50))  # Default padding for initial map view

            # Display the dynamic sentence with distance
            distance_text = f"The distance between {filtered_data.iloc[0]['away_team_city']} and {filtered_data.iloc[0]['home_team_city']} is {distance:,.1f} km"
            col_text.write(f"<div style='margin-top: 7px;'>{distance_text}</div>", unsafe_allow_html=True)

    # Render the map
    return m