'''
    Create NHL Team Geospatial Location through Streamlit

    To Start Locally: Turn on the server in terminal and run this code
        ~ % cd Documents 
        Documents % cd Github/NHL-Analytics/app 
        app % streamlit run team_mapper_app.py
        
        % cd Github/NHL-Analytics/app  
        % streamlit run team_mapper_app.py
'''

import streamlit as st
import folium
from folium import DivIcon
from streamlit_folium import st_folium
import json
import pandas as pd

# Load data from CSV file (ensure the 'temp' folder contains 'temp.csv')
csv_path = './data/teamlist_locations_distance_meters.csv'
df = pd.read_csv(csv_path)
csv_path = './data/teamlist.csv'
df_team = pd.read_csv(csv_path)

#df.sort_values(by="tricode_home", ascending=True, inplace=True)

# Join the two
df = df.merge(
    df_team, left_on="tricode_home", right_on="tricode"
).rename(columns={"team": "home_team", "city": "home_team_city"}).drop("tricode", axis=1)

df = df.merge(
    df_team, left_on="tricode_away", right_on="tricode"
).rename(columns={"team": "away_team", "city": "away_team_city"}).drop("tricode", axis=1)

# Initialize session state for home and away team selection
if 'home_team' not in st.session_state:
    st.session_state.home_team = ""
if 'away_team' not in st.session_state:
    st.session_state.away_team = ""

# Sidebar content
# Add logo image
st.sidebar.image("./assets/img_main.jpeg", use_container_width=True)

st.sidebar.markdown(
    """
    <hr style="border: none; height: 2px; background-color: #ffb703; margin-top: 5px; margin-bottom: 5px;">
    """,
    unsafe_allow_html=True
)

st.sidebar.title("NHL Analytics")
st.sidebar.markdown(
    """
    <div style="font-size: 14px;">
        This is my Streamlit application for my project on 
        <a href="https://github.com/justinjjlee/NHL-Analytics" target="_blank" style="text-decoration: none; color: #ffb703;">
            GitHub
        </a>
        and
        <a href="https://medium.com/@thinkingjustins" target="_blank" style="text-decoration: none; color: #ffb703;">
            Medium articles.
        </a>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar subsection
# Add spacing before the subheader
st.sidebar.markdown(
    """
    <div style="margin-top: 15px;"></div>
    """,
    unsafe_allow_html=True
)
st.sidebar.subheader("About my work")
st.sidebar.markdown(
    """
    <div style="font-size: 13px; text-align: justify;">
        My work focuses on analyzing NHL team and player performance, writing and visualizing insights, 
        and creating interactive tools to better understand team statistics. 
        This project integrates data science and geospatial mapping to explore game dynamics.
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown(
    """
    <hr style="border: none; height: 2px; background-color: #ffb703; margin-top: 5px; margin-bottom: 5px;">
    """,
    unsafe_allow_html=True
)

# .............................................................................................................
# (*) The map & distance
# .............................................................................................................
st.title("NHL Team Locations & Distance")

# Create a row with two columns for the refresh button and the distance string
col_button, col_text = st.columns([1, 4])  # Adjust the proportions as needed

# Button to refresh the map (reset zoom)
with col_button:
    refresh_button = st.button("Refresh Map")

# Handle refresh button click before rendering the widgets
if refresh_button:
    st.session_state.home_team = ""  # Reset home team
    st.session_state.away_team = ""  # Reset away team

# Create two columns for side-by-side dropdowns
col1, col2 = st.columns(2)

# Home team selection on the left column
with col1:
    home_team = st.selectbox("Select Home Team", [""] + df["home_team"].unique().tolist(), key="home_team")

# Away team selection on the right column
with col2:
    away_team = st.selectbox("Select Away Team", [""] + df["away_team"].unique().tolist(), key="away_team")

# Create a base map centered on North America
m = folium.Map(location=[39.8283, -98.5795], zoom_start=4)


# Show all team logos if no teams are selected
if (home_team == "" and away_team == "") | refresh_button:
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

        # Add the distance text as a label in red color, in one line
        #folium.Marker(
        #    location=[(home_coords[1] + away_coords[1]) / 2, 
        #              (home_coords[0] + away_coords[0]) / 2],  # mid-point between markers
        #    icon=folium.DivIcon(html=f'<div style="font-size: 12pt; color: red; white-space: nowrap;">Distance: {distance:.2f} km</div>')
        #).add_to(m)

        # Zoom to fit both markers with a little margin to ensure markers don't overlap the map edges
        bounds = [[home_coords[1], home_coords[0]], [away_coords[1], away_coords[0]]]

        m.fit_bounds(bounds, padding=(50, 50))  # Default padding for initial map view

        # Display the dynamic sentence with distance
        distance_text = f"The distance between {filtered_data.iloc[0]['away_team_city']} and {filtered_data.iloc[0]['home_team_city']} is {distance:,.1f} km"
        with col_text:
            st.write(f"<div style='margin-left: -25px; margin-top: 7px;'>{distance_text}</div>", unsafe_allow_html=True)

# Render the map in Streamlit
st_folium(m, width=700, height=500)
