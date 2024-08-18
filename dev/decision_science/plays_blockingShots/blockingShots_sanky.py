# %%
import pandas as pd

# Load the data into a pandas dataframe
df = pd.read_csv("./result/table - sequence blocked shot stats.csv")

# From EDA total count
total_values = df['seasonIdx'].sum()
# %%
# Extract nodes
nodes = list(
    set(df['blocker_team_lag1_now'].unique()) | 
    set(df['eventOwnerTeam_sequence'].unique()) |  
    set(df['group'].unique()) | 
    set(df['typeDescKey'].unique())
)

# Create dictionaries for nodes
node_indices = {node: i for i, node in enumerate(nodes)}

# Create links for each layer
links = {
    'source': [],
    'target': [],
    'value': [],
    'color': [],
    'hovertext': []
}


# Define colors
colors = {
    'Defense': '#1d3557',
    'Neutral': '#E9D8A6',
    'Offense': '#9B2226',

    'Opponent action': '#219EBC',
    'Own action': '#FB8500',

    'Shot':'#003049',
    'shot-on-goal':'#003049',
    'blocked-shot':'#003049',
    'missed-shot':'#003049',

    'Possession':'#d62828',
    'hit':'#d62828',
    'goal':'#d62828',
    'giveaway':'#d62828',
    'takeaway':'#d62828',

    'Goal':'#f77f00',
    'goal':'#f77f00',

    'Stoppage':'#fcbf49',
    'stoppage':'#fcbf49',
    'faceoff':'#fcbf49',
    'period-end':'#fcbf49',
    'game-end':'#fcbf49',

    'Penalty':'#eae2b7',
    'penalty':'#eae2b7',
    'delayed-penalty':'#eae2b7'
}

tempdf = df.groupby(['blocker_team_lag1_now', 'eventOwnerTeam_sequence']).seasonIdx.sum().reset_index()
# Add links from blocker_team_lag1_now to eventOwnerTeam_sequence
for i, row in tempdf.iterrows():
    source = node_indices[row['blocker_team_lag1_now']]
    target = node_indices[row['eventOwnerTeam_sequence']]
    value = row['seasonIdx']
    percent = (value / total_values) * 100

    links['source'].append(source)
    links['target'].append(target)
    links['value'].append(percent)
    links['color'].append(colors.get(row['blocker_team_lag1_now'], 'rgba(0,0,0,0.1)'))
    links['hovertext'].append(f"{value:,} ({percent:.2f}%)")

tempdf = df.groupby(['group', 'eventOwnerTeam_sequence']).seasonIdx.sum().reset_index()
# Add links from eventOwnerTeam_sequence to group
for i, row in tempdf.iterrows():
    source = node_indices[row['eventOwnerTeam_sequence']]
    target = node_indices[row['group']]
    value = row['seasonIdx']
    percent = (value / total_values) * 100

    links['source'].append(source)
    links['target'].append(target)
    links['value'].append(percent)
    links['color'].append(colors.get(row['eventOwnerTeam_sequence'], 'rgba(0,0,0,0.1)'))
    links['hovertext'].append(f"{value:,} ({percent:.2f}%)")

tempdf = df.groupby(['group', 'typeDescKey']).seasonIdx.sum().reset_index()
# Add links from group to typeDescKey
for i, row in tempdf.iterrows():
    source = node_indices[row['group']]
    target = node_indices[row['typeDescKey']]
    value = row['seasonIdx']
    percent = (value / total_values) * 100

    links['source'].append(source)
    links['target'].append(target)
    links['value'].append(percent)
    links['color'].append(colors.get(row['group'], 'rgba(0,0,0,0.1)'))
    links['hovertext'].append(f"{value:,} ({percent:.2f}%)")

# %%
from dash import Dash, dcc, html
import plotly.graph_objects as go

# Custom hover text formatting
def format_hovertext(value, total):
    percent = (value / total) * 100
    return f"{percent:.2f}%"

app = Dash(__name__)

# Calculate hover text
hovertext = [format_hovertext(value, total_values) for value in links['value']]

fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=nodes,
        color = [
            colors[node] if node in colors.keys()
            else '#0a9396' 
            for node in nodes
        ]
    ),
    link=dict(
        source=links['source'],
        target=links['target'],
        value=links['value'],
        color=links['color'],
        #hoverinfo='all',  # Display label and value; can't use custom text directly
        #text=hovertext           # Custom hover text
    ))])

# Add annotations for each layer
annotations = [
    dict(
        x=-0.04,  # Approximate x position for the first layer
        y=-0.1,  # Place label below the plot
        xref="paper",
        yref="paper",
        text="Posture of <br> the shot-blocking team <br>",
        showarrow=False,
        font=dict(size=14, color="black")
    ),
    dict(
        x=0.35,  # Approximate x position for the second layer
        y=-0.075,
        xref="paper",
        yref="paper",
        text="Action by",
        showarrow=False,
        font=dict(size=14, color="black")
    ),
    dict(
        x=0.65,  # Approximate x position for the third layer
        y=-0.075,
        xref="paper",
        yref="paper",
        text="Event - Group",
        showarrow=False,
        font=dict(size=14, color="black")
    ),
    dict(
        x=0.95,  # Approximate x position for the fourth layer
        y=-0.075,
        xref="paper",
        yref="paper",
        text="Event",
        showarrow=False,
        font=dict(size=14, color="black")
    )
]

fig.update_layout(
    title_text="After blocking opponent's shot, what do you do next?",
    title_x=0.985,  # Center title
    title_y= 1,
    title_font_size=24,
    annotations=[
        dict(
            x=1,  # Center annotation horizontally
            y=1.125,  # Place annotation below the plot
            xref="paper",
            yref="paper",
            text="Tracking posture and events after blocking shots (Percent)",
            showarrow=False,
            font=dict(size=16, color="gray")
        )
    ] + annotations,
    font_size=10,
    margin=dict(l=80, r=25, b=100, t=100),
)
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

if __name__ == '__main__':
    app.run_server(debug=True)

fig.write_html('./result/table - sequence blocked shot stats.html')