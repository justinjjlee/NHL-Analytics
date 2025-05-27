import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go

def get_data_path(rel_path):
    """Helper function to get correct data paths regardless of run location"""
    # Start with the app directory
    app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Go up to the repository root
    repo_root = os.path.dirname(app_dir)
    # Navigate to the data file
    return os.path.join(repo_root, "dev", "decision_science", "plays_blockingShots", rel_path)

def create_sankey_diagram():
    """Create and return the Sankey diagram figure"""
    try:
        # Load the data into a pandas dataframe
        df = pd.read_csv(get_data_path("result/table - sequence blocked shot stats.csv"))
        
        # If the file doesn't exist, display a message
        if df.empty:
            st.error("Data file is empty. Please check the path.")
            return None
            
        # From EDA total count
        total_values = df['seasonIdx'].sum()
        
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
        
        # Add links from blocker_team_lag1_now to eventOwnerTeam_sequence
        tempdf = df.groupby(['blocker_team_lag1_now', 'eventOwnerTeam_sequence']).seasonIdx.sum().reset_index()
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
        
        # Add links from eventOwnerTeam_sequence to group
        tempdf = df.groupby(['eventOwnerTeam_sequence', 'group']).seasonIdx.sum().reset_index()
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
        
        # Add links from group to typeDescKey
        tempdf = df.groupby(['group', 'typeDescKey']).seasonIdx.sum().reset_index()
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
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes,
                color=[
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
                customdata=links['hovertext'],
                hovertemplate='%{customdata}<extra></extra>'
            )
        )])
        
        # Add annotations for each layer
        annotations = [
            dict(
                x=-0.04,  # Approximate x position for the first layer
                y=-0.1,  # Place label below the plot
                xref="paper",
                yref="paper",
                text="Posture of <br> the shot-blocking team <br>",
                showarrow=False,
                font=dict(size=14, color="white")  # Changed to white
            ),
            dict(
                x=0.35,  # Approximate x position for the second layer
                y=-0.075,
                xref="paper",
                yref="paper",
                text="Action by",
                showarrow=False,
                font=dict(size=14, color="white")  # Changed to white
            ),
            dict(
                x=0.65,  # Approximate x position for the third layer
                y=-0.075,
                xref="paper",
                yref="paper",
                text="Event - Group",
                showarrow=False,
                font=dict(size=14, color="white")  # Changed to white
            ),
            dict(
                x=0.95,  # Approximate x position for the fourth layer
                y=-0.075,
                xref="paper",
                yref="paper",
                text="Event",
                showarrow=False,
                font=dict(size=14, color="white")  # Changed to white
            )
        ]
        
        fig.update_layout(
            title_text="Tracking posture and events after blocking shots (Percent)",
            title_x=0.,  # Right-align the title (0.95 instead of 0.5)
            title_y=0.98,
            title_font_size=20,
            annotations=annotations + [
                dict(
                    x=0.5,  # Keep this centered
                    y=1.05,  # Place annotation above the plot
                    xref="paper",
                    yref="paper",
                    text="",
                    showarrow=False,
                    font=dict(size=16, color="gray")
                )
            ],
            font=dict(
                size=10,
                color="white"  # Change all font color to white
            ),
            margin=dict(l=0, r=0, b=65, t=25),
            height=700,
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
            plot_bgcolor='rgba(0,0,0,0)'    # Transparent plot area
        )
        
        return fig
    except FileNotFoundError:
        st.error("Data file not found. Please check if the analysis files are in the correct location.")
        return None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def show():
    st.title("The Valor of Blocking Shots")
    
    st.markdown("""
    This analysis examines sequense of plays after a player blocks a shot. The insights from this analysis can help teams assess 
    the effectiveness and value of of shot-blocking strategies,
    and potentially reacting and adjusting strategies based on the speed/transition of each game.
    
    *Based on the analysis documented in ["Human Shields on Ice: The Valor of Blocking Shots"](https://medium.com/@thinkingjustin/human-shields-on-ice-the-valor-of-blocking-shots-2d55e2f0482c)*
    """)
    
    # Create tabs for different analyses
    tabs = st.tabs(["Interactive Flow Diagram", "Analysis Findings"])
    
    with tabs[0]:
        st.subheader("Plays After Shot Blocks")
        st.markdown("""
        The Sankey diagram below visualizes the flow of events after a player blocks a shot. The interactive chart can be interepreted in order (left to right):
        - **Posture of the Shot-Blocking Team**: The initial posture of the team that blocked the shot.
        - **The Next Action**: The team that performed the next action after the block.
        - **Action Category**: The general category of the subsequent action (e.g., shot, goal, hit).
        - **Action Type**: The pecific type of the subsequent action (e.g., shot on goal, blocked shot, missed shot).
        """)
        # Create the Sankey diagram
        fig = create_sankey_diagram()
        
        if fig:
            # Display the plot
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("""
            The diagram is based on play-by-play level data of 2022-2023 and 2023-2024 regular seasons. The data can be updated
            using the published code (see the Key Findings section).
            """)
        else:
            # If there was an error loading the data
            st.info("Analysis data is being prepared. Check back soon!")
    
    with tabs[1]:
        st.subheader("Key Findings")
        
        st.markdown("""
        While blocking shots is not followed by higher likelihood to offensive transition, 
        it significantly helps defensive success (lower sequential likelihood of getting scored against).

        ### What happens after a player blocks a shot?
        
        1. **Defensive Posture**: 60% of the time, teams remain in a defensive posture after blocking a shot
        2. **Offensive Transition**: Only 20% of shot blocks lead to an immediate transition to offensive posture
        4. **Shot Quality Impact**: Shot blocks significantly disrupt the offensive flow, with subsequent shots having lower scoring probability
        
        ### Methodology
        
        This analysis used NHL play-by-play data, tracking events/actions immediately after shot blocks. Events were categorized based on:
        
        - Team posture (offensive/defensive)
        - Subsequent actions and posture (which team initiated the next action)
        - Action type (shot, goal, hit, etc.)
        - Zone location
        
        For more details, check out the [full article on Medium](https://medium.com/@thinkingjustin/human-shields-on-ice-the-valor-of-blocking-shots-2d55e2f0482c) 
        or the [source code on GitHub](https://github.com/justinjjlee/NHL-Analytics/tree/main/dev/decision_science/plays_blockingShots).
        """)

if __name__ == "__main__":
    show()