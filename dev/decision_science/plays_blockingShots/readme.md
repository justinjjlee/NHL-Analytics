# Sequencing of player activities - blocking shots
Work related to analyzing sequence of data, particularly on blocking shots and how those subsequent activities impact subsequent activities.

![Diagram](./diagram/diagram.svg)

## Data & Measurements
These are data points and sequencial measurements that I caputre for this analysis
- [x] All shot types and outcomes
- [x] Team/players involved: shooter and blocker
- [x] Team posture and sequence of the shooting: before and after the blocked shots
- [x] Sequencial events in number of players: full strength or `power play`/`penalty kill`
- [x] Team posture and shifts after blocked shots

## Analysis
The complete analysis with narration is saved in my Medium page.

The Dash version of the interactive plot (Sankey chart) is not saved in the repository due to the lack of rendering capability (`.png` version is published instead). Users can replicate my work by generating the `.html` file locally through `blockingShots_sanky.py`.