'''
NHL Ice Rink Visualizer for Plotly
Renders accurate standard NHL rink geometry (200 ft x 85 ft)
with precise parametric rounded corners, authentic goal creases,
trapezoids, faceoff circles, and nets for spatial hockey analytics.
'''

import plotly.graph_objects as go
import numpy as np

def create_nhl_rink_figure(half_rink=False, theme="dark"):
    """
    Creates a Plotly figure with standard NHL rink markings and geometry.
    """
    fig = go.Figure()
    return add_rink_shapes(fig, half_rink=half_rink, theme=theme)

def _generate_rink_boundary_path(half_rink=False):
    """
    Generates an exact parametric SVG path for the NHL rink boards with 28 ft corner radii.
    Avoids browser/Plotly Cartesian SVG elliptical arc bugs by computing smooth coordinate points.
    """
    r = 28.0
    x_max = 100.0
    y_max = 42.5
    xc = x_max - r  # 72.0
    yc = y_max - r  # 14.5
    n_pts = 24

    # Top-right corner: 90 deg down to 0 deg
    t_tr = np.linspace(np.pi / 2, 0, n_pts)
    x_tr = xc + r * np.cos(t_tr)
    y_tr = yc + r * np.sin(t_tr)

    # Bottom-right corner: 0 deg down to -90 deg
    t_br = np.linspace(0, -np.pi / 2, n_pts)
    x_br = xc + r * np.cos(t_br)
    y_br = -yc + r * np.sin(t_br)

    if half_rink:
        # Starts at center ice top, curves around offensive zone, ends at center ice bottom
        path = f"M 0,{y_max:.2f} L {xc:.2f},{y_max:.2f}"
        for x, y in zip(x_tr, y_tr):
            path += f" L {x:.2f},{y:.2f}"
        path += f" L {x_max:.2f},{-yc:.2f}"
        for x, y in zip(x_br, y_br):
            path += f" L {x:.2f},{y:.2f}"
        path += f" L 0,{-y_max:.2f} Z"
        return path

    # Full rink
    # Bottom-left corner: -90 deg down to -180 deg
    t_bl = np.linspace(-np.pi / 2, -np.pi, n_pts)
    x_bl = -xc + r * np.cos(t_bl)
    y_bl = -yc + r * np.sin(t_bl)

    # Top-left corner: 180 deg down to 90 deg
    t_tl = np.linspace(np.pi, np.pi / 2, n_pts)
    x_tl = -xc + r * np.cos(t_tl)
    y_tl = yc + r * np.sin(t_tl)

    path = f"M {-xc:.2f},{y_max:.2f} L {xc:.2f},{y_max:.2f}"
    for x, y in zip(x_tr, y_tr):
        path += f" L {x:.2f},{y:.2f}"
    path += f" L {x_max:.2f},{-yc:.2f}"
    for x, y in zip(x_br, y_br):
        path += f" L {x:.2f},{y:.2f}"
    path += f" L {-xc:.2f},{-y_max:.2f}"
    for x, y in zip(x_bl, y_bl):
        path += f" L {x:.2f},{y:.2f}"
    path += f" L {-x_max:.2f},{yc:.2f}"
    for x, y in zip(x_tl, y_tl):
        path += f" L {x:.2f},{y:.2f}"
    path += " Z"
    return path

def _generate_crease_path(x_goal, direction=-1, radius=6.0):
    """
    Generates an authentic D-shaped goal crease projecting toward center ice.
    direction = -1: right net (projects left toward center ice)
    direction = +1: left net (projects right toward center ice)
    """
    n_pts = 18
    thetas = np.linspace(-np.pi / 2, np.pi / 2, n_pts)
    xs = x_goal + direction * radius * np.cos(thetas)
    ys = radius * np.sin(thetas)

    path = f"M {x_goal:.2f},{-radius:.2f}"
    for x, y in zip(xs, ys):
        path += f" L {x:.2f},{y:.2f}"
    path += f" L {x_goal:.2f},{radius:.2f} Z"
    return path

def add_rink_shapes(fig, half_rink=False, theme="dark"):
    """
    Adds NHL rink boundary and markings onto an existing or new Plotly figure.
    """
    ice_color = "#0f172a" if theme == "dark" else "#f8fafc"
    board_color = "#94a3b8" if theme == "dark" else "#64748b"
    red_line_color = "#ef4444"
    blue_line_color = "#3b82f6"
    crease_color = "rgba(56, 189, 248, 0.28)" if theme == "dark" else "rgba(186, 230, 253, 0.45)"
    net_color = "rgba(239, 68, 68, 0.2)"

    shapes = []

    # 1. Outer Boards (Rink Boundary with exact rounded corners)
    rink_path = _generate_rink_boundary_path(half_rink=half_rink)
    shapes.append(dict(
        type="path", path=rink_path,
        fillcolor=ice_color,
        line=dict(color=board_color, width=3),
        layer="below"
    ))

    # 2. Center Red Line & Neutral Zone Markings
    # Red line at X=0
    shapes.append(dict(
        type="line", x0=0, x1=0, y0=-42.5, y1=42.5,
        line=dict(color=red_line_color, width=3.5, dash="solid"),
        layer="below"
    ))

    # Center ice faceoff circle (Radius 15 ft)
    shapes.append(dict(
        type="circle", x0=-15, x1=15, y0=-15, y1=15,
        line=dict(color=blue_line_color, width=2),
        layer="below"
    ))
    # Center ice faceoff dot (Radius 1 ft, blue)
    shapes.append(dict(
        type="circle", x0=-1, x1=1, y0=-1, y1=1,
        fillcolor=blue_line_color, line=dict(color=blue_line_color, width=1),
        layer="below"
    ))

    # Referee crease at center ice players' bench side (Radius 10 ft, semi-circle at Y=-42.5)
    ref_thetas = np.linspace(0, np.pi, 16)
    ref_xs = 10.0 * np.cos(ref_thetas)
    ref_ys = -42.5 + 10.0 * np.sin(ref_thetas)
    ref_path = f"M {-10.0:.2f},-42.5"
    for x, y in zip(ref_xs, ref_ys):
        ref_path += f" L {x:.2f},{y:.2f}"
    ref_path += " L 10.0,-42.5"
    shapes.append(dict(
        type="path", path=ref_path,
        line=dict(color=red_line_color, width=1.5),
        layer="below"
    ))

    # 3. Blue Lines (X = +25, and -25 if full rink)
    shapes.append(dict(
        type="line", x0=25, x1=25, y0=-42.5, y1=42.5,
        line=dict(color=blue_line_color, width=3.5),
        layer="below"
    ))
    if not half_rink:
        shapes.append(dict(
            type="line", x0=-25, x1=-25, y0=-42.5, y1=42.5,
            line=dict(color=blue_line_color, width=3.5),
            layer="below"
        ))

    # 4. Goal Lines (X = +89, and -89 if full rink)
    # The goal line meets the curved boards exactly at Y = +/-36.75 ft
    y_goal_line_limit = 36.75
    shapes.append(dict(
        type="line", x0=89, x1=89, y0=-y_goal_line_limit, y1=y_goal_line_limit,
        line=dict(color=red_line_color, width=2),
        layer="below"
    ))
    if not half_rink:
        shapes.append(dict(
            type="line", x0=-89, x1=-89, y0=-y_goal_line_limit, y1=y_goal_line_limit,
            line=dict(color=red_line_color, width=2),
            layer="below"
        ))

    # 5. Goal Creases (D-Shape projecting out from goal line toward center ice)
    shapes.append(dict(
        type="path", path=_generate_crease_path(89.0, direction=-1, radius=6.0),
        fillcolor=crease_color, line=dict(color=red_line_color, width=1.5),
        layer="below"
    ))
    if not half_rink:
        shapes.append(dict(
            type="path", path=_generate_crease_path(-89.0, direction=1, radius=6.0),
            fillcolor=crease_color, line=dict(color=red_line_color, width=1.5),
            layer="below"
        ))

    # 6. Goal Nets (Behind goal lines, 6 ft wide x 3.5 ft deep)
    shapes.append(dict(
        type="rect", x0=89, x1=92.5, y0=-3, y1=3,
        fillcolor=net_color, line=dict(color=red_line_color, width=1.5),
        layer="below"
    ))
    if not half_rink:
        shapes.append(dict(
            type="rect", x0=-92.5, x1=-89, y0=-3, y1=3,
            fillcolor=net_color, line=dict(color=red_line_color, width=1.5),
            layer="below"
        ))

    # 7. Goal Trapezoids (Restricted Area behind net)
    # Right trapezoid: (89, -11) -> (100, -14) and (89, 11) -> (100, 14)
    shapes.append(dict(
        type="line", x0=89, x1=100, y0=11, y1=14,
        line=dict(color=red_line_color, width=1.5),
        layer="below"
    ))
    shapes.append(dict(
        type="line", x0=89, x1=100, y0=-11, y1=-14,
        line=dict(color=red_line_color, width=1.5),
        layer="below"
    ))
    if not half_rink:
        # Left trapezoid: (-89, -11) -> (-100, -14) and (-89, 11) -> (-100, 14)
        shapes.append(dict(
            type="line", x0=-89, x1=-100, y0=11, y1=14,
            line=dict(color=red_line_color, width=1.5),
            layer="below"
        ))
        shapes.append(dict(
            type="line", x0=-89, x1=-100, y0=-11, y1=-14,
            line=dict(color=red_line_color, width=1.5),
            layer="below"
        ))

    # 8. Faceoff Circles & Dots
    # Right zone faceoff circles (X=69, Y=+/-22, Radius 15 ft)
    for y_center in [22, -22]:
        shapes.append(dict(
            type="circle", x0=69-15, x1=69+15, y0=y_center-15, y1=y_center+15,
            line=dict(color=red_line_color, width=1.5),
            layer="below"
        ))
        shapes.append(dict(
            type="circle", x0=69-1, x1=69+1, y0=y_center-1, y1=y_center+1,
            fillcolor=red_line_color, line=dict(color=red_line_color, width=1),
            layer="below"
        ))
        # Neutral zone faceoff dots (X=20, Y=+/-22)
        shapes.append(dict(
            type="circle", x0=20-1, x1=20+1, y0=y_center-1, y1=y_center+1,
            fillcolor=red_line_color, line=dict(color=red_line_color, width=1),
            layer="below"
        ))

    if not half_rink:
        # Left zone faceoff circles (X=-69, Y=+/-22, Radius 15 ft)
        for y_center in [22, -22]:
            shapes.append(dict(
                type="circle", x0=-69-15, x1=-69+15, y0=y_center-15, y1=y_center+15,
                line=dict(color=red_line_color, width=1.5),
                layer="below"
            ))
            shapes.append(dict(
                type="circle", x0=-69-1, x1=-69+1, y0=y_center-1, y1=y_center+1,
                fillcolor=red_line_color, line=dict(color=red_line_color, width=1),
                layer="below"
            ))
            # Left neutral zone faceoff dots (X=-20, Y=+/-22)
            shapes.append(dict(
                type="circle", x0=-20-1, x1=-20+1, y0=y_center-1, y1=y_center+1,
                fillcolor=red_line_color, line=dict(color=red_line_color, width=1),
                layer="below"
            ))

    # Add ample padding around the boards so border and markers are never clipped
    x_range = [-6, 108] if half_rink else [-108, 108]
    y_range = [-48, 48]

    fig.update_layout(
        shapes=shapes,
        xaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=x_range, fixedrange=True
        ),
        yaxis=dict(
            showgrid=False, zeroline=False, showticklabels=False,
            range=y_range, scaleanchor="x", scaleratio=1, fixedrange=True
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=15, r=15, t=35, b=15)
    )
    return fig
