import matplotlib.pyplot as plt
import numpy as np


# ── Viz 1: Model Summary Stats Table ──────────────────────────────────────
def plot_model_summary_table(B_hat, B_lo, B_hi, event_cols, a0_hat, half_life):
    """
    Renders a formatted table figure summarising every estimated parameter.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    headers = ["Event", "B\u0302 (Point Est.)", "95% CI Low", "95% CI High"]
    rows = []
    for i, ev in enumerate(event_cols):
        rows.append([
            ev,
            f"{B_hat[i]:.4f}",
            f"{B_lo[i]:.4f}",
            f"{B_hi[i]:.4f}",
        ])
    # Append decay summary rows
    rows.append(["\u2500" * 20, "\u2500" * 12, "\u2500" * 12, "\u2500" * 12])
    rows.append(["Decay (a\u2080)", f"{a0_hat:.5f}", "", ""])
    rows.append(["Half-Life (sec)", f"{half_life:.1f}", "", ""])

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)

    # Style header row
    for j in range(len(headers)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Alternating row colours
    for i in range(1, len(rows) + 1):
        for j in range(len(headers)):
            if i <= len(event_cols):
                table[i, j].set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')
            else:
                table[i, j].set_facecolor('#f9e79f')  # separator / summary

    ax.set_title('MomentumSSM Estimated Parameters', fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()
    return fig


# ── Viz 2: Single-Game Momentum Timeline ──────────────────────────────────
def plot_game_momentum_timeline(smoothed_game, title_suffix=""):
    """
    Overlays the filtered/smoothed latent momentum state with the observed
    rolling zone-share proxy for a single game.
    """
    x_smooth = smoothed_game["x_smooth"]
    Y = smoothed_game["Y"]
    T = len(x_smooth)
    t_axis = np.arange(T)

    fig, ax1 = plt.subplots(figsize=(14, 5))

    color_obs = '#3498db'
    color_lat = '#e74c3c'

    ax1.plot(t_axis, Y, color=color_obs, alpha=0.35, linewidth=0.8, label='Observed Zone Share (90s)')
    ax1.set_xlabel('Event Index')
    ax1.set_ylabel('Zone Share', color=color_obs)
    ax1.tick_params(axis='y', labelcolor=color_obs)
    ax1.axhline(0.5, color='gray', linestyle='--', linewidth=0.5)

    ax2 = ax1.twinx()
    ax2.plot(t_axis, x_smooth, color=color_lat, linewidth=1.8, label='Smoothed Momentum (x_t)')
    ax2.set_ylabel('Latent Momentum', color=color_lat)
    ax2.tick_params(axis='y', labelcolor=color_lat)
    ax2.axhline(0, color='gray', linestyle=':', linewidth=0.5)

    game_id = smoothed_game.get("game_id", "")
    fig.suptitle(f'Game Momentum Timeline{" - " + str(game_id) if game_id else ""}{title_suffix}',
                 fontsize=13, fontweight='bold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    fig.tight_layout()
    return fig


# ── Viz 3: B Coefficients ─────────────────────────────────────────────────
def plot_B_coefficients(B_hat, B_lo, B_hi, event_cols):
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(event_cols))

    # Error bars (use absolute distance to prevent 1-sided bars if B_hat falls outside CI due to epoch mismatch)
    xerr = [np.abs(B_hat - B_lo), np.abs(B_hi - B_hat)]

    colors = []
    for ev in event_cols:
        if 'hit_for' in ev:
            colors.append('#27ae60')
        elif 'hit_against' in ev:
            colors.append('#e74c3c')
        elif 'shot' in ev:
            colors.append('#2980b9')
        elif ev in ('takeaway', 'block_for', 'faceoff_win'):
            colors.append('#f39c12')
        else:
            colors.append('#95a5a6')

    ax.barh(y_pos, B_hat, xerr=xerr, align='center', alpha=0.8,
            color=colors, ecolor='black', capsize=5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(event_cols)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Coefficient (Impact on Momentum, Latent Units)')
    ax.set_title('Estimated Event Impacts on Momentum (95% CI)')
    ax.axvline(x=0, color='red', linestyle='--')

    plt.tight_layout()
    return fig


# ── Viz 4: Impulse Response ───────────────────────────────────────────────
def plot_impulse_response(a0_val, b_vals, max_sec=180):
    fig, ax = plt.subplots(figsize=(10, 6))
    t = np.arange(0, max_sec)

    palette = ['#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6']
    for idx, (label, b) in enumerate(b_vals.items()):
        # Impulse response is just b * (a0^t)
        response = b * (a0_val ** t)
        ax.plot(t, response, label=f"{label} (b={b:.3f})",
                linewidth=2, color=palette[idx % len(palette)])

    half_life = np.log(0.5) / np.log(a0_val) if a0_val > 0 and a0_val < 1 else float('inf')
    ax.axvline(x=half_life, color='gray', linestyle=':', alpha=0.6, label=f'Half-life = {half_life:.1f}s')
    ax.set_xlabel('Time elapsed since event (seconds)')
    ax.set_ylabel('Residual Momentum Impact (Latent Units)')
    ax.set_title(f'Impulse Response Functions (Half-life = {half_life:.1f}s)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ── Viz 5: Hit Impact by Ice Zone ─────────────────────────────────────────
def plot_hit_impact_by_zone(B_hat, B_lo, B_hi, event_cols):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left panel: Hits FOR by zone
    for_cols = ["hit_for_O", "hit_for_D", "hit_for_N"]
    against_cols = ["hit_against_O", "hit_against_D", "hit_against_N"]
    zone_labels = ["Offensive Zone", "Defensive Zone", "Neutral Zone"]

    for ax, target_cols, panel_title, panel_color in [
        (axes[0], for_cols, "Hits FOR (by focal team)", ['#27ae60', '#2ecc71', '#a9dfbf']),
        (axes[1], against_cols, "Hits AGAINST (by opponent)", ['#e74c3c', '#ec7063', '#f5b7b1']),
    ]:
        y_pos = np.arange(len(target_cols))
        vals, los, his = [], [], []
        for col in target_cols:
            idx = event_cols.index(col)
            vals.append(B_hat[idx])
            los.append(B_lo[idx])
            his.append(B_hi[idx])

        vals = np.array(vals)
        los = np.array(los)
        his = np.array(his)
        xerr = [np.abs(vals - los), np.abs(his - vals)]

        ax.barh(y_pos, vals, xerr=xerr, align='center', alpha=0.8,
                color=panel_color, ecolor='black', capsize=5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(zone_labels)
        ax.invert_yaxis()
        ax.set_title(panel_title, fontsize=11)
        ax.set_xlabel('Impact on Momentum (Latent Units)')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.7)

    fig.suptitle('Hit Impact by Ice Zone', fontsize=14, fontweight='bold')
    fig.tight_layout()
    return fig


# ── Viz 6: Team Momentum Scatter ─────────────────────────────────────────
def plot_team_momentum_scatter(df_teams):
    """
    Scatter plot: hits per game vs average smoothed momentum.
    Works with a team-level aggregation DataFrame containing columns:
        - focal_team_abbrev
        - hits_per_game
        - avg_momentum
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    ax.scatter(df_teams['hits_per_game'], df_teams['avg_momentum'],
               alpha=0.7, s=100, edgecolors='black', linewidths=0.5, color='#3498db')

    for _, row in df_teams.iterrows():
        ax.annotate(row['focal_team_abbrev'],
                     (row['hits_per_game'], row['avg_momentum']),
                     textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax.set_xlabel('Hits per Game (Home Perspective)')
    ax.set_ylabel('Average Smoothed Momentum')
    ax.set_title('Team Hitting Volume vs Momentum Generated')
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    return fig


# ── Viz 7: Expected Goals Probability Surface ────────────────────────────
def plot_expected_goals_surface(model, x_range=(-5, 5), score_states=[-1, 0, 1]):
    fig, ax = plt.subplots(figsize=(10, 6))
    x_vals = np.linspace(x_range[0], x_range[1], 200)

    colors = {'Trailing (-1)': '#e74c3c', 'Tied (0)': '#3498db', 'Leading (+1)': '#27ae60'}
    for score, (label, color) in zip(score_states, colors.items()):
        # Features: [momentum, is_home=1, delta_pyth=0, score_state]
        Z = np.zeros((200, 4))
        Z[:, 0] = x_vals        # momentum
        Z[:, 1] = 1             # home
        Z[:, 2] = 0             # pythagorean delta
        Z[:, 3] = score         # score state

        probs = model.predict_proba(Z)[:, 1]
        ax.plot(x_vals, probs, label=label, linewidth=2, color=color)

    ax.set_xlabel('Smoothed Latent Momentum (x_t)')
    ax.set_ylabel('P(High-Danger Chance in next window)')
    ax.set_title('Expected Scoring Probability vs Momentum')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# ── Legacy compatibility alias ────────────────────────────────────────────
def plot_player_momentum_scatter(df_players):
    """Original stub kept for backwards-compatibility."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(df_players['hits_per_60'], df_players['momentum_per_60'], alpha=0.6)
    ax.set_xlabel('Hits per 60 mins')
    ax.set_ylabel('Momentum Generated per 60 mins')
    ax.set_title('Player Evaluation: Hitting Volume vs Impact')
    return fig
