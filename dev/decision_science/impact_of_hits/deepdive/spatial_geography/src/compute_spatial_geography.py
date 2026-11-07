"""
compute_spatial_geography.py
Analyzes the spatial geography of hits across 2D rink coordinates in the National
Hockey League (NHL), comparing corners vs neutral zone vs slot areas in terms of puck
recovery, shot generation, and downstream goal probabilities.

Abbreviations & Positional Taxonomy:
- NHL: National Hockey League
- SOG: Shots on Goal
- D-Zone: Defensive Zone
- O-Zone: Offensive Zone
- N-Zone: Neutral Zone
- F1: First Forward / Primary Forechecker (endboard pursuit & body check pin)
- F2: Second Forward / Support Forechecker (downhill loose-puck retrieval)
- F3: Third Forward / High Forecheck Safety & Slot Anchor (high slot shooting & rush safety)
- D1: First Defenseman / Retrieving Strong-Side Defender (puck retriever below goal line)
- D2: Second Defenseman / Weak-Side Partner (crease & low slot defender)
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR.parents[1] / "data"
TABLES_DIR = PROJECT_DIR / "tables"
FIGURES_DIR = PROJECT_DIR / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 1.0,
})

def draw_nhl_rink(ax):
    """Draws a standardized 200 x 85 ft NHL hockey rink on axes."""
    # Rink boundary (x in [-100, 100], y in [-42.5, 42.5])
    rink = patches.FancyBboxPatch(
        (-100, -42.5), 200, 85,
        boxstyle="round,pad=0,rounding_size=28",
        linewidth=2, edgecolor="#222222", facecolor="#f8fafc", zorder=1
    )
    ax.add_patch(rink)

    # Center red line
    ax.plot([0, 0], [-42.5, 42.5], color="#c93b2b", linewidth=2.5, zorder=2)

    # Blue lines (x = -25 and x = +25)
    ax.plot([-25, -25], [-42.5, 42.5], color="#2b5c8f", linewidth=3, zorder=2)
    ax.plot([25, 25], [-42.5, 42.5], color="#2b5c8f", linewidth=3, zorder=2)

    # Goal lines (x = -89 and x = +89)
    ax.plot([-89, -89], [-38, 38], color="#c93b2b", linewidth=1.5, zorder=2)
    ax.plot([89, 89], [-38, 38], color="#c93b2b", linewidth=1.5, zorder=2)

    # Center circle
    center_circle = plt.Circle((0, 0), 15, color="#2b5c8f", fill=False, linewidth=1.5, zorder=2)
    ax.add_patch(center_circle)

    # Faceoff circles
    for x in [-69, 69]:
        for y in [-22, 22]:
            circle = plt.Circle((x, y), 15, color="#c93b2b", fill=False, linewidth=1.2, zorder=2)
            ax.add_patch(circle)

    ax.set_xlim(-105, 105)
    ax.set_ylim(-47, 47)
    ax.set_aspect("equal")
    ax.set_xticks([-100, -75, -25, 0, 25, 75, 100])
    ax.set_yticks([-42.5, -20, 0, 20, 42.5])
    ax.set_xlabel("Rink Length (Feet, x: -100 Defensive Net to +100 Offensive Net)", fontweight="bold")
    ax.set_ylabel("Rink Width (Feet, y: -42.5 to +42.5)", fontweight="bold")

def main():
    print(f"Loading spatial data from {DATA_DIR}...")
    df_grid = pd.read_csv(DATA_DIR / "rink_grid_hit_sequences.csv")
    grid_all = df_grid[df_grid["season"] == "ALL"].copy()

    # ---------------------------------------------------------
    # 1. Spatial Partitioning & Cluster Summary
    # ---------------------------------------------------------
    def assign_cluster(row):
        x, y = row["x_bin"], row["y_bin"]
        if x <= -60 and abs(y) >= 20:
            return "1. D-Zone Deep Corners"
        elif x <= -25 and abs(y) >= 20:
            return "2. D-Zone Half-Wall / Boards"
        elif x <= -25 and abs(y) < 20:
            return "3. D-Zone Slot / High House"
        elif abs(x) < 25 and abs(y) >= 20:
            return "4. Neutral Zone Boards"
        elif abs(x) < 25 and abs(y) < 20:
            return "5. Neutral Zone Center Ice"
        elif x >= 25 and abs(y) < 20:
            return "6. O-Zone Slot / High House"
        elif x >= 25 and x < 60 and abs(y) >= 20:
            return "7. O-Zone Half-Wall / Points"
        elif x >= 60 and abs(y) >= 20:
            return "8. O-Zone Deep Corners / Behind Net"
        return "9. Other"

    grid_all["spatial_cluster"] = grid_all.apply(assign_cluster, axis=1)

    # Compute weighted cluster metrics
    cluster_metrics = grid_all.groupby("spatial_cluster").apply(lambda g: pd.Series({
        "Total_Hits": g["total_hits"].sum(),
        "Hit_Share_Pct": g["total_hits"].sum() / grid_all["total_hits"].sum() * 100,
        "Puck_Recovery_Pct": (g["puck_win_pct"] * g["total_hits"]).sum() / g["total_hits"].sum(),
        "SOG_For_Per_100": (g["sog_for_per_100"] * g["total_hits"]).sum() / g["total_hits"].sum(),
        "SOG_Against_Per_100": (g["sog_against_per_100"] * g["total_hits"]).sum() / g["total_hits"].sum(),
        "Net_SOG_Per_100": (g["net_sog_per_100"] * g["total_hits"]).sum() / g["total_hits"].sum(),
        "Goals_For_Per_100": (g["goals_for_per_100"] * g["total_hits"]).sum() / g["total_hits"].sum(),
        "Goals_Against_Per_100": (g["goals_against_per_100"] * g["total_hits"]).sum() / g["total_hits"].sum(),
        "Net_Goals_Per_100": (g["net_goals_per_100"] * g["total_hits"]).sum() / g["total_hits"].sum(),
    }), include_groups=False).round(2).reset_index()

    cluster_metrics.to_csv(TABLES_DIR / "table1_spatial_cluster_summary.csv", index=False)
    print("Saved table1_spatial_cluster_summary.csv")

    # ---------------------------------------------------------
    # 2. Neutral Zone vs. Corner Direct Comparison Table
    # ---------------------------------------------------------
    # Neutral Zone (all |x| < 25) vs Corners (|x| >= 60, |y| >= 20)
    grid_all["macro_landmark"] = np.where(
        grid_all["x_bin"].abs() < 25, "Neutral Zone (|x| < 25)",
        np.where((grid_all["x_bin"].abs() >= 60) & (grid_all["y_bin"].abs() >= 20),
                 np.where(grid_all["x_bin"] < 0, "D-Zone Deep Corners", "O-Zone Deep Corners"),
                 "Half-Wall / Interior Slot")
    )

    macro_summary = grid_all.groupby("macro_landmark").apply(lambda g: pd.Series({
        "Sample_Hits_N": int(g["total_hits"].sum()),
        "Puck_Recovery_Rate": f"{((g['puck_win_pct'] * g['total_hits']).sum() / g['total_hits'].sum()):.1f}%",
        "SOG_For_Per_100": round((g["sog_for_per_100"] * g["total_hits"]).sum() / g["total_hits"].sum(), 2),
        "SOG_Against_Per_100": round((g["sog_against_per_100"] * g["total_hits"]).sum() / g["total_hits"].sum(), 2),
        "Net_SOG_Per_100": round((g["net_sog_per_100"] * g["total_hits"]).sum() / g["total_hits"].sum(), 2),
        "Net_Goals_Per_100": round((g["net_goals_per_100"] * g["total_hits"]).sum() / g["total_hits"].sum(), 2),
        "Strategic_Outcome": np.where(
            g["net_sog_per_100"].mean() < -5, "High Slot Concession Risk",
            np.where(g["net_sog_per_100"].mean() > 5, "Strong Positive Forecheck Driver", "Neutral Transition Choke")
        )
    }), include_groups=False).reset_index()

    macro_summary.to_csv(TABLES_DIR / "table2_neutral_vs_corners_comparison.csv", index=False)
    print("Saved table2_neutral_vs_corners_comparison.csv")

    # ---------------------------------------------------------
    # Visual 1: Fig 1 - 2D Rink Spatial Heatmap (Net SOG Differential)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(14, 7))
    draw_nhl_rink(ax)

    # Scatter plot of spatial bins colored by net_sog_per_100
    sc = ax.scatter(
        grid_all["x_bin"], grid_all["y_bin"],
        s=grid_all["total_hits"] / 250 + 20,  # Size proportional to hit density
        c=grid_all["net_sog_per_100"],
        cmap="coolwarm_r",  # Red for negative (danger for defender), Blue for positive (advantage for attacker)
        vmin=-20, vmax=20,
        alpha=0.85, edgecolors="#333333", linewidths=0.5, zorder=3
    )

    cbar = plt.colorbar(sc, ax=ax, orientation="horizontal", pad=0.12, shrink=0.6)
    cbar.set_label("Net Shot Differential per 100 Hits (Blue = For Advantage, Red = Against Concession)", fontweight="bold")

    # Annotations on key rink landmarks
    ax.text(-80, 32, "D-Zone Corners\nNet SOG: -14.8/100\nRecovery: 39.9%", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="#ffebee", ec="#c93b2b", lw=1.2), fontsize=8.5, fontweight="bold", zorder=4)
    ax.text(80, 32, "O-Zone Corners\nNet SOG: +7.2/100\nRecovery: 54.1%", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="#e3f2fd", ec="#2b5c8f", lw=1.2), fontsize=8.5, fontweight="bold", zorder=4)
    ax.text(0, 32, "Neutral Zone Boards\nNet SOG: -1.8/100\nRecovery: 47.0%", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="#f5f5f5", ec="#666666", lw=1.2), fontsize=8.5, fontweight="bold", zorder=4)
    ax.text(-50, 0, "D-Zone Slot Danger\nHigh Exposure", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="#ffebee", ec="#c93b2b", lw=1), fontsize=8, color="#c93b2b", zorder=4)

    ax.set_title("The Geography of Contact: 2D Spatial Shot Differential Across the NHL Rink\nBubble Size Indicates Collision Frequency; Color Maps Net Shot Advantage", fontweight="bold", pad=14)
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "fig1_rink_spatial_heatmap.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig1_path}")

    # ---------------------------------------------------------
    # Visual 2: Fig 2 - Neutral Zone vs. Corners Comparison
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    landmarks = ["D-Zone Deep Corners", "Neutral Zone Boards", "O-Zone Deep Corners"]
    sub_df = cluster_metrics[cluster_metrics["spatial_cluster"].isin([
        "1. D-Zone Deep Corners", "4. Neutral Zone Boards", "8. O-Zone Deep Corners / Behind Net"
    ])].copy()
    
    # Order properly
    sub_df["landmark_label"] = ["D-Zone Corners", "Neutral Zone Boards", "O-Zone Corners"]

    colors = ["#c93b2b", "#666666", "#2b5c8f"]

    # Panel A: Net SOG Differential
    bars1 = axes[0].bar(sub_df["landmark_label"], sub_df["Net_SOG_Per_100"], color=colors, width=0.55, edgecolor="#222222")
    axes[0].axhline(0, color="#333333", linestyle="-", linewidth=1.2)
    axes[0].set_ylabel("Net Shots on Goal / 100 Hits", fontweight="bold")
    axes[0].set_ylim(-19, 12)
    axes[0].set_title("Net Shot Differential by Rink Location", fontweight="bold", pad=12)
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars1:
        yval = bar.get_height()
        va = "bottom" if yval >= 0 else "top"
        offset = 0.6 if yval >= 0 else -1.2
        axes[0].text(bar.get_x() + bar.get_width()/2, yval + offset, f"{yval:+.1f}",
                     ha="center", va=va, fontweight="bold", fontsize=11)

    # Panel B: Immediate Puck Win Rate (%)
    bars2 = axes[1].bar(sub_df["landmark_label"], sub_df["Puck_Recovery_Pct"], color=colors, width=0.55, edgecolor="#222222")
    axes[1].axhline(50, color="#666666", linestyle=":", linewidth=1.5, label="50% Equilibrium Baseline")
    axes[1].set_ylabel("Immediate Puck Recovery Rate (%)", fontweight="bold")
    axes[1].set_ylim(20, 75)
    axes[1].set_title("Puck Recovery Probability by Rink Location", fontweight="bold", pad=12)
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)
    # Generous vertical headroom ensures legend does not crowd bars or data labels
    axes[1].legend(loc="upper left", frameon=True, fontsize=10, facecolor="#ffffff", edgecolor="#cccccc")
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, yval + 1.2, f"{yval:.1f}%",
                     ha="center", va="bottom", fontweight="bold", fontsize=11)

    plt.suptitle("Neutral Zone Blue-Line Collisions vs. Endboard Battles: Strategic Asymmetry", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig2_path = FIGURES_DIR / "fig2_corners_vs_neutral_bar.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig2_path}")

if __name__ == "__main__":
    main()
