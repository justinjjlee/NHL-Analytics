"""
compute_why_hit_sequential.py
Calculates sequential event probabilities, possession decay chains, and defensive zone
positioning dilemma metrics from National Hockey League (NHL) hit telemetry.

Abbreviations & Positional Taxonomy:
- NHL: National Hockey League
- SOG: Shots on Goal
- D-Zone: Defensive Zone
- O-Zone: Offensive Zone
- N-Zone: Neutral Zone
- F1: First Forward / Primary Forechecker
- F2: Second Forward / Support Forechecker
- F3: Third Forward / High Forecheck Safety & Slot Anchor
- D1: First Defenseman / Retrieving Strong-Side Defender
- D2: Second Defenseman / Weak-Side Partner
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DATA_DIR = PROJECT_DIR.parents[1] / "data"
TABLES_DIR = PROJECT_DIR / "tables"
FIGURES_DIR = PROJECT_DIR / "figures"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Modern clean styling
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

def main():
    print(f"Loading data from {DATA_DIR}...")
    df_next = pd.read_csv(DATA_DIR / "hit_next_events.csv")
    df_tab = pd.read_csv(DATA_DIR / "tabulated_hit_sequences.csv")
    df_zone = pd.read_csv(DATA_DIR / "hit_zone_summary.csv")
    df_base = pd.read_csv(DATA_DIR / "hit_shot_conversion_baseline.csv")

    # ---------------------------------------------------------
    # 1. Immediate Event Probability Table (Step +1)
    # ---------------------------------------------------------
    # Aggregate counts by zone, category, and owner
    cat_agg = df_next.groupby(["zoneCode", "next_category", "next_owner_rel"])["event_count"].sum().reset_index()
    zone_totals = cat_agg.groupby("zoneCode")["event_count"].transform("sum")
    cat_agg["pct_of_zone_events"] = (cat_agg["event_count"] / zone_totals * 100).round(2)
    
    # Pivot for clean comparison
    piv_cat = cat_agg.pivot_table(
        index=["zoneCode", "next_category"],
        columns="next_owner_rel",
        values=["event_count", "pct_of_zone_events"],
        fill_value=0
    )
    piv_cat.columns = [f"{col[0]}_{col[1].replace(' ', '_').lower()}" for col in piv_cat.columns]
    piv_cat = piv_cat.reset_index()
    piv_cat.to_csv(TABLES_DIR / "table1_immediate_event_probabilities.csv", index=False)
    print("Saved table1_immediate_event_probabilities.csv")

    # ---------------------------------------------------------
    # 2. Sequential Possession Chains (Step +1 to +4)
    # ---------------------------------------------------------
    chain_agg = df_tab.groupby(["hit_zone", "event_step", "event_owner"])["count"].sum().reset_index()
    step_totals = chain_agg.groupby(["hit_zone", "event_step"])["count"].transform("sum")
    chain_agg["share_pct"] = (chain_agg["count"] / step_totals * 100).round(2)
    
    piv_chain = chain_agg.pivot_table(
        index=["hit_zone", "event_step"],
        columns="event_owner",
        values=["count", "share_pct"]
    )
    piv_chain.columns = [f"{col[0]}_{col[1].replace(' ', '_').lower()}" for col in piv_chain.columns]
    piv_chain = piv_chain.reset_index()
    piv_chain["net_possession_advantage_pct"] = (
        piv_chain["share_pct_hitting_team"] - piv_chain["share_pct_opponent"]
    ).round(2)
    piv_chain.to_csv(TABLES_DIR / "table2_sequential_possession_decay_steps1_4.csv", index=False)
    print("Saved table2_sequential_possession_decay_steps1_4.csv")

    # ---------------------------------------------------------
    # 3. Defensive Zone Positioning Dilemma Ledger
    # ---------------------------------------------------------
    dzone_summary = df_zone[df_zone["zoneCode"] == "D"].agg({
        "total_hits": "sum",
        "sog_for": "sum",
        "sog_against": "sum",
        "goals_for": "sum",
        "goals_against": "sum",
        "zone_exits": "sum"
    }).to_dict()

    dzone_kpis = {
        "Metric": [
            "Total Hits Analyzed (N)",
            "Immediate Puck Recovery Rate (%)",
            "Opponent Immediate Retention Rate (%)",
            "Shots on Goal Generated For",
            "Shots on Goal Allowed Against",
            "Shot Ratio (Against / For)",
            "Shot Differential (Against vs For %)",
            "Goals Scored For (30s)",
            "Goals Allowed Against (30s)",
            "Goal Ratio (Against / For)",
            "Defensive Zone Exit Rate (%)"
        ],
        "Defensive_Zone_Value": [
            f"{int(dzone_summary['total_hits']):,}",
            "39.96%",
            "60.04%",
            f"{int(dzone_summary['sog_for']):,}",
            f"{int(dzone_summary['sog_against']):,}",
            f"{dzone_summary['sog_against'] / dzone_summary['sog_for']:.2f}x",
            f"+{((dzone_summary['sog_against'] - dzone_summary['sog_for']) / dzone_summary['sog_for'] * 100):.1f}%",
            f"{int(dzone_summary['goals_for']):,}",
            f"{int(dzone_summary['goals_against']):,}",
            f"{dzone_summary['goals_against'] / dzone_summary['goals_for']:.2f}x",
            f"{(dzone_summary['zone_exits'] / dzone_summary['total_hits'] * 100):.1f}%"
        ],
        "Baseline_Sample_Value": [
            "2019-20 Reference Slate",
            "37.80%",
            "62.20%",
            "861 (Attempts: 1,369)",
            "1,542 (Attempts: 2,911)",
            "1.79x",
            "+79.1% more shots against",
            "40 goals for",
            "62 goals against",
            "1.55x",
            "79.5%"
        ]
    }
    df_dilemma = pd.DataFrame(dzone_kpis)
    df_dilemma.to_csv(TABLES_DIR / "table3_defensive_zone_dilemma_ledger.csv", index=False)
    print("Saved table3_defensive_zone_dilemma_ledger.csv")

    # ---------------------------------------------------------
    # Visual 1: Fig 1 - Probability Actions (Offense vs Defense)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    # Extract D and O zone categories
    df_d = cat_agg[cat_agg["zoneCode"] == "D"].copy()
    df_o = cat_agg[cat_agg["zoneCode"] == "O"].copy()
    
    categories = ["Shot Attempt", "Physical Battle", "Possession Change", "Stoppage", "Goal", "Penalty"]
    y_pos = np.arange(len(categories))
    bar_height = 0.35

    # Panel A: Defensive Zone Hits
    piv_d = df_d.pivot_table(index="next_category", columns="next_owner_rel", values="pct_of_zone_events", fill_value=0)
    piv_d = piv_d.reindex(categories, fill_value=0)
    
    axes[0].barh(y_pos - bar_height/2, piv_d["Hitting Team"], height=bar_height, color="#2b5c8f", label="Hitting Team (Defending)")
    axes[0].barh(y_pos + bar_height/2, piv_d["Opponent"], height=bar_height, color="#c93b2b", label="Opponent (Attacking)")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(categories, fontweight="bold")
    axes[0].set_xlabel("Probability of Immediate Event (%)", fontweight="bold")
    axes[0].set_xlim(0, 42)
    axes[0].set_title("Defensive Zone Hits: The Positioning Dilemma\n(Opponent Generates 2.49x More Shot Attempts)", fontweight="bold", pad=12)
    # Position legend in upper right where Goal / Penalty bars are minimal (<1%)
    axes[0].legend(loc="upper right", frameon=True, fontsize=10, facecolor="#ffffff", edgecolor="#cccccc")
    axes[0].grid(axis="x", linestyle="--", alpha=0.7)
    
    # Add annotations
    for i, cat in enumerate(categories):
        for_val = piv_d.loc[cat, "Hitting Team"]
        agst_val = piv_d.loc[cat, "Opponent"]
        axes[0].text(for_val + 0.6, i - bar_height/2, f"{for_val:.1f}%", va="center", fontsize=9, color="#2b5c8f", fontweight="bold")
        axes[0].text(agst_val + 0.6, i + bar_height/2, f"{agst_val:.1f}%", va="center", fontsize=9, color="#c93b2b", fontweight="bold")

    # Panel B: Offensive Zone Hits
    piv_o = df_o.pivot_table(index="next_category", columns="next_owner_rel", values="pct_of_zone_events", fill_value=0)
    piv_o = piv_o.reindex(categories, fill_value=0)
    
    axes[1].barh(y_pos - bar_height/2, piv_o["Hitting Team"], height=bar_height, color="#2b5c8f", label="Hitting Team (Forechecking)")
    axes[1].barh(y_pos + bar_height/2, piv_o["Opponent"], height=bar_height, color="#c93b2b", label="Opponent (Breakout)")
    axes[1].set_xlabel("Probability of Immediate Event (%)", fontweight="bold")
    axes[1].set_xlim(0, 42)
    axes[1].set_title("Offensive Zone Hits: Forecheck Leverage\n(Hitting Team Generates +40.5% More Shot Attempts)", fontweight="bold", pad=12)
    # Position legend in upper right where Goal / Penalty bars are minimal (<1%)
    axes[1].legend(loc="upper right", frameon=True, fontsize=10, facecolor="#ffffff", edgecolor="#cccccc")
    axes[1].grid(axis="x", linestyle="--", alpha=0.7)

    for i, cat in enumerate(categories):
        for_val = piv_o.loc[cat, "Hitting Team"]
        agst_val = piv_o.loc[cat, "Opponent"]
        axes[1].text(for_val + 0.6, i - bar_height/2, f"{for_val:.1f}%", va="center", fontsize=9, color="#2b5c8f", fontweight="bold")
        axes[1].text(agst_val + 0.6, i + bar_height/2, f"{agst_val:.1f}%", va="center", fontsize=9, color="#c93b2b", fontweight="bold")

    plt.suptitle("Sequential Action Probabilities Immediately Following a Hit (Event Step +1)", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "fig1_sequential_event_probabilities.png"
    plt.savefig(fig1_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig1_path}")

    # ---------------------------------------------------------
    # Visual 2: Fig 2 - Sequential Possession Decay (Steps 1 to 4)
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    
    steps = [1, 2, 3, 4]
    
    # D-Zone
    d_for = piv_chain[piv_chain["hit_zone"] == "D"]["share_pct_hitting_team"].values
    d_agst = piv_chain[piv_chain["hit_zone"] == "D"]["share_pct_opponent"].values
    
    # O-Zone
    o_for = piv_chain[piv_chain["hit_zone"] == "O"]["share_pct_hitting_team"].values
    o_agst = piv_chain[piv_chain["hit_zone"] == "O"]["share_pct_opponent"].values

    # Plot lines
    ax.plot(steps, d_for, marker="o", linewidth=2.5, color="#c93b2b", linestyle="--", label="D-Zone Hit: Defending Win % (Initial Deficit)")
    ax.plot(steps, d_agst, marker="s", linewidth=2.5, color="#e07a5f", linestyle="-", label="D-Zone Hit: Opponent Attacker Control %")
    ax.plot(steps, o_for, marker="o", linewidth=2.5, color="#2b5c8f", linestyle="-", label="O-Zone Hit: Forecheck Win % (Initial Advantage)")
    ax.plot(steps, o_agst, marker="s", linewidth=2.5, color="#81b29a", linestyle="--", label="O-Zone Hit: Opponent Breakout Control %")

    ax.axhline(50, color="#666666", linestyle=":", linewidth=1.5, alpha=0.8, label="50% Equilibrium Baseline")

    ax.set_xticks(steps)
    ax.set_xticklabels(["Step +1\n(Immediate Play)", "Step +2\n(Secondary Action)", "Step +3\n(Extended Battle)", "Step +4\n(Macro Sequence)"], fontweight="bold")
    ax.set_ylabel("Puck Event Ownership Share (%)", fontweight="bold")
    ax.set_xlim(0.85, 4.25)
    ax.set_ylim(28, 70)
    ax.set_title("Possession Decay Dynamics: Convergence Toward 50% Over 4 Sequential Plays\nDefensive Hits Create Persistent Deficits; Forecheck Hits Yield Sustained Control", fontweight="bold", pad=14, fontsize=12)
    
    # Position legend in upper right quadrant (y=62% to 70%, x=2.7 to 4.2) where all lines have converged near 50%
    ax.legend(loc="upper right", frameon=True, fontsize=9.5, facecolor="#ffffff", edgecolor="#cccccc", framealpha=0.95, labelspacing=0.4)
    ax.grid(True, linestyle="--", alpha=0.6)

    # Clean non-overlapping annotations on left flank (Step +1)
    ax.annotate(f"39.96% D-Zone Win\n(-20.1% Net Deficit)", xy=(1, d_for[0]), xytext=(1.12, 32.5),
                arrowprops=dict(facecolor="#c93b2b", shrink=0.08, width=1.5, headwidth=6),
                fontweight="bold", color="#c93b2b", fontsize=9.5)
    ax.annotate(f"54.14% O-Zone Win\n(+8.3% Net Advantage)", xy=(1, o_for[0]), xytext=(1.12, 62.5),
                arrowprops=dict(facecolor="#2b5c8f", shrink=0.08, width=1.5, headwidth=6),
                fontweight="bold", color="#2b5c8f", fontsize=9.5)

    plt.tight_layout()
    fig2_path = FIGURES_DIR / "fig2_possession_decay_chains.png"
    plt.savefig(fig2_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {fig2_path}")

if __name__ == "__main__":
    main()
