"""
Computation & Visualization Script for Blog Post 5:
dev/player_team_success/rebuild/deepdive/modern_shift/
Spacious Full-Width Vertical Layout with Plain-English Labels (Zero Crowding)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[1]
REBUILD_DATA_DIR = (
    PROJECT_DIR.parents[1] / "data" / "processed"
    if (PROJECT_DIR.parents[1] / "data" / "processed").exists()
    else PROJECT_DIR.parent / "rebuild" / "data" / "processed"
)
FIG_DIR = PROJECT_DIR / "figures"
TAB_DIR = PROJECT_DIR / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.sans-serif"] = "DejaVu Sans"


def style_axis(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94A3B8")
    ax.spines["bottom"].set_color("#94A3B8")
    ax.tick_params(colors="#1E293B", labelsize=10.5)
    ax.grid(True, axis="y", color="#E2E8F0", linestyle="-", linewidth=0.9, zorder=0)
    if grid_axis == "both":
        ax.grid(True, axis="x", color="#F1F5F9", linestyle="--", linewidth=0.8, zorder=0)


def main() -> None:
    df_ep = pd.read_parquet(REBUILD_DATA_DIR / "panel_rebuild_episodes.parquet")
    df_ts = pd.read_parquet(REBUILD_DATA_DIR / "panel_team_season.parquet")

    completed = df_ep[df_ep["outcome_tier_label"] != "In-Progress / Incomplete Cycle"].copy()

    arch_sum = (
        completed.groupby("primary_archetype")
        .agg(
            n_episodes=("episode_id", "count"),
            mean_duration_yrs=("duration_years", "mean"),
            median_duration_yrs=("duration_years", "median"),
            draft_core_share=("pathway_a_draft_share", "mean"),
            trade_ufa_core_share=("pathway_b_trade_share", "mean"),
            tier1_playoff_rate=("achieved_tier_1", "mean"),
            tier2_final_four_rate=("achieved_tier_2", "mean"),
            tier3_cup_rate=("achieved_tier_3", "mean"),
            purgatory_failure_rate=("is_purgatory_trap", "mean"),
            mean_edv_invested=("edv_invested", "mean"),
            mean_net_surplus=("net_surplus_generated", "mean"),
        )
        .reindex(["SCORCHED_EARTH_DRAFT", "HYBRID_REBUILD", "COMPOSITIONAL_RETOOL"])
        .reset_index()
        .round(3)
    )
    arch_sum.to_csv(TAB_DIR / "era_shift_rebuild_outcomes_2000_2025.csv", index=False)

    era_sum = (
        completed.groupby("era_code")
        .agg(
            n_episodes=("episode_id", "count"),
            mean_duration_yrs=("duration_years", "mean"),
            draft_core_share=("pathway_a_draft_share", "mean"),
            trade_ufa_core_share=("pathway_b_trade_share", "mean"),
            tier1_playoff_rate=("achieved_tier_1", "mean"),
            purgatory_failure_rate=("is_purgatory_trap", "mean"),
        )
        .reset_index()
        .round(3)
    )
    era_sum.to_csv(TAB_DIR / "era_evolution_summary.csv", index=False)

    modern_teams = ["FLA", "VGK", "CAR", "DAL", "COL", "MTL", "OTT", "UTA", "DET", "BUF", "CHI", "ANA", "SJS"]
    sub_mod = (
        df_ts[(df_ts["franchise_id"].isin(modern_teams)) & (df_ts["season"] >= 2021)]
        .groupby("franchise_id")
        .agg(
            mean_pts_pct=("pts_pct", "mean"),
            pts_pct_latest=("pts_pct", "last"),
            total_playoff_games=("playoff_games_won", "sum"),
            total_playoff_rounds=("playoff_rounds_won", "sum"),
            draft_core_share=("pathway_a_draft_core_share", "mean"),
            trade_core_share=("pathway_b_trade_core_share", "mean"),
            prime_bridge_b2_b3=("s_b2_prime_4_6", lambda s: s.mean()),
            elc_share_b1=("s_b1_elc_0_3", "mean"),
            surplus_value_avg=("surplus_value_total", "mean"),
            spsi_synergy=("spsi_style_synergy", "mean"),
        )
        .reindex(modern_teams)
        .reset_index()
        .round(3)
    )
    sub_mod["pts_pct_2024"] = sub_mod["pts_pct_latest"]
    sub_mod.to_csv(TAB_DIR / "modern_rebuilds_trade_vs_draft_matrix.csv", index=False)

    # Full-Width Vertical 2-Row Infographic (Extra-Generous Spacing, Zero Label Overlaps)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14.0, 12.6), dpi=240)
    fig.patch.set_facecolor("#F8FAFC")
    style_axis(ax1, "y")
    style_axis(ax2, "both")

    arch_labels = [
        f"1. Pure 'Scorched-Earth' Draft Tanking\n{arch_sum.loc[0, 'draft_core_share']*100:.0f}% Draft Core · Avg Wait: {arch_sum.loc[0, 'mean_duration_yrs']:.1f} Yrs ({int(arch_sum.loc[0, 'n_episodes'])} Cases)",
        f"2. Hybrid Draft + Trade Rebuild\n{arch_sum.loc[1, 'draft_core_share']*100:.0f}% Draft / {arch_sum.loc[1, 'trade_ufa_core_share']*100:.0f}% Trade · Avg Wait: {arch_sum.loc[1, 'mean_duration_yrs']:.1f} Yrs ({int(arch_sum.loc[1, 'n_episodes'])} Cases)",
        f"3. Compositional On-the-Fly Retool\nVeteran Core · Avg Wait: {arch_sum.loc[2, 'mean_duration_yrs']:.1f} Yrs ({int(arch_sum.loc[2, 'n_episodes'])} Cases)",
    ]
    x = np.arange(len(arch_labels))
    w = 0.24

    t1 = arch_sum["tier1_playoff_rate"] * 100
    t2 = arch_sum["tier2_final_four_rate"] * 100
    fail = arch_sum["purgatory_failure_rate"] * 100

    ax1.bar(x - w, t1, width=w, color="#2563EB", label="Reached Playoff Contention (Tier 1, %)", zorder=3)
    ax1.bar(x, t2, width=w, color="#16A34A", label="Reached Conference Finals (Tier 2, %)", zorder=3)
    ax1.bar(x + w, fail, width=w, color="#DC2626", label="Collapsed Into Rebuild Purgatory (Failed, %)", zorder=3)

    for i in range(len(x)):
        ax1.text(x[i] - w, t1[i] + 1.8, f"{t1[i]:.1f}%", ha="center", fontsize=9.8, fontweight="bold", color="#1D4ED8")
        ax1.text(x[i], t2[i] + 1.8, f"{t2[i]:.1f}%", ha="center", fontsize=9.8, fontweight="bold", color="#15803D")
        ax1.text(x[i] + w, fail[i] + 1.8, f"{fail[i]:.1f}%", ha="center", fontsize=9.8, fontweight="bold", color="#B91C1C")

    ax1.set_xticks(x)
    ax1.set_xticklabels(arch_labels, fontsize=9.2, fontweight="bold")
    ax1.set_ylim(0, 118)
    ax1.set_ylabel("Historical Outcome Rate (%)", fontsize=11.0, fontweight="bold", color="#0F172A")
    ax1.set_title(
        f"Panel A: 25-Year Rebuild Strategy Scorecard — Pure Draft Tanking Fails {fail.iloc[0]:.1f}% of the Time vs. {fail.iloc[1]:.1f}% for Hybrid",
        fontsize=12.8,
        fontweight="bold",
        color="#0F172A",
        pad=42,
    )
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=3,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        fontsize=9.2,
    )

    # Panel B: Modern Era (2021-2025) External Trade Core Share vs 2024/25 Points %
    ax2.axvspan(42, 72, color="#DCFCE7", alpha=0.45, label="Optimal Hybrid Trade-Sourcing Zone (42%–72% of Core Ice Time Acquired via Trade/UFA)", zorder=1)
    ax2.axhline(55.0, color="#16A34A", linestyle="--", lw=1.5, label="Playoff Contention Threshold (.550 Points %)", zorder=2)

    team_names_short = {
        "FLA": "Florida (FLA · 2 Cups)",
        "VGK": "Vegas (VGK · 1 Cup)",
        "CAR": "Carolina (CAR · Cup Final)",
        "DAL": "Dallas (DAL · Cup Final)",
        "COL": "Colorado (COL · 1 Cup)",
        "MTL": "Montréal (MTL · 9 PO Wins)",
        "OTT": "Ottawa (OTT · Playoff Core)",
        "UTA": "Utah (UTA · Emerging)",
        "DET": "Detroit (DET · Emerging)",
        "BUF": "Buffalo (BUF · Emerging)",
        "CHI": "Chicago (CHI · Purgatory)",
        "ANA": "Anaheim (ANA · Emerging)",
        "SJS": "San Jose (SJS · Purgatory)",
    }

    custom_offsets = {
        "DAL": (0, 10),
        "OTT": (0, 10),
        "COL": (0, 10),
        "CAR": (0, -18),
        "FLA": (18, -16),
        "VGK": (-28, -16),
        "MTL": (38, -14),
        "UTA": (-38, 10),
        "DET": (-48, 10),
        "ANA": (-42, -16),
        "BUF": (38, 10),
        "CHI": (0, -18),
        "SJS": (-36, -16),
    }

    for r in sub_mod.itertuples():
        if r.franchise_id == "MTL":
            c = "#C8102E"
        elif r.franchise_id == "CHI":
            c = "#1E293B"
        elif r.pts_pct_2024 >= 0.55:
            c = "#15803D"
        else:
            c = "#B91C1C"

        sz = 180 if r.franchise_id in ("MTL", "CHI", "FLA", "VGK", "BUF") else 120
        ax2.scatter(
            r.trade_core_share * 100,
            r.pts_pct_2024 * 100,
            s=sz,
            color=c,
            alpha=0.92,
            edgecolors="#FFFFFF",
            linewidth=1.8,
            zorder=4,
        )
        dx, dy = custom_offsets.get(r.franchise_id, (0, 9))
        lbl_str = f"{team_names_short.get(r.franchise_id, r.franchise_id)} (.{int(round(r.pts_pct_2024*1000)):03d})"
        ax2.annotate(
            lbl_str,
            (r.trade_core_share * 100, r.pts_pct_2024 * 100),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="center",
            fontsize=8.8,
            fontweight="bold",
            color=c,
            bbox=dict(boxstyle="round,pad=0.22", fc="#FFFFFF", ec="#E2E8F0", lw=0.7, alpha=0.90),
            zorder=5,
        )

    ax2.set_xlim(22, 104)
    ax2.set_ylim(36, 82)
    ax2.set_xlabel("Share of Core Ice Time Acquired via External Trade / Cap Arbitrage (2021–2026 Avg, %)", fontsize=11.0, fontweight="bold", color="#0F172A")
    ax2.set_ylabel("2025/26 Regular-Season Points % (x100)", fontsize=11.0, fontweight="bold", color="#0F172A")
    ax2.set_title(
        "Panel B: The Modern Hybrid Blueprint (2021–2025) — Blending Entry-Level Draft Hits With Prime Trade Acquisitions",
        fontsize=12.8,
        fontweight="bold",
        color="#0F172A",
        pad=42,
    )
    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=2,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        fontsize=9.2,
    )

    plt.suptitle(
        "Why Modern NHL Rebuilds Are Won Through Hybrid Trade & Cap Arbitrage — Not Endless Tanking",
        fontsize=14.5,
        fontweight="bold",
        color="#0F172A",
        y=0.988,
    )
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.955], h_pad=4.2)
    fig.savefig(FIG_DIR / "fig1_scorched_earth_vs_hybrid_trade_era_shift.png", dpi=240)
    plt.close(fig)
    print("Blog 5 vertical spacious chart saved.")


if __name__ == "__main__":
    main()
