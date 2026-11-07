"""
Computation & Visualization Script for Blog Post 2:
dev/player_team_success/rebuild/deepdive/explain/
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


def style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94A3B8")
    ax.spines["bottom"].set_color("#94A3B8")
    ax.tick_params(colors="#1E293B", labelsize=10.0)
    ax.grid(True, axis="y", color="#E2E8F0", linestyle="-", linewidth=0.9, zorder=0)


def main() -> None:
    df_ts = pd.read_parquet(REBUILD_DATA_DIR / "panel_team_season.parquet")

    metrics = [
        "pts_pct",
        "playoff_games_won",
        "playoff_rounds_won",
        "ali_score",
        "yis_score",
        "rci_score",
        "ccm_to_prior_year",
        "s_b1_elc_0_3",
        "s_b2_prime_4_6",
        "s_b3_peak_7_9",
        "s_b4_vet_10_plus",
        "shannon_exp_entropy",
        "vet_insulation_ratio",
        "pathway_a_draft_core_share",
        "pathway_b_trade_core_share",
        "xg_share_5v5",
        "ozone_hit_pct",
        "proactive_forecheck_share",
        "spsi_style_synergy",
    ]

    macro_grp = (
        df_ts.groupby("macro_regime")[metrics]
        .mean()
        .reindex(["Peak", "Rebuild", "Inaction"])
        .round(4)
    )
    macro_grp["n_team_seasons"] = (
        df_ts.groupby("macro_regime").size().reindex(["Peak", "Rebuild", "Inaction"])
    )
    macro_grp.reset_index().to_csv(
        TAB_DIR / "model_parameter_estimates_and_clusters.csv", index=False
    )

    gran_grp = (
        df_ts.groupby("rebuild_regime")[metrics]
        .mean()
        .reindex(
            [
                "R0_CONTENDER",
                "R1_SLUMP",
                "R2_TEARDOWN",
                "R3_INCUBATION",
                "R4_EMERGENCE",
                "R5_PURGATORY",
            ]
        )
        .round(4)
    )
    gran_grp["n_team_seasons"] = (
        df_ts.groupby("rebuild_regime")
        .size()
        .reindex(
            [
                "R0_CONTENDER",
                "R1_SLUMP",
                "R2_TEARDOWN",
                "R3_INCUBATION",
                "R4_EMERGENCE",
                "R5_PURGATORY",
            ]
        )
    )
    gran_grp.reset_index().to_csv(TAB_DIR / "granular_substate_centroids.csv", index=False)

    sub_24 = df_ts[df_ts["season"] == 2024].copy()
    mtl_24 = sub_24[sub_24["franchise_id"] == "MTL"][metrics].iloc[0]
    chi_24 = sub_24[sub_24["franchise_id"] == "CHI"][metrics].iloc[0]
    peak_bench = df_ts[df_ts["macro_regime"] == "Peak"][metrics].mean()
    inaction_bench = df_ts[df_ts["macro_regime"] == "Inaction"][metrics].mean()

    comp_df = pd.DataFrame(
        {
            "metric": metrics,
            "MTL_2024": mtl_24.values.round(4),
            "CHI_2024": chi_24.values.round(4),
            "League_Peak_Avg": peak_bench.values.round(4),
            "League_Inaction_Avg": inaction_bench.values.round(4),
        }
    )
    comp_df.to_csv(TAB_DIR / "mtl_chi_component_decomposition.csv", index=False)

    # Full-Width Vertical 2-Row Infographic (Extra-Generous Spacing, Zero Label Overlaps)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14.0, 12.6), dpi=240)
    fig.patch.set_facecolor("#F8FAFC")
    style_axis(ax1)
    style_axis(ax2)

    states = [
        "R0_CONTENDER",
        "R4_EMERGENCE",
        "R3_INCUBATION",
        "R2_TEARDOWN",
        "R1_SLUMP",
        "R5_PURGATORY",
    ]
    state_labels = [
        "R0: Contender\n(Peak Contention)",
        "R4: Emergence\n(Playoff Breakout)",
        "R3: Incubation\n(Developing Core)",
        "R2: Teardown\n(Veteran Sell-Off)",
        "R1: Slump\n(Mild Inaction)",
        "R5: Purgatory\n(Stalled 5+ Years)",
    ]
    g_sub = gran_grp.loc[states]
    x = np.arange(len(states))
    w = 0.18

    b1 = ax1.bar(
        x - 1.5 * w,
        g_sub["pts_pct"] * 100,
        width=w,
        color="#15803D",
        label="Standings Points % (0–100)",
        zorder=3,
    )
    b2 = ax1.bar(
        x - 0.5 * w,
        g_sub["ali_score"] * 100,
        width=w,
        color="#EA580C",
        label="Veteran Sell-Off / Asset Liquidation (ALI)",
        zorder=3,
    )
    b3 = ax1.bar(
        x + 0.5 * w,
        g_sub["s_b1_elc_0_3"] * 100,
        width=w,
        color="#4F46E5",
        label="Young Skater Ice-Time Share (0–3 Yrs NHL, %)",
        zorder=3,
    )
    b4 = ax1.bar(
        x + 1.5 * w,
        g_sub["spsi_style_synergy"] * 100,
        width=w,
        color="#DB2777",
        label="Skill × Forecheck Hit Synergy (SPSI)",
        zorder=3,
    )

    for bars, col in [(b1, "#15803D"), (b2, "#EA580C"), (b3, "#4F46E5"), (b4, "#DB2777")]:
        for bar in bars:
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                h + 1.2,
                f"{h:.0f}",
                ha="center",
                fontsize=8.2,
                fontweight="bold",
                color=col,
            )

    ax1.set_xticks(x)
    ax1.set_xticklabels(state_labels, fontsize=10.0, fontweight="bold")
    ax1.set_ylim(0, 84)
    ax1.set_ylabel("Average Score / Percentage (0–100 Scale)", fontsize=11.0, fontweight="bold", color="#0F172A")
    ax1.set_title(
        f"Panel A: How the Model Defines the 6 Franchise Sub-States (Averages Across {len(df_ts)} NHL Team-Seasons)",
        fontsize=12.8,
        fontweight="bold",
        color="#0F172A",
        pad=46,
    )
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=2,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        fontsize=9.2,
    )

    # Panel B: 2024/25 Component Comparison (Plain-English Multi-Line Category Labels)
    cat_labels = [
        "Standings\nPoints %\n(x100)",
        "Prime Bridge Ice Time\n(4–6 Yrs Experience,\n% of Team TOI)",
        "5-on-5 Expected\nGoals Share\n(xGF%)",
        "Offensive-Zone\nForecheck Hit Share\n(% of All Hits)",
        "Skill × Physicality\nSynergy Score\n(SPSI x100)",
        "Year-Over-Year\nCore Continuity\n(CCM x100)",
    ]
    mtl_vals = [
        mtl_24["pts_pct"] * 100,
        mtl_24["s_b2_prime_4_6"] * 100,
        mtl_24["xg_share_5v5"] * 100,
        mtl_24["ozone_hit_pct"],
        mtl_24["spsi_style_synergy"] * 100,
        mtl_24["ccm_to_prior_year"] * 100,
    ]
    chi_vals = [
        chi_24["pts_pct"] * 100,
        chi_24["s_b2_prime_4_6"] * 100,
        chi_24["xg_share_5v5"] * 100,
        chi_24["ozone_hit_pct"],
        chi_24["spsi_style_synergy"] * 100,
        chi_24["ccm_to_prior_year"] * 100,
    ]
    pk_vals = [
        peak_bench["pts_pct"] * 100,
        peak_bench["s_b2_prime_4_6"] * 100,
        peak_bench["xg_share_5v5"] * 100,
        peak_bench["ozone_hit_pct"],
        peak_bench["spsi_style_synergy"] * 100,
        peak_bench["ccm_to_prior_year"] * 100,
    ]

    x2 = np.arange(len(cat_labels))
    w2 = 0.27
    ax2.bar(x2 - w2, mtl_vals, width=w2, color="#C8102E", label="Montréal 2024/25 (R4 Emergence)", zorder=3)
    ax2.bar(x2, chi_vals, width=w2, color="#1E293B", label="Chicago 2024/25 (R2 Teardown)", zorder=3)
    ax2.bar(x2 + w2, pk_vals, width=w2, color="#15803D", alpha=0.85, label="Historical Peak Contender Average", zorder=3)

    for i in range(len(x2)):
        # If first two bars are identical (e.g. 100.0 and 100.0), stagger vertical height slightly so numbers never touch
        y_off_m = 4.6 if abs(mtl_vals[i] - chi_vals[i]) < 2.0 else 1.8
        ax2.text(x2[i] - w2, mtl_vals[i] + y_off_m, f"{mtl_vals[i]:.1f}", ha="center", fontsize=8.8, fontweight="bold", color="#C8102E")
        ax2.text(x2[i], chi_vals[i] + 1.8, f"{chi_vals[i]:.1f}", ha="center", fontsize=8.8, fontweight="bold", color="#1E293B")
        ax2.text(x2[i] + w2, pk_vals[i] + 1.8, f"{pk_vals[i]:.1f}", ha="center", fontsize=8.8, fontweight="bold", color="#15803D")

    ax2.set_xticks(x2)
    ax2.set_xticklabels(cat_labels, fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, 118)
    ax2.set_ylabel("Measured Parameter Value (0–100 Scale)", fontsize=11.0, fontweight="bold", color="#0F172A")
    ax2.set_title(
        "Panel B: 2024/25 Telemetry Comparison — Why Montréal Emerged While Chicago Stayed in Teardown",
        fontsize=12.8,
        fontweight="bold",
        color="#0F172A",
        pad=42,
    )
    ax2.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=3,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        fontsize=9.2,
    )

    plt.suptitle(
        "Inside the Rebuild Model: Player Experience Clusters, Physicality & Regime Telemetry",
        fontsize=14.5,
        fontweight="bold",
        color="#0F172A",
        y=0.988,
    )
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.955], h_pad=4.2)
    fig.savefig(FIG_DIR / "fig1_four_pillar_architecture_mtl_chi.png", dpi=240)
    plt.close(fig)
    print("Blog 2 vertical spacious chart saved.")


if __name__ == "__main__":
    main()
