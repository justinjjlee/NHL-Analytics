"""
Computation & Visualization Script for Blog Post 1:
dev/player_team_success/rebuild/deepdive/basics/
Spacious Full-Width Vertical Layout with Plain-English Labels (Zero Crowding)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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

STATE_PLAIN = {
    "R0_CONTENDER": "Contender (Peak)",
    "R1_SLUMP": "Slump (Inaction)",
    "R2_TEARDOWN": "Teardown (Rebuild)",
    "R3_INCUBATION": "Incubation (Rebuild)",
    "R4_EMERGENCE": "Emergence (Breakout)",
    "R5_PURGATORY": "Purgatory (Stalled 5+ Yrs)",
}

STATE_SHORT = {
    "R0_CONTENDER": "Contender",
    "R1_SLUMP": "Slump",
    "R2_TEARDOWN": "Teardown",
    "R3_INCUBATION": "Incubation",
    "R4_EMERGENCE": "Emergence",
    "R5_PURGATORY": "Purgatory",
}


def style_axis(ax: plt.Axes, grid_axis: str = "both") -> None:
    ax.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#94A3B8")
    ax.spines["bottom"].set_color("#94A3B8")
    ax.tick_params(colors="#1E293B", labelsize=10.5)
    if grid_axis in ("both", "y"):
        ax.grid(True, axis="y", color="#E2E8F0", linestyle="-", linewidth=0.9, zorder=0)
    if grid_axis in ("both", "x"):
        ax.grid(True, axis="x", color="#F1F5F9", linestyle="--", linewidth=0.8, zorder=0)


def main() -> None:
    df_ts = pd.read_parquet(REBUILD_DATA_DIR / "panel_team_season.parquet")

    sub_mc = (
        df_ts[(df_ts["franchise_id"].isin(["MTL", "CHI"])) & (df_ts["season"] >= 2018)]
        .sort_values(["franchise_id", "season"])
        .copy()
    )

    cols_export = [
        "franchise_id",
        "season",
        "season_id",
        "pts_pct",
        "pyth_win_pct",
        "xg_share_5v5",
        "playoff_games_won",
        "playoff_rounds_won",
        "macro_regime",
        "rebuild_regime",
        "ali_score",
        "yis_score",
        "rci_score",
        "s_b1_elc_0_3",
        "s_b2_prime_4_6",
        "s_b3_peak_7_9",
        "s_b4_vet_10_plus",
        "vet_insulation_ratio",
        "ozone_hit_pct",
        "spsi_style_synergy",
    ]
    sub_mc[cols_export].to_csv(
        TAB_DIR / "mtl_chi_regime_identification_2019_2025.csv", index=False
    )

    cohort_ids = ["MTL", "CHI", "BUF", "DET", "OTT", "ANA", "SJS", "CBJ", "UTA"]
    sub_cohort = df_ts[
        (df_ts["franchise_id"].isin(cohort_ids)) & (df_ts["season"] >= 2019)
    ].copy()

    cohort_summary = []
    for fid, grp in sub_cohort.groupby("franchise_id"):
        grp = grp.sort_values("season")
        pts_2019 = grp.loc[grp["season"] == 2019, "pts_pct"].values[0]
        pts_trough = grp["pts_pct"].min()
        trough_yr = int(grp.loc[grp["pts_pct"].idxmin(), "season"])
        latest_yr = int(grp["season"].max())
        pts_latest = grp.loc[grp["season"] == latest_yr, "pts_pct"].values[0]
        pts_2024 = grp.loc[grp["season"] == 2024, "pts_pct"].values[0] if (grp["season"] == 2024).any() else np.nan
        pts_2025 = grp.loc[grp["season"] == 2025, "pts_pct"].values[0] if (grp["season"] == 2025).any() else np.nan
        regime_latest = grp.loc[grp["season"] == latest_yr, "rebuild_regime"].values[0]
        macro_latest = grp.loc[grp["season"] == latest_yr, "macro_regime"].values[0]
        po_games = int(grp.loc[grp["season"] >= 2021, "playoff_games_won"].sum())
        po_series = int(grp.loc[grp["season"] >= 2021, "playoff_rounds_won"].sum())
        fname = grp["franchise_name"].iloc[0]
        cohort_summary.append(
            {
                "franchise_id": fid,
                "franchise_name": fname,
                "pts_pct_2019": round(pts_2019, 3),
                "trough_pts_pct": round(pts_trough, 3),
                "trough_season": trough_yr,
                "pts_pct_2024": round(pts_2024, 3) if not np.isnan(pts_2024) else round(pts_latest, 3),
                "pts_pct_2025": round(pts_2025, 3) if not np.isnan(pts_2025) else round(pts_latest, 3),
                "pts_pct_latest": round(pts_latest, 3),
                "rebound_from_trough": round(pts_latest - pts_trough, 3),
                "net_change_from_2019": round(pts_latest - pts_2019, 3),
                "net_change_2019_2024": round(pts_latest - pts_2019, 3),
                "playoff_games_won": po_games,
                "playoff_rounds_won": po_series,
                "mean_ali_2019_2024": round(grp["ali_score"].mean(), 3),
                "mean_yis_2019_2024": round(grp["yis_score"].mean(), 3),
                "mean_elc_share": round(grp["s_b1_elc_0_3"].mean(), 3),
                "mean_bridge_b2_share": round(grp["s_b2_prime_4_6"].mean(), 3),
                "inaction_seasons_since_2019": int((grp["macro_regime"] == "Inaction").sum()),
                "rebuild_seasons_since_2019": int((grp["macro_regime"] == "Rebuild").sum()),
                "peak_seasons_since_2019": int((grp["macro_regime"] == "Peak").sum()),
                "state_latest": regime_latest,
                "macro_latest": macro_latest,
                "state_2024": regime_latest,
                "macro_2024": macro_latest,
            }
        )

    df_cohort_sum = pd.DataFrame(cohort_summary).sort_values(
        "rebound_from_trough", ascending=False
    )
    df_cohort_sum.to_csv(TAB_DIR / "cohort_2020s_rebuilds_comparison.csv", index=False)

    # Full-Width Vertical 2-Row Infographic (Extra-Generous Spacing, Zero Label Overlaps)
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14.0, 12.6), dpi=240, gridspec_kw={"height_ratios": [1.05, 1.0]}
    )
    fig.patch.set_facecolor("#F8FAFC")

    style_axis(ax1, "both")
    style_axis(ax2, "x")

    mtl = sub_mc[sub_mc["franchise_id"] == "MTL"].sort_values("season")
    chi = sub_mc[sub_mc["franchise_id"] == "CHI"].sort_values("season")

    seasons = mtl["season"].values
    x_labels = [f"{s}/{str(s+1)[2:]}" for s in seasons]
    x_idx = np.arange(len(seasons))

    # Background Regime Bands in Panel A (Placed in clean whitespace away from data points)
    ax1.axhspan(0.550, 0.705, color="#DCFCE7", alpha=0.55, zorder=0)
    ax1.axhspan(0.435, 0.550, color="#FEF3C7", alpha=0.45, zorder=0)
    ax1.axhspan(0.255, 0.435, color="#FEE2E2", alpha=0.45, zorder=0)

    ax1.text(0.08, 0.678, "PEAK CONTENDER ZONE (Points % >= .550)", fontsize=8.8, fontweight="bold", color="#15803D", ha="left")
    ax1.text(7.45, 0.528, "TRANSITION / MEDIOCRITY BAND (.435 to .550)", fontsize=8.8, fontweight="bold", color="#B45309", ha="right")
    ax1.text(0.08, 0.270, "LOTTERY TEARDOWN BASEMENT (Points % < .435)", fontsize=8.8, fontweight="bold", color="#B91C1C", ha="left")

    ax1.axhline(0.550, color="#16A34A", linestyle="--", linewidth=1.2, alpha=0.75, zorder=1)
    ax1.axhline(0.435, color="#DC2626", linestyle="--", linewidth=1.2, alpha=0.55, zorder=1)

    ax1.plot(
        x_idx,
        mtl["pts_pct"],
        marker="o",
        linewidth=3.5,
        markersize=10,
        color="#C8102E",
        markeredgecolor="#FFFFFF",
        markeredgewidth=1.8,
        label=f"Montréal Canadiens (MTL) — Rebounded +.311 from 2021/22 Bottom to {mtl['pts_pct'].iloc[-1]:.3f} Contender ({int(mtl['playoff_games_won'].iloc[-1])} Playoff Wins in 2025/26)",
        zorder=4,
    )
    ax1.plot(
        x_idx,
        chi["pts_pct"],
        marker="s",
        linewidth=3.2,
        markersize=9.5,
        linestyle="--",
        color="#1E293B",
        markeredgecolor="#FFFFFF",
        markeredgewidth=1.8,
        label=f"Chicago Blackhawks (CHI) — 5 Straight Non-Playoff Rebuild Seasons ({chi['pts_pct'].iloc[-1]:.3f} in 2025/26)",
        zorder=3,
    )

    mtl_offsets = [
        (0, 14),     # 2018/19 (.585)
        (-36, -28),  # 2019/20 (.500)
        (0, 14),     # 2020/21 (.527)
        (0, -28),    # 2021/22 (.335)
        (16, 16),    # 2022/23 (.415) - shifted right/up to avoid CHI 2021/22
        (-12, 16),   # 2023/24 (.463)
        (0, 14),     # 2024/25 (.555)
        (-18, 13),   # 2025/26 (.646)
    ]
    chi_offsets = [
        (0, -28),    # 2018/19 (.512)
        (24, 14),    # 2019/20 (.514)
        (-18, -28),  # 2020/21 (.491)
        (-28, 16),   # 2021/22 (.415) - shifted left/up to avoid MTL 2022/23
        (0, -28),    # 2022/23 (.360)
        (0, -28),    # 2023/24 (.317)
        (0, -28),    # 2024/25 (.372)
        (-12, -26),  # 2025/26 (.476)
    ]

    for i, r in enumerate(mtl.itertuples()):
        lbl = STATE_SHORT.get(r.rebuild_regime, r.rebuild_regime)
        dx, dy = mtl_offsets[i]
        ax1.annotate(
            f"MTL: {lbl} ({r.pts_pct:.3f})",
            (x_idx[i], r.pts_pct),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="center",
            fontsize=8.6,
            fontweight="bold",
            color="#991B1B",
            bbox=dict(boxstyle="round,pad=0.23", fc="#FFF1F2", ec="#FECDD3", lw=0.9, alpha=0.95),
            zorder=5,
        )

    for i, r in enumerate(chi.itertuples()):
        lbl = STATE_SHORT.get(r.rebuild_regime, r.rebuild_regime)
        dx, dy = chi_offsets[i]
        ax1.annotate(
            f"CHI: {lbl} ({r.pts_pct:.3f})",
            (x_idx[i], r.pts_pct),
            textcoords="offset points",
            xytext=(dx, dy),
            ha="center",
            fontsize=8.6,
            fontweight="bold",
            color="#0F172A",
            bbox=dict(boxstyle="round,pad=0.23", fc="#F1F5F9", ec="#CBD5E1", lw=0.9, alpha=0.95),
            zorder=5,
        )

    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(x_labels, fontsize=10.5, fontweight="bold")
    ax1.set_xlim(-0.55, 7.60)
    ax1.set_ylim(0.255, 0.705)
    ax1.set_ylabel("Regular-Season Points Percentage", fontsize=11.0, fontweight="bold", color="#0F172A")
    ax1.set_title(
        "Panel A: Standings Trajectory & Classified Regime States (2018/19 to 2025/26)",
        fontsize=12.8,
        fontweight="bold",
        color="#0F172A",
        pad=48,
    )
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=1,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        fontsize=9.5,
    )

    # Panel B: 2020s Rebuilding Cohort Rebound From Trough
    df_plot = df_cohort_sum.sort_values("rebound_from_trough", ascending=True)
    y_pos = np.arange(len(df_plot))

    bar_colors = []
    for r in df_plot.itertuples():
        if r.franchise_id == "MTL":
            bar_colors.append("#C8102E")
        elif r.franchise_id == "CHI":
            bar_colors.append("#1E293B")
        elif r.state_2024 in ("R0_CONTENDER", "R4_EMERGENCE"):
            bar_colors.append("#16A34A")
        else:
            bar_colors.append("#64748B")

    bars = ax2.barh(y_pos, df_plot["rebound_from_trough"] * 100, color=bar_colors, height=0.62, alpha=0.92, zorder=3)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(
        [f"{r.franchise_id}  —  {STATE_PLAIN.get(r.state_2024, r.state_2024)}" for r in df_plot.itertuples()],
        fontsize=10.0,
        fontweight="bold",
        color="#0F172A",
    )

    for bar, r in zip(bars, df_plot.itertuples()):
        w = bar.get_width()
        is_focus = r.franchise_id in ("MTL", "CHI", "BUF", "OTT", "UTA")
        ax2.text(
            w + 0.45,
            bar.get_y() + bar.get_height() / 2,
            f"+{w:.1f} pct pts  (Bottom: {r.trough_pts_pct:.3f} in {r.trough_season}/{str(r.trough_season+1)[2:]} → 2025/26: {r.pts_pct_2024:.3f} | {r.playoff_games_won} Playoff W)",
            va="center",
            fontsize=8.8,
            fontweight="bold" if is_focus else "normal",
            color="#991B1B" if r.franchise_id == "MTL" else ("#0F172A" if is_focus else "#334155"),
        )

    ax2.set_xlim(0, 62.0)
    ax2.set_xlabel("Points % Recovery From Rebuild Bottom to 2025/26 (Percentage Points)", fontsize=11.0, fontweight="bold", color="#0F172A")
    ax2.set_title(
        "Panel B: Comparing All 9 Rebuilding Franchises of the 2020s (Recovery & Playoff Game Wins Through 2025/26)",
        fontsize=12.8,
        fontweight="bold",
        color="#0F172A",
        pad=46,
    )

    p_mtl = mpatches.Patch(color="#C8102E", label="Montréal (R0 Contender · 9 Playoff Wins in 2025/26)")
    p_ott = mpatches.Patch(color="#16A34A", label="Graduated to Playoff Peak/Emergence (BUF, OTT, ANA, DET, UTA)")
    p_purg = mpatches.Patch(color="#64748B", label="Inaction / Purgatory (CBJ, SJS)")
    p_chi = mpatches.Patch(color="#1E293B", label="Chicago (R5 Purgatory · 0 Playoff Wins Since 2020)")
    ax2.legend(
        handles=[p_mtl, p_ott, p_purg, p_chi],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=2,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        fontsize=9.2,
    )

    plt.suptitle(
        "Quantifying Generational Struggle vs. Rebuild Escape Velocity (NHL 2018/19–2025/26)",
        fontsize=14.5,
        fontweight="bold",
        color="#0F172A",
        y=0.988,
    )
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.955], h_pad=4.2)
    fig.savefig(FIG_DIR / "fig1_mtl_vs_chi_regime_trajectories.png", dpi=240)
    plt.close(fig)
    print("Blog 1 vertical spacious chart saved.")


if __name__ == "__main__":
    main()
