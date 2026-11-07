"""
Computation & Visualization Script for Blog Post 3:
dev/player_team_success/rebuild/deepdive/timeline/
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

    completed = df_ep[df_ep["outcome_tier_label"] != "In-Progress / Incomplete Cycle"].copy()
    n_comp = len(completed)

    years = list(range(1, 11))
    rows = []
    for y in years:
        cif_t1 = ((completed["achieved_tier_1"]) & (completed["duration_years"] <= y)).sum() / n_comp
        cif_t2 = ((completed["achieved_tier_2"]) & (completed["duration_years"] <= y)).sum() / n_comp
        cif_t3 = ((completed["achieved_tier_3"]) & (completed["duration_years"] <= y)).sum() / n_comp
        cif_purg = ((completed["is_purgatory_trap"]) & (completed["duration_years"] <= y)).sum() / n_comp
        still_unresolved = max(0.0, 1.0 - cif_t1 - cif_purg)
        rows.append(
            {
                "rebuild_year_clock": y,
                "cum_prob_tier1_playoff_core": round(cif_t1, 4),
                "cum_prob_tier2_final_four": round(cif_t2, 4),
                "cum_prob_tier3_stanley_cup": round(cif_t3, 4),
                "cum_prob_purgatory_failure": round(cif_purg, 4),
                "prob_still_unresolved": round(still_unresolved, 4),
            }
        )
    df_clock = pd.DataFrame(rows)
    df_clock.to_csv(TAB_DIR / "actuarial_rebuild_clock_by_year.csv", index=False)

    completed["continuity_bucket"] = np.where(
        completed["mean_ccm"] >= completed["mean_ccm"].median(),
        "High Core Stability (CCM >= 0.87)",
        "High Turnover / Regime Reset (CCM < 0.87)",
    )
    ccm_summary = (
        completed.groupby(["primary_archetype", "continuity_bucket"])
        .agg(
            n_episodes=("episode_id", "count"),
            median_duration_yrs=("duration_years", "median"),
            mean_duration_yrs=("duration_years", "mean"),
            tier1_success_rate=("achieved_tier_1", "mean"),
            tier2_final_four_rate=("achieved_tier_2", "mean"),
            tier3_cup_rate=("achieved_tier_3", "mean"),
            purgatory_failure_rate=("is_purgatory_trap", "mean"),
            mean_ccm=("mean_ccm", "mean"),
            mean_vir=("mean_vet_insulation_ratio", "mean"),
        )
        .reset_index()
        .round(3)
    )
    ccm_summary.to_csv(TAB_DIR / "episode_benchmark_comparison.csv", index=False)

    # Full-Width Vertical 2-Row Infographic (Extra-Generous Spacing, Zero Label Overlaps)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14.0, 12.6), dpi=240)
    fig.patch.set_facecolor("#F8FAFC")
    style_axis(ax1, "both")
    style_axis(ax2, "y")

    t1_ceil = df_clock["cum_prob_tier1_playoff_core"].iloc[-1] * 100
    t2_ceil = df_clock["cum_prob_tier2_final_four"].iloc[-1] * 100
    t3_ceil = df_clock["cum_prob_tier3_stanley_cup"].iloc[-1] * 100
    purg_ceil = df_clock["cum_prob_purgatory_failure"].iloc[-1] * 100
    t1_yr4_cum = df_clock.loc[df_clock["rebuild_year_clock"] == 4, "cum_prob_tier1_playoff_core"].values[0] * 100
    purg_yr8_cum = df_clock.loc[df_clock["rebuild_year_clock"] == 8, "cum_prob_purgatory_failure"].values[0] * 100
    t1_episodes = completed[completed["achieved_tier_1"]]
    t1_breakout_yr4_pct = (t1_episodes["duration_years"] <= 4).mean() * 100

    y_vals = df_clock["rebuild_year_clock"].values
    ax1.step(
        y_vals,
        df_clock["cum_prob_tier1_playoff_core"] * 100,
        where="post",
        linewidth=3.4,
        color="#2563EB",
        label=f"Tier 1: Sustained Playoff Core ({t1_ceil:.1f}% Ceiling)",
        zorder=4,
    )
    ax1.step(
        y_vals,
        df_clock["cum_prob_tier2_final_four"] * 100,
        where="post",
        linewidth=3.0,
        color="#16A34A",
        label=f"Tier 2: Conference Finalist / Final Four ({t2_ceil:.1f}%)",
        zorder=4,
    )
    ax1.step(
        y_vals,
        df_clock["cum_prob_tier3_stanley_cup"] * 100,
        where="post",
        linewidth=2.8,
        color="#EA580C",
        label=f"Tier 3: Stanley Cup Champion ({t3_ceil:.1f}%)",
        zorder=4,
    )
    ax1.step(
        y_vals,
        df_clock["cum_prob_purgatory_failure"] * 100,
        where="post",
        linewidth=3.2,
        linestyle="--",
        color="#DC2626",
        label=f"Failed: Collapsed Into Rebuild Purgatory ({purg_ceil:.1f}%)",
        zorder=4,
    )

    ax1.axvspan(3.5, 5.5, color="#FEF08A", alpha=0.35, label="Critical Year 4–5 Inflection Window", zorder=1)

    ax1.annotate(
        f"Montréal (2021–2025 Rebuild): Year 4 Breakout\n{t1_breakout_yr4_pct:.1f}% of all successful rebuilds break through by Year 4\n(MTL reached .555 Points % & R4 Emergence in Year 4)",
        xy=(4.0, t1_yr4_cum),
        xytext=(1.25, 72.0),
        arrowprops=dict(arrowstyle="->", color="#C8102E", lw=2.2, connectionstyle="arc3,rad=-0.12"),
        fontsize=9.5,
        fontweight="bold",
        color="#991B1B",
        bbox=dict(boxstyle="round,pad=0.4", fc="#FFF1F2", ec="#C8102E", lw=1.2),
        zorder=6,
    )
    ax1.annotate(
        "Chicago (Post-2017/2020 Rebuild): Year 5–8 Danger Zone\nOnce a rebuild drags past Year 5, Purgatory failure odds surge\n(CHI remained at .372 Points % in R2 Teardown)",
        xy=(8.0, purg_yr8_cum),
        xytext=(5.25, 34.0),
        arrowprops=dict(arrowstyle="->", color="#0F172A", lw=2.2, connectionstyle="arc3,rad=0.12"),
        fontsize=9.5,
        fontweight="bold",
        color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.4", fc="#F1F5F9", ec="#334155", lw=1.2),
        zorder=6,
    )

    ax1.set_xticks(y_vals)
    ax1.set_xticklabels([f"Year {y}" for y in y_vals], fontsize=10.2, fontweight="bold")
    ax1.set_ylim(0, 94)
    ax1.set_xlabel("Years Elapsed Since Rebuild Started (Year 1 to Year 10)", fontsize=11.0, fontweight="bold", color="#0F172A")
    ax1.set_ylabel("Cumulative Probability of Outcome (%)", fontsize=11.0, fontweight="bold", color="#0F172A")
    ax1.set_title(
        f"Panel A: The Fan Waiting Clock — Cumulative Probability of Playoff Success vs. Purgatory Failure ({n_comp} Completed Rebuilds)",
        fontsize=12.8,
        fontweight="bold",
        color="#0F172A",
        pad=46,
    )
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        fontsize=9.0,
    )

    # Panel B: Success vs Purgatory Rate by Rebuild Duration Window
    dur_bins = [
        ("1–2 Years\n(Quick On-the-Fly Retool)", completed[completed["duration_years"] <= 2]),
        ("3–4 Years\n(Optimal Rebuild Sweet Spot)", completed[(completed["duration_years"] >= 3) & (completed["duration_years"] <= 4)]),
        ("5–6 Years\n(Danger / Cap-Crunch Zone)", completed[(completed["duration_years"] >= 5) & (completed["duration_years"] <= 6)]),
        ("7+ Years\n(Generational Purgatory Rut)", completed[completed["duration_years"] >= 7]),
    ]
    b_labels = [b[0] for b in dur_bins]
    t1_rates = [b[1]["achieved_tier_1"].mean() * 100 for b in dur_bins]
    t2_rates = [b[1]["achieved_tier_2"].mean() * 100 for b in dur_bins]
    fail_rates = [b[1]["is_purgatory_trap"].mean() * 100 for b in dur_bins]

    xb = np.arange(len(b_labels))
    wb = 0.24
    ax2.bar(xb - wb, t1_rates, width=wb, color="#2563EB", label="Reached Playoff Contention (Tier 1, %)", zorder=3)
    ax2.bar(xb, t2_rates, width=wb, color="#16A34A", label="Reached Conference Finals (Tier 2, %)", zorder=3)
    ax2.bar(xb + wb, fail_rates, width=wb, color="#DC2626", label="Collapsed Into Rebuild Purgatory (Failed, %)", zorder=3)

    for i in range(len(xb)):
        ax2.text(xb[i] - wb, t1_rates[i] + 1.8, f"{t1_rates[i]:.0f}%", ha="center", fontsize=9.5, fontweight="bold", color="#1D4ED8")
        ax2.text(xb[i], t2_rates[i] + 1.8, f"{t2_rates[i]:.0f}%", ha="center", fontsize=9.5, fontweight="bold", color="#15803D")
        ax2.text(xb[i] + wb, fail_rates[i] + 1.8, f"{fail_rates[i]:.0f}%", ha="center", fontsize=9.5, fontweight="bold", color="#B91C1C")

    ax2.set_xticks(xb)
    ax2.set_xticklabels(b_labels, fontsize=10.2, fontweight="bold")
    ax2.set_ylim(0, 116)
    ax2.set_ylabel("Outcome Rate Within Duration Group (%)", fontsize=11.0, fontweight="bold", color="#0F172A")
    ax2.set_title(
        "Panel B: Why Year 5 Is the Cliff — Success vs. Failure Rates by Total Rebuild Length",
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
        "Actuarial Survival Analysis of NHL Rebuilds: How Long Should Fans Wait?",
        fontsize=14.5,
        fontweight="bold",
        color="#0F172A",
        y=0.988,
    )
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.955], h_pad=4.2)
    fig.savefig(FIG_DIR / "fig1_cumulative_incidence_fan_wait_clock.png", dpi=240)
    plt.close(fig)
    print("Blog 3 vertical spacious chart saved.")


if __name__ == "__main__":
    main()
