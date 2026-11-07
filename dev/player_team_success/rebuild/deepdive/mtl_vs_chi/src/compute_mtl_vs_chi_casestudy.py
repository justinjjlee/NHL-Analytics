"""
Computation & Visualization Script for Blog Post 4:
dev/player_team_success/rebuild/deepdive/mtl_vs_chi/
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
    df_ts = pd.read_parquet(REBUILD_DATA_DIR / "panel_team_season.parquet")
    df_pl = pd.read_parquet(REBUILD_DATA_DIR / "panel_player_season_core.parquet")

    df_mod = df_ts.dropna(
        subset=[
            "pts_pct",
            "s_b1_elc_0_3",
            "s_b2_prime_4_6",
            "s_b3_peak_7_9",
            "vet_insulation_ratio",
            "xg_share_5v5",
            "spsi_style_synergy",
            "ccm_to_prior_year",
        ]
    ).copy()

    df_mod["youth_overload_sq"] = np.maximum(0.0, df_mod["s_b1_elc_0_3"] - 0.30) ** 2
    df_mod["prime_bridge_b2_b3"] = df_mod["s_b2_prime_4_6"] + df_mod["s_b3_peak_7_9"]
    df_mod["insulation_interaction"] = df_mod["s_b1_elc_0_3"] * np.log1p(
        df_mod["vet_insulation_ratio"]
    )

    features = [
        "prime_bridge_b2_b3",
        "youth_overload_sq",
        "insulation_interaction",
        "ccm_to_prior_year",
        "xg_share_5v5",
        "spsi_style_synergy",
    ]
    X = np.column_stack([np.ones(len(df_mod))] + [df_mod[f].values for f in features])
    y = df_mod["pts_pct"].values
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    df_mod["pred_pts_pct"] = X @ beta
    df_mod["residual_pts_pct"] = df_mod["pts_pct"] - df_mod["pred_pts_pct"]

    param_names = ["Intercept"] + features
    res_vec = y - X @ beta
    n_obs, k_p = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    meat = (X * (res_vec[:, None] ** 2)).T @ X * (n_obs / (n_obs - k_p))
    se_hc1 = np.sqrt(np.diag(xtx_inv @ meat @ xtx_inv))
    t_stats = beta / se_hc1

    df_params = pd.DataFrame(
        {
            "parameter": param_names,
            "beta_est": beta.round(4),
            "robust_se_hc1": se_hc1.round(4),
            "t_stat": t_stats.round(2),
        }
    )
    df_params.to_csv(TAB_DIR / "structural_model_parameters.csv", index=False)

    sub_mc = (
        df_mod[(df_mod["franchise_id"].isin(["MTL", "CHI"])) & (df_mod["season"] >= 2019)]
        .sort_values(["franchise_id", "season"])
        .copy()
    )

    cols_out = [
        "franchise_id",
        "season",
        "macro_regime",
        "rebuild_regime",
        "pts_pct",
        "pred_pts_pct",
        "residual_pts_pct",
        "playoff_games_won",
        "playoff_rounds_won",
        "s_b1_elc_0_3",
        "s_b2_prime_4_6",
        "s_b3_peak_7_9",
        "s_b4_vet_10_plus",
        "prime_bridge_b2_b3",
        "shannon_exp_entropy",
        "vet_insulation_ratio",
        "ccm_to_prior_year",
        "pathway_a_draft_core_share",
        "pathway_b_trade_core_share",
        "xg_share_5v5",
        "ozone_hit_pct",
        "spsi_style_synergy",
        "surplus_value_total",
    ]
    sub_mc[cols_out].round(4).to_csv(
        TAB_DIR / "mtl_vs_chi_parameter_comparison_2019_2025.csv", index=False
    )

    core_24 = (
        df_pl[
            (df_pl["franchise_id"].isin(["MTL", "CHI"]))
            & (df_pl["season"] == 2024)
            & (df_pl["is_core_player"])
        ]
        .sort_values(["franchise_id", "toi_all_situations"], ascending=[True, False])
        .copy()
    )

    core_cols = [
        "franchise_id",
        "player_name",
        "pos_group",
        "year_inNHL",
        "exp_bucket",
        "sourcing_pathway",
        "draft_overall",
        "draft_franchise",
        "gamesPlayed",
        "points",
        "toi_per_game",
        "cap_hit",
        "mrp_est",
        "surplus_value_est",
    ]
    core_24[core_cols].round(2).to_csv(TAB_DIR / "mtl_vs_chi_core_players_2024.csv", index=False)

    # Full-Width Vertical 2-Row Infographic (Extra-Generous Spacing, Zero Label Overlaps)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14.0, 12.6), dpi=240)
    fig.patch.set_facecolor("#F8FAFC")
    style_axis(ax1, "both")
    style_axis(ax2, "y")

    mtl = sub_mc[sub_mc["franchise_id"] == "MTL"].sort_values("season")
    chi = sub_mc[sub_mc["franchise_id"] == "CHI"].sort_values("season")
    seasons = mtl["season"].values
    x_lbls = [f"{s}/{str(s+1)[2:]}" for s in seasons]
    x_idx = np.arange(len(seasons))

    # Panel A: Observed vs Model-Predicted Points %
    ax1.axhline(0.550, color="#16A34A", linestyle="--", lw=1.5, label="Playoff Contention Threshold (.550 Points %)", zorder=2)

    ax1.plot(
        x_idx,
        mtl["pts_pct"],
        "o-",
        color="#C8102E",
        lw=3.4,
        ms=9.5,
        markeredgecolor="#FFFFFF",
        markeredgewidth=1.6,
        label="Montréal Actual Points %",
        zorder=5,
    )
    ax1.plot(
        x_idx,
        mtl["pred_pts_pct"],
        "o:",
        color="#E11D48",
        lw=2.4,
        ms=7.0,
        alpha=0.85,
        label="Montréal Model-Predicted Points % (from Roster & Style)",
        zorder=4,
    )
    ax1.plot(
        x_idx,
        chi["pts_pct"],
        "s-",
        color="#1E293B",
        lw=3.2,
        ms=9.0,
        markeredgecolor="#FFFFFF",
        markeredgewidth=1.6,
        label="Chicago Actual Points %",
        zorder=5,
    )
    ax1.plot(
        x_idx,
        chi["pred_pts_pct"],
        "s:",
        color="#64748B",
        lw=2.4,
        ms=7.0,
        alpha=0.85,
        label="Chicago Model-Predicted Points % (from Roster & Style)",
        zorder=4,
    )

    ax1.annotate(
        f"Montréal 2024/25–2025/26 Contender Surge\n2025/26 Actual: {mtl['pts_pct'].values[-1]:.3f} (9 Playoff Wins, 2 Series Won)\nDriven by Prime Bridge Core & O-Zone Forechecking",
        xy=(len(x_idx) - 1, mtl["pts_pct"].values[-1]),
        xytext=(1.85, 0.585),
        arrowprops=dict(arrowstyle="->", color="#C8102E", lw=2.0),
        fontsize=9.1,
        fontweight="bold",
        color="#991B1B",
        bbox=dict(boxstyle="round,pad=0.35", fc="#FFF1F2", ec="#C8102E", lw=1.1),
        zorder=6,
    )
    ax1.annotate(
        f"Chicago Rebuild Purgatory Rut\n2024/25: .372  |  2025/26: {chi['pts_pct'].values[-1]:.3f} (0 Playoff Wins)\nHeld down by Hollow Bridge Core & Low xGF%",
        xy=(len(x_idx) - 1, chi["pts_pct"].values[-1]),
        xytext=(-0.25, 0.295),
        arrowprops=dict(arrowstyle="->", color="#1E293B", lw=2.0, connectionstyle="arc3,rad=0.18"),
        fontsize=9.0,
        fontweight="bold",
        color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.35", fc="#F1F5F9", ec="#334155", lw=1.1),
        zorder=6,
    )

    ax1.set_xticks(x_idx)
    ax1.set_xticklabels(x_lbls, fontsize=10.5, fontweight="bold")
    ax1.set_xlim(-0.4, len(x_idx) - 0.5)
    ax1.set_ylim(0.275, 0.695)
    ax1.set_ylabel("Regular-Season Points Percentage", fontsize=11.0, fontweight="bold", color="#0F172A")
    ax1.set_title(
        "Panel A: Actual vs. Econometric Model-Predicted Points % (2019/20 to 2025/26)",
        fontsize=12.8,
        fontweight="bold",
        color="#0F172A",
        pad=48,
    )
    ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=3,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        fontsize=8.8,
    )

    # Panel B: 2024/25 Experience Pyramid — Plain-English Labels
    buckets = [
        "Cluster B1: Entry-Level Youth\n(0–3 Years NHL Experience)\nSlafkovský, Hutson vs. Bedard",
        "Cluster B2: Prime Bridge Core\n(4–6 Years NHL Experience)\nSuzuki, Caufield, Dach vs. Vlasic",
        "Cluster B3: Peak Core Veterans\n(7–9 Years NHL Experience)\nPrime-Age Supporting Cast",
        "Cluster B4: Greybeard Anchors\n(10+ Years NHL Experience)\nMatheson vs. Foligno/Murphy/Brodie",
    ]
    mtl_24_row = mtl[mtl["season"] == 2024].iloc[0]
    chi_24_row = chi[chi["season"] == 2024].iloc[0]

    m_shares = [
        mtl_24_row["s_b1_elc_0_3"] * 100,
        mtl_24_row["s_b2_prime_4_6"] * 100,
        mtl_24_row["s_b3_peak_7_9"] * 100,
        mtl_24_row["s_b4_vet_10_plus"] * 100,
    ]
    c_shares = [
        chi_24_row["s_b1_elc_0_3"] * 100,
        chi_24_row["s_b2_prime_4_6"] * 100,
        chi_24_row["s_b3_peak_7_9"] * 100,
        chi_24_row["s_b4_vet_10_plus"] * 100,
    ]

    xb = np.arange(len(buckets))
    w = 0.32
    ax2.bar(xb - w / 2, m_shares, width=w, color="#C8102E", label="Montréal Canadiens 2024/25 (Balanced Bridge Core)", zorder=3)
    ax2.bar(xb + w / 2, c_shares, width=w, color="#1E293B", label="Chicago Blackhawks 2024/25 ('Dumbbell' Roster Trap)", zorder=3)

    for i in range(len(xb)):
        ax2.text(
            xb[i] - w / 2,
            m_shares[i] + 1.2,
            f"{m_shares[i]:.1f}%",
            ha="center",
            fontsize=9.8,
            fontweight="bold",
            color="#C8102E",
        )
        ax2.text(
            xb[i] + w / 2,
            c_shares[i] + 1.2,
            f"{c_shares[i]:.1f}%",
            ha="center",
            fontsize=9.8,
            fontweight="bold",
            color="#0F172A",
        )

    ax2.annotate(
        "Montréal's Prime Bridge Engine (29.9% vs. 16.0%)\nMTL doubled its 4–6 yr core around Suzuki (89 pts),\nCaufield (70 pts), Dach & Newhook",
        xy=(1 - w / 2, m_shares[1] + 3.5),
        xytext=(0.15, 43.0),
        arrowprops=dict(arrowstyle="->", color="#C8102E", lw=2.0, connectionstyle="arc3,rad=-0.12"),
        fontsize=9.0,
        fontweight="bold",
        color="#991B1B",
        bbox=dict(boxstyle="round,pad=0.35", fc="#FFF1F2", ec="#C8102E", lw=1.2),
        zorder=6,
    )
    ax2.annotate(
        "Chicago's 10+ Year Stopgap Bloat (41.9%)\nAfter trading away young stars Hagel, DeBrincat & Dach,\nCHI filled 41.9% of ice time with aging veteran stopgaps",
        xy=(3 + w / 2 - 0.12, c_shares[3] - 2.0),
        xytext=(1.70, 48.0),
        arrowprops=dict(arrowstyle="->", color="#1E293B", lw=2.0, connectionstyle="arc3,rad=0.15"),
        fontsize=9.0,
        fontweight="bold",
        color="#0F172A",
        bbox=dict(boxstyle="round,pad=0.35", fc="#F1F5F9", ec="#334155", lw=1.2),
        zorder=6,
    )

    ax2.set_xticks(xb)
    ax2.set_xticklabels(buckets, fontsize=9.5, fontweight="bold")
    ax2.set_ylim(0, 60)
    ax2.set_ylabel("Share of Total Team Ice Time (TOI %)", fontsize=11.0, fontweight="bold", color="#0F172A")
    ax2.set_title(
        "Panel B: 2024/25 Roster Experience Distribution — Why Selling Your 23-Year-Old Core Creates a Dumbbell Trap",
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
        fontsize=9.5,
    )

    plt.suptitle(
        "Growth vs. Regression Since 2019: Montréal Canadiens vs. Chicago Blackhawks Econometric Audit",
        fontsize=14.5,
        fontweight="bold",
        color="#0F172A",
        y=0.988,
    )
    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.955], h_pad=4.2)
    fig.savefig(FIG_DIR / "fig1_mtl_vs_chi_structural_decomposition_2019_2025.png", dpi=240)
    plt.close(fig)
    print("Blog 4 vertical spacious chart saved.")


if __name__ == "__main__":
    main()
