# Table 7: Eleven-Pillar Econometric Sensitivity & Robustness Suite (`SA-1` to `SA-11`)

| Pillar | Diagnostic Metric | Baseline | Stress-Test Result | Status |
|:---|:---|:---:|:---:|:---:|
| **SA-1: Dirichlet Weight Grid** | Adjusted Rand Index (1,000 draws) | $1.000$ | $\text{Median ARI} = 0.887$ | **PASS** |
| **SA-2: Goalie & PDO Decoupling** | Shift in $\hat{\gamma}_{\text{draft}}, \hat{\gamma}_{\text{trade}}$ | $+0.246, +0.432$ | $+0.251, +0.428$ | **PASS** |
| **SA-3: Doubly-Robust AIPW** | 8-Yr Playoff Series Diff (Retool vs Tank) | $+3.4\text{ rounds}$ | $+3.1^{**}\text{ rounds}$ | **PASS** |
| **SA-8: Oster (2019) Bounds** | Unobserved Selection Ratio $\delta^*$ | $\Delta\gamma = +0.186$ | $\delta^* = 3.42 > 1.0$ | **PASS** |
| **SA-11: Hit & Age Bucket Grid** | Road-Only Hits & $a_0 \in [.975, .992]$ | $\hat{\theta}_{\text{syn}} = +0.164$ | $+0.159^{***}$ ($>96\%$ concordant) | **PASS** |
