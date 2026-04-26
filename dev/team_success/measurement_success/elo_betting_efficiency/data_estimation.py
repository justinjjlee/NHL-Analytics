# Databricks notebook source
# MAGIC %md
# MAGIC # NHL Betting Market Efficiency: Data Estimation and Analysis
# MAGIC This notebook executes the analytical framework outlined in our documentation. It evaluates whether the NHL betting market incorporates publicly available information efficiently, comparing market-implied probabilities against Elo ratings and process-based forecasting.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Environment Setup and Data Ingestion

# COMMAND ----------

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

# Load our primary analytical matrix prepared from proc_data
df_analytical = spark.table("`nhl-databricks`.compute.elo_betting_efficiency").toPandas()
df_analytical = df_analytical.sort_values(by="date").reset_index(drop=True)

# Define our prediction targets
# For static/dynamic efficiency tests, our binary target is home_win. Using goal differential indicator for evaluation.
df_analytical['target'] = df_analytical['home_win'].astype(int)

# Drop games missing essential metrics or shin conversions
model_data = df_analytical.dropna(subset=['prob_home_shin', 'delta_CF_pct', 'delta_pythagorean']).copy()

# COMMAND ----------

# DBTITLE 1,Create Results Directory
import os

# Create results directory
res_dir = 'res'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)
    print(f"Created directory: {res_dir}")
else:
    print(f"Directory already exists: {res_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Theoretical Framework: ELO Specifications
# MAGIC We maintain two concurrent Elo variations:
# MAGIC 1. **Standard Elo**: Latent skill tracking relying purely on game outcome expected score.
# MAGIC 2. **BORS Elo**: Latent skill tracking initialized substituting the Shin-corrected market probability for the expectation calculation.

# COMMAND ----------

class EloTracker:
    def __init__(self, k_factor=10, home_ice_adv=25, is_bors=False):
        self.k_factor = k_factor
        self.home_ice_adv = home_ice_adv
        self.is_bors = is_bors
        self.ratings = {}
        
    def get_rating(self, team):
        return self.ratings.get(team, 1500)
        
    def expected_score(self, r_home, r_away):
        return 1 / (1 + 10 ** (-(r_home - r_away + self.home_ice_adv) / 400))
        
    def track_and_update(self, df):
        forecasts = []
        for idx, row in df.iterrows():
            home, away = row['home_team'], row['away_team']
            r_home = self.get_rating(home)
            r_away = self.get_rating(away)
            
            # Predict
            prob_home = self.expected_score(r_home, r_away)
            forecasts.append(prob_home)
            
            # Outcome
            outcome = row['target']
            
            # Formulate expectation
            if self.is_bors:
                # BORS uses the Shin probability as expected score
                expected = row['prob_home_shin']
            else:
                expected = prob_home
                
            # Update mechanism
            self.ratings[home] = r_home + self.k_factor * (outcome - expected)
            self.ratings[away] = r_away + self.k_factor * ((1 - outcome) - (1 - expected))
            
        return forecasts

# Compute Standard Elo
elo_engine = EloTracker(k_factor=10, home_ice_adv=25, is_bors=False)
model_data['prob_home_elo'] = elo_engine.track_and_update(model_data)

# Compute BORS Elo
bors_engine = EloTracker(k_factor=10, home_ice_adv=25, is_bors=True)
model_data['prob_home_bors'] = bors_engine.track_and_update(model_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Process-Based Forecasting Metrics
# MAGIC The third framework runs sequential Rolling Logistic Regressions targeting the predictive power of public Corsi/Fenwick metrics. 

# COMMAND ----------

# Calculate expanding rolling logit to evaluate informational content accumulation
process_forecasts = []
process_features = ['delta_CF_pct', 'delta_FF_pct', 'delta_pythagorean']

# We need a minimum seed of games to spin up the process regressor (e.g., 50 cumulative league games)
burn_in = 50 

for i in range(len(model_data)):
    if i < burn_in:
        process_forecasts.append(np.nan)
        continue
    
    # Train on all strictly historical priors up to game g - 1
    X_train = model_data.loc[:i-1, process_features]
    y_train = model_data.loc[:i-1, 'target']
    
    # Fit Logit
    clf = LogisticRegression(solver='liblinear')
    clf.fit(X_train, y_train)
    
    # Predict current game g
    X_test = model_data.loc[i:i, process_features]
    prob = clf.predict_proba(X_test)[0][1]
    process_forecasts.append(prob)

model_data['prob_home_process'] = process_forecasts

# Trim burn in wrapper
eval_data = model_data.dropna(subset=['prob_home_process']).copy()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Estimations & Calibration
# MAGIC Evaluate Brier Scores and Log Loss algorithms across the model variations.

# COMMAND ----------

# DBTITLE 1,Calculate Scoring Rules and Save Results
def calc_scoring_rules(y_true, y_pred):
    return {
        'Brier Score': brier_score_loss(y_true, y_pred),
        'Log Loss': log_loss(y_true, y_pred)
    }

scores = {
    'Shin Market': calc_scoring_rules(eval_data['target'], eval_data['prob_home_shin']),
    'Standard Elo': calc_scoring_rules(eval_data['target'], eval_data['prob_home_elo']),
    'BORS Elo': calc_scoring_rules(eval_data['target'], eval_data['prob_home_bors']),
    'Process Metrics': calc_scoring_rules(eval_data['target'], eval_data['prob_home_process'])
}

df_scores = pd.DataFrame(scores).T
df_scores = df_scores.reset_index().rename(columns={'index': 'Model'})
display(df_scores)

# Save to file
with open('res/scoring_metrics.txt', 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("MODEL SCORING METRICS\n")
    f.write("=" * 60 + "\n\n")
    f.write(df_scores.to_string(index=False))
    f.write("\n\n")
print("\nResults saved to res/scoring_metrics.txt")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Statistical Testings
# MAGIC ### Mincer-Zarnowitz Forecast Rationality Test
# MAGIC Null Hypothesis: Intercept is 0 and slope is 1 (forecast is perfectly rational). High Wald stat p-value means we cannot reject efficiency map.

# COMMAND ----------

# DBTITLE 1,Mincer-Zarnowitz Test and Save Results
def run_mincer_zarnowitz(model_prob_col, name="Model"):
    X = sm.add_constant(eval_data[model_prob_col])
    y = eval_data['target']
    
    # HC1 is equivalent to robust standard errors protecting against heteroskedasticity
    ols = sm.OLS(y, X).fit(cov_type='HC1')
    
    # Wald Test for Joint Null: interecept = 0, slope = 1
    wald_test = ols.wald_test("const = 0, {} = 1".format(model_prob_col))
    
    result_text = f"--- Mincer-Zarnowitz: {name} ---\n"
    result_text += f"Alpha (bias): {ols.params.iloc[0]:.4f} (p-val: {ols.pvalues.iloc[0]:.4f})\n"
    result_text += f"Beta (calibration): {ols.params.iloc[1]:.4f}\n"
    if np.isnan(wald_test.pvalue):
         result_text += "Joint Wald Test Failed due to singular matrices.\n"
    else:
         result_text += f"Joint Wald Test P-Value: {wald_test.pvalue:.4e}\n\n"
    
    print(result_text)
    return result_text

# Run tests and collect results
results = []
results.append(run_mincer_zarnowitz('prob_home_shin', "Shin Corrected Market"))
results.append(run_mincer_zarnowitz('prob_home_process', "Process Based Forecast"))

# Save to file
with open('res/mincer_zarnowitz_tests.txt', 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("MINCER-ZARNOWITZ FORECAST RATIONALITY TESTS\n")
    f.write("=" * 60 + "\n\n")
    f.write("Null Hypothesis: Intercept is 0 and slope is 1 (forecast is perfectly rational).\n")
    f.write("High Wald stat p-value means we cannot reject efficiency map.\n\n")
    for result in results:
        f.write(result)
print("Results saved to res/mincer_zarnowitz_tests.txt")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Overidentified GMM & J-Statistic
# MAGIC Tests if additional instruments (Elo, game count, quadratic probabilities) predict residuals not captured by the Shin model. (Section 4.3)

# COMMAND ----------

# DBTITLE 1,GMM J-Test and Save Results
from scipy import stats

eval_data['prob_shin_sq'] = eval_data['prob_home_shin'] ** 2
eval_data['const'] = 1.0

# Ensure no NaNs drift into the instrumental matrix
gmm_df = eval_data.dropna(subset=['prob_home_shin', 'prob_shin_sq', 'prob_home_elo', 'n_g_home']).copy()

result_text = ""
try:
    # We build the 2-step optimal GMM analytically to bypass broken sandbox packages
    y_mat = gmm_df['target'].values
    X_mat = gmm_df[['const', 'prob_home_shin']].values
    Z_mat = gmm_df[['const', 'prob_home_shin', 'prob_shin_sq', 'prob_home_elo', 'n_g_home']].values

    # Step 1: Unweighted / identity instrument projection
    W_step1 = np.linalg.inv(Z_mat.T @ Z_mat)
    ZX = Z_mat.T @ X_mat
    Zy = Z_mat.T @ y_mat
    theta_1 = np.linalg.inv(ZX.T @ W_step1 @ ZX) @ (ZX.T @ W_step1 @ Zy)

    # Calculate first stage residuals
    resid_1 = y_mat - X_mat @ theta_1

    # Step 2: Establish the Heteroskedasticity-Robust Moment Covariance Matrix (S)
    # S = Z' * diag(e^2) * Z
    S = Z_mat.T @ np.diag(resid_1**2) @ Z_mat
    W_step2 = np.linalg.inv(S)

    # Compute optimal two-step GMM estimate
    theta_gmm = np.linalg.inv(ZX.T @ W_step2 @ ZX) @ (ZX.T @ W_step2 @ Zy)
    
    # Calculate generalized final residuals
    resid_gmm = y_mat - X_mat @ theta_gmm

    # J-statistic calculation: J = (e'Z) W_optimal (Z'e)
    # Represents the minimized objective function assessing orthogonality
    J_stat = (resid_gmm.T @ Z_mat) @ W_step2 @ (Z_mat.T @ resid_gmm)
    
    # Degrees of freedom: total instruments - active parameters (5 - 2 = 3)
    df = Z_mat.shape[1] - X_mat.shape[1]
    j_pval = 1.0 - stats.chi2.cdf(J_stat, df=df)

    result_text = "--- GMM Efficiency Estimations (Analytical 2-Step) ---\n"
    result_text += f"Alpha (bias mapping): {theta_gmm[0]:.4f}\n"
    result_text += f"Beta (efficiency projection): {theta_gmm[1]:.4f}\n\n"
    
    result_text += f"J-statistic: {J_stat:.4f} (p-value: {j_pval:.4e}, df: {df})\n"
    if j_pval < 0.05:
        result_text += "Conclusion: Reject the null. Instruments contain information not priced by the market.\n"
    else:
        result_text += "Conclusion: Fail to reject. No evidence that instruments contain unpriced information.\n"
    
    print(result_text)
        
except Exception as e:
    result_text = f"GMM analytical estimation failed: {e}\n"
    print(result_text)

# Save to file
with open('res/gmm_j_test.txt', 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("GMM OVERIDENTIFIED J-STATISTIC TEST\n")
    f.write("=" * 60 + "\n\n")
    f.write("Tests if additional instruments (Elo, game count, quadratic probabilities)\n")
    f.write("predict residuals not captured by the Shin model.\n\n")
    f.write(result_text)
print("\nResults saved to res/gmm_j_test.txt")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Diebold-Mariano Test (Pairwise Predictability)
# MAGIC Evaluates whether point-differences in Brier score residuals between models are statistically significant. (Section 7)

# COMMAND ----------

# DBTITLE 1,Diebold-Mariano Test and Save Results
import scipy.stats as stats

def dm_test(model1_col, model2_col):
    # Calculate pointwise Brier scores
    loss1 = (eval_data['target'] - eval_data[model1_col]) ** 2
    loss2 = (eval_data['target'] - eval_data[model2_col]) ** 2
    
    # Difference in loss
    d = loss1 - loss2
    
    # 1-sample t-test (null hypothesis is that mean difference = 0)
    stat, pval = stats.ttest_1samp(d, 0)
    
    result_text = f"DM Test: {model1_col} vs {model2_col}\n"
    result_text += f"Mean Loss Diff: {d.mean():.6f} | T-stat: {stat:.4f} | P-val: {pval:.4e}\n"
    if pval < 0.05:
        better_model = model1_col if d.mean() < 0 else model2_col
        result_text += f"Result: {better_model} is systematically superior.\n\n"
    else:
        result_text += "Result: No statistically significant difference.\n\n"
    
    print(result_text)
    return result_text

# Run tests and collect results
results = []
results.append(dm_test('prob_home_shin', 'prob_home_process'))
results.append(dm_test('prob_home_shin', 'prob_home_bors'))
results.append(dm_test('prob_home_elo', 'prob_home_bors'))

# Save to file
with open('res/diebold_mariano_tests.txt', 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("DIEBOLD-MARIANO PAIRWISE PREDICTABILITY TESTS\n")
    f.write("=" * 60 + "\n\n")
    f.write("Evaluates whether point-differences in Brier score residuals\n")
    f.write("between models are statistically significant.\n\n")
    for result in results:
        f.write(result)
print("Results saved to res/diebold_mariano_tests.txt")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Forecast Encompassing Regression
# MAGIC Test to find whether the process estimates (Corsi/Fenwick) hold statistically **incremental** predictive weighting outside of what the Market has already formulated.

# COMMAND ----------

# DBTITLE 1,Forecast Encompassing Regression and Save Results
# Encompassing Regression: W_g = d0 + d1(Shin) + d2(Process) + d3(Elo) + e
X_enc = eval_data[['prob_home_shin', 'prob_home_process', 'prob_home_elo']]
X_enc = sm.add_constant(X_enc)
y_enc = eval_data['target']

enc_model = sm.OLS(y_enc, X_enc).fit(cov_type='HC1')
print("--- Forecast Encompassing Regression ---")
print(enc_model.summary())

# Save to file
with open('res/encompassing_regression.txt', 'w') as f:
    f.write("=" * 60 + "\n")
    f.write("FORECAST ENCOMPASSING REGRESSION\n")
    f.write("=" * 60 + "\n\n")
    f.write("Tests whether the process estimates (Corsi/Fenwick) hold statistically\n")
    f.write("incremental predictive weighting outside of what the Market has already formulated.\n\n")
    f.write(str(enc_model.summary()))
    f.write("\n")
print("\nResults saved to res/encompassing_regression.txt")

# COMMAND ----------

# MAGIC %md
# MAGIC ### GMM Efficiency Estimations (RQ2 - Dynamic Efficiency)
# MAGIC Execute rolling linear evaluations mimicking sample moment condition parameters to evaluate structural efficiencies dynamically.

# COMMAND ----------

# DBTITLE 1,Dynamic Market Efficiency Plot and Save
import matplotlib.pyplot as plt

window_size = 50
rolling_betas = []
rolling_dates = []

for i in range(window_size, len(eval_data)):
    window_df = eval_data.iloc[i-window_size:i]
    X_roll = sm.add_constant(window_df['prob_home_shin'])
    y_roll = window_df['target']
    
    roll_model = sm.OLS(y_roll, X_roll).fit(cov_type='HC1')
    rolling_betas.append(roll_model.params.iloc[1])
    rolling_dates.append(pd.to_datetime(window_df['date'].iloc[-1]))

# This array represents the learning convergence parameter beta as season iterates.
# Approaching 1 represents total efficiency.

import seaborn as sns
import matplotlib.dates as mdates

fig, ax = plt.subplots(figsize=(10, 6))
# Create rolling plot
sns.lineplot(x=rolling_dates, y=rolling_betas, ax=ax, color='navy', linewidth=2)
ax.axhline(1.0, color='red', linestyle='--', label='Perfect Efficiency (Beta = 1)')
# Formatting
ax.set_title('Dynamic Market Efficiency: Rolling 50-Game Mincer-Zarnowitz $\\beta$', fontsize=14, pad=15)
ax.set_ylabel('Market Coefficient ($\\beta$)', fontsize=12)
ax.set_xlabel('Date', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend()
# Format x-axis dates nicely
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)
plt.tight_layout()

# Save figure
plt.savefig('res/dynamic_efficiency_plot.png', dpi=300, bbox_inches='tight')
print("Plot saved to res/dynamic_efficiency_plot.png")

display(fig) # Display inline to Databricks

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Calibration/Reliability Diagrams
# MAGIC To visually inspect whether the calculated probabilities correspond to empirical win frequencies across discrete buckets.

# COMMAND ----------

# DBTITLE 1,Model Calibration Diagrams and Save
from sklearn.calibration import calibration_curve

fig, ax = plt.subplots(figsize=(10, 8))

# Define model names and their representative probability columns
calibration_models = {
    'Shin Market': 'prob_home_shin',
    'Standard Elo': 'prob_home_elo',
    'BORS Elo': 'prob_home_bors',
    'Process Forecast': 'prob_home_process'
}

colors = ['navy', 'forestgreen', 'darkorange', 'purple']

for (name, col), color in zip(calibration_models.items(), colors):
    # Use quantile strategy to ensure each bin has sufficient samples
    # This is especially important for process model which has concentrated predictions
    prob_true, prob_pred = calibration_curve(eval_data['target'], eval_data[col], n_bins=10, strategy='quantile')
    
    # Plot curve
    ax.plot(prob_pred, prob_true, marker='o', linewidth=2, label=name, color=color)

# Plot perfectly calibrated reference line
ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')

ax.set_title('Model Calibration (Reliability) Diagrams', fontsize=15, pad=15)
ax.set_xlabel('Mean Predicted Probability', fontsize=12)
ax.set_ylabel('Fraction of True Wins / Empirical Probability', fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.legend(loc='lower right', fontsize=11)

plt.tight_layout()

# Save figure
plt.savefig('res/calibration_diagrams.png', dpi=300, bbox_inches='tight')
print("Plot saved to res/calibration_diagrams.png")
print("Note: Using 'quantile' binning strategy to ensure adequate samples per bin")

#display(fig) # Display inline to Databricks

# COMMAND ----------

# DBTITLE 1,Calibration Issue Investigation
# MAGIC %md
# MAGIC ### Calibration Diagnostic Analysis
# MAGIC
# MAGIC **Issue Identified:** The process-based model (`prob_home_process`) produces predictions that are highly concentrated around 0.5, with 99% of predictions falling in the narrow range [0.4, 0.6].
# MAGIC
# MAGIC **Root Cause:** The rolling logistic regression trained on process metrics (delta_CF_pct, delta_FF_pct, delta_pythagorean) produces weakly-dispersed predictions. This is typical when predictive features have limited discriminative power, causing the model to regress toward the base rate (~0.5 for balanced outcomes).
# MAGIC
# MAGIC **Impact:** When using 'uniform' binning strategy for calibration curves:
# MAGIC * Bins are evenly spaced across [0, 1]
# MAGIC * Most bins contain 0-2 samples (insufficient for reliable estimates)
# MAGIC * Bins with few samples produce extreme empirical probabilities (0.0 or 1.0)
# MAGIC * Creates misleading "staircase" pattern in calibration plot
# MAGIC
# MAGIC **Solution:** Changed calibration curve binning from `'uniform'` to `'quantile'` strategy:
# MAGIC * Creates bins based on actual prediction distribution
# MAGIC * Ensures each bin has adequate sample size (~1,748 samples per bin)
# MAGIC * Produces reliable empirical probability estimates
# MAGIC * Allows fair comparison across all models

# COMMAND ----------

# DBTITLE 1,Diagnose Process Model Calibration Issues
# Investigate the prob_home_process calibration issue
print("Process Model Diagnostics:")
print("="*60)
print(f"\nTotal samples: {len(eval_data)}")
print(f"\nProb_home_process distribution:")
print(eval_data['prob_home_process'].describe())
print(f"\nNumber of predictions by probability range:")
print(f"  < 0.3: {(eval_data['prob_home_process'] < 0.3).sum()}")
print(f"  0.3-0.4: {((eval_data['prob_home_process'] >= 0.3) & (eval_data['prob_home_process'] < 0.4)).sum()}")
print(f"  0.4-0.6: {((eval_data['prob_home_process'] >= 0.4) & (eval_data['prob_home_process'] < 0.6)).sum()}")
print(f"  0.6-0.7: {((eval_data['prob_home_process'] >= 0.6) & (eval_data['prob_home_process'] < 0.7)).sum()}")
print(f"  > 0.7: {(eval_data['prob_home_process'] >= 0.7).sum()}")

# Check calibration with different bin strategies
from sklearn.calibration import calibration_curve

print("\n" + "="*60)
print("Calibration Analysis by Bins:")
print("="*60)

# Use more bins to see finer detail
prob_true, prob_pred = calibration_curve(
    eval_data['target'], 
    eval_data['prob_home_process'], 
    n_bins=10, 
    strategy='uniform'
)

print("\nBin-by-bin breakdown (uniform strategy, 10 bins):")
for i, (pred, true) in enumerate(zip(prob_pred, prob_true)):
    # Count samples in this bin
    bin_width = 1.0 / 10
    bin_start = i * bin_width
    bin_end = (i + 1) * bin_width
    n_in_bin = ((eval_data['prob_home_process'] >= bin_start) & 
                (eval_data['prob_home_process'] < bin_end)).sum()
    print(f"Bin {i+1}: Pred={pred:.3f}, Empirical={true:.3f}, N={n_in_bin}")

# Try quantile strategy
print("\n" + "="*60)
prob_true_q, prob_pred_q = calibration_curve(
    eval_data['target'], 
    eval_data['prob_home_process'], 
    n_bins=10, 
    strategy='quantile'
)

print("\nBin-by-bin breakdown (quantile strategy, 10 bins):")
for i, (pred, true) in enumerate(zip(prob_pred_q, prob_true_q)):
    print(f"Bin {i+1}: Pred={pred:.3f}, Empirical={true:.3f}")

# Check for NaN or extreme values
print("\n" + "="*60)
print("Data Quality Checks:")
print("="*60)
print(f"NaN values in prob_home_process: {eval_data['prob_home_process'].isna().sum()}")
print(f"Values < 0: {(eval_data['prob_home_process'] < 0).sum()}")
print(f"Values > 1: {(eval_data['prob_home_process'] > 1).sum()}")
print(f"\nMin value: {eval_data['prob_home_process'].min():.6f}")
print(f"Max value: {eval_data['prob_home_process'].max():.6f}")

# Save diagnostic report
with open('res/calibration_diagnostics.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("CALIBRATION DIAGNOSTICS REPORT\n")
    f.write("="*60 + "\n\n")
    f.write("ISSUE IDENTIFIED:\n")
    f.write("The process-based model predictions are highly concentrated around 0.5.\n")
    f.write(f"99% of predictions fall in the range [0.4, 0.6].\n\n")
    f.write("IMPACT:\n")
    f.write("When using 'uniform' binning strategy, most bins have 0-2 samples,\n")
    f.write("causing unreliable empirical probability estimates (e.g., 0.0 or 1.0).\n\n")
    f.write("SOLUTION:\n")
    f.write("Changed to 'quantile' binning strategy which creates bins based on\n")
    f.write("the actual distribution, ensuring adequate samples per bin.\n\n")
    f.write("="*60 + "\n")
    f.write(f"Total samples: {len(eval_data)}\n\n")
    f.write("Distribution by range:\n")
    f.write(f"  < 0.3: {(eval_data['prob_home_process'] < 0.3).sum()} samples\n")
    f.write(f"  0.3-0.4: {((eval_data['prob_home_process'] >= 0.3) & (eval_data['prob_home_process'] < 0.4)).sum()} samples\n")
    f.write(f"  0.4-0.6: {((eval_data['prob_home_process'] >= 0.4) & (eval_data['prob_home_process'] < 0.6)).sum()} samples (99%)\n")
    f.write(f"  0.6-0.7: {((eval_data['prob_home_process'] >= 0.6) & (eval_data['prob_home_process'] < 0.7)).sum()} samples\n")
    f.write(f"  > 0.7: {(eval_data['prob_home_process'] >= 0.7).sum()} samples\n")

print("\nDiagnostic report saved to res/calibration_diagnostics.txt")
