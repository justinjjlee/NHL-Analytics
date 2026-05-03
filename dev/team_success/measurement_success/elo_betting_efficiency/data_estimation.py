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

def optimize_elo_params(df, is_bors=False, burn_in_frac=0.25):
    best_k = 10
    best_hfa = 25
    best_brier = float('inf')
    
    train_idx = int(len(df) * burn_in_frac)
    eval_df = df.iloc[train_idx:].copy()
    
    results = []
    
    print(f"Optimizing {'BORS ' if is_bors else ''}Elo...")
    for k in range(4, 34, 4):
        for hfa in range(10, 55, 5):
            tracker = EloTracker(k_factor=k, home_ice_adv=hfa, is_bors=is_bors)
            forecasts = tracker.track_and_update(df)
            
            eval_forecasts = forecasts[train_idx:]
            
            valid_mask = ~pd.isna(eval_forecasts) & ~pd.isna(eval_df['target'].values)
            if valid_mask.sum() > 0:
                y_true = eval_df['target'].values[valid_mask]
                y_pred = np.array(eval_forecasts)[valid_mask]
                
                brier = brier_score_loss(y_true, y_pred)
                results.append({'K': k, 'HFA': hfa, 'Brier': brier})
                
                if brier < best_brier:
                    best_brier = brier
                    best_k = k
                    best_hfa = hfa
                    
    results_df = pd.DataFrame(results)
    return best_k, best_hfa, results_df

# Compute Standard Elo
opt_k, opt_hfa, elo_grid = optimize_elo_params(model_data, is_bors=False)
hfa_prob_shift = (1 / (1 + 10 ** (-opt_hfa / 400))) - 0.5
print(f"Optimal Standard Elo -> K: {opt_k}, HFA: {opt_hfa}")
print(f"Implied isolated Home-Ice Advantage Win Probability Bump: +{hfa_prob_shift*100:.2f}%")

elo_engine = EloTracker(k_factor=opt_k, home_ice_adv=opt_hfa, is_bors=False)
model_data['prob_home_elo'] = elo_engine.track_and_update(model_data)

# Compute BORS Elo
opt_k_bors, opt_hfa_bors, bors_grid = optimize_elo_params(model_data, is_bors=True)
bors_hfa_prob_shift = (1 / (1 + 10 ** (-opt_hfa_bors / 400))) - 0.5
print(f"Optimal BORS Elo -> K: {opt_k_bors}, HFA: {opt_hfa_bors}")
print(f"BORS Implied isolated Home-Ice Advantage Win Probability Bump: +{bors_hfa_prob_shift*100:.2f}%")

bors_engine = EloTracker(k_factor=opt_k_bors, home_ice_adv=opt_hfa_bors, is_bors=True)
model_data['prob_home_bors'] = bors_engine.track_and_update(model_data)

# COMMAND ----------

# DBTITLE 1,Home Ice Advantage Calibration Plot
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, grid, title in zip(axes, [elo_grid, bors_grid], ["Standard Elo", "BORS Elo"]):
    pivot_table = grid.pivot(index="K", columns="HFA", values="Brier")
    sns.heatmap(pivot_table, annot=True, fmt=".4f", cmap="YlGnBu_r", ax=ax)
    ax.set_title(f"{title} Parameter Calibration (Brier Score)", fontsize=14)
    ax.set_xlabel("Home Ice Advantage (Elo Points)")
    ax.set_ylabel("K-Factor")
    
plt.tight_layout()
plt.savefig('res/elo_hfa_calibration_heatmap.png', dpi=300, bbox_inches='tight')
print("HFA Calibration Heatmap saved to res/elo_hfa_calibration_heatmap.png")
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Process-Based Forecasting Metrics
# MAGIC The third framework runs sequential Rolling Logistic Regressions targeting the predictive power of public Corsi/Fenwick metrics. 

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

# Calculate expanding rolling logit to evaluate informational content accumulation
process_forecasts = []
process_features = ['delta_CF_pct', 'delta_FF_pct', 'delta_pythagorean']

# We need a minimum seed of games to spin up the process regressor (e.g., 50 cumulative league games)
burn_in = 50 

for i in range(len(model_data)):
    if i < burn_in:
        process_forecasts.append(np.nan)
        continue
    
    current_date = model_data.loc[i, 'date']
    
    # Train strictly on historical dates to avoid intra-day temporal leakage
    historical_mask = model_data['date'] < current_date
    if historical_mask.sum() < burn_in:
        process_forecasts.append(np.nan)
        continue
        
    X_train_raw = model_data.loc[historical_mask, process_features]
    y_train = model_data.loc[historical_mask, 'target']
    
    # Standardize to avoid default L2 penalization neutering unscaled fractional features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    
    # Fit Logit mapping pure coefficient shifts without extensive regularization bias
    clf = LogisticRegression(solver='lbfgs', C=1e5, max_iter=500)
    clf.fit(X_train, y_train)
    
    # Predict current game g
    X_test_raw = model_data.loc[i:i, process_features]
    X_test = scaler.transform(X_test_raw)
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
import seaborn as sns
import matplotlib.dates as mdates

window_size = 50
rolling_betas = []
rolling_dates = []
rolling_bse = []

# Prepare data for 2-step analytic GMM without NaNs
gmm_eval_data = eval_data.copy()
gmm_eval_data['prob_shin_sq'] = gmm_eval_data['prob_home_shin'] ** 2
gmm_eval_data['const'] = 1.0

rolling_gmm_df = gmm_eval_data.dropna(subset=['prob_home_shin', 'prob_shin_sq', 'prob_home_elo', 'n_g_home']).reset_index(drop=True)

for i in range(window_size, len(rolling_gmm_df)):
    window_df = rolling_gmm_df.iloc[i-window_size:i]
    
    y_mat = window_df['target'].values
    X_mat = window_df[['const', 'prob_home_shin']].values
    Z_mat = window_df[['const', 'prob_home_shin', 'prob_shin_sq', 'prob_home_elo', 'n_g_home']].values
    
    try:
        # Step 1: Unweighted instrument projection
        W_step1 = np.linalg.inv(Z_mat.T @ Z_mat)
        ZX = Z_mat.T @ X_mat
        Zy = Z_mat.T @ y_mat
        theta_1 = np.linalg.inv(ZX.T @ W_step1 @ ZX) @ (ZX.T @ W_step1 @ Zy)
        
        # Calculate first stage residuals
        resid_1 = y_mat - X_mat @ theta_1
        
        # Step 2: Establish robust Covariance Matrix (S)
        S = Z_mat.T @ np.diag(resid_1**2) @ Z_mat
        W_step2 = np.linalg.inv(S)
        
        # Compute optimal two-step GMM estimate
        cov_matrix_gmm = np.linalg.inv(ZX.T @ W_step2 @ ZX)
        theta_gmm = cov_matrix_gmm @ (ZX.T @ W_step2 @ Zy)
        
        # Extract Standard Error for Beta
        beta_bse = np.sqrt(cov_matrix_gmm[1, 1])
        
        rolling_betas.append(theta_gmm[1])
        rolling_bse.append(beta_bse)
        rolling_dates.append(pd.to_datetime(window_df['date'].iloc[-1]))
        
    except np.linalg.LinAlgError:
        pass # Collinearity anomaly, skip window

rolling_betas = np.array(rolling_betas)
rolling_bse = np.array(rolling_bse)

fig, ax = plt.subplots(figsize=(10, 6))
# Create rolling plot with 95% Confidence Bounds
sns.lineplot(x=rolling_dates, y=rolling_betas, ax=ax, color='navy', linewidth=2, label='GMM Estimate $\\hat{\\beta}_n$')
ax.fill_between(rolling_dates, 
                rolling_betas - 1.96 * rolling_bse, 
                rolling_betas + 1.96 * rolling_bse, 
                color='navy', alpha=0.2, label='95% Confidence Interval')

ax.axhline(1.0, color='red', linestyle='--', label='Perfect Efficiency (Beta = 1)')
# Formatting
ax.set_title('Dynamic Market Efficiency: Rolling 50-Game 2-Step GMM $\\beta$', fontsize=14, pad=15)
ax.set_ylabel('Market Coefficient ($\\beta$)', fontsize=12)
ax.set_xlabel('Date', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower left')
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

#display(fig) # Display inline to Databricks

# COMMAND ----------
