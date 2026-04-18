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

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Statistical Testings
# MAGIC ### Mincer-Zarnowitz Forecast Rationality Test
# MAGIC Null Hypothesis: Intercept is 0 and slope is 1 (forecast is perfectly rational). High Wald stat p-value means we cannot reject efficiency map.

# COMMAND ----------

def run_mincer_zarnowitz(model_prob_col, name="Model"):
    X = sm.add_constant(eval_data[model_prob_col])
    y = eval_data['target']
    
    # HC1 is equivalent to robust standard errors protecting against heteroskedasticity
    ols = sm.OLS(y, X).fit(cov_type='HC1')
    
    # Wald Test for Joint Null: interecept = 0, slope = 1
    wald_test = ols.wald_test("const = 0, {} = 1".format(model_prob_col))
    
    print(f"--- Mincer-Zarnowitz: {name} ---")
    print(f"Alpha (bias): {ols.params.iloc[0]:.4f} (p-val: {ols.pvalues.iloc[0]:.4f})")
    print(f"Beta (calibration): {ols.params.iloc[1]:.4f}")
    if np.isnan(wald_test.pvalue):
         print("Joint Wald Test Failed due to singular matrices.")
    else:
         print(f"Joint Wald Test P-Value: {wald_test.pvalue:.4e}\n")

run_mincer_zarnowitz('prob_home_shin', "Shin Corrected Market")
run_mincer_zarnowitz('prob_home_process', "Process Based Forecast")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Overidentified GMM & J-Statistic
# MAGIC Tests if additional instruments (Elo, game count, quadratic probabilities) predict residuals not captured by the Shin model. (Section 4.3)

# COMMAND ----------

from scipy import stats

eval_data['prob_shin_sq'] = eval_data['prob_home_shin'] ** 2
eval_data['const'] = 1.0

# Ensure no NaNs drift into the instrumental matrix
gmm_df = eval_data.dropna(subset=['prob_home_shin', 'prob_shin_sq', 'prob_home_elo', 'n_g_home']).copy()

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

    print("--- GMM Efficiency Estimations (Analytical 2-Step) ---")
    print(f"Alpha (bias mapping): {theta_gmm[0]:.4f}")
    print(f"Beta (efficiency projection): {theta_gmm[1]:.4f}")
    
    print(f"\\nJ-statistic: {J_stat:.4f} (p-value: {j_pval:.4e}, df: {df})")
    if j_pval < 0.05:
        print("Conclusion: Reject the null. Instruments contain information not priced by the market.")
    else:
        print("Conclusion: Fail to reject. No evidence that instruments contain unpriced information.")
        
except Exception as e:
    print(f"GMM analytical estimation failed: {e}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Diebold-Mariano Test (Pairwise Predictability)
# MAGIC Evaluates whether point-differences in Brier score residuals between models are statistically significant. (Section 7)

# COMMAND ----------

import scipy.stats as stats

def dm_test(model1_col, model2_col):
    # Calculate pointwise Brier scores
    loss1 = (eval_data['target'] - eval_data[model1_col]) ** 2
    loss2 = (eval_data['target'] - eval_data[model2_col]) ** 2
    
    # Difference in loss
    d = loss1 - loss2
    
    # 1-sample t-test (null hypothesis is that mean difference = 0)
    stat, pval = stats.ttest_1samp(d, 0)
    print(f"DM Test: {model1_col} vs {model2_col}")
    print(f"Mean Loss Diff: {d.mean():.6f} | T-stat: {stat:.4f} | P-val: {pval:.4e}")
    if pval < 0.05:
        better_model = model1_col if d.mean() < 0 else model2_col
        print(f"Result: {better_model} is systematically superior.\\n")
    else:
        print("Result: No statistically significant difference.\\n")

dm_test('prob_home_shin', 'prob_home_process')
dm_test('prob_home_shin', 'prob_home_bors')
dm_test('prob_home_elo', 'prob_home_bors')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Forecast Encompassing Regression
# MAGIC Test to find whether the process estimates (Corsi/Fenwick) hold statistically **incremental** predictive weighting outside of what the Market has already formulated.

# COMMAND ----------

# Encompassing Regression: W_g = d0 + d1(Shin) + d2(Process) + d3(Elo) + e
X_enc = eval_data[['prob_home_shin', 'prob_home_process', 'prob_home_elo']]
X_enc = sm.add_constant(X_enc)
y_enc = eval_data['target']

enc_model = sm.OLS(y_enc, X_enc).fit(cov_type='HC1')
print("--- Forecast Encompassing Regression ---")
print(enc_model.summary())

# COMMAND ----------

# MAGIC %md
# MAGIC ### GMM Efficiency Estimations (RQ2 - Dynamic Efficiency)
# MAGIC Execute rolling linear evaluations mimicking sample moment condition parameters to evaluate structural efficiencies dynamically.

# COMMAND ----------

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
display(fig) # Display inline to Databricks

# COMMAND ----------

# MAGIC %md
# MAGIC ### Model Calibration/Reliability Diagrams
# MAGIC To visually inspect whether the calculated probabilities correspond to empirical win frequencies across discrete buckets.

# COMMAND ----------

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
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(eval_data['target'], eval_data[col], n_bins=10, strategy='uniform')
    
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
display(fig) # Display inline to Databricks

