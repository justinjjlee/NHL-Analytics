'''
    Statistical test of difference by sequential impact on goal scoring by offense/defense
'''

# %% Settings
import os
import pandas as pd
import numpy as np
import itertools

import pymc as pm
from scipy.stats import ks_2samp

import matplotlib.pyplot as plt
import seaborn as sns
# Save the working directory for results savings
str_cwd = os.getcwd()

# Import data
tabp = pd.read_csv(str_cwd + '/result/tabulated - sequence.csv')

# Additional data process
tabp['category_bi'] = ["blocked-shots" if iter=='blocked-shot' else 'other-shots' for iter in tabp['category']]
# %% (1) Kolmogorov-Smirnov Test for two-group testings
iter_category = tabp.category.unique()
iter_agent    = tabp.agent.unique()

# Generate all combinations (Cartesian product), then unique pairs
combinations = list(itertools.product(iter_category, iter_agent))
unique_pairs = list(itertools.combinations(combinations, 2))

fig, axs = plt.subplots(figsize=(16,13), nrows=4, ncols=4)
fig.suptitle('Evaluating sequential likelihood of goal-scoring for previous shot-attempt event')
axs = axs.flatten()

for cnt_pair, iter_pair in enumerate(unique_pairs):
    # get criteria vectors
    category_bases = tabp.loc[(tabp.category == iter_pair[0][0]) & (tabp.agent == iter_pair[0][1]), :]
    category_alter = tabp.loc[(tabp.category == iter_pair[1][0]) & (tabp.agent == iter_pair[1][1]), :]

    # Perform the Kolmogorov-Smirnov test
    ks_stat, p_value = ks_2samp(category_bases['p_game'], category_alter['p_game'])

    #eCDF plot
    # Empirical Cumulative Distribution Function (ECDF) 
    sns.ecdfplot(
        ax=axs[cnt_pair], 
        data=pd.concat([category_bases, category_alter]), 
        x='p_game', hue='Sequence',
        palette=['#e63946', '#1d3557']
    )
    # Add text to the subplot
    axs[cnt_pair].text(
        0.15, 0.4, 
        f'Kolmogorov-Smirnov Test \n Test Statistics: {np.round(ks_stat, 2)} \n p-value: {np.round(p_value, 2)}', 
        transform=axs[cnt_pair].transAxes, 
        fontsize=12, ha='left', va='top'
    )
    
    axs[cnt_pair].set_title('')
    axs[cnt_pair].set_xlabel('Probability of Goal Scoring')
    axs[cnt_pair].set_ylabel('ECDF')
    sns.move_legend(axs[cnt_pair], 'lower center', bbox_to_anchor=(0.5, 1))

axs[-1].axis('off')

plt.tight_layout()
# Show the plot
plt.savefig(str_cwd + '/result/figure - eda - hypothesis test - KS eCDF.png', dpi=600)

# %% (2) Multigroup difference using Bayesian estimation of difference iin groups

# Filter the data for the categories
categories = tabp['category'].unique()
data_dict = {category: tabp[tabp['category'] == category]['p_game'].values for category in categories}
# %%
# Bayesian model

# Define ad-hoc distribution
# Maxwell-Boltzmann distribution
def maxwell_boltzmann_pdf(x, loc, scale):
    return np.sqrt(2/np.pi) * ((x - loc)**2 / scale**3) * np.exp(-((x - loc)**2) / (2 * scale**2))

def maxwell_boltzmann_logpdf(x, loc, scale):
    return np.log(np.sqrt(2/np.pi)) + 2*np.log(x - loc) - 3*np.log(scale) - ((x - loc)**2) / (2 * scale**2)


with pm.Model() as model:
    # Priors for group means
    mus = pm.Normal('mus', mu=0, sigma=1, shape=len(categories))
    
    # Priors for group standard deviations
    sigmas = pm.HalfNormal('sigmas', sigma=1, shape=len(categories))

    # Priors for unknown parameters
    loc = pm.Normal('loc', mu=0, sigma=10)
    scale = pm.HalfNormal('scale', sigma=10)
    
    # Likelihoods for each group
    likelihoods = {}
    for i, category in enumerate(categories):
        likelihoods[category] = pm.Normal(category, mu=mus[i], sigma=sigmas[i], observed=data_dict[category])
    
    # Differences in means
    mean_diffs = pm.Deterministic('mean_diffs', mus[:, None] - mus)

    # Likelihood (sampling distribution) of observations
    y = pm.DensityDist('y', lambda value: maxwell_boltzmann_logpdf(value, loc, scale), observed=data)
    
    # Sampling
    trace = pm.sample(2000, return_inferencedata=True)

# Summary of the posterior distribution
summary = pm.summary(trace, hdi_prob=0.95)
print(summary)

# Plot the posterior distribution of the differences in means
pm.plot_posterior(trace, var_names=['mean_diffs'])
plt.title('Posterior distribution of the differences in means')
plt.xlabel('Difference in means')
plt.show()