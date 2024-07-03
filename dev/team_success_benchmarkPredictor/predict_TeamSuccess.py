# %%

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FixedFormatter, FixedLocator
plt.rcParams.update({'font.size': 16})

str_dirc = '/Google Drive/Learning/sports/nhl/';

# import relevant functions
str_dir_sourceCode = str_origin+"/GitHub/NHL-Analytics/src/measurement"
exec(open(str_dir_sourceCode + "/measurement_teamSuccess.py").read())

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
#model = LogisticRegression(solver='liblinear', random_state=0)
# Or different approach
import numpy as np
import statsmodels.api as sm
# print result in latex
from stargazer.stargazer import Stargazer, LineLocation
from IPython.core.display import HTML


def win_prediction_eval(y, x, classify_results):
    # Model Evaluation for scikit learn
    '''
    model.score(x, y)
    p_pred = model.predict_proba(x)
    y_pred = model.predict(x)
    score_ = model.score(x, y)
    conf_m = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred)
    '''
    # Fit model
    model = sm.Logit(y, x)
    result = model.fit(method='newton')

    # Predictor get
    p_pred = result.predict(x)
    y_pred = (result.predict(x) >= 0.5).astype(int)

    print(result.summary())
    conf_m = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred)
    eval = classification_report(y, y_pred, output_dict = True)
    classify_results['precision_win'] = classify_results['precision_win'] \
                                        + [round(eval['1.0']['precision'],3)]
    classify_results['precision_loss'] = classify_results['precision_loss'] \
                                        + [round(eval['0.0']['precision'],3)]
    classify_results['recall_win'] = classify_results['recall_win'] \
                                        + [round(eval['1.0']['recall'],3)]
    classify_results['recall_loss'] = classify_results['recall_loss'] \
                                        + [round(eval['0.0']['recall'],3)]
    classify_results['f1_win'] = classify_results['f1_win'] \
                                + [round(eval['1.0']['f1-score'],3)]
    classify_results['f1_loss'] = classify_results['f1_loss'] \
                                + [round(eval['0.0']['f1-score'],3)]
    classify_results['support_win'] = classify_results['support_win'] \
                                + [round(eval['1.0']['support'],3)]
    classify_results['support_loss'] = classify_results['support_loss'] \
                                + [round(eval['0.0']['support'],3)]
    print("Report: ", report, sep = "\n")

    return result, classify_results

# %%
# Pull historical data

# Last year season progress
idx_yr_last = 2022
list_years = np.arange(2011, idx_yr_last)

df = []
for iter in list_years:
    tempdf = pd.read_csv(str_data + f'/{iter}_02_box.csv')
    if iter == 2011:
        df = tempdf
    else:
        df = pd.concat([df, tempdf])

# Regress tie or overtime game 
dfregress = df[df[ybool] != 0.5]
# Only use home game (home and away records duplicate)
dfregress = dfregress[dfregress.team_vis_for == 'home']

# %% 
# Define KPI

# set saving results for classifications
classify_results = {
    'precision_win':[],
    'precision_loss':[],
    'recall_win':[],
    'recall_loss':[],
    'f1_win':[],
    'f1_loss':[],
    'support_win':[],
    'support_loss':[]
}

# Use to filter out to be each game,
idx_game = 'gameIdx_now_against'
idx_home = 'team_vis_against'

# Define KPI
ybool = 'win_for'
ycum = 'rwin'
# Control variable, if needed
xcontrol = ['yr_season']

# Baselines: Post-hoc team predictor
# Model 1-0. Season win percentage
x_1_0 = ['kpi_wp', 'kpi_wp_against']
# Model 1-1. Pythagorean expectation
x_1_1 = ['kpi_pe', 'kpi_pe_against']
# Model 1-2. Corsi (normalized scale)
x_1_2 = ['corsi_for', 'corsi_against']
# Model 1-3. Fenwick (normalized scale)
x_1_3 = ['fenwick_for', 'fenwick_against']
# Model 1-4. RPI
x_1_4 = ['rpi', 'rpi_against']
# Model 1-5. Fenwick
x_1_5 = ['pairwise_win', 'pairwise_win_against']
# Model 1-6 . Better and good?
x_1_6 = ['idx_OppoGood', 'idx_OppoBetter']


# Real-time predictor

# %%
# Predictor
# .....................................................................
# Data choice
x = dfregress[x_1_0]*100
y = dfregress[ybool]

# Run and save logit regression
#model_1, classify_results = win_prediction_eval(y, x, classify_results)

# .....................................................................
# Data choice
x = (dfregress[x_1_0[0]] - dfregress[x_1_0[1]])*100
x.name = 'kpi_wp_diff'
y = dfregress[ybool]

# Run and save logit regression
#model_2, classify_results = win_prediction_eval(y, x, classify_results)

# .....................................................................
# Data choice
x = dfregress[x_1_1]
y = dfregress[ybool]

# Run and save logit regression
#model_1_1, classify_results = win_prediction_eval(y, x, classify_results)

# .....................................................................
# Data choice
x = dfregress[x_1_2]*1.0
# Fill NA with mean
x = x.fillna(0)
y = dfregress[ybool]

# Run and save logit regression
model_3, classify_results = win_prediction_eval(y, x, classify_results)

# Data choice
x = dfregress['corsi_lvl']*1.0
# Fill NA with mean
x = x.fillna(0)
y = dfregress[ybool]

# Run and save logit regression
#model_4, classify_results = win_prediction_eval(y, x, classify_results)

# .....................................................................
# Data choice
x = dfregress[x_1_3]*1.0
# Fill NA with mean
x = x.fillna(0)
y = dfregress[ybool]

# Run and save logit regression
model_5, classify_results = win_prediction_eval(y, x, classify_results)

# Data choice
x = dfregress['fenwick_lvl']*1.0
# Fill NA with mean
x = x.fillna(0)
y = dfregress[ybool]

# Run and save logit regression
#model_6, classify_results = win_prediction_eval(y, x, classify_results)

# .....................................................................
# Data choice
x = dfregress[x_1_4]*1.0
# Fill NA with mean
x = x.fillna(0)
y = dfregress[ybool]

# Run and save logit regression
model_7, classify_results = win_prediction_eval(y, x, classify_results)

# .....................................................................
# Data choice
x = dfregress[x_1_5]*1.0
# Fill NA with mean
x = x.fillna(0)
y = dfregress[ybool]

# Run and save logit regression
model_8, classify_results = win_prediction_eval(y, x, classify_results)

# .....................................................................
# Data choice
x = dfregress[x_1_6]*1.0
# Fill NA with mean
x = x.fillna(0)
y = dfregress[ybool]

# Run and save logit regression
#model_9, classify_results = win_prediction_eval(y, x, classify_results)

# .....................................................................
# Data choice
x = dfregress[x_1_6 + x_1_0 + x_1_1]*1.0
# Fill NA with mean
x = x.fillna(0)
y = dfregress[ybool]

# Run and save logit regression
model_10, classify_results = win_prediction_eval(y, x, classify_results)

stargazer = Stargazer([model_10, model_3, 
                       model_5, 
                       model_7, model_8])
#stargazer.custom_columns(['', 'Post-hoc Predictor', '', '', '', ''], [1, 1, 1, 1, 1, 1])
stargazer.rename_covariates({'kpi_wp': 'Win Rate: Own', 
                        'kpi_wp_against': 'Win Rate: Against',
                        'kpi_wp_diff': 'Win Rate: Own - Against',
                        'kpi_pe': "P. Expectation: Own",
                        'kpi_pe_against': 'P. Expectation: Against',
                        'corsi_for': "Corsi: Own",
                        'corsi_against': "Corsi: Against",
                        'corsi_lvl': "Corsi: Own - Against",
                        'fenwick_for': "Fenwick: Own",
                        'fenwick_against': "Fenwick: Against",
                        'fenwick_lvl': "Fenwick: Own - Against",
                        'pairwise_win': "Pairwise: Own",
                        'pairwise_win_against': "Pairwise: Against",
                        'rpi': "RPI$^{a}$: Own",
                        'rpi_against': "RPI$^{a}$: Against",
                        'idx_OppoBetter': "Index: Better opponent",
                        'idx_OppoGood': "Index: Expected win"
                        })
stargazer.add_line('Precision - Win', classify_results['precision_win'], LineLocation.FOOTER_TOP)
stargazer.add_line('Precision - Loss', classify_results['precision_loss'], LineLocation.FOOTER_TOP)
stargazer.add_line('Accuracy - Win', classify_results['recall_win'], LineLocation.FOOTER_TOP)
stargazer.add_line('Accuracy - Loss', classify_results['recall_loss'], LineLocation.FOOTER_TOP)
stargazer.add_line('F1 - Win', classify_results['f1_win'], LineLocation.FOOTER_TOP)
stargazer.add_line('F1 - Loss', classify_results['f1_loss'], LineLocation.FOOTER_TOP)
stargazer.add_line('Support/Count - Win', classify_results['support_win'], LineLocation.FOOTER_TOP)
stargazer.add_line('Support/Count - Loss', classify_results['support_loss'], LineLocation.FOOTER_TOP)
#stargazer.show_confidence_intervals(True)
stargazer.title('Table 1. Baseline predictor')
HTML(stargazer.render_html())

#with open("yourhtmlfile.html", "w") as file:
#    file.write(html_res)

# %%
# Real-time prediction

# set saving results for classifications
classify_results = {
    'precision_win':[],
    'precision_loss':[],
    'recall_win':[],
    'recall_loss':[],
    'f1_win':[],
    'f1_loss':[],
    'support_win':[],
    'support_loss':[]
}

x_2_1 = ['rpe', 'rpe_against', 'rgame']	
x_2_2 = ['rpe_10', 'rpe_10_oppo', 'rgame']
x_2_3 = ['rwp', 'rwp_oppo', 'rgame']
x_2_4 = ['rfenwick_for', 'rfenwick_against', 'rgame']
x_2_5 = ['rcorsi_for', 'rcorsi_against', 'rcorsi_pct', 'rgame']

# .....................................................................
# Data choice
x = dfregress[x_2_1]*1.0
# Fill NA with mean
x = x.fillna(0)
y = dfregress[ybool]

# Run and save logit regression
model_1, classify_results = win_prediction_eval(y, x, classify_results)

# .....................................................................
# Data choice
x = dfregress[x_2_2]*1.0
# Fill NA with mean
x = x.fillna(0)
y = dfregress[ybool]

# Run and save logit regression
model_2, classify_results = win_prediction_eval(y, x, classify_results)

# .....................................................................
# Data choice
x = dfregress[x_2_3]*100.0
# Fill NA with mean
x = x.fillna(0)
y = dfregress[ybool]

# Run and save logit regression
model_3, classify_results = win_prediction_eval(y, x, classify_results)

# .....................................................................
# Data choice
x = dfregress[x_2_4]*1.0
# Fill NA with mean
x = x.fillna(0)
y = dfregress[ybool]

# Run and save logit regression
model_4, classify_results = win_prediction_eval(y, x, classify_results)

# .....................................................................
# Data choice
x = dfregress[x_2_5]*1.0
# Fill NA with mean
x = x.fillna(0)
y = dfregress[ybool]

# Run and save logit regression
model_5, classify_results = win_prediction_eval(y, x, classify_results)

stargazer = Stargazer([model_1,model_2,model_3,model_4,model_5])
#stargazer.custom_columns(['', 'Post-hoc Predictor', '', '', '', ''], [1, 1, 1, 1, 1, 1])
stargazer.rename_covariates({'kpi_wp': 'Win Rate: Own', 
                        'kpi_wp_against': 'Win Rate: Against',
                        'kpi_wp_diff': 'Win Rate: Own - Against',
                        'rpe': "P. Expectation: Own",
                        'rpe_10': "P. Expectation: Own (Past 10 games)",
                        'rpe_against': 'P. Expectation: Against',
                        'rpe_10_oppo': "P. Expectation: Against (Past 10 games)",
                        'rcorsi_for': "Corsi: Own",
                        'rcorsi_against': "Corsi: Against",
                        'rcorsi_pct': "Corsi: Own - Against",
                        'rfenwick_for': "Fenwick: Own",
                        'rfenwick_against': "Fenwick: Against",
                        'rgame':'Count Games Played',
                        'rwp':'Win Rate Own',
                        'rwp_oppo':'Win Rate: Against'
                        })
stargazer.add_line('Precision - Win', classify_results['precision_win'], LineLocation.FOOTER_TOP)
stargazer.add_line('Precision - Loss', classify_results['precision_loss'], LineLocation.FOOTER_TOP)
stargazer.add_line('Accuracy - Win', classify_results['recall_win'], LineLocation.FOOTER_TOP)
stargazer.add_line('Accuracy - Loss', classify_results['recall_loss'], LineLocation.FOOTER_TOP)
stargazer.add_line('F1 - Win', classify_results['f1_win'], LineLocation.FOOTER_TOP)
stargazer.add_line('F1 - Loss', classify_results['f1_loss'], LineLocation.FOOTER_TOP)
stargazer.add_line('Support/Count - Win', classify_results['support_win'], LineLocation.FOOTER_TOP)
stargazer.add_line('Support/Count - Loss', classify_results['support_loss'], LineLocation.FOOTER_TOP)
#stargazer.show_confidence_intervals(True)
stargazer.title('Table 1. Baseline predictor')
HTML(stargazer.render_html())

#with open("yourhtmlfile.html", "w") as file:
#    file.write(html_res)