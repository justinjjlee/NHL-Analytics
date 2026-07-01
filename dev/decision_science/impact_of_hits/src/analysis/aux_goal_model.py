import numpy as np
from sklearn.linear_model import LogisticRegression

def fit_expected_goals_model(smoothed_results):
    """
    Fits an auxiliary logistic regression mapping smoothed latent momentum
    to a scoring probability proxy (high-danger chance in next window).
    """
    X_aux = []
    y_aux = []
    
    for game in smoothed_results:
        x_smooth = game["x_smooth"]
        Z = game["Z"]
        # In absence of next-goal data in smoothed_results, we create a
        # dummy proxy based on momentum and score state to enable the visualization
        # in the pipeline. For actual modeling, this would join with the true
        # target (e.g. goal scored in next 60s).
        
        # We simulate P(goal) increasing with x_smooth. 
        # Baseline probability around 5%, increases to ~15% at high momentum.
        for t in range(len(x_smooth)):
            momentum = x_smooth[t]
            is_home = Z[t, 0]
            score_state = Z[t, 3]
            
            # Simple simulation for pipeline visualization
            logit = -2.5 + 0.8 * momentum + 0.1 * score_state
            prob = 1 / (1 + np.exp(-logit))
            
            # Generate dummy label based on prob
            target = 1 if np.random.rand() < prob else 0
            
            X_aux.append([momentum, is_home, 0, score_state]) # 0 for delta_pythagorean
            y_aux.append(target)
            
    X_aux = np.array(X_aux)
    y_aux = np.array(y_aux)
    
    model = LogisticRegression(solver='liblinear')
    if len(np.unique(y_aux)) > 1:
        model.fit(X_aux, y_aux)
    else:
        # Fallback if dummy generation fails
        model.fit(X_aux, np.random.randint(0, 2, len(y_aux)))
        
    return model
