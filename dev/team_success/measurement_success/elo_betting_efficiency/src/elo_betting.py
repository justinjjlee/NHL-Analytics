'''
Data Process

'''

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pymc as pm

class DataProcessOdds:
    '''
        Object to process odds data
    '''
    
    def __init__(self):
        '''
        df - Data frame object of the betting odds data
            Can be from .csv or databricks df
        '''

    def line_to_odds(self, df, linetype: str): -> pd.DataFrame:
        '''
        Convert line to odds

        Also filter out 
        '''
        # Select odds type to transform
        df_odds = self.df[self.df['odds_description'] == linetype]
        # Convert line to odds (win probability), if line's negative then multiply by -1
        df_odds['odds_home'] = np.where(
            df_odds['home_odds_value'] < 0, 
            abs(df_odds['home_odds_value']) / (abs(df_odds['home_odds_value']) + 100), 
            100 / (df_odds['home_odds_value'] + 100)
        )
        df_odds['odds_away'] = np.where(
            df_odds['away_odds_value'] < 0, 
            abs(df_odds['away_odds_value']) / (abs(df_odds['away_odds_value']) + 100), 
            100 / (df_odds['away_odds_value'] + 100)
        )

        df_odds = df_odds[['game_id', 'homeTeam.abbrev', 'awayTeam.abbrev', 'home_odds_value', 'away_odds_value', 'odds_home', 'odds_away']]
        return df_odds

    def to3d(self):
        '''
        Create 3-dimension data
            rows of Teams, home/away (2), and number of games (if total of 82 games played, it would be 41)
        '''
        # Pivot to create 3D array: teams x home/away x time
        teams = pd.unique(self.df[['homeTeam.abbrev', 'awayTeam.abbrev']].values.ravel('K'))
        teams.sort()
        games = self.df.sort_values('game_id')
        odds_3d = np.full((len(teams), 2, games['game_id'].nunique()), np.nan)

        team_idx = {team: i for i, team in enumerate(teams)}
        game_ids = games['game_id'].unique()
        game_idx = {gid: i for i, gid in enumerate(game_ids)}

        for _, row in games.iterrows():
            g = game_idx[row['game_id']]
            h = team_idx[row['homeTeam.abbrev']]
            a = team_idx[row['awayTeam.abbrev']]
            odds_3d[h, 0, g] = row['odds_home']
            odds_3d[a, 1, g] = row['odds_away']

        return odds_3d, teams, game_ids



        # Create 3-dimension data
        df_3d = self.df.groupby(['homeTeam.abbrev', 'awayTeam.abbrev'])['odds_home', 'odds_away'].agg(list).reset_index()
        df_3d.columns = ['homeTeam', 'awayTeam', 'odds_home', 'odds_away']
        return df_3d

class DataProcessBoxScore:
    '''
        Object to process box score data
    '''
    
    def __init__(self, df):
        '''
        df - Data frame object of the box score data
        '''
        self.df = df

class elocalibration:
    '''
        Logit model for calibrating ELO model using PyMC
    '''

    def __init__(self, df):
        '''
        df - Data frame object of the betting odds data processed
        '''
        self.df = df
    
    def model(self):
        '''
        Create model for PyMC
        '''
        # Create model
        with pm.Model() as model:
            # Define priors
            intercept = pm.Normal('intercept', mu=0, sd=10)
            slope = pm.Normal('slope', mu=0, sd=10)
            # Define likelihood
            y = pm.Bernoulli('y', p=pm.math.sigmoid(intercept + slope * self.df['odds_home']), observed=self.df['odds_away'])
            # Inference
            trace = pm.sample(1000, tune=1000)


class logitpredictor:
    '''
        Logit-based model for predictor using PyTorch
    '''

    def __init__(self, df):
        '''
        df - Data frame object of the betting odds data processed
        '''
        self.df = df

    def dataformat(self):
        '''
        Format data for PyTorch
        '''
        # Convert to PyTorch tensor
        self.df = torch.tensor(self.df.values, dtype=torch.float32)
        # Split into features and labels
        self.X = self.df[:, :-1]
        self.y = self.df[:, -1]
        # Split into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = torch.utils.data.random_split(self.X, self.y, train_size=0.8, test_size=0.2)
        # Create PyTorch dataset
        self.dataset = torch.utils.data.TensorDataset(self.X_train, self.y_train)
        # Create PyTorch dataloader
        self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=32, shuffle=True)

    def model(self):
        '''
        Create model for PyTorch
        '''
        # Create model
        self.model = nn.Sequential(
            nn.Linear(2, 1),
            nn.Sigmoid()
        )
        # Create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        # Create loss function
        self.loss_fn = nn.BCEWithLogitsLoss()

    def train(self):
        '''
        Train model for PyTorch
        '''
        # Train model
        for epoch in range(100):
            for X_batch, y_batch in self.dataloader:
                # Zero gradients
                self.optimizer.zero_grad()
                # Forward pass
                y_pred = self.model(X_batch)
                # Compute loss
                loss = self.loss_fn(y_pred, y_batch)
                # Backward pass
                loss.backward()
                # Update weights
                self.optimizer.step()
        # Evaluate model
        y_pred = self.model(self.X_test)    
