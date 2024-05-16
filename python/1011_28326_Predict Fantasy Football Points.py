import sqlalchemy
from sqlalchemy.orm import create_session
from sklearn import preprocessing
from collections import namedtuple
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='ticks', palette='Set2')
get_ipython().magic('matplotlib inline')

engine = sqlalchemy.create_engine('postgresql://localhost/fantasyfootball')    
session = create_session(bind=engine)

class Prediction:

    
    def __init__(self, test_proj_week, position='ALL'):
        self.train_proj_week = test_proj_week - 1
        self.test_proj_week = test_proj_week
        self.position = position


    def make_prediction(self):
        encoders = self.create_position_factors()
        
        self.train_data = self.week_df(self.train_proj_week, encoders.position,
                             encoders.team, self.position)
        test_data = self.week_df(self.test_proj_week, encoders.position, encoders.team, self.position)
        
        clf = RandomForestRegressor(n_estimators=5000, max_depth=5)
        clf.fit(self.train_data.X, self.train_data.y)
        model_predicted_points = clf.predict(test_data.X)
        
        results = self.rmean_sq(test_data.y.values, model_predicted_points)
        espn = self.rmean_sq(test_data.y.values, test_data.espn_proj.values)
        
        # Put some variables in self for easier access
        self.my_prediction = model_predicted_points
        self.model = clf
        self.results = results
        self.espn_results = espn
        self.actual = test_data.y.values
        self.espn_prediction = test_data.espn_proj.values
        
        self.get_combined_df(test_data)
      
    
    def get_combined_df(self, data):
        df=pd.concat([data.X, data.index, data.espn_proj, data.y], axis=1)
        df['name'] = df['index'].str.split("_").str.get(0)
        df['team'] = df['index'].str.split("_").str.get(1)
        df['position'] = df['index'].str.split("_").str.get(2)
        df['my_prediction'] = self.my_prediction
        self.combined_test_df = df
        
    
    def report(self):
        print("Prediction for Week {0} for {1} position(s)".format(self.test_proj_week, self.position))
        print("My RMSE: {}".format(self.results.rmse))
        print("ESPN RMSE: {}".format(self.espn_results.rmse))
        self.plot_feature_importance()
        plt.title("Feature Importance", fontsize=20)
        plt.show()
        self.plot_dist_comp()
        plt.title("Distribution of RMSE", fontsize=20)
        plt.show()
        self.scatter_plot(self.actual, self.my_prediction)
        plt.title("My Predictions", fontsize=20)
        plt.show()
        self.scatter_plot(self.actual, self.espn_prediction)
        plt.title("ESPN Predictions", fontsize=20)
        plt.show()
        
    
    def plot_feature_importance(self):
        plt.figure(figsize=(8,5))
        df = pd.DataFrame()
        df['fi'] = self.model.feature_importances_
        df['name'] = self.train_data.X.columns
        df = df.sort('fi')
        sns.barplot(x=df.name, y=df.fi)
        plt.xticks(rotation='vertical')
        sns.despine()
        
    
    def plot_dist_comp(self):
        plt.figure(figsize=(8,5))
        sns.distplot(self.results.array, label="Me")
        sns.distplot(self.espn_results.array, label="ESPN")
        plt.legend()
        sns.despine()
        
    
    def scatter_plot(self, x, y):
        plt.figure(figsize=(8,5))
        max_v = 40
        x_45 = np.arange(0, max_v)
        plt.scatter(x, y)
        plt.plot(x_45, x_45)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.ylim([0, max_v])
        plt.xlim([0, max_v])
    
    
    def week_df(self, proj_week, position_encoder, team_encoder, position = 'ALL'):
        
        # Get actual data for all previous weeks
        actual_data = pd.read_sql_query("""select name, team, position, opponent, week,
                                    at_home, total_points, won_game, opponent_score, team_score
                                    from scoring_leaders_weekly""", engine)
        
        # Calculate how team's perform on average against positions for fantasy points
        team_data = pd.DataFrame(actual_data.groupby(['opponent', 'position']).total_points.mean())
        team_data.reset_index(level=0, inplace=True)
        team_data.reset_index(level=0, inplace=True)
        team_data.rename(columns={'total_points': 'opponent_points'}, inplace=True)
        actual_data = actual_data.merge(team_data, on=['opponent', 'position'], how='left')
        team_data.rename(columns={'opponent_points': 'next_opponent_points', 'opponent': 'next_opponent'}, inplace=True)
        
        actual_data['index'] = actual_data.name + "_" + actual_data.team + "_" + actual_data.position
        actual_data.week = actual_data.week.astype(int)
        actual_data = actual_data[actual_data.week < proj_week]

        # Calculate the average values for previous week metrics
        wgt_df = actual_data[['opponent_points', 'index', 'at_home', 'total_points',
                              'won_game', 'opponent_score', 'team_score']]
        group_wgt_df = wgt_df.groupby('index')
        player_df = group_wgt_df.mean()
        player_df.reset_index(level=0, inplace=True)
        
        # Get the opponent data for the next week as well as espn projection
        predicted_data = pd.read_sql_query("""select name, team, position, opponent as next_opponent,
                                            at_home as next_at_home, total_points as predicted_points
                                            from next_week_projections
                                            where week = '{0}'""".format(proj_week), engine)
        predicted_data['index'] = predicted_data.name + "_" + predicted_data.team + "_" + predicted_data.position
        predicted_data.drop(['name', 'team'], axis=1, inplace=True)

        # Start combining everything - messy - sorry...
        X = player_df.merge(predicted_data, on='index', how='left')
        X = X.dropna()

        # Get the actual result as our target
        actual_result = pd.read_sql_query("""select name, team, position, total_points as actual_points
                                            from scoring_leaders_weekly
                                            where week = '{0}'""".format(proj_week), engine)
        actual_result['index'] = actual_result.name + "_" + actual_result.team + "_" + actual_result.position
        actual_result.drop(['name', 'team', 'position'], axis=1, inplace=True)

        X = X.merge(actual_result, on='index', how='left')
        X = X.merge(team_data, on=['position', 'next_opponent'], how='left')
        X = X.dropna()
        if position != 'ALL':
            X = X[X.position == position]
        y = X.actual_points

        X['team'] = X['index'].str.split("_").str.get(1)

        # Sklearn won't create factors for you, so encode the categories to integers
        X['team_factor'] = team_encoder.transform(X.team)
        if position == 'ALL':
            X['position_factor'] = position_encoder.transform(X.position)
        X['next_opponent_factor'] = team_encoder.transform(X.next_opponent)

        espn = X['predicted_points']
        index = X['index']
        X.drop(['predicted_points', 'actual_points', 'team', 'position', 'next_opponent', 'index'], axis=1, inplace=True)
        
        # Return named tuple of all the data I need
        week_tuple = namedtuple('Week', ['X', 'y', 'espn_proj', 'index'])
        return week_tuple(X, y, espn, index)
    
    
    def create_position_factors(self):
        # Convert positions into integer categories
        position_encoder = preprocessing.LabelEncoder()
        positions = np.ravel(pd.read_sql_query("""select distinct position from scoring_leaders_weekly;""", engine).values)
        position_encoder.fit(positions)

        # Convert team names into integer categories
        team_encoder = preprocessing.LabelEncoder()
        teams = np.ravel(pd.read_sql_query("""select distinct team from scoring_leaders_weekly;""", engine).values)
        team_encoder.fit(teams)
        encoders = namedtuple('encoders', ['team', 'position'])
        return encoders(team_encoder, position_encoder)
    
    
    def rmean_sq(self, y_true, y_pred):
        rmse = namedtuple('rmse', ['rmse', 'array'])
        sq_error = []
        assert len(y_true) == len(y_pred)
        for i in range(len(y_true)):
            sq_error.append((y_true[i] - y_pred[i])**2)
        return rmse(np.sqrt(np.mean(sq_error)), np.sqrt(sq_error))

week_3_proj_all = Prediction(3)
week_3_proj_all.make_prediction()
week_3_proj_all.report()

data = week_3_proj_all.combined_test_df
data['my_error'] = np.sqrt((data['my_prediction'] - data['actual_points'])**2)
data['espn_error'] = np.sqrt((data['predicted_points'] - data['actual_points'])**2)
data = data[['position', 'my_error', 'espn_error']]
data = pd.melt(data, id_vars=['position'], value_vars=['my_error', 'espn_error'])
data.columns = ['position', 'type', 'error']

sns.barplot(x='position', y='error', hue='type', data=data)
sns.despine()

