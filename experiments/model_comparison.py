import pandas as pd
import numpy as np
import sklearn.metrics as skm
import sqlite3
import time
from config import DAILIES, DB_NAME

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from sklearn.ensemble import RandomForestRegressor

from lightgbm import LGBMRegressor

from sklearn.neural_network import MLPRegressor

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))
os.chdir(ROOT_DIR)

with sqlite3.connect(DB_NAME) as conn:
    query = "SELECT * FROM historical_weather WHERE city = 'Krakow' ORDER BY time ASC" 
    weather = pd.read_sql(query, conn)

weather['time'] = pd.to_datetime(weather['time'])

weather['temp_max_lag1'] = weather['temperature_2m_max'].shift(1)
weather['temp_max_lag2'] = weather['temperature_2m_max'].shift(2)
weather['temp_diff'] = weather['temp_max_lag1'] - weather['temp_max_lag2']

weather['temp_rolling_mean7'] = weather['temp_max_lag1'].rolling(window = 7).mean()

weather['pressure_lag1'] = weather['surface_pressure_mean'].shift(1)
weather['pressure_lag2'] = weather['surface_pressure_mean'].shift(2)
weather['pressure_diff'] = weather['pressure_lag1'] - weather['pressure_lag2']

weather['wind_lag1'] = weather['wind_speed_10m_max'].shift(1)
weather['precipit_lag1'] = weather['precipitation_sum'].shift(1)
weather['cloudcover_lag1'] = weather['cloudcover_mean' ].shift(1)
weather['humidity_lag1'] = weather['relative_humidity_2m_mean' ].shift(1)
weather['wind_dir_lag1'] = weather['winddirection_10m_dominant' ].shift(1)

days = weather['time'].dt.dayofyear
weather['sin_day'] = np.sin( 2 * np.pi * days / 365.25 )
weather['cos_day'] = np.cos( 2 * np.pi * days / 365.25 )

weather.dropna(inplace = True)

features_lin = ['temp_max_lag1',  
                 'temp_diff', 
                 'temp_rolling_mean7', 
                 'pressure_lag1', 
                 'pressure_diff', 
                 'wind_lag1', 
                 'precipit_lag1',
                 'sin_day',
                 'cloudcover_lag1', 
                 'humidity_lag1', 
                 'wind_dir_lag1',
                 'cos_day']

features_tree = features_lin.copy()
features_tree.extend(['temp_max_lag2', 'pressure_lag2'])

target = 'temperature_2m_max'
mask = weather['time'] < '2025-01-01'

X_lin_train = weather.loc[mask, features_lin]
X_lin_test = weather.loc[~mask, features_lin]

X_tree_train = weather.loc[mask, features_tree]
X_tree_test = weather.loc[~mask, features_tree]

y_train = weather.loc[mask, target]
y_test = weather.loc[~mask, target]

print("Lin trening:", X_lin_train.shape, "y trening:", y_train.shape)
print("Lin test:   ", X_lin_test.shape, "y test:   ", y_test.shape)
print("Tree trening:", X_tree_train.shape)
print("Tree test:   ", X_tree_test.shape)

baseline_pred = X_lin_test['temp_max_lag1'].copy()

same_temp_mae = skm.mean_absolute_error( y_test, baseline_pred)

# Ridge
pip_ridge = make_pipeline( StandardScaler(), Ridge() )
pip_ridge.fit( X_lin_train, y_train )

ridge_prediction = pip_ridge.predict( X_lin_test )
ridge_mae = skm.mean_absolute_error( y_test, ridge_prediction )

# Random Forrest Regressor
rfr = RandomForestRegressor( random_state = 42 ).fit( X_tree_train, y_train )

rfr_prediction = rfr.predict( X_tree_test )
rfr_mae = skm.mean_absolute_error( y_test, rfr_prediction )

# lightGBM
lgbm = LGBMRegressor( random_state = 42, verbose = -1).fit( X_tree_train, y_train )

lgbm_prediction = lgbm.predict( X_tree_test )
lgbm_mae = skm.mean_absolute_error( y_test, lgbm_prediction )

# MLPRegressor
pip_mlp = make_pipeline( StandardScaler(), MLPRegressor(hidden_layer_sizes = (64, 32), max_iter = 1000, random_state = 42) )
pip_mlp.fit( X_lin_train, y_train )

mlp_prediction = pip_mlp.predict( X_lin_test )
mlp_mae = skm.mean_absolute_error( y_test, mlp_prediction )

print( 'Same temperature as yesterday: ', same_temp_mae )
print( 'Ridge: ', ridge_mae )
print( 'Random forest: ', rfr_mae )
print( 'LightGbm: ', lgbm_mae )
print( 'MLP: ', mlp_mae )
