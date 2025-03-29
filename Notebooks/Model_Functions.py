import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import itertools
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from prophet import Prophet

# preparation

def parse_month_year(date_string):
    month_map = {
    "Jan": 1,
    "Feb": 2,
    "Mai": 3,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Okt": 10,
    "Nov": 11,
    "Dez": 12
    }

    month_abbr = date_string[:3]
    year_suffix = date_string[-2:]
    month_num = month_map.get(month_abbr)

    if not month_num:
        raise ValueError(f"Unknown month abbreviation: {month_abbr}")
    
    return datetime.strptime(f"{month_num} 01 {year_suffix}", "%m %d %y").date()

# create features 
def create_lag_features(df, max_lag):
    """
    Adds lag features for all original columns in the DataFrame from lag 1 to max_lag.

    Parameters:
    df (pd.DataFrame): Input DataFrame. Must contain a 'date' column.
    max_lag (int): Maximum lag value to generate features for.

    Returns:
    pd.DataFrame: DataFrame with new lag features.
    """
    df_lagging = df.copy()
    original_cols = df_lagging.columns.tolist()  # Store original column names

    for lag in range(1, max_lag + 1):
        for col in original_cols:
            df_lagging[f'{col}_lag_{lag}'] = df_lagging[col].shift(lag)
    return df_lagging

# Train_test_split

def train_val_split(df_target, df_market, target_column = "Sales_EUR" ,train_size = 0.8):

    """
    this function takes in two dataframes, one with the target variable and the other with the market data
    and returns the train and test sets for the target variable and the market data
    parameter:
        df_target: dataframe with the target variable
        df_market: dataframe with the market data
        train_size: the ratio of the train set to the entire dataset
    return:
        X_train: the train set for the market data
        X_test: the test set for the market data
        y_train: the train set for the target variable
        y_test: the test set for the target variable
    """
    df = pd.merge(df_target, df_market, left_index=True, right_index=True, how='inner')
    df.sort_index(inplace=True)
    split_index = int(len(df) * train_size)

    train = df.iloc[:split_index]
    val = df.iloc[split_index:]

    X_train = train.drop(target_column, axis=1)
    y_train = train[target_column]
    X_val = val.drop(target_column, axis=1)
    y_val = val[target_column]

    return X_train, X_val, y_train, y_val

## Model Functions

def model_evaluation_XGboost(X_train, y_train, X_val, y_val, param_grid):
    """
    Performs manual grid search using validation set (not cross-validation).
    
    Parameters:
    - X_train, y_train: training data
    - X_val, y_val: validation data
    - param_grid: dictionary of hyperparameters

    Returns:
    - best_model: trained model with best params
    - best_params: dict of best hyperparameters
    - best_rmse: RMSE score on validation set
    """
    keys, values = zip(*param_grid.items())
    best_rmse = float('inf')
    best_params = None
    best_model = None

    for param_combination in itertools.product(*values):
        params = dict(zip(keys, param_combination))
        
        model = XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            **params
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        print(f"Params: {params} => RMSE: {rmse:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
            best_model = model

    return best_model, best_params, best_rmse

def model_evaluation_prophet(X_train, y_train, X_val, y_val, param_grid):
    """
    Performs manual grid search using validation set (not cross-validation) for Prophet model.
    
    Parameters:
    - X_train: training data features (DataFrame)
    - y_train: training data target (Series)
    - X_val: validation data features (DataFrame)
    - y_val: validation data target (Series)
    - param_grid: dictionary of hyperparameters

    Returns:
    - best_model: trained model with best params
    - best_params: dict of best hyperparameters
    - best_rmse: RMSE score on validation set
    """
    # Combine X and y for Prophet
    train_df = pd.DataFrame({'ds': X_train.index, 'y': y_train.values})
    val_df = pd.DataFrame({'ds': X_val.index, 'y': y_val.values})

    keys, values = zip(*param_grid.items())
    best_rmse = float('inf')
    best_params = None
    best_model = None

    for param_combination in itertools.product(*values):
        params = dict(zip(keys, param_combination))
        
        model = Prophet(**params)
        
        model.fit(train_df)
        future = model.make_future_dataframe(periods=len(val_df), freq='D')
        forecast = model.predict(future)
        y_pred = forecast['yhat'][-len(val_df):].values
        y_true = val_df['y'].values
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        print(f"Params: {params} => RMSE: {rmse:.4f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
            best_model = model

    return best_model, best_params, best_rmse

## select best model 

def select_best_model(model_1, params_1, rmse_1, name_1,
                      model_2, params_2, rmse_2, name_2):
    """
    Compares two models based on RMSE and returns the best one.

    Args:
        model_1: First trained model
        params_1: Best parameters for model 1
        rmse_1: Validation RMSE for model 1
        name_1: Name of model 1 (e.g. 'Prophet')

        model_2: Second trained model
        params_2: Best parameters for model 2
        rmse_2: Validation RMSE for model 2
        name_2: Name of model 2 (e.g. 'XGBoost')

    Returns:
        best_model: The model with the lower RMSE
        best_params: Best parameters for the best model
        best_rmse: RMSE of the best model
        best_model_name: Name of the best model
    """
    if rmse_1 < rmse_2:
        return model_1, params_1, rmse_1, name_1
    else:
        return model_2, params_2, rmse_2, name_2