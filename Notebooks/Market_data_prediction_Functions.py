import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_squared_error
## grid search 
def grid_search_prophet_for_feature(series, forecast_horizon=10):
    """
    Performs grid search for one time series (a market feature).
    Splits the series into training (all except the last forecast_horizon points)
    and testing (the last forecast_horizon points), fits Prophet models with different
    hyperparameters, and returns the best combination based on RMSE.
    
    Parameters:
        series (pd.Series): The time series data (with datetime index).
        forecast_horizon (int): Number of future periods to hold out for testing.
        
    Returns:
        best_params (dict): The best hyperparameter combination.
        best_rmse (float): The RMSE for the best combination.
        results (list): A list of dicts with all combinations and their RMSE.
    """
    series = series.dropna()
    if len(series) < forecast_horizon + 2:
        raise ValueError("Not enough data points for grid search.")
    
    # Split into training and testing sets
    train = series.iloc[:-forecast_horizon]
    test = series.iloc[-forecast_horizon:]
    
    # Prepare training data for Prophet
    df_train = pd.DataFrame({'ds': train.index, 'y': train.values})
    
    best_rmse = float('inf')
    best_params = None
    results = []
    
    # Define hyperparameter grid
    seasonality_modes = ['multiplicative', 'additive']
    cps_values = [0.2, 0.5, 1.0, 2.0, 5.0]
    
    for seasonality_mode in seasonality_modes:
        for cps in cps_values:
            try:
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=False,
                    daily_seasonality=False,
                    seasonality_mode=seasonality_mode,
                    changepoint_prior_scale=cps
                )
                model.fit(df_train)
                
                # Forecast for the holdout period
                future = model.make_future_dataframe(periods=forecast_horizon, freq='MS')
                forecast = model.predict(future)
                
                # Extract predictions for test period
                forecast_test = forecast.set_index('ds').loc[test.index, 'yhat']
                rmse = np.sqrt(mean_squared_error(test, forecast_test))
                
                results.append({
                    'seasonality_mode': seasonality_mode,
                    'changepoint_prior_scale': cps,
                    'rmse': rmse
                })
                
                # Update best hyperparameters if RMSE is lower
                if rmse < best_rmse:
                    best_rmse = rmse
                    best_params = {'seasonality_mode': seasonality_mode, 'changepoint_prior_scale': cps}
            except Exception as e:
                print(f"Error for parameters (seasonality_mode={seasonality_mode}, cps={cps}): {e}")
                continue
                
    return best_params, best_rmse, results

def grid_search_all_features(market_df, date_column='date', forecast_horizon=10):
    """
    Performs grid search for every feature (all columns except the date column)
    in market_df.
    
    Parameters:
        market_df (pd.DataFrame): DataFrame containing market data.
        date_column (str): Name of the column containing dates.
        forecast_horizon (int): Number of future periods to hold out for testing.
    
    Returns:
        dict: Mapping of each feature to its best hyperparameters, RMSE, and grid search details.
    """
    results_dict = {}
    df = market_df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.set_index(date_column).asfreq("MS")
    
    for feature in df.columns:
        series = df[feature]
        try:
            best_params, best_rmse, param_results = grid_search_prophet_for_feature(series, forecast_horizon)
            results_dict[feature] = {
                'best_params': best_params,
                'best_rmse': best_rmse,
                'param_results': param_results
            }
            print(f"Feature: {feature} | Best Params: {best_params} | RMSE: {best_rmse:.4f}")
        except Exception as e:
            print(f"Error processing feature '{feature}': {e}")
            results_dict[feature] = None
    return results_dict

## make the pediction 
def specific_single_forecast_market_features_prophet(market_df, best_params_dict, date_column='date', forecast_horizon=10, plot=True):
    """
    Forecasts every feature in market_df for the next forecast_horizon periods using Prophet.
    For each feature, the model uses the best hyperparameters provided in best_params_dict.
    
    Parameters:
        market_df (pd.DataFrame): Market data with a datetime column and numeric features.
        best_params_dict (dict): Dictionary mapping each feature to its grid search results.
            Expected format: { feature_name: {'best_params': { 'seasonality_mode': str, 
                                                               'changepoint_prior_scale': float },
                                               'best_rmse': float,
                                               'param_results': [...] } }
        date_column (str): Name of the datetime column.
        forecast_horizon (int): Number of future periods to forecast.
        plot (bool): Whether to plot the forecast for each feature.
    
    Returns:
        dict: Mapping of each feature to a DataFrame of forecasted values.
              Each DataFrame has an index of dates and a 'yhat' column.
    """
    forecasts = {}
    
    # Prepare the DataFrame: convert the date column to datetime and set it as index with monthly frequency.
    df = market_df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.set_index(date_column).asfreq("MS")
    
    for feature in df.columns:
        series = df[feature].dropna()
        
        # Prepare the data for Prophet
        df_prophet = pd.DataFrame({'ds': series.index, 'y': series.values})
        
        # Get best hyperparameters for the feature; use defaults if not found.
        if feature in best_params_dict and best_params_dict[feature] is not None and best_params_dict[feature]['best_params'] is not None:
            best_params = best_params_dict[feature]['best_params']
        else:
            best_params = {'seasonality_mode': 'multiplicative', 'changepoint_prior_scale': 0.5}
        
        # Initialize and fit the Prophet model with the best hyperparameters.
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode=best_params['seasonality_mode'],
            changepoint_prior_scale=best_params['changepoint_prior_scale']
        )
        model.fit(df_prophet)
        
        # Create a future DataFrame and generate forecasts.
        future = model.make_future_dataframe(periods=forecast_horizon, freq='MS')
        forecast = model.predict(future)
        
        # Keep only the forecasted part for the specified horizon.
        forecast_feature = forecast[['ds', 'yhat']].set_index('ds').iloc[-forecast_horizon:]
        forecasts[feature] = forecast_feature
        
        # Optionally, plot the historical and forecasted values.
        if plot:
            plt.figure(figsize=(10, 4))
            plt.plot(df_prophet['ds'], df_prophet['y'], label='History')
            plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', linestyle='--')
            plt.title(f"Forecast for '{feature}' using best hyperparameters")
            plt.xlabel("Date")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
    
    return forecasts

## extract the prediction 
def append_forecasts_to_market_df(market_df, forecasts, date_column='date'):
    """
    Appends 10-month forecasts to market_df using forecast results per feature.

    Parameters:
        market_df (pd.DataFrame): Original market data.
        forecasts (dict): Dictionary of {feature: pd.Series or pd.DataFrame} with date index.
        date_column (str): Name of the datetime column in original market_df.

    Returns:
        pd.DataFrame: market_df extended with forecasted rows.
    """
    # Make sure the original date column is datetime and index is set
    df = market_df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    df = df.set_index(date_column).asfreq("MS")

    # Create empty DataFrame for forecasts
    forecast_df = pd.DataFrame(index=next(iter(forecasts.values())).index)

    for feature, forecast_series in forecasts.items():
        if isinstance(forecast_series, pd.DataFrame):
            forecast_df[feature] = forecast_series['yhat']
        else:
            forecast_df[feature] = forecast_series

    # Combine original + forecasted data
    extended_df = pd.concat([df, forecast_df])
    extended_df = extended_df.reset_index().rename(columns={'index': date_column})

    return extended_df