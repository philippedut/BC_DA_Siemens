import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import shapiro
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.api as sm

def get_sales_by_product_group(df_sales, product_group):
    sales_agg_group = df_sales[df_sales['Mapped_GCK'] == product_group].groupby('DATE')['Sales_EUR'].sum().reset_index()
    return sales_agg_group

"""
We performed the Shapiro-Wilk test for normality on the sales data. The hypothesis are:

-H0: The data is normally distributed.
-H1: The data is not normally distributed.
"""

def shapiro_wilk_test(sales_data):
    """
    Perform the Shapiro-Wilk test for normality on a given product group’s sales data.
    
    Args:
    sales_data (pd.Series): Time series sales data for a single product group.
    
    Returns:
    dict: Contains the Shapiro-Wilk statistic and p-value.
    """

    # Run Shapiro-Wilk test
    shapiro_results = shapiro(sales_data)

    # Store results
    result = {'Statistic': shapiro_results.statistic, 'p-value': shapiro_results.pvalue}

    # Print results
    print(f"Shapiro-Wilk Test for Normality:")
    print(f"Statistic: {shapiro_results.statistic:.6f}")
    print(f"p-value: {shapiro_results.pvalue:.6f}")

    # Interpretation
    if shapiro_results.pvalue > 0.05:
        print(f"\n✅ The p-value ({shapiro_results.pvalue:.6f}) is greater than 0.05.")
        print(f"   We fail to reject the null hypothesis, meaning the data appears normally distributed.\n")
    else:
        print(f"\n❌ The p-value ({shapiro_results.pvalue:.6f}) is less than 0.05.")
        print(f"   We reject the null hypothesis, meaning the data is not normally distributed.\n")

    return result

def plot_distribution(sales_data, product_group):
    """
    Plots the histogram and KDE for a product group's sales data, to visualize the distribution of the Product Groups that are not normally distributed.
    
    Args:
    sales_data (pd.Series): Sales data for a single product group.
    product_group (int): The product group identifier.
    """

    plt.figure(figsize=(8, 5))

    # Histogram & KDE
    sns.histplot(sales_data, bins=30, kde=True, color="blue")
    plt.title(f"Histogram & KDE - Product Group {product_group}")
    plt.xlabel("Sales (€)")
    plt.ylabel("Frequency")

    plt.show()

def adf_test(sales_data, product_group):
    """
    Perform the Augmented Dickey-Fuller (ADF) test for stationarity on a product group's sales data.

    Args:
    sales_data (pd.Series): Time series sales data for a single product group.
    product_group (int): The product group identifier.

    Returns:
    dict: Contains ADF Statistic, p-value, and critical values.
    """

    # Run ADF Test
    adf_result = adfuller(sales_data)

    # Store results
    result = {
        'ADF Statistic': adf_result[0],
        'p-value': adf_result[1],
        'Critical Values': adf_result[4]
    }

    # Print results in a structured way
    print(f"Augmented Dickey-Fuller (ADF) Test - Product Group {product_group}")
    print(f"ADF Statistic: {adf_result[0]:.6f}")
    print(f"p-value: {adf_result[1]:.6f}")
    print("Critical Values:")
    for key, value in adf_result[4].items():
        print(f"\t{key}: {value:.3f}")

    if adf_result[1] <= 0.05:
        print(f"\n✅ The p-value ({adf_result[1]:.6f}) is ≤ 0.05.")
        print("   We reject the null hypothesis, meaning the data is stationary.\n")
    else:
        print(f"\n❌ The p-value ({adf_result[1]:.6f}) is > 0.05.")
        print("   We fail to reject the null hypothesis, meaning the data is non-stationary.\n")

    return result

def seasonal_decomposition(sales_data, product_group, date_column, period=12):
    """
    Performs seasonal decomposition on a product group's sales data without modifying the original dataset.

    Parameters:
        sales_data (pd.DataFrame): DataFrame containing Date and Sales values.
        product_group (int): The product group ID for reference.
        date_column (str): The column name containing date values.
        period (int): The expected seasonality period (default=12 for monthly data).

    Returns:
        None (Displays the decomposition plots).
    """
    # Perform seasonal decomposition without modifying the original DataFrame
    decomposition = seasonal_decompose(sales_data.set_index(date_column)['Sales_EUR'], model="additive", period=period)

    # Extract components
    observed = decomposition.observed
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    resid = decomposition.resid

    # Plot decomposition with correct date-axis
    fig, axs = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

    axs[0].plot(observed, label="Observed", color="tab:blue")
    axs[0].legend()
    
    axs[1].plot(trend, label="Trend", color="tab:orange")
    axs[1].legend()
    
    axs[2].plot(seasonal, label="Seasonality", color="tab:green")
    axs[2].legend()
    
    axs[3].plot(resid, label="Residuals", color="tab:red")
    axs[3].legend()

    fig.suptitle(f"Seasonal Decomposition - Product Group {product_group}", fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_acf_pacf(sales_data, product_group, max_lag=12):
    """
    Plots Auto-Correlation (ACF) and Partial Auto-Correlation (PACF) for a product group's sales data.
    
    Parameters:
        sales_data (pd.DataFrame): DataFrame containing Date and Sales values.
        product_group (int): The product group ID for reference.
        max_lag (int): Number of lags to display in the plots.
        
    Returns:
        None (Displays ACF, PACF, and Lag Plot).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ACF Plot
    plot_acf(sales_data['Sales_EUR'], lags=max_lag, zero=False, ax=axes[0])
    axes[0].set_title(f'ACF - Product Group {product_group}')

    # PACF Plot
    plot_pacf(sales_data['Sales_EUR'], lags=max_lag, zero=False, ax=axes[1])
    axes[1].set_title(f'PACF - Product Group {product_group}')

    # Lag Plot (Optional but useful)
    pd.plotting.lag_plot(sales_data['Sales_EUR'], lag=1, ax=axes[2])
    axes[2].set_title(f'Lag Plot - Product Group {product_group}')

    plt.tight_layout()
    plt.show()

## featurre selection function 

def plot_cross_correlation(sales_df, market_df, product_group, date_sales_column='DATE', date_market_column='date', sales_column='Sales_EUR', max_lag=12, threshold=0.5):
    """
    Plots cross-correlation between sales of a product group and each market feature.

    Parameters:
        sales_df (pd.DataFrame): Aggregated sales data for one product group.
        market_df (pd.DataFrame): Market data for all time periods.
        product_group (int): Product group number (used for titles).
        date_column (str): Common date column name in both dataframes.
        sales_column (str): Column name containing sales values.
        max_lag (int): Max number of lags to test.
        threshold (float): Minimum absolute correlation to consider as relevant.

    Returns:
        Tuple: 
        dict: {feature: [correlations by lag]} for relevant features
        list: names of relevant features with correlation ≥ threshold"""
    # Temporarily set DATE as index
    sales_series = sales_df.set_index(date_sales_column)[sales_column]
    market_df_indexed = market_df.set_index(date_market_column)

    # Align market and sales by DATE
    aligned_market = market_df_indexed.loc[sales_series.index]
    lags = np.arange(0, max_lag + 1)

    relevant_correlations = {}
    relevant_features = []

    for feature in aligned_market.columns:
        series = aligned_market[feature]
        correlations = [sales_series.corr(series.shift(lag)) for lag in lags]
        max_corr = np.nanmax(np.abs(correlations))

        if max_corr >= threshold:
            relevant_correlations[feature] = correlations
            relevant_features.append(feature)

            # Plot
            plt.figure(figsize=(10, 4))
            plt.stem(lags, correlations, basefmt="b")
            plt.title(f"Cross-Correlation with '{feature}' - Product Group {product_group}")
            plt.xlabel("Lag (months)")
            plt.ylabel("Correlation")
            plt.grid(True)
            plt.show()

    print("Selected Features for Product Group:")
    print(relevant_features)

    return relevant_correlations, relevant_features

def select_features_via_autocorrelation(df_target, df_features, target_col, target_date_col='DATE', feature_date_col='date', max_lag=10, threshold=0.2):
    """
    Selects exogenous features based on their autocorrelation with the target variable.

    Parameters:
    df_target (pd.DataFrame): DataFrame containing the target variable (e.g. sales).
    df_features (pd.DataFrame): DataFrame containing the exogenous/market features.
    target_col (str): Name of the target column in df_target.
    target_date_col (str): Date column name in df_target.
    feature_date_col (str): Date column name in df_features.
    max_lag (int): Maximum lag to consider for autocorrelation.
    threshold (float): Minimum absolute ACF value to select a feature.

    Returns:
    dict: Selected features with a list of lags that passed the threshold.
    """

    # Convert date columns to datetime
    df_target[target_date_col] = pd.to_datetime(df_target[target_date_col])
    df_features[feature_date_col] = pd.to_datetime(df_features[feature_date_col])

    # Merge the two dataframes on the date columns
    df_combined = pd.merge(df_target[[target_date_col, target_col]], df_features, left_on=target_date_col, right_on=feature_date_col, how='inner')

    selected_features_with_lags = {}

    for col in df_features.columns:
        if col not in [feature_date_col]:
            series = df_combined[col].dropna()

            if not series.empty:
                acf_values = acf(series, nlags=max_lag, fft=True)

                relevant_lags = [lag for lag, value in enumerate(acf_values[1:], start=1) if abs(value) > threshold]

                if relevant_lags:
                    selected_features_with_lags[col] = relevant_lags

    print("Selected Features based on ACF:", selected_features_with_lags)
    print(f"number of selected features is {len(selected_features_with_lags)}")

    selected_features = list(selected_features_with_lags.keys())

    return selected_features_with_lags, selected_features

## model prep function 
def create_lag_features(df, targets_with_lags):
    """
    Adds lag features for multiple target columns, each with custom lag values.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    targets_with_lags (dict): Dictionary where keys are target column names
                              and values are lists of lag values.

    Returns:
    pd.DataFrame: DataFrame with new lag features.
    """
    df = df.copy()
    for target_col, lags in targets_with_lags.items():
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df