import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import shapiro
from statsmodels.tsa.stattools import adfuller

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

## featurre selection function 
def select_features_via_autocorrelation(df, target_col, max_lag=10, threshold=0.2):
    """
    Selects features based on their autocorrelation with the target variable.

    Parameters:
    df (pd.DataFrame): Time series DataFrame.
    target_col (str): Target variable column name.
    max_lag (int): Maximum lag to consider for autocorrelation.
    threshold (float): Minimum absolute ACF value to select a feature.

    Returns:
    dict: Selected features with a list of lags that passed the threshold.
    """
    selected_features_with_lags = {}

    for col in df.columns:
        if col != target_col:
            acf_values = acf(df[col], nlags=max_lag, fft=True)

            # Check which lags exceed the threshold (excluding lag 0)
            relevant_lags = [lag for lag, value in enumerate(acf_values[1:], start=1) if abs(value) > threshold]

            if relevant_lags:
                selected_features_with_lags[col] = relevant_lags

    return selected_features_with_lags

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