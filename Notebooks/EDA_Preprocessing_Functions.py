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
    sns.histplot(sales_data, bins=30, kde=True, color='#BED62F')
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

def rolling_zscore_outlier_detection(df, column_name, window_size=12, z_threshold=3):
    """
    Function to detect and handle outliers using rolling Z-score, replacing them with interpolated values.

    Parameters:
    - df (pd.DataFrame): The dataset containing the time-series sales data.
    - column_name (str): The name of the column to apply the rolling Z-score on (e.g., 'Adjusted Sales (€)').
    - window_size (int): The rolling window size for calculating mean and standard deviation (default is 12 months).
    - z_threshold (float): The Z-score threshold for outlier detection (default is 3).
    
    Returns:
    - pd.DataFrame: The dataset with outliers replaced by interpolated values.
    """
    original_rows = df.shape[0]

    # Compute rolling mean and standard deviation for the middle section first
    df['Rolling_Mean'] = df[column_name].rolling(window=window_size, min_periods=1).mean()
    df['Rolling_Std'] = df[column_name].rolling(window=window_size, min_periods=1).std()

    # Initialize Rolling Z-score column
    df['Rolling_Zscore'] = np.nan

    # Compute rolling Z-score for the middle rows (where we have full rolling windows)
    middle_rows = df.index[window_size:-window_size]  # Middle section where full windows exist
    df.loc[middle_rows, 'Rolling_Zscore'] = (df.loc[middle_rows, column_name] - df.loc[middle_rows, 'Rolling_Mean']) / df.loc[middle_rows, 'Rolling_Std']

    # Handle first 12 and last 12 rows using global statistics
    global_mean = df[column_name].mean()
    global_std = df[column_name].std()
    
    first_rows = df.index[:window_size]
    last_rows = df.index[-window_size:]

    df.loc[first_rows, 'Rolling_Zscore'] = (df.loc[first_rows, column_name] - global_mean) / global_std
    df.loc[last_rows, 'Rolling_Zscore'] = (df.loc[last_rows, column_name] - global_mean) / global_std

    # Mark outliers (Z-score > 3 or < -3) and replace with NaN
    df.loc[df['Rolling_Zscore'].abs() > z_threshold, column_name] = np.nan  

    # Store new row count after outliers were removed
    new_rows = df[column_name].notna().sum()

    # Fill NaNs using linear interpolation
    df[column_name] = df[column_name].interpolate(method='linear')

    df.drop(columns=['Rolling_Mean', 'Rolling_Std', 'Rolling_Zscore'], inplace=True)
    print(f"{original_rows - new_rows} out of {original_rows} rows were considered outliers and adjusted.")

    return df