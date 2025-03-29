import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

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
    lags = np.arange(1, max_lag + 1)

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


def select_features_decision_tree(
    sales_df, market_df, product_group,
    date_sales_column='DATE', date_market_column='date',
    sales_column='Sales_EUR', importance_threshold=0.01, max_depth=4):
    """
    Selects market features for a given product group using Decision Tree Regressor.
    
    Parameters:
        sales_df (pd.DataFrame): Aggregated sales data for the product group.
        market_df (pd.DataFrame): Market data with matching dates.
        product_group (int): Product group number for identification.
        date_column (str): Date column name (should match in both dataframes).
        sales_column (str): Name of the sales target column.
        importance_threshold (float): Minimum feature importance to be selected.
        max_depth (int): Max depth for the decision tree to control overfitting.

    Returns:
        list: Names of selected features above importance threshold.
    """
    # Set date as index temporarily
    sales_df = sales_df.set_index(date_sales_column)
    market_df = market_df.set_index(date_market_column)

    # Align datasets by date
    aligned_df = sales_df[[sales_column]].join(market_df, how='inner')

    # Drop rows with missing values
    aligned_df.dropna(inplace=True)

    # Split into features and target
    X = aligned_df.drop(columns=[sales_column])
    y = aligned_df[sales_column]

    # Fit decision tree
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=0)
    model.fit(X, y)

    # Get feature importances
    importances = pd.Series(model.feature_importances_, index=X.columns)
    selected_features = importances[importances >= importance_threshold].sort_values(ascending=False)

    print(f"\nSelected features for Product Group {product_group}:")
    print(selected_features)

    return selected_features.index.tolist()



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