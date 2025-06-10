import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib # To save/load preprocessor objects

def separate_target(df, target_column='SalePrice'):
    """
    Separates the target variable from the DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        target_column (str): The name of the target variable column.

    Returns:
        tuple: (features_df, target_series) - DataFrame of features and Series of target.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    target_series = df[target_column]
    features_df = df.drop(columns=[target_column])
    print(f"Separated target column '{target_column}'.")
    return features_df, target_series

def identify_column_types(df):
    """
    Identifies numeric and categorical columns within a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to analyze.

    Returns:
        tuple: (numeric_cols, categorical_cols) - Lists of numeric and categorical column names.
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    print(f"Identified {len(numeric_cols)} numeric and {len(categorical_cols)} categorical columns.")
    return numeric_cols, categorical_cols