import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def missing_info(df, drop_threshold=50, drop=False, name="DataFrame"):
    """
    Display missing value information for a DataFrame.
    """
    missing_count = df.isnull().sum()
    missing_percent = 100 * missing_count / len(df)
    missing_df = pd.DataFrame({
        'Missing Count': missing_count,
        'Missing Percentage': missing_percent
    })
    missing_df = missing_df[missing_df['Missing Count'] > 0].sort_values(by='Missing Percentage', ascending=False)

    print(f"\n{name} missing values:")
    print(missing_df)

    if drop:
        cols_to_drop = missing_df[missing_df['Missing Percentage'] > drop_threshold].index
        print(f"\nDropping columns from {name} with > {drop_threshold}% missing values:")
        print(list(cols_to_drop))
        df = df.drop(columns=cols_to_drop)

    return df

def impute_numeric(X_train, other_sets, numeric_cols):
    """
    Impute missing numeric values using Iterative Imputer.
    """
    imputer = IterativeImputer(random_state=42)
    X_train[numeric_cols] = imputer.fit_transform(X_train[numeric_cols])
    for df in other_sets:
        df[numeric_cols] = imputer.transform(df[numeric_cols])
    return X_train, other_sets

def impute_categorical(X_train, other_sets, categorical_cols):
    """
    Impute missing categorical values using mode.
    """
    for col in categorical_cols:
        mode = X_train[col].mode()[0]
        X_train[col] = X_train[col].fillna(mode)
        for df in other_sets:
            df[col] = df[col].fillna(mode)
    return X_train, other_sets