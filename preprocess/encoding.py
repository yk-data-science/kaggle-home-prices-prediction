from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd

def encode_features(X_train, X_val, X_cal, test, categorical_cols, numeric_cols):
    """
    Encode categorical and numeric features using OneHotEncoder and StandardScaler.
    """
    encoder = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ('num', StandardScaler(), numeric_cols)
    ])
    X_train_encoded = encoder.fit_transform(X_train)
    X_val_encoded = encoder.transform(X_val)
    X_cal_encoded = encoder.transform(X_cal)
    test_encoded = encoder.transform(test)
    
    feature_names = encoder.get_feature_names_out()
    return (
        pd.DataFrame(X_train_encoded, columns=feature_names, index=X_train.index),
        pd.DataFrame(X_val_encoded, columns=feature_names, index=X_val.index),
        pd.DataFrame(X_cal_encoded, columns=feature_names, index=X_cal.index),
        pd.DataFrame(test_encoded, columns=feature_names, index=test.index),
        feature_names
    )