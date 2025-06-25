import pandas as pd
from preprocess.missing import missing_info, impute_numeric, impute_categorical
from preprocess.outlier import remove_outliers_iqr
from preprocess.features import engineer_features
from preprocess.encoding import encode_features
from models.random_forest import train_rf, eval_rf
from models.xgboost_model import train_xgb, evaluate_model as eval_xgb
from utils.helpers import rmse_scorer
import numpy as np

# Load data
from utils.data_loader import load_kaggle_data
train, test = load_kaggle_data('data/home-data-for-ml-course/')

# Separate target
y = train['SalePrice']
X = train.drop(['SalePrice', 'Id'], axis=1)

# Detect types
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Missing
X = missing_info(X, drop_threshold=45, drop=True)
test = missing_info(test, drop_threshold=45, drop=True)
X_train, X_val, X_cal = np.split(X.sample(frac=1, random_state=42), [int(.7*len(X)), int(.85*len(X))])
y_train, y_val, y_cal = y.loc[X_train.index], y.loc[X_val.index], y.loc[X_cal.index]

numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

# Imputation
X_train, [X_val, X_cal, test] = impute_numeric(X_train, [X_val, X_cal, test], numeric_cols)
X_train, [X_val, X_cal, test] = impute_categorical(X_train, [X_val, X_cal, test], categorical_cols)

# Outliers
X_train = remove_outliers_iqr(X_train, numeric_cols)
y_train = y_train.loc[X_train.index]

# Feature engineering
X_train = engineer_features(X_train)
X_val = engineer_features(X_val)
X_cal = engineer_features(X_cal)
test = engineer_features(test)

# Add engineered features
categorical_cols += ["age_bin"]
numeric_cols += ["price_per_sqft", "age", "room_to_bath_ratio", "garage_score", "overall_score"]

# Encode
X_train_enc, X_val_enc, X_cal_enc, test_enc, feature_names = encode_features(
    X_train, X_val, X_cal, test, categorical_cols, numeric_cols
)

# Log transform
y_train_log = np.log1p(y_train)
y_val_log = np.log1p(y_val)

# Train & Evaluate
rf_model = train_rf(X_train_enc, y_train_log)
rf_mean_rmse, rf_std_rmse = eval_rf(X_train_enc, y_train_log)
print(f"RandomForest RMSE: {rf_mean_rmse:.2f} ± {rf_std_rmse:.2f}")

xgb_model = train_xgb(X_train_enc, y_train_log)
xgb_rmse, xgb_std = eval_xgb(xgb_model, X_train_enc, y_train_log, rmse_scorer)
print(f"XGBoost RMSE: {xgb_rmse:.2f} ± {xgb_std:.2f}")