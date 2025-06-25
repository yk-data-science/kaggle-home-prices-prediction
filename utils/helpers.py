from sklearn.metrics import mean_squared_error, make_scorer
import numpy as np

def rmse_exp(y_true, y_pred_log):
    """
    Calculate the root mean squared error (RMSE) after applying the exponential transformation
    """
    y_true_exp = np.expm1(y_true)
    y_pred_exp = np.expm1(y_pred_log)
    return np.sqrt(mean_squared_error(y_true_exp, y_pred_exp))

rmse_scorer = make_scorer(rmse_exp, greater_is_better=False)