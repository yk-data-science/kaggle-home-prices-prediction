import pandas as pd
import os

def load_kaggle_data(input_dir):
    """
    Loads train.csv and test.csv from the specified Kaggle input directory.

    Args:
        input_dir (str): Path to the Kaggle input data directory.
                         Example: '/kaggle/input/home-data-for-ml-course/'

    Returns:
        tuple: (train_df, test_df) - Loaded training and test DataFrames.
    """
    train_path = os.path.join(input_dir, 'train.csv')
    test_path = os.path.join(input_dir, 'test.csv')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train data not found at: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test data not found at: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    print(f"Loaded train data from: {train_path}")
    print(f"Loaded test data from: {test_path}")

    return train_df, test_df