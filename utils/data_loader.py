import pandas as pd
import os

def load_kaggle_data(data_dir):
    """Load training and test data from Kaggle competition dataset."""
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    return train, test