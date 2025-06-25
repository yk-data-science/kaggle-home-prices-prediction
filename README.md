# House Price Prediction Project

This project is based on the Kaggle competition [Housing Prices Competition for Kaggle Learn Users](https://www.kaggle.com/competitions/home-data-for-ml-course). The goal is to predict house prices using machine learning methods.

The main purpose of this project is to practice ML fundamentals, organise code properly with folder structure, and build reproducible pipelines. The project currently includes data preprocessing, exploratory data analysis (EDA), and model building using random forest and XGBoost.

**Future work:** Grid search and hyperparameter tuning will be implemented to improve model performance. Additionally, the notebook will be split into smaller, modular files for better maintainability.

## Project structure

- .gitignore  
- main.py  
- README.md  
- requirements.txt   
- data  
  - home-data-for-ml-course  
    - data_description.txt  
    - sample_submission.csv  
    - sample_submission.csv.gz  
    - test.csv  
    - test.csv.gz  
    - train.csv  
    - train.csv.gz  
- models  
  - evaluate.py  
  - random_forest.py  
  - xgboost_model.py  
  - __init__.py  
- notebooks  
  - 01_eda_and_initial_analysis.ipynb  
- preprocess  
  - encoding.py  
  - features.py  
  - missing.py  
  - outlier.py  
  - __init__.py  
- utils  
  - data_loader.py  
  - helpers.py  
  - __init__.py  

## Notes  
- The notebook currently includes all steps but will be split into smaller modules for clarity.  
- Grid search and hyperparameter tuning are planned future steps.  
