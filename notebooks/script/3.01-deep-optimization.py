#!/usr/bin/env python
# coding: utf-8

# ## Variable Descriptions Guide
# 
# - **encd_df**: This is the one-hot encoded dataframe used for model training.
# - **val_set**: The validation set used to validate the performance of the models during training.
# - **train_set_splitted**: The remaining part of the training set after splitting out the validation set.
# - **train_set**: The final training set used for training the models.
# - **test_set**: The final test set used to evaluate the performance of the trained models.
# - **X_train_smoted, y_train_smoted**: The training sets after applying SMOTE to handle class imbalance.
# 
# 
# ### Transformed Sets
# - **transformed_train_set, transformed_test_set**: These are the transformed training and validation sets without feature engineering.
# i.e transformed datasets on one hot encoded dataframe.
# - **transformed_featured_train_set, transformed_featured_val_set**: These are the transformed training and validation sets after feature engineering and transformation.
# - **transformed_featured_final_train_set, transformed_featured_test_set**: The transformed training set (without splitting) and the test set.
# - **transformed_featured_smoted_train_set, transformed_featured_smoted_test_set**: The transformed SMOTEd training and test sets.
# 
# ### Optimization
# 
# - **featured_lgb_study, featured_xgb_study, featured_cat_study, featured_ann_study**: These are the optimized studies of the models on the feature-engineered sets.
# - **org_lgb_study, org_xgb_study, org_cat_study, org_nn_study**: These are the optimized studies of the models on the original sets (i.e., without feature engineering).
# 

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join('..', 'src')))


# This cell imports all the necessary libraries and modules required 
import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
import warnings

from features.generate_and_transform_features import FeatureTransformer
from optimization.model_optimizer import ModelOptimizer,save_results,load_results


# Suppress all warnings for a cleaner output
# Set seed for numpy and tensorflow for reproducibility
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.CRITICAL)
np.random.seed(42)
tf.random.set_seed(42)


# # Training and Optimization

#  let's optimize the models using the engineered dataset and oiginal(without engineered)
#  I am not runnig the below code as i have already optimized and saved the result if you want to run the code please change the variables **run_featured_trials,run_org_trials** to ***True***

# ### ***Warning*** :- It may take few  Hours to run trials.

run_featured_trials = False
run_org_trials = False


transformed_featured_train_set = pd.read_csv(r"../data/processed/transformed_featured_train_set.csv")
transformed_featured_test_set = pd.read_csv(r"../data/processed/transformed_featured_test_set.csv")
transformed_featured_final_train_set = pd.read_csv(r"../data/processed/transformed_featured_final_train_set.csv")
transformed_featured_smoted_train_set = pd.read_csv(r"../data/processed/transformed_featured_smoted_train_set.csv")
transformed_featured_val_set = pd.read_csv(r"../data/processed/transformed_featured_val_set.csv")
val_set = pd.read_csv(r"../data/interim/val_set.csv")
train_set_splitted = pd.read_csv(r"../data/interim/train_set_splitted.csv")


if run_featured_trials:
    optimizer = ModelOptimizer(transformed_featured_train_set,transformed_featured_val_set)
    featured_xgb_study = optimizer.optimize_xgb()
    featured_cat_study = optimizer.optimize_catboost()
    featured_lgb_study = optimizer.optimize_lgb()
    featured_nn_study = optimizer.optimize_nn()
    
    save_results([featured_lgb_study,featured_xgb_study,featured_cat_study,featured_nn_study],[r"../reports/optemization-study-reports/lgb_featured_study.csv",r"../reports/optemization-study-reports/xgb_featured_study.csv",r"../reports/optemization-study-reports/catboost_featured_study.csv",r"../reports/optemization-study-reports/nn_featured_study.csv"])
else:
    # loading the study which was done using featured dataset
    featured_lgb_study , featured_xgb_study , featured_cat_study ,featured_nn_study = load_results([r"../reports/optemization-study-reports/lgb_featured_study.csv",r"../reports/optemization-study-reports/xgb_featured_study.csv",r"../reports/optemization-study-reports/catboost_featured_study.csv",r"../reports/optemization-study-reports/nn_featured_study.csv"])


# The `ModelOptimizer` class is designed for optimizing machine learning models using various algorithms like CatBoost, LightGBM, XGBoost, and neural networks. The optimization is performed using the Optuna framework, focusing on maximizing a custom metric called weighted recall. This metric is a combination of recall and F1 score, providing a balanced evaluation of model performance. The class also logs additional evaluation metrics like accuracy, precision, recall, F1 score, and ROC AUC to give a comprehensive view of model effectiveness.
# 

# For more detailed information, refer to the [documentation](../docs/ModelOptimizer.md) or check out the [source code](../src/optimization/model_optimizer.py).

if run_org_trials:
    transfm  = FeatureTransformer(train_set_splitted,val_set)
    transformed_train_set,transformed_test_set = transfm.transform()
    org_optimizer = ModelOptimizer(transformed_featured_train_set,transformed_featured_val_set)
    org_xgb_study = org_optimizer.optimize_xgb()
    org_cat_study = org_optimizer.optimize_catboost()
    org_lgb_study = org_optimizer.optimize_lgb()
    org_nn_study = org_optimizer.optimize_nn()

    # save studies to reports directory
    save_results([org_lgb_study,org_xgb_study,org_cat_study,org_nn_study],[r"../reports/optemization-study-reports/lgb_org_study.csv",r"../reports/optemization-study-reports/xgb_org_study.csv",r"../reports/optemization-study-reports/catboost_org_study.csv",r"../reports/optemization-study-reports/nn_org_study.csv"])

else:
    # loading the study which was done using original dataset
    org_lgb_study , org_xgb_study,org_cat_study,org_nn_study = load_results([r"../reports/optemization-study-reports/lgb_org_study.csv",r"../reports/optemization-study-reports/xgb_org_study.csv",r"../reports/optemization-study-reports/catboost_org_study.csv",r"../reports/optemization-study-reports/nn_org_study.csv"])


featured_lgb_study.iloc[featured_lgb_study['user_attrs_recall'].idxmax()]


featured_nn_study.iloc[featured_nn_study['user_attrs_recall'].idxmax()]


featured_cat_study.iloc[featured_cat_study['user_attrs_recall'].idxmax()]


featured_xgb_study.iloc[featured_xgb_study['user_attrs_recall'].idxmax()]


# **Results are imporved**.You can explore more as per your wish.
