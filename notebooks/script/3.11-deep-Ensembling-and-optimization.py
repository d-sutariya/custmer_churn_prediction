#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join('..', 'src')))


import numpy as np
import pandas as pd
import optuna
import re
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,f1_score,precision_score
from optimization.ensemble_utils import *


transformed_featured_train_set = pd.read_csv(r"../data/processed/transformed_featured_train_set.csv")
transformed_featured_val_set = pd.read_csv(r"../data/processed/transformed_featured_val_set.csv")
transformed_featured_test_set = pd.read_csv(r"../data/processed/transformed_featured_test_set.csv")
transformed_featured_final_train_set = pd.read_csv(r"../data/processed/transformed_featured_final_train_set.csv")
lgb_featured_study = pd.read_csv(r"../reports/optemization-study-reports/lgb_featured_study.csv")
xgb_featured_study = pd.read_csv(r"../reports/optemization-study-reports/xgb_featured_study.csv")
catboost_featured_study = pd.read_csv(r"../reports/optemization-study-reports/catboost_featured_study.csv")
nn_featured_study = pd.read_csv(r"../reports/optemization-study-reports/nn_featured_study.csv")


# This notebook includes several functions that handle tasks such as dropping unnecessary columns from a DataFrame, removing json words from feature naems, generating predictions using multiple models, performing soft and weighted voting for ensemble methods, and cleaning hyperparameters. Additionally, it contains a function to create and compile a neural network model using TensorFlow/Keras. 

# For more details, refer to the [documentation](../docs/index.html) or explore the [source code](../src/optimization/ensemble_utils.py) of this functions.
# 

def objective(trial):
    X_train, y_train = transformed_featured_train_set.drop('Churn', axis=1), transformed_featured_train_set['Churn']  # Extract training features and labels
    X_val, y_val = transformed_featured_val_set.drop('Churn', axis=1), transformed_featured_val_set['Churn']  # Extract validation features and labels
    
    # Sample indices for hyperparameters from different models
    lgb_study = drop_unnessesary_columns(lgb_featured_study.copy())  # Prepare LightGBM study data
    xgb_study = drop_unnessesary_columns(xgb_featured_study.copy())  # Prepare XGBoost study data
    cat_study = drop_unnessesary_columns(catboost_featured_study.copy())  # Prepare CatBoost study data
    nn_study = drop_unnessesary_columns(nn_featured_study.copy())  # Prepare Neural Network study data
    
    lgb_idx = trial.suggest_int('lgb_idx', 0, len(lgb_study) - 1)  # Sample index for LightGBM hyperparameters
    xgb_idx = trial.suggest_int('xgb_idx', 0, len(xgb_study) - 1)  # Sample index for XGBoost hyperparameters
    cat_idx = trial.suggest_int('cat_idx', 0, len(cat_study) - 1)  # Sample index for CatBoost hyperparameters
    nn_idx = trial.suggest_int('nn_idx', 0, len(nn_study) - 1)  # Sample index for Neural Network hyperparameters
    
    # Fetch hyperparameters and clean them
    lgb_params = clean_hyperparameters(lgb_study.iloc[lgb_idx].to_dict())  # Get LightGBM parameters
    xgb_params = clean_hyperparameters(xgb_study.iloc[xgb_idx].to_dict())  # Get XGBoost parameters
    cat_params = clean_hyperparameters(cat_study.iloc[cat_idx].to_dict())  # Get CatBoost parameters
    nn_params = clean_hyperparameters(nn_study.iloc[nn_idx].to_dict())  # Get Neural Network parameters
    
    # Set additional fixed hyperparameters
    lgb_params['verbose'] = -1  # Silence LightGBM output
    xgb_params['verbose'] = 0  # Silence XGBoost output
    cat_params['early_stopping_rounds'] = 3000  # Set early stopping rounds for CatBoost
    cat_params['iterations'] = 200  # Set number of iterations for CatBoost

    # Weights for the voting classifier
    lgb_weight = trial.suggest_float('lgb_weight', 0.1, 1.0)  # Suggest weight for LightGBM
    xgb_weight = trial.suggest_float('xgb_weight', 0.1, 1.0)  # Suggest weight for XGBoost
    cat_weight = trial.suggest_float('cat_weight', 0.1, 1.0)  # Suggest weight for CatBoost
    nn_weight = trial.suggest_float('nn_weight', 0.1, 1.0)  # Suggest weight for Neural Network

    weights = {
        'lgb': lgb_weight,
        'xgb': xgb_weight,
        'cat': cat_weight,
        'nn': nn_weight
    }  # Store weights in a dictionary
    
    #Let’s give these models some workout time. First up, LightGBM. 
    #It's like the gym but for data—let’s see if it can lift those predictions high!
    
    # Train and predict with LightGBM
    lgb_train = lgb.Dataset(X_train, label=y_train)  # Prepare LightGBM dataset
    lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=100)  # Train LightGBM model
    lgb_preds = lgb_model.predict(X_val)  # Predict on validation set with LightGBM
    
    #Phew, LightGBM is done. Now let’s see if XGBoost can boost our mood... or just our predictions!"
    
    # Train and predict with XGBoost
    xgb_model = xgb.XGBClassifier(**xgb_params)  # Initialize XGBoost model with parameters
    xgb_model.fit(X_train, y_train)  # Train XGBoost model
    xgb_preds = xgb_model.predict_proba(X_val)[:, 1]  # Predict on validation set with XGBoost and get probabilities

    # Train and predict with CatBoost
    cat_model = cb.CatBoostClassifier(**cat_params, verbose=0)  # Initialize CatBoost model with parameters
    cat_model.fit(X_train, y_train)  # Train CatBoost model
    cat_preds = cat_model.predict_proba(X_val)[:, 1]  # Predict on validation set with CatBoost and get probabilities

    # Train and predict with Neural Network using TensorFlow/Keras
    nn_model = create_nn_model(nn_params, transformed_featured_train_set.shape[1] - 1)  # Create Neural Network model
    nn_model.fit(X_train, y_train, epochs=50, batch_size=int(nn_params['batch_size']), verbose=0)  # Train Neural Network model
    nn_preds = nn_model.predict(transformed_featured_val_set.drop('Churn', axis=1)).ravel()  # Predict on validation set with Neural Network

    # Combine predictions using weighted soft voting
    predictions = {
        'lgb': lgb_preds,
        'xgb': xgb_preds,
        'cat': cat_preds,
        'nn': nn_preds
    }  # Store predictions in a dictionary
    combined_preds = weighted_voting(predictions, weights)  # Perform weighted voting to combine predictions
    preds_digits = [1 if pred >= 0.4 else 0 for pred in combined_preds]  # Convert probabilities to binary predictions with a threshold of 0.4
    
    # Calculate evaluation metrics
    roc_auc = roc_auc_score(y_val, combined_preds)  # Calculate ROC AUC score
    f1 = f1_score(y_val, preds_digits)  # Calculate F1 score
    recall = recall_score(y_val, preds_digits)  # Calculate recall score
    accuracy = accuracy_score(y_val, preds_digits)  # Calculate accuracy score
    weighted_recall = 0.65 * recall + 0.35 * f1  # Calculate weighted recall combining recall and F1 score
    prec = precision_score(y_val, preds_digits)  # Calculate precision score
    
    # Store metrics as trial user attributes
    trial.set_user_attr('roc', roc_auc)  # Store ROC AUC score in the study
    trial.set_user_attr('f1', f1)  # Store F1 score in the study object
    trial.set_user_attr('accuracy', accuracy)  # Store accuracy score
    trial.set_user_attr('recall', recall)  # Store recall score
    trial.set_user_attr('precision', prec)  # Store precision score
    
    return weighted_recall  # Return weighted recall as the objective value for optimization


run_ensemble_trials = False


if run_ensemble_trials:
    # Create Optuna study and optimize
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=5 , n_jobs =-1)

    # Best trial
    best_trial = study.best_trial
    print(f'Best metric: {best_trial.value}')
    print('Best hyperparameters and weights:', best_trial.params)
    trials = study.trials

    # Extract trial data
    data = {
        'trial_number': [trial.number for trial in trials],
        'value': [trial.value for trial in trials],
        'params': [trial.params for trial in trials],
        'datetime_start': [trial.datetime_start for trial in trials],
        'datetime_complete': [trial.datetime_complete for trial in trials],
        'f1': [trial.user_attrs.get('f1', None) for trial in trials],
        'accuracy': [trial.user_attrs.get('accuracy', None) for trial in trials],
        'roc': [trial.user_attrs.get('roc', None) for trial in trials],
        'recall': [trial.user_attrs.get('recall', None) for trial in trials],
        'precision': [trial.user_attrs.get('precision', None) for trial in trials]
        
    }

    # Convert to DataFrame
    ensemble_results_df = pd.DataFrame(data)
    
else:
    ensemble_results_df = pd.read_csv(r"../reports/optemization-study-reports/ensemble_study.csv")
    ensemble_results_df = ensemble_results_df.drop(columns="Unnamed: 0")


# model's performance which has highest weighted recall
ensemble_results_df.iloc[ensemble_results_df['value'].idxmax()]


# model's performance which has highest recall
ensemble_results_df.iloc[ensemble_results_df['recall'].idxmax()]


# 

# model's performance which has highest precision
ensemble_results_df.iloc[ensemble_results_df['precision'].idxmax()]


ann_model = create_nn_model(clean_hyperparameters(nn_featured_study.iloc[87].to_dict()),input_shape=transformed_featured_train_set.drop(columns='Churn').shape[1])
ann_model.fit(transformed_featured_final_train_set.drop(columns='Churn'),transformed_featured_final_train_set['Churn'],epochs=20,validation_data=(transformed_featured_test_set.drop(columns='Churn'),transformed_featured_test_set['Churn']))


#let's see final model's performance
preds = ann_model.predict(transformed_featured_test_set.drop(columns='Churn'))
print('roc_auc score is:-',roc_auc_score(transformed_featured_test_set['Churn'],preds))
preds_digits = [1 if pred >= 0.4 else 0 for pred in preds]
print('recall score is :-',recall_score(transformed_featured_test_set['Churn'],preds_digits))
print('precision score is:-',precision_score(transformed_featured_test_set['Churn'],preds_digits))

