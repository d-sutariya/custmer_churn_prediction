#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import optuna
import mlflow
import warnings
import os
import uuid
import dagshub
import lightgbm as lgb
import mlflow.lightgbm
from IPython.display import FileLink,display,Image
from mlflow.tracking import MlflowClient
from pathlib import Path
from dotenv import load_dotenv

from sklearn.metrics import roc_auc_score,f1_score,accuracy_score,recall_score,precision_score
warnings.filterwarnings("ignore")


env_path = Path('.env')
load_dotenv(env_path)
root_dir = Path(os.getenv('ROOT_DIRECTORY'))


os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_TRACKING_PASSWORD")
os.environ['MLFLOW_TRACKING_URI'] = os.getenv("MLFLOW_TRACKING_URI")
os.environ['DAGSHUB_USER_TOKEN'] = os.getenv("DAGSHUB_USER_TOKEN")


dagshub.init(repo_name='mlflow_experiment_tracking',repo_owner='tnbmarketplace',mlflow=True)


transformed_featured_train_set = pd.read_csv(root_dir/'data'/'processed'/"transformed_featured_train_set.csv")
transformed_featured_val_set = pd.read_csv(root_dir/'data'/'processed'/"transformed_featured_val_set.csv")
transformed_featured_test_set = pd.read_csv(root_dir/'data'/'processed'/"transformed_featured_test_set.csv")
transformed_featured_final_train_set = pd.read_csv(root_dir/'data'/'processed'/"transformed_featured_final_train_set.csv")


# ### Optimization

mlflow.set_tracking_uri("https://dagshub.com/tnbmarketplace/mlflow_experiment_tracking.mlflow")


# **If you want to run trials make the below variable True.**

run_trials = False


def objective(trial):
    try:
        # Disable auto-logging for LightGBM in MLflow (manual logging is used)
        mlflow.lightgbm.autolog(disable=True)
        
        # Create a new MLflow run for this trial
        run = client.create_run(experiment.experiment_id)
        print(f"Starting trial {trial.number} with run ID: {run.info.run_id}")

        # Define hyperparameters for LightGBM, using Optuna suggestions
        param = {
            'objective': 'binary',
            'metric': 'auc',
            'verbose': -1,
            'boosting_type': 'gbdt',
            'scale_pos_weight': 3,
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-2, 1e-1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 150),
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'early_stopping_round': 200,
            'num_boost_rounds': 3000
        }

        # Log each hyperparameter to MLflow
        for key, value in param.items():
            client.log_param(run.info.run_id, key, value)

        # Prepare LightGBM datasets for training and validation
        dtrain = lgb.Dataset(transformed_featured_train_set.drop(columns='Churn'), label=transformed_featured_train_set['Churn'])
        dtest = lgb.Dataset(transformed_featured_val_set.drop(columns='Churn'), label=transformed_featured_val_set['Churn'])

        # Train the LightGBM model with the specified parameters
        model = lgb.train(param, dtrain, 3000, [dtest])

        # Make predictions and evaluate the model
        preds = model.predict(transformed_featured_val_set.drop(columns='Churn'))
        preds_digits = [0 if pred < 0.4 else 1 for pred in preds]
        y_true = transformed_featured_val_set['Churn']

        # Compute various performance metrics
        auc = roc_auc_score(y_true, preds)
        recall = recall_score(y_true, preds_digits)
        precision = precision_score(y_true, preds_digits)
        f1 = f1_score(y_true, preds_digits)
        weighted_recall = 0.6 * recall + 0.4 * f1
        accuracy = accuracy_score(y_true, preds_digits)

        # Log the performance metrics to MLflow
        metrics = {
            'auc': auc,
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'weighted_recall': weighted_recall,
            'accuracy': accuracy
        }

        for key, value in metrics.items():
            client.log_metric(run.info.run_id, key, value)

        # Create a directory to save the model for this trial
        trial_id = f"trial_{trial.number}"
        trial_dir = os.path.join('/kaggle/working/experiments', trial_id)
        os.makedirs(trial_dir, exist_ok=True)

        # Save the trained model to a file and log it to MLflow
        model_path = os.path.join(trial_dir, 'artifacts', 'model.txt')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save_model(model_path)
        
        trial.set_user_attr("model", model)
        
        client.log_artifact(run.info.run_id, model_path)

    except Exception as e:
        # Print error message and return a low score if an exception occurs
        print(f"Trial {trial.number} failed: {e}")
        return 0.0

    # Return the weighted recall as the objective value for Optuna
    return weighted_recall


# ### **Visualizing Experiments on MLflow UI**
# 
# 1. Copy your experiment name as indicated in below image. If you are not running trials you can use my experiment(name = my experiment).
# 
# 
# 

display(Image(root_dir/'reports'/'figures'/"Screenshot (154).png"))


# 2. Go to the by clicking here https://dagshub.com/tnbmarketplace/mlflow_experiment_tracking.mlflow 
# 
# 3. paste your experiment name in serach bar and search it then click on the your experiment as shown below image. If you want to run experiment done by me write my experiment in the name of the search bar and click on it.

display(Image(root_dir/'reports'/'figures'/"Screenshot (155).png"))


if run_trials:

    # Initialize MLflow Client
    client = MlflowClient()

    # Define your experiment (use an existing one or create a new one)
    experiment_name =  str(uuid.uuid4())
#     experiment_name ="my experiment"
    
    try:
        # Attempt to create a new experiment
        experiment_id = client.create_experiment(experiment_name)  # Create or use existing experiment
    except mlflow.exceptions.MlflowException as e:
        # If the experiment already exists, retrieve its ID
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    # Retrieve experiment details
    experiment = client.get_experiment(experiment_id)

    # Create an Optuna study for hyperparameter optimization
    study = optuna.create_study(direction='maximize')
    # Optimize the objective function over 100 trials
    study.optimize(objective, n_trials=100,n_jobs = -1) # change trials as per your wish
    
    # Prepare training and test datasets
    train_set = lgb.Dataset(transformed_featured_final_train_set.drop(columns='Churn'), label=transformed_featured_final_train_set['Churn'])
    test_set = lgb.Dataset(transformed_featured_test_set.drop(columns='Churn'), label=transformed_featured_test_set['Churn'])

    # Train the final model using the best parameters from Optuna
    final_model = lgb.train(study.best_params,
                           train_set,
                           num_boost_round=3000,
                           valid_sets=[test_set],
                           callbacks=[
                               lgb.early_stopping(stopping_rounds=200)
                           ])
    
    print("experiment name is",experiment_name)
    print("Save it somewhere to access experiments.")
else:
    # Skip trials and load the pre-existing best model
    print("Skipping trials. Loading pre-existing best model...")

    # Load the best model from a saved file
    best_model = lgb.Booster(model_file=root_dir/'models'/"model.txt")
    
    # Prepare training and test datasets
    train_set = lgb.Dataset(transformed_featured_final_train_set.drop(columns='Churn'), label=transformed_featured_final_train_set['Churn'])
    test_set = lgb.Dataset(transformed_featured_test_set.drop(columns='Churn'), label=transformed_featured_test_set['Churn'])

    # Train the final model using the pre-loaded best model
    final_model = lgb.train(best_model.params,
                           train_set,
                           num_boost_round=500,
                           valid_sets=[test_set],
                           init_model=best_model,
                           callbacks=[
                               lgb.early_stopping(stopping_rounds=50)
                           ])


# ### **Warning**:-
# **You should save experiment name to somwhere to access that perticular experiment Other wise once you Close this notebook You will not able to access it.**

# Predictions and metrics
preds = final_model.predict(transformed_featured_test_set.drop(columns='Churn'))
preds_digits = [0 if pred < 0.5 else 1 for pred in preds]
y_true = transformed_featured_test_set['Churn']

auc = roc_auc_score(y_true, preds)
recall = recall_score(y_true, preds_digits)
precision = precision_score(y_true, preds_digits)
f1 = f1_score(y_true, preds_digits)
weighted_recall = 0.6 * recall + 0.4 * f1
accuracy = accuracy_score(y_true, preds_digits)

metrics = {
    'auc': auc,
    'recall': recall,
    'precision': precision,
    'f1': f1,
    'weighted_recall': weighted_recall,
    'accuracy': accuracy
}

for key, value in metrics.items():
    print(f"{key}:-", value)

