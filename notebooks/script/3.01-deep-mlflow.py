#!/usr/bin/env python
# coding: utf-8

# ### Series Introduction:
# 
# Welcome to the **Kaggle Customer Churn Master Series**, where we've journeyed through the essentials and beyond in predicting customer churn.If you are unaware about this series please take a look into first two notebooks . Throughout this series, we've explored data analysis, feature engineering, and model optimization to equip you with the tools and knowledge to tackle churn prediction with confidence. 
# 
# ### Notebook Introduction:
# 
# In **Notebook 3**, the final installment of this series, we bring everything together with a focus on model tracking and deployment. Using **MLflow** and **DagsHub**, we track experiments and visualize results to fine-tune our models to perfection. This notebook showcases the final steps in refining our predictions, ensuring that our churn model is both accurate and ready for real-world application.
# 

# # 🚨 **Warning: You're Missing Out!** 🚨 
# 
# If you've jumped straight into this notebook, you're skipping some crucial steps that could significantly impact your understanding of MLflow. In the previous notebook, we laid the foundation for everything we're about to do here.
# 
# 
# ### **Why the Previous Notebook Matters**
# 
# In the previous notebook, we covered foundational concepts related to **model optimization** and **feature engineering**. These are not just technical details—they are pivotal for managing machine learning workflows effectively.**```Feature Engineering is the most important phase of ML cycle. ```** Indeed, I have reproduced most of the code from the priviouse notebook.  Missing this part means missing out on critical techniques and the context necessary for fully grasping the advanced concepts we’ll explore here.
# 
# [**Go back to the previous notebook now!**](https://www.kaggle.com/code/deepsutariya/churn-prediction-featuretools-optuna-mastery)
# 
# 

# ### Navigation:
# 
# - **Previous Notebook:** [Churn Prediction Featuretools Optuna Mastery](https://www.kaggle.com/code/deepsutariya/churn-prediction-featuretools-optuna-mastery)
# 
# ---
# 
# ### Series Navigation:
# 
# - **First Notebook:** [Explore Churn Insights: Plotly EDA for Beginners](https://www.kaggle.com/code/deepsutariya/explore-churn-insights-plotly-eda-for-beginners)
# - **Second Notebook:** [Churn Prediction Featuretools Optuna Mastery](https://www.kaggle.com/code/deepsutariya/churn-prediction-featuretools-optuna-mastery)
# - **Third Notebook:** [Churn Modeling to Deployment: MLflow & DagsHub](https://www.kaggle.com/code/deepsutariya/churn-modeling-to-deployment-mlflow-DagsHub)
# 

import optuna
import mlflow
import warnings
import shutil
import re
import os
import zipfile
import subprocess
import uuid
import dagshub
import lightgbm as lgb
import mlflow.lightgbm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import featuretools as ft
import matplotlib.image as mpimg
from IPython.display import FileLink,display,Image
from mlflow.tracking import MlflowClient
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder,StandardScaler , OrdinalEncoder 
from sklearn.utils.class_weight import compute_class_weight
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score,f1_score,accuracy_score,recall_score,precision_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
warnings.filterwarnings("ignore")


os.environ['MLFLOW_TRACKING_USERNAME'] = 'tnbmarketplace'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '0d957e7b20c38643e8fd8de6d9d8e1de130caf90'
os.environ['MLFLOW_TRACKING_URI'] = 'https://dagshub.com/tnbmarketplace/mlflow_experiment_tracking.mlflow'
os.environ['DAGSHUB_USER_TOKEN'] = "fc957a0e9846b45be51bcea1a3ea28f7a3f236aa"
get_ipython().system('dagshub login --token "fc957a0e9846b45be51bcea1a3ea28f7a3f236aa"')


dagshub.init(repo_name='mlflow_experiment_tracking',repo_owner='tnbmarketplace',mlflow=True)


transformed_featured_train_set = pd.read_csv(r"../data/processed/transformed_featured_train_set.csv")
transformed_featured_val_set = pd.read_csv(r"../data/processed/transformed_featured_val_set.csv")
transformed_featured_test_set = pd.read_csv(r"../data/processed/transformed_featured_test_set.csv")
transformed_featured_final_train_set = pd.read_csv(r"../data/processed/transformed_featured_final_train_set.csv")


# ## Understanding the Feature Engineering Code
# 
# If you're new to feature engineering or need a quick refresher, here's a breakdown of the code provided. This code is part of a class designed to streamline the process of feature engineering using the **Featuretools** library.
# 
# ### **Class Overview**
# 
# The `Feature_Engineering` class is used to manage and transform datasets for machine learning tasks. It has several key components:
# 
# - **Initialization (`__init__`)**: 
#   - Sets up the training and test datasets.
#   - Initializes placeholders for the entity set and dataset names.
# 
# ### **Key Methods**
# 
# 1. **`Create_Entityset`**:
#    - **Purpose**: Creates an entity set, which is a data structure used by Featuretools to manage and transform data.
#    - **How It Works**:
#      - Adds the training and test datasets to the entity set.
#      - Allows the option to specify an index column if it exists, or creates a new one.
# 
# 2. **`__clean_feature_names`**:
#    - **Purpose**: Cleans up feature names by replacing special characters with underscores.
#    - **How It Works**: Uses regular expressions to ensure that column names are consistent and free from problematic characters.
# 
# 3. **`__remove_duplicate_columns`**:
#    - **Purpose**: Removes duplicate columns from a DataFrame.
#    - **How It Works**: Ensures that there are no repeated columns, which can occur during feature engineering.
# 
# 4. **`Generate_Features`**:
#    - **Purpose**: Generates new features for both the training and test datasets.
#    - **How It Works**:
#      - Uses Featuretools’ `dfs` (Deep Feature Synthesis) function to create new features.
#      - Allows customization of transformation and aggregation primitives.
#      - Cleans and aligns features to ensure consistency between training and test datasets.
#      - Optionally only returns feature names without generating actual feature values.
# 
# ### **What This Code Achieves**
# 
# - **Feature Engineering**: Transforms raw data into meaningful features that improve model performance.
# - **Data Alignment**: Ensures that training and test data are aligned, with consistent columns and no duplicates.
# - **Data Cleaning**: Addresses issues such as duplicate columns and special characters in feature names, ensuring cleaner and more reliable data.
# 
# ### **Why This Matters**
# 
# Understanding this code is crucial because feature engineering significantly impacts your model's performance. Well-engineered features can lead to better insights and more accurate predictions. Additionally, managing data consistency and cleanliness is fundamental to building robust machine learning models.
# 
# By mastering these techniques, you enhance your ability to handle real-world data effectively, making you a more skilled data scientist or machine learning engineer.
# 

# ### Optimization

# ## Why Model Tuning is Crucial and How Optuna Helps
# 
# In the world of machine learning, model tuning is a critical step that can significantly impact your model's performance. Whether you’re working on a startup project or a large-scale production system, finding the optimal hyperparameters can make the difference between a mediocre model and one that excels at solving real-world problems.
# 
# ### **The Importance of Model Tuning**
# 
# Model tuning involves adjusting various hyperparameters to find the best combination for your machine learning model. This process is essential because:
# 
# - **Enhanced Performance**: Proper tuning ensures that your model performs at its best, leading to better predictions and more accurate results.
# - **Efficient Use of Resources**: Fine-tuning your model can lead to more efficient use of computational resources, saving time and cost in the long run.
# - **Real-World Impact**: For companies and startups dealing with real-world data, optimized models can lead to actionable insights and improved decision-making.
# 
# ### **How Optuna Facilitates Effective Tuning**
# 
# **Optuna** is an open-source hyperparameter optimization framework that simplifies and automates the model tuning process. Here’s how it works:
# 
# - **Automatic Search**: Optuna employs advanced optimization algorithms to automatically search for the best hyperparameter values. This helps in discovering optimal configurations without extensive manual intervention.
# - **Efficient Trials**: By intelligently exploring different hyperparameter combinations, Optuna reduces the number of trials needed to find a high-performing model.
# - **Scalable Optimization**: Suitable for both small-scale experiments and large production systems, Optuna can handle complex tuning tasks with ease.
# 
# ### **The Power of Experiment Logging with MLflow**
# 
# Logging experiments is a crucial practice for tracking and managing machine learning experiments effectively. **MLflow** stands out as a powerful tool in this area, offering several benefits:
# 
# - **Comprehensive Tracking**: MLflow logs every aspect of your experiments, including hyperparameters, metrics, and artifacts. This comprehensive tracking allows you to revisit and compare past experiments easily.
# - **Seamless Integration**: With MLflow, you can integrate experiment tracking into your workflow effortlessly. This is especially useful for startups and companies that need to maintain a robust and reproducible process for model development.
# - **Production Readiness**: MLflow's capabilities extend to managing experiments in production environments, ensuring that your models are well-documented and easily accessible.
# 
# ### **How MLflow Can Help**
# 
# In the provided code, **MLflow** is used to manage experiments and log important metrics and model artifacts. Here’s a brief overview of the process:
# 
# 1. **Setup**: MLflow is configured with a tracking URI, directing where experiments will be logged.
#    
# 2. **Logging Parameters and Metrics**: During each trial of the hyperparameter optimization process, parameters and metrics are logged using `mlflow.log_param` and `mlflow.log_metric`. This allows you to monitor how different parameter choices affect model performance.
# 
# 3. **Model Artifacts**: The trained model is saved and logged as an artifact, making it easy to access and deploy later.
# 
# 4. **Experiment Management**: MLflow’s UI enables you to view and compare different experiments, providing insights into how various configurations impact model performance.
# 
# By leveraging Optuna for tuning and MLflow for logging, you ensure that your machine learning process is both efficient and well-documented. This combination of tools not only enhances model performance but also improves the overall manageability and reproducibility of your experiments.
# 
# ---
# 
# For a more detailed look at how these tools are implemented in practice, check out the code examples and explanations in the notebook!
# 

mlflow.set_tracking_uri("https://dagshub.com/tnbmarketplace/mlflow_experiment_tracking.mlflow")


# **If you want to run trials make the below variable True.**

run_trials = False


# ### Overview
# 
# The `objective` function is an integral part of the Optuna optimization process, designed to tune the hyperparameters of a LightGBM model. This function achieves the following:
# 
# 1. **Setup for MLflow**: Initializes MLflow logging to manually track experiment parameters and metrics, overriding LightGBM's automatic logging.
# 2. **Hyperparameter Definition**: Uses Optuna to suggest and define a range of hyperparameters for the LightGBM model. These include learning rate, number of leaves, and regularization parameters.
# 3. **Model Training**: Trains the LightGBM model with the suggested hyperparameters on the training dataset and evaluates its performance on a validation dataset.
# 4. **Evaluation and Metrics**: Calculates performance metrics such as AUC, recall, precision, F1 score, weighted recall, and accuracy, logging these metrics to MLflow.
# 5. **Artifact Logging**: Saves the trained model and logs it as an artifact in MLflow, along with the hyperparameters and metrics.
# 6. **Error Handling**: Catches and prints exceptions if any issues arise during the training or logging process.
# 
# This structured approach helps in systematically optimizing model performance while keeping track of all relevant details, making it suitable for production environments and real-world applications.
# 

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

display(Image(r"../reports/figures/Screenshot (154).png"))


# 2. Go to the by clicking here https://dagshub.com/tnbmarketplace/mlflow_experiment_tracking.mlflow 
# 
# 3. paste your experiment name in serach bar and search it then click on the your experiment as shown below image. If you want to run experiment done by me write my experiment in the name of the search bar and click on it.

display(Image(r"../reports/figures/Screenshot (155).png"))


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
    best_model = lgb.Booster(model_file=r"../models/model.txt")
    
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

# If you find this notebook helpful and inspiring, don’t forget to give it a thumbs up! 👍

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


#  ***Most importantly,*** You can deploy model from **DagsHub directly**. This notebook will become very long if i will do it here . Let me know in comment if you want ***Deployment's notebook*** as well.

# ✨ **Found this series helpful?** 
# 
# - **Show your appreciation!**: If this notebook brought you value, don't forget to give it an **Upvote**. It means the world to me!
# 
# 🎯 **Your feedback matters!** 
# 
# - **Comments, questions, or suggestions?**: Share your thoughts below! I’m here to help and learn together.
# 
# 🚀 **Explore the journey again!** 
# 
# - **Missed the earlier notebooks?**: Revisit the [EDA](https://www.kaggle.com/code/deepsutariya/explore-churn-insights-plotly-eda-for-beginners) or [Advanced Feature Engineering](https://www.kaggle.com/code/deepsutariya/churn-prediction-featuretools-optuna-mastery) notebooks to refresh your memory and sharpen your skills!
# 
