


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
from pathlib import Path
from dotenv import load_dotenv

from features.generate_and_transform_features import FeatureTransformer
from optimization.model_optimizer import ModelOptimizer,save_results,load_results


# Suppress all warnings for a cleaner output
# Set seed for numpy and tensorflow for reproducibility
warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.CRITICAL)
np.random.seed(42)
tf.random.set_seed(42)

env_path = Path('.env')
load_dotenv(env_path)

root_dir = Path(os.getenv('ROOT_DIRECTORY'))


# # Training and Optimization

#  let's optimize the models using the engineered dataset and oiginal(without engineered)
#  I am not runnig the below code as i have already optimized and saved the result if you want to run the code please change the variables **run_featured_trials,run_org_trials** to ***True***

# ### ***Warning*** :- It may take few  Hours to run trials.

run_featured_trials = True
run_org_trials = True


transformed_featured_train_set = pd.read_csv(root_dir/'data'/'processed'/"transformed_featured_train_set.csv")
transformed_featured_test_set = pd.read_csv(root_dir/'data'/'processed'/"transformed_featured_test_set.csv")
transformed_featured_final_train_set = pd.read_csv(root_dir/'data'/'processed'/"transformed_featured_final_train_set.csv")
transformed_featured_smoted_train_set = pd.read_csv(root_dir/'data'/'processed'/"transformed_featured_smoted_train_set.csv")
transformed_featured_val_set = pd.read_csv(root_dir/'data'/'processed'/"transformed_featured_val_set.csv")
val_set = pd.read_csv(root_dir/'data'/'interim'/"val_set.csv")
train_set_splitted = pd.read_csv(root_dir/'data'/'interim'/"train_set_splitted.csv")


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
