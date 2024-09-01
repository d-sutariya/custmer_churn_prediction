import numpy as np
import pandas as pd
import lightgbm as lgb
import os
from dotenv import load_dotenv
from pathlib import Path


env_path = Path('.env')
load_dotenv(dotenv_path = env_path)
root_dir = Path(os.getenv('ROOT_DIRECTORY'))
model_path =root_dir/'models'/'model.txt'
train_df = pd.read_csv(root_dir/'data'/'processed'/'transformed_featured_final_train_set.csv')
test_df = pd.read_csv(root_dir/'data'/'processed'/'transformed_featured_test_set.csv')

best_model = lgb.Booster(model_file=model_path)
train_set = lgb.Dataset(train_df.drop(columns='Churn'),train_df['Churn'],params={'verbose':-1})
test_set = lgb.Dataset(test_df.drop(columns='Churn'),test_df['Churn'],params={'verbose':-1},reference=train_set)

final_model = lgb.train(best_model.params,
                           train_set,
                           num_boost_round=500,
                           valid_sets=[test_set],
                           init_model=best_model,
                           callbacks=[
                               lgb.early_stopping(stopping_rounds=50)
                           ])

final_model_path = root_dir/'models'/'final_model.txt'
final_model.save_model(final_model_path)


