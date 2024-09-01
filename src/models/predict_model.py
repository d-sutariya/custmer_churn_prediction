import os
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from dotenv import load_dotenv

env_path = Path('.env')
load_dotenv(env_path)
root_dir = os.getenv('ROOT_DIRECTORY')
print(root_dir)
final_model_path = Path(root_dir)/'models'/'final_model.txt'
file_path = Path(root_dir)/'data'/'processed'/'transformed_featured_test_set.csv'
test_set = pd.read_csv(file_path)
model = lgb.Booster(model_file=final_model_path)
preds = model.predict(test_set.drop(columns='Churn'))

