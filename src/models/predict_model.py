import os
import pandas as pd
from pathlib import Path
import lightgbm as lgb
from dotenv import load_dotenv
from argparse import ArgumentParser

def main(file_path,model_path):
    env_path = Path('.env')
    load_dotenv(env_path)
    root_dir = Path(os.getenv('ROOT_DIRECTORY'))

    test_set = pd.read_csv(root_dir/file_path)
    model = lgb.Booster(model_file=root_dir/model_path)
    preds = model.predict(test_set.drop(columns='Churn'))
    preds_df = pd.DataFrame({
        'predictions':preds
    })
    preds_df.to_csv(root_dir/'reports'/'predictions'/'preds.csv',index=False)
    print("predictions saved in project_root/reports/predictions/preds.csv")

if __name__ == "__main__":
    parser = ArgumentParser(description="this file is used for predicting values")
    parser.add_argument('--test_file_path',type=str,help="test file path relative to root directory :")
    parser.add_argument("--model_path",type=str,help="model path relative to the root directory :")
    args = parser.parse_args()

    if not args.test_file_path:
        args.test_file_path = input("Please Enter test file path relative to the root directory :")
    if not args.model_path : 
        args.model_path = input("Please enter model path relative to the root directory :")

    main(args.test_file_path,args.model_path)