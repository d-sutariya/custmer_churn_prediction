import lightgbm as lgb
from pathlib import Path
import os
import pandas as pd
import sys
import warnings
from dotenv import load_dotenv
from argparse import ArgumentParser

sys.path.append(str(Path(__file__).resolve().parents[2]/'src'))
warnings.filterwarnings('ignore')


def main(batch_train_path,model_path):

    root_dir = Path(os.getenv('ROOT_DIRECTORY'))
    transformed_batch_set = pd.read_csv(root_dir/batch_train_path)
    model_path = root_dir/model_path

    final_model = lgb.Booster(model_file=model_path)
    train_set = lgb.Dataset(transformed_batch_set.drop(columns=['Churn','customerID']),transformed_batch_set['Churn'],params={'verbose':-1})

    trained_model = lgb.train(
        params=final_model.params,
        train_set=train_set,
        init_model=final_model,
        valid_sets=[train_set],
        num_boost_round=200,
        callbacks=[
            lgb.early_stopping(stopping_rounds=30)
        ]
    )

    model_path = root_dir/'post_deployment'/'models'/'final_model.txt'
    trained_model.save_model(model_path)

if __name__ == '__main__':
    parser = ArgumentParser(description="this file is for model retraing")
    parser.add_argument('--model_path',type=str,help='model path relative to project root directory')
    parser.add_argument('--batch_train_path',type=str,help='batch train path relative to project root directory')

    args = parser.parse_args()
    if not args.model_path:
        args.model_path =  input("Please enter model path relative to project root directory")

    if not args.batch_train_path:
        args.batch_train_path = input("Please enter batch train path relative to project root directory")
    
    env_path = Path('.env')
    load_dotenv(env_path)

    main(args.batch_train_path,args.model_path)