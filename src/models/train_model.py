import numpy as np
import pandas as pd
import lightgbm as lgb
import os
from dotenv import load_dotenv
from pathlib import Path
from argparse import ArgumentParser
import warnings


def main(model_path,dataset_path):

    """
    Train a LightGBM model using an existing model as a starting point and save the final model.
    
    Parameters
    ----------
    model_path : str
        Path to the pre-trained model file relative to the root directory.
    dataset_path : str
        Path to the training dataset file relative to the root directory.
    
    Returns
    -------
    None
    """
    root_dir = Path(os.getenv('ROOT_DIRECTORY'))
    model_path =root_dir/model_path
    train_df = pd.read_csv(root_dir/dataset_path)


    best_model = lgb.Booster(model_file=model_path)
    train_set = lgb.Dataset(train_df.drop(columns=['Churn']),train_df['Churn'],params={'verbose':-1})
    
    final_model = lgb.train(best_model.params,
                            train_set,
                            num_boost_round=500,
                            valid_sets=[train_set],
                            init_model=best_model,
                            callbacks=[
                                lgb.early_stopping(stopping_rounds=50)
                            ])

    final_model_path = root_dir/'models'/'final_trained_model.txt'
    final_model.save_model(final_model_path)
    print("model saved in project_root/models/final_trained_model.txt")

if __name__ == '__main__':
    parser = ArgumentParser(description="this file is for training the model")
    parser.add_argument('--model_path',type=str,help="model path relative to project root directory")
    parser.add_argument('--dataset_path',type=str,help='dataset path relative to the project root directory')
    args =  parser.parse_args()

    if not args.model_path:
        args.model_path = input("Please enter model path relative to project root directory: ")
    
    if not args.dataset_path:
        args.dataset_path = input("Please Enter dataset path relative to project root directory: ")

    env_path = Path('.env')
    load_dotenv(dotenv_path = env_path)

    warnings.filterwarnings('ignore')

    main(args.model_path,args.dataset_path)