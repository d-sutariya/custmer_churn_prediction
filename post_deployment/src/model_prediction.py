import os
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from dotenv import load_dotenv
import warnings 
from argparse import ArgumentParser

warnings.filterwarnings('ignore')

def main(model_path , pred_path):
    root_dir = Path(os.getenv('ROOT_DIRECTORY'))
    model_path = root_dir/model_path
    pred_set = pd.read_csv(root_dir/pred_path)
    final_model = lgb.Booster(model_file=model_path)

    if 'Churn' in pred_set.columns:

        preds = final_model.predict(pred_set.drop(columns=['customerID','Churn']))
    else:
        preds = final_model.predict(pred_set.drop(columns=['customerID']))
        
    pred_digits = [0 if pred <= 0.4 else 1 for pred in preds]
    preds_df = pd.DataFrame({
        'customerID':pred_set['customerID'],
        'predictions':pred_digits
    })
    preds_df.to_csv(root_dir/'post_deployment'/'predictions.csv')



if __name__=='__main__':

    paraser = ArgumentParser(description=" model prediction")
    paraser.add_argument('--model_path',type=str,help='model path relative to project root directory')
    paraser.add_argument('--prediction_set_path',type=str,help='prediction set path relative to project root')

    args = paraser.parse_args()

    if not args.model_path:
        args.model_path = input("Please enter the model path relative to project root directory:  ")
    if not args.prediction_set_path:
        args.prediction_set_path = input("Please provide prediction set path relative to project root directory:  ")


    env_path = Path('.env')
    load_dotenv(env_path)

    main(args.model_path,args.prediction_set_path)