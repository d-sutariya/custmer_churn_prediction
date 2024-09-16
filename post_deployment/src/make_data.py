import numpy as np
import pandas as pd
import os 
from pathlib import Path
import featuretools as ft
import sys
import warnings
import argparse

sys.path.append(str(Path(__file__).resolve().parents[2]/'src'))

warnings.filterwarnings('ignore')

from data.data_utils import DataLoader
from features.generate_and_transform_features import FeatureGenerater,FeatureTransformer
from dotenv import load_dotenv


def main(batch_train_file_path,sliding_window_path):

    """
    Main function to process and transform batch training data.
    
    Parameters
    ----------
    batch_train_file_path : str
        Path to the batch training data file.
    sliding_window_path : str
        Path to the sliding window data file.
    
    Returns
    -------
    None
        This function saves the transformed and aligned batch data to a CSV file.
    """

    root_dir = Path(os.getenv('ROOT_DIRECTORY'))
    feature_name_path = root_dir/'reports'/'feature_dfs'/'featured_final_train.json'
    train_set = pd.read_csv(root_dir/'data'/'interim'/'train_set.csv')
    transformed_featured_final_train_set = pd.read_csv(root_dir/sliding_window_path)

    d = DataLoader(root_dir/batch_train_file_path)
    batch_set = d.load_data(drop_customer_id=False)
    encd_batch_set = d.preprocess_data()
    encd_batch_set = encd_batch_set.reset_index()

    combined_set = pd.concat([train_set,encd_batch_set])
    combined_set = combined_set.drop(columns='index').reset_index(drop=True).reset_index()

    with open(feature_name_path,'r') as f:
        feature_defs = ft.load_features(f)

    new_es = ft.EntitySet(id="new_entity")

    new_es.add_dataframe(
        dataframe=combined_set,
        dataframe_name='final_train',
        index='index'
    )

    featured_batch = ft.calculate_feature_matrix(
        features=feature_defs,
        entityset=new_es
    )

    featured_batch = featured_batch.reset_index()
    final_featured_batch = featured_batch.reset_index(drop=True).loc[train_set.shape[0]+encd_batch_set['index'],:]

    cleaned_featured_batch = FeatureGenerater(combined_set,final_featured_batch).clean_dataframes(final_featured_batch)
    customerid = cleaned_featured_batch['customerID']

    transformed_batch = FeatureTransformer(cleaned_featured_batch.drop(columns='customerID')).transform()
    _,transformed_aligned_batch = transformed_featured_final_train_set.align(transformed_batch,join='inner',axis=1)

    transformed_aligned_batch['customerID'] = customerid.reset_index(drop=True)

    transformed_aligned_batch.to_csv(root_dir/'post_deployment'/'data'/'processed'/'transformed_aligned_batch.csv',index=False)
    print("trnasformed data is stored at:- ",root_dir/'post_deployment'/'data'/'processed'/'transformed_aligned_batch.csv')

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="convert raw data into processed")
    parser.add_argument('--batch_train_input',type=str,help='path of batch data')
    parser.add_argument('--window_train_input',type=str,help="path for window data")
    args = parser.parse_args()

    if not args.batch_train_input:
        args.batch_train_input = input("please provide relative path of the batch data (relative to project root directory):  ")
    
    if not args.window_train_input:     

        args.window_train_input = input("please provide relative path of the window data (relative to project root directory):  ")



    env_path = Path('.env')
    load_dotenv(env_path)

    main(args.batch_train_input,args.window_train_input)
