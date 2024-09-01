# -*- coding: utf-8 -*-

import sys
import os
from pathlib import Path

# Add the src directory to the Python path
# print(sys.path)
sys.path.append(str(Path(__file__).resolve().parents[2] / 'src'))

import click
import logging

from dotenv import find_dotenv, load_dotenv , set_key
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import warnings

from features.generate_and_transform_features import FeatureGenerater,FeatureTransformer
from data.data_utils import DataLoader,split_data




@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
def main(input_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    output_dir = Path(os.getenv('ROOT_DIRECTORY'))/'data'
    output_dir.mkdir(exist_ok=True,parents=True)

    processed_dir = output_dir / 'processed'
    interim_dir = output_dir / 'interim'
    raw_dir = output_dir/ 'raw'

    processed_dir.mkdir(parents=True,exist_ok=True)
    interim_dir.mkdir(parents=True,exist_ok=True)
    raw_dir.mkdir(parents=True,exist_ok=True)

    load_and_split_data(input_filepath,output_dir)

def load_and_split_data(input_filepath,output_dir):
    
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    data_loader = DataLoader(input_filepath)
    df = data_loader.load_data()
    encd_df = data_loader.preprocess_data()


    train_set,test_set,train_set_splitted,val_set = split_data(encd_df.dropna())
    X_train_splitted , y_train_splitted, X_val,y_val = train_set_splitted.drop('Churn',axis = 1 ) , train_set_splitted['Churn'] , val_set.drop('Churn',axis = 1) , val_set['Churn']


    # there is class imbalance that can affect results
    # To experiment with balance dataset I am using SMOTE algorithm.
    X_train_smoted,y_train_smoted = SMOTE().fit_resample(X_train_splitted,y_train_splitted)


    smoted_df = X_train_smoted
    smoted_df['Churn'] = y_train_smoted
    smoted_df = smoted_df.drop(columns='index').reset_index()

    #save to interim folder.
    encd_df.to_csv(output_dir/'interim'/'encd_df.csv',index=False)
    smoted_df.to_csv(output_dir/'interim'/'smoted_df.csv',index=False)
    train_set.to_csv(output_dir/'interim'/'train_set.csv',index=False)
    test_set.to_csv(output_dir/'interim'/'test_set.csv',index=False)
    val_set.to_csv(output_dir/'interim'/'val_set.csv',index=False)
    train_set_splitted.to_csv(output_dir/'interim'/'train_set_splitted.csv',index=False)

    transform_and_save_data(encd_df,train_set,test_set,train_set_splitted,val_set,smoted_df,output_dir)

def transform_and_save_data(encd_df,train_set,test_set,train_set_splitted,val_set,smoted_df,output_dir):
    
    trans_list =  [
    'absolute_diff',
    'exponential_weighted_variance',
    'modulo_numeric',
    'greater_than',
    'multiply_numeric_scalar',
    'less_than_equal_to',
    'cosine',
    'subtract_numeric',
    'rolling_max',
    'subtract_numeric_scalar',
    'less_than_scalar',
    'exponential_weighted_std',
    'natural_logarithm',
    'add_numeric_scalar'
    ]


    # while feature generating we shouldn't use Churn and index.
    # Using Churn will cause data leak
    # Index is redundant feature (it is used to map Churn values with dataset after shuffling)

    ignore_columns = {
        'smoted_train':['Churn','index'],
        'val_test':['Churn','index',]
    }
    val_copy = val_set.copy()

    # let's generate features for the smoted dataset

    feature_gen = FeatureGenerater(smoted_df,val_copy,encd_df)
    feature_gen.Create_Entityset('smoted','smoted_train','val_test','index')

    # the below sets will be used for the evaluation of the smoted datasets
    smoted_featured_train_set , smoted_featured_test_set = feature_gen.Generate_Features(trans_list,ignore_columns = ignore_columns , names_only=False)
    # let's generate features for the splitted_train , val set
    ignore_columns = {
        'train':['Churn','index'],
        'test':['Churn','index',]
    }
    feature_gen = FeatureGenerater(train_set_splitted,val_set,encd_df)
    feature_gen.Create_Entityset('validation','train','test','index')

    # the below sets will be used for the training and validation process
    featured_train_set , featured_val_set = feature_gen.Generate_Features(trans_list,ignore_columns = ignore_columns , names_only=False)


    # below code generate the features for the final_train , test set
    ignore_columns = {
        'final_train':['Churn','index'],
        'final_test':['Churn','index',]
    }
    feature_gen = FeatureGenerater(train_set,test_set,encd_df)
    feature_gen.Create_Entityset('final','final_train','final_test','index')
    
    # the below sets will be used for the train and test the final model that we will get from the val set
    featured_final_train_set , featured_test_set = feature_gen.Generate_Features(trans_list,ignore_columns = ignore_columns , names_only=False)

    feature_transformer = FeatureTransformer(featured_train_set,featured_val_set)
    transformed_featured_train_set , transformed_featured_val_set = feature_transformer.transform()


    # transforming final train set and test set
    feature_transformer = FeatureTransformer(featured_final_train_set,featured_test_set)
    transformed_featured_final_train_set , transformed_featured_test_set = feature_transformer.transform()


    feature_transformer = FeatureTransformer(smoted_featured_train_set , smoted_featured_test_set)
    transformed_featured_smoted_train_set , transformed_featured_smoted_test_set = feature_transformer.transform()


    transformed_featured_final_train_set.to_csv(output_dir/'processed'/'transformed_featured_final_train_set.csv',index=False)
    transformed_featured_smoted_train_set.to_csv(output_dir/'processed'/'transformed_featured_smoted_train_set.csv',index=False)
    transformed_featured_train_set.to_csv(output_dir/'processed'/'transformed_featured_train_set.csv',index=False)
    transformed_featured_test_set.to_csv(output_dir/'processed'/'transformed_featured_test_set.csv',index=False)
    transformed_featured_val_set.to_csv(output_dir/'processed'/'transformed_featured_val_set.csv',index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

    
    