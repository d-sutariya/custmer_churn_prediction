#!/usr/bin/env python
# coding: utf-8

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join('..', 'src')))


import numpy as np
import pandas as pd
import warnings
import re

from features.build_and_transform_features import FeatureGenerater,FeatureTransformer


warnings.filterwarnings("ignore")


encd_df = pd.read_csv(r"../data/interim/encd_df.csv")
smoted_df = pd.read_csv(r"../data/interim/smoted_df.csv")
val_set = pd.read_csv(r"../data/interim/val_set.csv")
train_set = pd.read_csv(r"../data/interim/train_set.csv")
test_set = pd.read_csv(r"../data/interim/test_set.csv")
train_set_splitted = pd.read_csv(r"../data/interim/train_set_splitted.csv")


# For simplicity i am not taking any aggeragation primitives.
trans_list =  [
 'multiply_numeric_boolean',
 'absolute_diff',
 'email_address_to_domain',
 'exponential_weighted_variance',
 'modulo_numeric',
 'rate_of_change',
 'url_to_protocol',
 'greater_than',
 'multiply_numeric_scalar',
 'less_than_equal_to',
 'longitude',
 'age',
 'cosine',
 'subtract_numeric',
 'week',
 'cityblock_distance',
 'rolling_max',
 'subtract_numeric_scalar',
 'is_quarter_end',
 'less_than_scalar',
 'exponential_weighted_std',
 'natural_logarithm',
 'add_numeric_scalar',
 'percent_change',
 'subtract_numeric',
 'is_lunch_time'
]


# while feature generating we shouldn't use Churn and index.
# Using Churn will cause data leak
# Index is redundant feature
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


# Apply the feature engineering pipeline to the training and validation sets
# - Fit and transform the training set using the pipeline
# - Transform the validation set using the pipeline
# - Ensure that the same transformations are applied to both sets

feature_transformer = FeatureTransformer(featured_train_set,featured_val_set)
transformed_featured_train_set , transformed_featured_val_set = feature_transformer.transform()


# transforming final train set and test set
feature_transformer = FeatureTransformer(featured_final_train_set,featured_test_set)
transformed_featured_final_train_set , transformed_featured_test_set = feature_transformer.transform()


feature_transformer = FeatureTransformer(smoted_featured_train_set , smoted_featured_test_set)
transformed_featured_smoted_train_set , transformed_featured_smoted_test_set = feature_transformer.transform()

