#!/usr/bin/env python
# coding: utf-8

import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join('..', 'src')))


import numpy as np
import pandas as pd
from sklearn.metrics import recall_score,accuracy_score,precision_score
from imblearn.over_sampling import SMOTE
import warnings

from data.data_utils import DataLoader,split_data


warnings.filterwarnings("ignore")
np.random.seed(42)


#load the data
data_loader = DataLoader("../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = data_loader.load_data()
encd_df = data_loader.preprocess_data()


train_set,test_set,train_set_splitted,val_set = split_data(encd_df.dropna())
X_train , y_train , X_test , y_test = train_set.drop(columns=['Churn','index']) , train_set['Churn'] , test_set.drop(columns=['index','Churn']) , test_set['Churn']
X_train_splitted , y_train_splitted, X_val,y_val = train_set_splitted.drop('Churn',axis = 1 ) , train_set_splitted['Churn'] , val_set.drop('Churn',axis = 1) , val_set['Churn']



# there is class imbalance that can affect results
# To experiment with balance dataset I am using SMOTE algorithm.
X_train_smoted,y_train_smoted = SMOTE().fit_resample(X_train_splitted,y_train_splitted)


smoted_df = X_train_smoted
smoted_df['Churn'] = y_train_smoted
smoted_df = smoted_df.drop(columns='index').reset_index()

