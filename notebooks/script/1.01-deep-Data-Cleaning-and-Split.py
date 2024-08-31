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
from sklearn.metrics import recall_score,accuracy_score,precision_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb
from lightgbm import LGBMClassifier 
import warnings


from data.data_utils import DataLoader,split_data


warnings.filterwarnings("ignore")
np.random.seed(42)


# # Data preprocessing

# The `DataLoader` class is responsible for loading data from a CSV file and performing basic preprocessing steps, including handling categorical variables and missing values. It also provides a method to preprocess data by creating dummy variables for categorical features. The `split_data` function then splits the preprocessed data into training, testing, and validation datasets, ensuring a balanced distribution of the target variable.
# 

# For more detailed information, you can refer to the [documentation](../docs/DataLoader.md) or check out the [source code](../src/data/data_utils.py).
# 

#load the data
data_loader = DataLoader("../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = data_loader.load_data()
encd_df = data_loader.preprocess_data()


df.info()


# ## Data Splitting

train_set,test_set,train_set_splitted,val_set = split_data(encd_df.dropna())
X_train , y_train , X_test , y_test = train_set.drop(columns=['Churn','index']) , train_set['Churn'] , test_set.drop(columns=['index','Churn']) , test_set['Churn']
X_train_splitted , y_train_splitted, X_val,y_val = train_set_splitted.drop('Churn',axis = 1 ) , train_set_splitted['Churn'] , val_set.drop('Churn',axis = 1) , val_set['Churn']
X_train_splitted.shape


# there is class imbalance that can affect results
# To experiment with balance dataset I am using SMOTE algorithm.
X_train_smoted,y_train_smoted = SMOTE().fit_resample(X_train_splitted,y_train_splitted)


smoted_df = X_train_smoted
smoted_df['Churn'] = y_train_smoted
smoted_df = smoted_df.drop(columns='index').reset_index()


# ### Let us define the base line score for our future work.

model = LGBMClassifier(verbose=-1).fit(X_train_splitted,y_train_splitted)
y_preds = model.predict(X_val)


# I am going to use weighted recall (it's formula is ``0.65 * recall + 0.35 * precision``) as my primary evaluation metric. As recall is more important in this Project.

print("accuracy score is " ,accuracy_score(y_true=y_val,y_pred=y_preds))

print("precision score is " ,precision_score(y_true=y_val,y_pred=y_preds))

print("recall score is " ,recall_score(y_true=y_val,y_pred=y_preds))

print("weighted recall  score is " , 0.65 *recall_score(y_true=y_val,y_pred=y_preds) + 0.35 * precision_score(y_true=y_val,y_pred=y_preds))

