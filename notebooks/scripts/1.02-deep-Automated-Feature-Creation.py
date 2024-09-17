

import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join('..', 'src')))


import numpy as np
import pandas as pd
import warnings
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score
from dotenv import load_dotenv
from pathlib import Path

from features.generate_and_transform_features import FeatureGenerater,FeatureTransformer

warnings.filterwarnings("ignore")
np.random.seed(42)


env_path = Path('.env')
load_dotenv(env_path)

root_dir = Path(os.getenv('ROOT_DIRECTORY'))


encd_df = pd.read_csv(root_dir/'data'/'interim'/"encd_df.csv")
smoted_df = pd.read_csv(root_dir/'data'/'interim'/"smoted_df.csv")
val_set = pd.read_csv(root_dir/'data'/'interim'/"val_set.csv")
train_set = pd.read_csv(root_dir/'data'/'interim'/"train_set.csv")
test_set = pd.read_csv(root_dir/'data'/'interim'/"test_set.csv")
train_set_splitted = pd.read_csv(root_dir/'data'/'interim'/"train_set_splitted.csv")


# The `FeatureGenerater` class is designed to generate features automatically. it entity set from the training and testing datasets and generate features using the **Featuretools** library. It offers methods to handle feature cleaning, remove duplicates, and align the datasets. Additionally, it provides flexibility in choosing transformation and aggregation primitives for feature engineering.
# 

# For more detailed information, you can refer to the [documentation](../docs/FeatureGenerater.md) or check out the [source code](../src/features/generate_and_transform_features.py).
# 

# For simplicity i am not taking any aggeragation primitives.
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
 'add_numeric_scalar',
 'percent_change',
 'subtract_numeric',
]


# while feature generating we shouldn't use Churn and index.
# Using Churn will cause data leak
# Index is redundant feature (it is used to map Churn values with dataset after shuffling)

ignore_columns = {
    'smoted_train':['Churn','index'],
    'val_test':['Churn','index']
}
val_copy = val_set.copy()

# let's generate features for the smoted dataset

feature_gen = FeatureGenerater(encd_df,smoted_df,val_copy)
feature_gen.Create_Entityset('smoted','smoted_train','val_test','index')

# the below sets will be used for the evaluation of the smoted datasets
smoted_featured_train_set , smoted_featured_test_set = feature_gen.Generate_Features(trans_list,ignore_columns = ignore_columns , names_only=False)
# let's generate features for the splitted_train , val set
ignore_columns = {
    'train':['Churn','index'],
    'test':['Churn','index']
}
feature_gen = FeatureGenerater(encd_df,train_set_splitted,val_set)
feature_gen.Create_Entityset('validation','train','test','index')

# the below sets will be used for the training and validation process
featured_train_set , featured_val_set = feature_gen.Generate_Features(trans_list,ignore_columns = ignore_columns , names_only=False)


# below code generate the features for the final_train , test set
ignore_columns = {
    'final_train':['Churn','index'],
    'final_test':['Churn','index']
}
feature_gen = FeatureGenerater(encd_df,train_set,test_set)
feature_gen.Create_Entityset('final','final_train','final_test','index')
 
# the below sets will be used for the train and test the final model that we will get from the val set
featured_final_train_set , featured_test_set = feature_gen.Generate_Features(trans_list,ignore_columns = ignore_columns , names_only=False)


# The `FeatureTransformer` class is responsible for transforming the features of training and testing datasets. It handles imputation of missing values, scaling of numerical features, and preserves categorical data. The class also ensures that the transformed datasets are properly formatted and ready for model training and evaluation.
# 
# 

# For more detailed information, you can refer to the [documentation](../docs/FeatureTransformer.md) or check out the [source code](../src/features/generate_and_transform_features.py).

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


# Let's Check performance of the featured datasets.

model = LGBMClassifier(verbose=-1).fit(featured_train_set.drop(columns='Churn'),featured_train_set['Churn'])
y_preds = model.predict(featured_val_set.drop(columns='Churn'))


print("accuracy score is " ,accuracy_score(y_true=featured_val_set['Churn'],y_pred=y_preds))

print("precision score is " ,precision_score(y_true=featured_val_set['Churn'],y_pred=y_preds))

print("recall score is " ,recall_score(y_true=featured_val_set['Churn'],y_pred=y_preds))

print("weighted recall  score is " , 0.65 *recall_score(y_true=featured_val_set['Churn'],y_pred=y_preds) + 0.35 * precision_score(y_true=transformed_featured_val_set['Churn'],y_pred=y_preds))

