import numpy as np
import pandas as pd
import featuretools as ft
import re
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

class GenerateFeatures:
    def __init__(self, train_set, test_set, df):
        self.train_set = train_set  # Training dataset
        self.test_set = test_set  # Test dataset
        self.entity_set = None  # Placeholder for entity set
        self.train_set_name = None  # Name of training dataset in entity set
        self.test_set_name = None  # Name of test dataset in entity set
        self.df = df  # Original DataFrame

    def Create_Entityset(self, entity_id, train_set_name, test_set_name, index_name=None):
        # Check if index_name is not present in train and test sets
        if index_name not in self.train_set.columns and index_name not in self.test_set.columns:
            es = ft.EntitySet(id=entity_id)
            # Add train_set to the entity set with make_index=True
            es.add_dataframe(
                dataframe_name=train_set_name,
                dataframe=self.train_set, 
                make_index=True,
                index=index_name,
                # time_index='tenure'  # Uncomment if using time index
            ) 
            # Add test_set to the entity set with make_index=True
            es.add_dataframe(
                dataframe_name=test_set_name,
                dataframe=self.test_set,
                make_index=True,
                index=index_name,
                # time_index='tenure'  # Uncomment if using time index
            )
        else:
            es = ft.EntitySet(id=entity_id)
            # Add train_set to the entity set with existing index
            es.add_dataframe(
                dataframe_name=train_set_name,
                dataframe=self.train_set,
                index=index_name,
                # time_index='tenure'  # Uncomment if using time index
            )
            # Add test_set to the entity set with existing index
            es.add_dataframe(
                dataframe_name=test_set_name,
                dataframe=self.test_set,
                index=index_name,
                # time_index='tenure'  # Uncomment if using time index
            )
        self.entity_set = es  # Store the entity set
        self.train_set_name = train_set_name  # Store the training set name
        self.test_set_name = test_set_name  # Store the test set name

    def __clean_feature_names(self, df):
        # Clean column names by replacing special characters with underscore
        cleaned_names = []
        for col in df.columns:
            clean_name = re.sub(r'[^A-Za-z0-9_]+', '_', col)
            cleaned_names.append(clean_name)
        df.columns = cleaned_names
        return df

    def __remove_duplicate_columns(self, df):
        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    def Generate_Features(self, trans_list=None, agg_list=None, ignore_columns=None, names_only=True):
        if names_only == False:
            # Generate features for training set
            # dfs generate features from data and merge them with target dataframe
            feature_df, feature_names = ft.dfs(
                entityset=self.entity_set, # for which entity set do you want to generate features.
                target_dataframe_name=self.train_set_name, # To Which dataframe features must be merged? 
                trans_primitives=trans_list, # list of the transformations that you want to apply .
                agg_primitives = agg_list, # Aggregation primitives means methods that combine multiple rows like median,average,sum etc.
                max_depth=2, # Number of  transformation,aggregation  primitves  can be stacked upon each other.
                ignore_columns=ignore_columns, # columns that shouldn't use for feature engineering . Example 'Churn','index'
                features_only=names_only, # should it generate features or just provide names of the
                ignore_dataframes=[self.test_set_name] # which dataframes should avoid to mitigate data leak.
            )
            # Ensure 'index' column in feature_df is treated as integer index
            feature_df = feature_df.reset_index()
            feature_df['index'] = feature_df['index'].astype(int)

            # Align feature_df index to avoid out-of-range or mismatch
            aligned_churn = self.df.loc[feature_df['index'], 'Churn'].reset_index(drop=True)

            # Assign aligned Churn values to feature_df
            feature_df['Churn'] = aligned_churn

            # Generate features for test set
            feature_df_test, features_test_name = ft.dfs(
                entityset=self.entity_set,
                target_dataframe_name=self.test_set_name,
                trans_primitives=trans_list,
                max_depth=2,
                ignore_columns=ignore_columns,
                features_only=names_only,
                ignore_dataframes=[self.train_set_name]
            )
            # Ensure 'index' column in feature_df_test is treated as integer index
            feature_df_test = feature_df_test.reset_index()
            feature_df_test['index'] = feature_df_test['index'].astype(int)

            # Align feature_df_test index to avoid out-of-range or mismatch
            aligned_churn_test = self.df.loc[feature_df_test['index'], 'Churn'].reset_index(drop=True)

            # Assign aligned Churn values to feature_df_test
            feature_df_test['Churn'] = aligned_churn_test

            # Drop 'index' column from feature dataframes
            feature_df = feature_df.drop('index', axis=1)
            feature_df_test = feature_df_test.drop('index', axis=1)

            # Clean feature names for both train and test datasets
            feature_df = self.__clean_feature_names(feature_df)
            feature_df_test = self.__clean_feature_names(feature_df_test)

            # Remove duplicate columns from both datasets
            feature_df = self.__remove_duplicate_columns(feature_df)
            feature_df_test = self.__remove_duplicate_columns(feature_df_test)

            # Remove columns with a single unique value
            single_col_list = [col for col in feature_df.columns if feature_df[col].nunique() == 1]
            feature_df = feature_df.drop(columns=single_col_list)

            # Replace infinite values with NaN
            feature_df = feature_df.replace([-np.inf, np.inf], np.nan)
            feature_df_test = feature_df_test.replace([-np.inf, np.inf], np.nan)

            # Align the datasets to keep only common columns
            featured_train_labels = feature_df['Churn']
            feature_df_aligned, feature_df_test_aligned = feature_df.align(feature_df_test, join='inner', axis=1)
            feature_df_aligned['Churn'] = featured_train_labels

            return feature_df_aligned, feature_df_test_aligned
        else:
            # Generate feature names only for training set
            feature_names = ft.dfs(
                entityset=self.entity_set,
                target_dataframe_name=self.train_set_name,
                trans_primitives=trans_list,
                max_depth=2,
                ignore_columns=ignore_columns,
                features_only=names_only,
                ignore_dataframes=[self.test_set_name]
            )
            # Generate feature names only for test set
            feature_test_names = ft.dfs(
                entityset=self.entity_set,
                target_dataframe_name=self.test_set_name,
                trans_primitives=trans_list,
                max_depth=2,
                ignore_columns=ignore_columns,
                features_only=names_only,
                ignore_dataframes=[self.train_set_name]
            )
            return feature_names, feature_test_names
        

class FeatureTransformer:
    def __init__(self, train_set, test_set):
        self.train_set = train_set  # Training dataset
        self.test_set = test_set  # Test dataset

        # Lists to store numerical and categorical feature names
        self.num_list = [col for col in train_set.columns if train_set[col].dtype != 'bool']
        self.cat_list = [col for col in train_set.columns if train_set[col].dtype == 'bool']
        
        self.num_list.remove('Churn')  # Remove the target variable from numerical features

        # Define the column transformer
        self.col_transfm = ColumnTransformer(
            transformers=[
                # Pipeline for numerical features: imputation and scaling
                ("num", Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy='median')),  # Impute missing values with median
                        ("scaler", StandardScaler())  # Standardize numerical features
                    ]), self.num_list)
            ],
            n_jobs=-1,  # Use all available cores
            verbose=True,  # Verbose output
            verbose_feature_names_out=True,  # Verbose feature names
            remainder='passthrough'  # Keep remaining columns as they are
        )

    def transform(self):
        # Fit and transform the training set
        self.train_set = self.col_transfm.fit_transform(self.train_set)
        # Transform the test set
        self.test_set = self.col_transfm.transform(self.test_set)

        # Convert transformed data to DataFrames with feature names
        self.train_set = pd.DataFrame(self.train_set, columns=self.col_transfm.get_feature_names_out())
        self.test_set = pd.DataFrame(self.test_set, columns=self.col_transfm.get_feature_names_out())

        # Rename the Churn column
        self.train_set = self.train_set.rename(columns={'remainder__Churn': 'Churn'})
        self.test_set = self.test_set.rename(columns={'remainder__Churn': 'Churn'})

        # Ensure all data is of type float64
        self.train_set, self.test_set = self.train_set.astype('float64'), self.test_set.astype('float64')

        return self.train_set, self.test_set  # Return the transformed datasets
    