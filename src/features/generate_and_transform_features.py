import numpy as np
import pandas as pd
import featuretools as ft
import re
import os
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

class FeatureGenerater:
    """
    A class used to generate and transform features for customer churn prediction.
    
    Parameters      
    ----------
    df : pandas.DataFrame
        The original DataFrame containing the data.
    train_set : pandas.DataFrame
        The training dataset.
    test_set : pandas.DataFrame, optional
        The test dataset (default is None).
    entity_set : featuretools.EntitySet
        The entity set used for feature engineering.
    train_set_name : str
        The name of the training dataset in the entity set.
    test_set_name : str
        The name of the test dataset in the entity set.
    """
    def __init__(self, df , train_set,test_set = None):
        """
        Initializes the feature generation and transformation class.

        Parameters:
        df (pd.DataFrame): The original DataFrame containing the data.
        train_set (pd.DataFrame): The training dataset.
        test_set (pd.DataFrame, optional): The test dataset. Defaults to None.

        Attributes:
        train_set (pd.DataFrame): The training dataset.
        test_set (pd.DataFrame): The test dataset.
        entity_set (any): Placeholder for entity set.
        train_set_name (str): Name of the training dataset in the entity set.
        test_set_name (str): Name of the test dataset in the entity set.
        df (pd.DataFrame): The original DataFrame.
        """
        self.train_set = train_set  # Training dataset
        self.test_set = test_set  # Test dataset
        self.entity_set = None  # Placeholder for entity set
        self.train_set_name = None  # Name of training dataset in entity set
        self.test_set_name = None  # Name of test dataset in entity set
        self.df = df  # Original DataFrame

    def Create_Entityset(self, entity_id, train_set_name, test_set_name = None, index_name=None):
  
        """
        Create an EntitySet for feature engineering using Featuretools.

        This method creates an EntitySet and adds the training and optionally the test datasets to it.
        It checks if the specified index column is present in the datasets and creates the index if not present.

        Parameters
        ----------
        entity_id : str
            Unique identifier for the EntitySet.
        train_set_name : str
            Name to assign to the training dataset within the EntitySet.
        test_set_name : str, optional
            Name to assign to the test dataset within the EntitySet. Default is None.
        index_name : str, optional
            Name of the index column. If not present in the datasets, it will be created. Default is None.

        Returns
        -------
        None
        """
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
            if test_set_name:
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
            if test_set_name:
                es.add_dataframe(
                    dataframe_name=test_set_name,
                    dataframe=self.test_set,
                    index=index_name,
                    # time_index='tenure'  # Uncomment if using time index
                )
                self.test_set_name = test_set_name
        self.entity_set = es  # Store the entity set
        self.train_set_name = train_set_name  # Store the training set name
          # Store the test set name

    def __clean_feature_names(self, df):
        """
        Clean column names in a DataFrame by replacing special characters with underscores.

        This method takes a DataFrame and processes its column names to ensure they contain
        only alphanumeric characters and underscores. Any special characters are replaced
        with underscores.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame whose column names need to be cleaned.

        Returns
        -------
        pandas.DataFrame
            The DataFrame with cleaned column names.
        """
        # Clean column names by replacing special characters with underscore
        cleaned_names = []
        for col in df.columns:
            clean_name = re.sub(r'[^A-Za-z0-9_]+', '_', col)
            cleaned_names.append(clean_name)
        df.columns = cleaned_names
        return df

    def __remove_duplicate_columns(self, df):
        """
        Remove duplicate columns from a DataFrame.

        This method removes columns that are duplicated in the given DataFrame,
        retaining only the first occurrence of each column.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame from which duplicate columns need to be removed.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with duplicate columns removed.
        """
        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        return df

    def Generate_Features(self, trans_list=None, agg_list=None, ignore_columns=None, names_only=True):
        """
        Generate features for training and test datasets using featuretools.
        
        Parameters
        ----------
        trans_list : list, optional
            List of transformation primitives to apply.
        agg_list : list, optional
            List of aggregation primitives to apply.
        ignore_columns : list, optional
            List of columns to ignore during feature generation.
        names_only : bool, default=True
            If True, only feature names are generated. If False, features are generated and saved.
        
        Returns
        -------
        tuple or DataFrame
            If names_only is True, returns a tuple containing feature names for training and test sets.
            If names_only is False, returns cleaned feature DataFrames for training and test sets.
        """
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
            root_dir = Path(os.getenv('ROOT_DIRECTORY'))

            # Ensure that the directory exists
            feature_dir = root_dir / 'reports' / 'feature_definations'
            feature_dir.mkdir(parents=True, exist_ok=True)

            # Now save the features
            with open(feature_dir / f'featured_{self.train_set_name}.json', 'w') as f:
                ft.save_features(feature_names, f)

            # Generate features for test set
            if self.test_set is not None:
                feature_df_test, features_test_name = ft.dfs(
                    entityset=self.entity_set,
                    target_dataframe_name=self.test_set_name,
                    trans_primitives=trans_list,
                    max_depth=2,
                    ignore_columns=ignore_columns,
                    features_only=names_only,
                    ignore_dataframes=[self.train_set_name]
                )

                with open(feature_dir/f'featured_{self.test_set_name}.json','w') as f:
                    ft.save_features(features_test_name,f)

                return self.clean_dataframes(feature_df,feature_df_test)
                
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
            if self.test_set  is not None:
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
                # Ensure 'index' column in feature_df_test is treated as integer index
            return feature_names
            
    def clean_dataframes(self, feature_df, feature_df_test=None):
        """
        Clean and transform feature dataframes for customer churn prediction.
        This method performs several cleaning and transformation steps on the provided
        feature dataframes, including resetting indices, aligning with the main dataframe,
        cleaning feature names, removing duplicate columns, and handling infinite values.
        
        Parameters
        ----------
        feature_df : pandas.DataFrame
            The main feature dataframe to be cleaned and transformed.
        feature_df_test : pandas.DataFrame, optional
            An optional test feature dataframe to be cleaned and transformed. If provided,
            both the training and test dataframes will be aligned to have the same columns.
        
        Returns
        -------
        pandas.DataFrame or tuple of pandas.DataFrame
            If `feature_df_test` is not provided, returns the cleaned and transformed `feature_df`.
            If `feature_df_test` is provided, returns a tuple of the cleaned and aligned
            `(feature_df, feature_df_test)`.
        Raises
        ------
        KeyError
            If some indices in `feature_df` or `feature_df_test` are missing in the main dataframe `self.df`.
        Notes
        -----
        - The method assumes that the main dataframe `self.df` contains a 'Churn' column and optionally a 'customerID' column.
        - The method ensures that the indices in `feature_df` and `feature_df_test` exist in `self.df`.
        - The method handles infinite values by replacing them with NaN.
        - The method removes columns with a single unique value from the feature dataframes.
        """
        # Reset and ensure integer index in feature_df
        feature_df = feature_df.reset_index()
        feature_df['index'] = feature_df['index'].astype(int)

        if 'customerID' in self.df.columns:
            # Ensure index exists in self.df
            if not all(idx in self.df.index for idx in feature_df['index']):
                raise KeyError("Some indices in feature_df are missing in self.df.")
            
            # Extract 'Churn' and 'customerID' columns based on the index
            aligned_churn = pd.DataFrame()
            aligned_churn['Churn'] = self.df.loc[feature_df['index'],'Churn'].reset_index(drop=True)
            aligned_churn['customerID'] = self.df.loc[feature_df['index'],'customerID'].reset_index(drop=True)
            feature_df['customerID'] = aligned_churn['customerID']
        else:    
            aligned_churn = pd.DataFrame()
            aligned_churn['Churn'] = self.df.loc[feature_df['index'],'Churn'].reset_index(drop=True)

        feature_df['Churn'] = aligned_churn['Churn']

        # Drop 'index' column and clean feature names
        feature_df = feature_df.drop('index', axis=1)
        feature_df = self.__clean_feature_names(feature_df)
        feature_df = self.__remove_duplicate_columns(feature_df)
        single_col_list = [col for col in feature_df.columns if feature_df[col].nunique() == 1]
        feature_df = feature_df.drop(columns=single_col_list)
        feature_df = feature_df.replace([-np.inf, np.inf], np.nan)

        if feature_df_test is not None:
            feature_df_test = feature_df_test.reset_index()
            feature_df_test['index'] = feature_df_test['index'].astype(int)
            
            if 'customerID' in self.df.columns:
                # Ensure index exists in self.df
                if not all(idx in self.df.index for idx in feature_df['index']):
                    raise KeyError("Some indices in feature_df are missing in self.df.")
                
                # Extract 'Churn' and 'customerID' columns based on the index
                aligned_churn = pd.DataFrame()
                aligned_churn['Churn'] = self.df.loc[feature_df_test['index'],'Churn'].reset_index(drop=True)
                aligned_churn['customerID'] = self.df.loc[feature_df_test['index'],'customerID'].reset_index(drop=True)
                feature_df_test['customerID'] = aligned_churn['customerID']
            else:
                aligned_churn = pd.DataFrame()
                aligned_churn['Churn'] = self.df.loc[feature_df_test['index'],'Churn'].reset_index(drop=True)
            feature_df_test['Churn'] = aligned_churn['Churn']
            feature_df_test = feature_df_test.drop('index', axis=1)
            feature_df_test = self.__clean_feature_names(feature_df_test)
            feature_df_test = self.__remove_duplicate_columns(feature_df_test)
            feature_df_test = feature_df_test.replace([-np.inf, np.inf], np.nan)

            # Align the datasets to keep only common columns
            featured_train_labels = feature_df['Churn']
            feature_df_aligned, feature_df_test_aligned = feature_df.align(feature_df_test, join='inner', axis=1)
            feature_df_aligned['Churn'] = featured_train_labels

            return feature_df_aligned, feature_df_test_aligned
        
        return feature_df


class FeatureTransformer:
    """
    A class used to transform features for a customer churn prediction model.
    
    Parameters
    ----------
    train_set : pandas.DataFrame
        The training dataset.
    test_set : pandas.DataFrame, optional
        The test dataset (default is None).
    num_list : list
        List of numerical feature names.
    cat_list : list
        List of categorical feature names.
    col_transfm : sklearn.compose.ColumnTransformer
        Column transformer for preprocessing features.
    
    """ 
    def __init__(self, train_set, test_set=None):
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
        """
        Transforms the training and test datasets using the column transformer.
        This method performs the following steps:   
        
        1. Fits and transforms the training set using the column transformer.
        2. Converts the transformed training set to a DataFrame with feature names.
        3. Renames the 'remainder__Churn' column to 'Churn' and 'remainder__customerID' to 'customerID' if present.
        4. Ensures all data in the training set is of type float64.
        5. If a test set is provided:
        
            a. Transforms the test set using the already fitted column transformer.
            b. Converts the transformed test set to a DataFrame with feature names.
            c. Renames the 'remainder__Churn' column to 'Churn' and 'remainder__customerID' to 'customerID' if present.
            d. Ensures all data in the test set is of type float64.
            e. Returns both the transformed training and test sets.
        
        Returns:
            pd.DataFrame: Transformed training set.
            tuple(pd.DataFrame, pd.DataFrame): Transformed training and test sets if test set is provided.
        """
        # Fit and transform the training set
        self.train_set = self.col_transfm.fit_transform(self.train_set)

        # Convert transformed data to DataFrames with feature names
        self.train_set = pd.DataFrame(self.train_set, columns=self.col_transfm.get_feature_names_out())
        # Rename the Churn column
        if 'customerID' in self.train_set:
            self.train_set = self.train_set.rename(columns={'remainder__Churn': 'Churn','remainder__customerID':'customerID'})
        else:
            self.train_set = self.train_set.rename(columns={'remainder__Churn': 'Churn'})

        # Ensure all data is of type float64
        self.train_set  = self.train_set.astype('float64')

        if self.test_set  is not None :
            # Transform the test set
            self.test_set = self.col_transfm.transform(self.test_set)
            
            self.test_set = pd.DataFrame(self.test_set, columns=self.col_transfm.get_feature_names_out())
            
            if 'customerID' in self.test_set:
                self.test_set = self.test_set.rename(columns={'remainder__Churn': 'Churn','remainder__customerID':'customerID'})
            else:
                self.test_set = self.test_set.rename(columns={'remainder__Churn': 'Churn'})
        
            self.test_set =  self.test_set.astype('float64')
            
            return self.train_set,self.test_set


        return self.train_set  # Return the transformed datasets
    