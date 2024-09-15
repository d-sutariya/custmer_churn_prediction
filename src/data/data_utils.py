# The below class loads the data and performs basic preprocessing steps
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

class DataLoader:
    """
    A class used to load and preprocess customer churn data.
    Parameters
    ----------
    file_path : str
        The path to the CSV file containing the data.
    df : pandas.DataFrame or None
        The DataFrame that holds the loaded data.
    Methods
    -------
    load_data(drop_customer_id=True)
        Loads data from the CSV file, performs initial preprocessing, and returns the DataFrame.
    preprocess_data()
        Further preprocesses the data by handling missing values, creating dummy variables, and returning the processed DataFrame.
    """


    def __init__(self, file_path):
        self.file_path = file_path  # Store the file path
        self.df = None  # Initialize an empty DataFrame

    def load_data(self,drop_customer_id=True):

        """
        Load and preprocess customer churn data from a CSV file.
        This method performs the following operations:
        1. Loads data from the CSV file specified by `self.file_path`.
        2. Replaces the 'SeniorCitizen' column values: 0 -> "No", 1 -> "Yes".
        3. Converts the 'TotalCharges' column to numeric, coercing errors to NaN.
        4. Replaces the 'Churn' column values: "Yes" -> 1, "No" -> 0.
        5. Optionally drops the 'customerID' column and resets the index.
        Parameters
        ----------
        drop_customer_id : bool, optional
            If True, drops the 'customerID' column from the DataFrame (default is True).
        Returns
        -------
        pd.DataFrame
            The preprocessed DataFrame.
        """
        # Load data from the CSV file
        self.df = pd.read_csv(self.file_path)
        
        # Replace the 'SeniorCitizen' column values: 0 -> "No", 1 -> "Yes"
        self.df['SeniorCitizen'] = self.df['SeniorCitizen'].replace({0: "No", 1: "Yes"})
        
        # Convert 'TotalCharges' to numeric, coercing errors to NaN
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        
        # Replace the 'Churn' column values: "Yes" -> 1, "No" -> 0
        self.df['Churn'] = self.df['Churn'].replace({"Yes": 1, "No": 0})
        
        if drop_customer_id:
            # Drop the 'customerID' column and reset the index
            self.df = self.df.drop('customerID', axis=1).reset_index()
        
        return self.df

    def preprocess_data(self):
        """
        Preprocesses the data by performing the following steps:
        1. Drops columns with more than 70% missing values.
        2. Selects numerical columns and drops those with fewer than 5 unique values (excluding 'index').
        3. Imputes missing values in numerical columns using the median strategy.
        4. Creates dummy variables for categorical features, excluding 'Churn' and 'customerID'.
        5. Adds the 'Churn' column back to the processed DataFrame.
        Returns:
            pd.DataFrame: The preprocessed DataFrame with dummy variables and imputed values.
        """

        # Create dummy variables for categorical features (excluding 'Churn')
        df_length = len(self.df)
        for col in self.df.columns:
            if (self.df[col].isnull().sum())/df_length > 0.7:
                self.df = self.df.drop(columns=col)

        num_df = self.df.select_dtypes(include=['int64', 'float64'])

        drop_list = []
        for col in num_df.columns:
            if num_df[col].nunique() < 5 and col !='index':
                drop_list.append(col)

        num_df = num_df.drop(columns=drop_list)        
        
        imputed_arr = SimpleImputer(strategy='median').fit_transform(num_df)
        imputed_df = pd.DataFrame(imputed_arr,columns=num_df.columns)
        imputed_df = pd.concat([imputed_df , self.df.drop(columns=imputed_df.columns)],axis=1)
        

        if 'customerID' in imputed_df.columns:
            dummy_df = pd.get_dummies(imputed_df.drop(columns=['Churn','customerID']))
            dummy_df['customerID'] = self.df['customerID']
            
        else:    
            dummy_df = pd.get_dummies(imputed_df.drop(columns='Churn'))

        # Add the 'Churn' column back to the dummy DataFrame
        dummy_df['Churn'] = self.df['Churn']
        return dummy_df

# split into train , test ,validation datasets
def split_data(dummy_df):
    """
    Splits the input DataFrame into training, validation, and test sets.

    The function performs the following splits:
    1. Splits the input DataFrame into a training set and a test set.
    2. Further splits the training set into a smaller training set and a validation set.

    The splits are stratified based on the 'Churn' column to ensure that the class distribution is preserved in each subset.

    Parameters
    ----------
    dummy_df : pandas.DataFrame
        The input DataFrame containing the data to be split. It must include a 'Churn' column.

    Returns
    -------
    tuple
        A tuple containing four DataFrames:
        - train_set : pandas.DataFrame
            The initial training set before the second split.
        - test_set : pandas.DataFrame
            The test set.
        - train_set_splitted : pandas.DataFrame
            The training set after the second split.
        - val_set : pandas.DataFrame
            The validation set.
    """
    train_set, test_set = train_test_split(dummy_df, test_size=0.2, shuffle=True, random_state=42, stratify=dummy_df['Churn'])
    train_set_splitted, val_set = train_test_split(train_set, test_size=0.15, shuffle=True, random_state=42, stratify=train_set['Churn'])
    return train_set, test_set, train_set_splitted, val_set