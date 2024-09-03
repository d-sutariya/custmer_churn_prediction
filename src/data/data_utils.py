# The below class loads the data and performs basic preprocessing steps
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path  # Store the file path
        self.df = None  # Initialize an empty DataFrame

    def load_data(self,drop_customer_id=True):
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
    train_set, test_set = train_test_split(dummy_df, test_size=0.2, shuffle=True, random_state=42, stratify=dummy_df['Churn'])
    train_set_splitted, val_set = train_test_split(train_set, test_size=0.15, shuffle=True, random_state=42, stratify=train_set['Churn'])
    return train_set, test_set, train_set_splitted, val_set