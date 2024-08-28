# The below class loads the data and performs basic preprocessing steps
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path  # Store the file path
        self.df = None  # Initialize an empty DataFrame

    def load_data(self):
        # Load data from the CSV file
        self.df = pd.read_csv(self.file_path)
        
        # Replace the 'SeniorCitizen' column values: 0 -> "No", 1 -> "Yes"
        self.df['SeniorCitizen'] = self.df['SeniorCitizen'].replace({0: "No", 1: "Yes"})
        
        # Convert 'TotalCharges' to numeric, coercing errors to NaN
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
        
        # Replace the 'Churn' column values: "Yes" -> 1, "No" -> 0
        self.df['Churn'] = self.df['Churn'].replace({"Yes": 1, "No": 0})
        
        # Drop the 'customerID' column and reset the index
        self.df = self.df.drop('customerID', axis=1).reset_index()
        
        return self.df

    def preprocess_data(self):
        # Create dummy variables for categorical features (excluding 'Churn')
        dummy_df = pd.get_dummies(self.df.drop('Churn', axis=1))
        
        # Add the 'Churn' column back to the dummy DataFrame
        dummy_df['Churn'] = self.df['Churn']
        
        return dummy_df

# split into train , test ,validation datasets
def split_data(dummy_df):
    train_set, test_set = train_test_split(dummy_df, test_size=0.2, shuffle=True, random_state=42, stratify=dummy_df['Churn'])
    train_set_splitted, val_set = train_test_split(train_set, test_size=0.15, shuffle=True, random_state=42, stratify=train_set['Churn'])
    return train_set, test_set, train_set_splitted, val_set