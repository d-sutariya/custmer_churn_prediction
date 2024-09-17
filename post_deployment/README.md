# Post-Deployment Overview

This section is dedicated to the **post-deployment** phase of the Customer Churn Prediction project. It ensures the project remains effective and relevant by handling new data for predictions and retraining models when necessary.

## Directory Structure

### 1. `data/`
This directory is organized into three subdirectories to manage data flow for post-deployment activities:

- **`raw/`**: Contains the unprocessed data collected post-deployment for making predictions or retraining models.
- **`interim/`**: Holds data that has been partially processed, serving as a checkpoint before final processing.
- **`processed/`**: The final cleaned and transformed data, ready for predictions or model retraining.

### 2. `models/`
This folder stores the trained models used for post-deployment predictions and retraining:

- Models in this folder are leveraged for generating predictions and can be retrained on new data as needed to maintain performance.

### 3. `src/`
The `src/` folder contains essential scripts for post-deployment activities:

- **`make_data.py`**: This script handles the transformation of raw data into processed data, aligning it with the format required for predictions or retraining.
  
- **`model_prediction.py`**: Generates predictions using the latest trained model and outputs the results in a `.csv` file for further analysis or action.

- **`retrain_model.py`**: This script retrains the machine learning models on newly processed data to ensure they stay updated and continue to perform well in real-world scenarios.

## Purpose of Post-Deployment

The goal of this section is to keep the churn prediction system dynamic and adaptable to new data. By continuously processing new raw data, making predictions, and retraining the models when necessary, the system can provide accurate, real-time insights.

