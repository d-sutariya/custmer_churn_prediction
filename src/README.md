# Source Code (`src`) Directory Overview

The `src/` directory is the heart and brain of the Customer Churn Prediction project, containing essential scripts and utilities for various stages of the project. Here’s a detailed overview of its contents:

## Directory Structure

### 1. `config/`
This folder contains configuration files crucial for setting up and initializing the project environment.

- **`setup_project.py`**: A script to set up the project environment, including installing dependencies and configuring settings.

### 2. `data/`
Stores utility scripts related to data processing and transformation.

- **`data_utils.py`**: A modular file containing classes and functions for data processing. It includes utility functions to handle various data operations.
- **`make_dataset.py`**: Transforms raw data into a training-ready format. This script is responsible for cleaning and preparing the data for modeling.

### 3. `features/`
Contains utility scripts for feature engineering.

- **`generate_and_transform_features.py`**: Includes classes and functions related to feature engineering using Featuretools. This script automates the generation and transformation of features from raw data.

### 4. `optimization/`
Holds utility scripts for model optimization and tracking.

- **`ensemble_utils.py`**: Contains classes and functions related to model ensembling. This script helps in combining predictions from multiple models to improve overall performance.
- **`model_optimizer.py`**: Includes the `ModelOptimizer` class and functions for performing hyperparameter optimization using Optuna.
- **`tuning_and_tracking.py`**: Manages MLflow experiments and tracks model performance metrics. This script helps in monitoring and optimizing model performance throughout the development process.

### 5. `pipeline/`
Contains the pipeline configuration for the entire data processing and modeling workflow.

- **`dvc.yaml`**: Defines the Data Version Control (DVC) pipeline, detailing the steps from data cleaning to model predictions. This file is crucial for reproducing and managing the entire pipeline.

## Additional Information

For more details on each folder and its contents, refer to the [`docs/`](https://github.com/d-sutariya/custmer_churn_prediction/tree/main/docs/_build/html/source/src.html) directory, which provides comprehensive documentation and guides for each component of the project.
