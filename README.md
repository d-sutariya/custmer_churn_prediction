# Customer Churn Analysis

This project focuses on telecom customer churn analysis and prediction using machine learning techniques.

## Project Organization

```bash
├── README.md           <- The top-level README for developers using this project.
├── data
│   ├── external        <- Data from third-party sources.
│   ├── interim         <- Intermediate data that has been transformed.
│   ├── processed       <- Final, canonical datasets ready for modeling.
│   └── raw             <- The original, immutable data dump.
│
├── docs                <- Project documentation.
│
├── models              <- Trained and serialized models.
│
├── notebooks           <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── post_deployment     <- Scripts related to post-deployment activities.
│
├── reports             <- Feature transformation definitions, predictions, and mlflow runs.
│
├── requirements.txt    <- The requirements file for reproducing the analysis environment.
│
├── setup.py            <- Makes project pip installable (pip install -e .) so src can be imported.
│
├── src                 <- Source code for use in this project.
│   ├── config          <- Script for setting up the project locally.
│   ├── data            <- Scripts to download or generate data.
│   │   ├── make_dataset.py
│   │   └── data_utils.py <- Data processing utilities.
│   ├── features        <- Scripts to turn raw data into features for modeling.
│   │   └── generate_and_transform_features.py <- Generate and transform features using Featuretools.
│   ├── models          <- Scripts to train models and use them for predictions.
│   │   ├── predict_model.py
│   │   └── train_model.py
│   ├── optimization    <- Scripts related to model optimization.
│   │   ├── ensemble_utils.py <- Utilities for ensembling models.
│   │   ├── model_optimization.py <- Manual model optimization.
│   │   └── tuning_and_tracking.py <- Hyperparameter tuning and tracking using MLflow and DagsHub.
│   ├── pipeline        <- DVC pipeline for data cleaning to model predictions.
│   │   └── dvc.yaml    <- Full pipeline configuration.
│
└── tox.ini             <- Tox file with settings for running tests and managing environments.
