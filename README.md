# Customer Churn Analysis

## Overview

The objective of this project is to identify potential telecom churners of TNB telco inc. (hypothetical compony), enabling the company to take proactive measures to retain these customers. Using the  Telco Customer Churn dataset provided by TNB (dataset is available on kaggle), various machine learning models have been evaluated to achieve this goal. You can view the detailed report on ![TNB customer churn report](https://github.com/d-sutariya/custmer_churn_prediction/blob/main/Customer_Churn_Prediction_TNB_Project_Report.pdf)

## Objective

The primary goal is to identify possible telecom churners so that the company can implement strategies to retain these customers. While the implementation of retention strategies is outside the scope of this project, the insights provided can greatly inform decision-making.


## Dataset

This project utilizes the Kaggle Telco Customer Churn dataset, which contains comprehensive information about telecom customers, including their usage patterns, payment methods, and service preferences.

## Frameworks and Tools

This project leverages a range of powerful frameworks and tools to ensure cutting-edge performance and efficiency. Here are the key technologies used:

### Core Technologies

- **Plotly** ![Plotly](https://img.shields.io/badge/Plotly-3C0D3F?style=flat-square&logo=plotly&logoColor=white): Interactive data visualization library that brings your data to life.

- **Featuretools** ![Featuretools](https://img.shields.io/badge/Featuretools-1F77B4?style=flat-square&logo=featuretools&logoColor=white): Automated feature engineering for creating meaningful features from raw data.

- **LightGBM** ![LightGBM](https://img.shields.io/badge/LightGBM-F9A828?style=flat-square&logo=lightgbm&logoColor=black): Gradient boosting framework that uses tree-based learning algorithms.

- **Optuna** ![Optuna](https://img.shields.io/badge/Optuna-00C1D4?style=flat-square&logo=optuna&logoColor=white): Hyperparameter optimization framework to enhance model performance.

- **MLflow** ![MLflow](https://img.shields.io/badge/MLflow-FFA500?style=flat-square&logo=mlflow&logoColor=white): Platform for managing the end-to-end machine learning lifecycle.

- **Dagshub** ![Dagshub](https://img.shields.io/badge/Dagshub-1F77B4?style=flat-square&logo=dagshub&logoColor=white): Collaborative data science platform for versioning and managing datasets and models.

### Additional Tools

- **Sphinx** ![Sphinx](https://img.shields.io/badge/Sphinx-FF0000?style=flat-square&logo=sphinx&logoColor=white): Documentation generator for creating beautiful project docs.

- **DVC** ![DVC](https://img.shields.io/badge/DVC-003D00?style=flat-square&logo=dvc&logoColor=white): Data version control system for managing data and model versions.

- **Scikit-learn** ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-FABF00?style=flat-square&logo=scikit-learn&logoColor=black): Machine learning library for Python providing simple and efficient tools.

- **TensorFlow** ![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white): Open-source platform for machine learning and artificial intelligence.

- **Pandas** ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white): Data analysis and manipulation library for Python.

- **XGBoost** ![XGBoost](https://img.shields.io/badge/XGBoost-FF9900?style=flat-square&logo=xgboost&logoColor=white): Scalable and flexible gradient boosting library.

- **CatBoost** ![CatBoost](https://img.shields.io/badge/CatBoost-5C4F8C?style=flat-square&logo=catboost&logoColor=white): Gradient boosting library that handles categorical features efficiently.

- **Seaborn** ![Seaborn](https://img.shields.io/badge/Seaborn-9C27B0?style=flat-square&logo=seaborn&logoColor=white): Statistical data visualization library built on top of Matplotlib.

- **Keras** ![Keras](https://img.shields.io/badge/Keras-FF6F00?style=flat-square&logo=keras&logoColor=white): High-level neural networks API, written in Python and capable of running on top of TensorFlow.

 
<details>
  <summary><strong>Click here for more details on the Methodology</strong></summary>


### Methodology

To ensure a thorough analysis and implementation, I explored multiple models and techniques, which demonstrates my adaptability and desire to leave no stone unturned. Below is a brief look into the methodologies that helped drive the project’s success:

#### Models
Several machine learning models were tested to predict customer churn, including LightGBM (LGB), XGBoost (XGB), CatBoost (Cat), and Artificial Neural Networks (ANN). After thorough comparison, **LightGBM** and **ANN** outperformed the rest, offering the best balance of accuracy and interpretability.

#### Feature Engineering
Featuretools was used for automatic feature construction, which proved to be highly effective. The top 15 features were mostly generated by Featuretools, highlighting the benefits of automated feature engineering.

 *I believe that sophisticated feature engineering techniques, a key to improving model accuracy.*

#### Data Imputation
Missing values were handled using median imputation for numerical data, while categorical features received a special "missing" category. This was achieved using the `ColumnTransformer` and `Imputer` classes.

#### Performance Metrics
To balance recall and precision, I used a custom weighted recall metric:  
- **Weighted Recall** = 0.65 * Recall + 0.35 * F1 Score.

The model achieved a **recall of 0.80** and a **precision of 0.54**. Emphasizing the recall ensures that the model captures as many churners as possible, which is crucial for customer retention strategies. 

*I've optimized the metrics based on the project’s goals, ensuring I’m providing real-world, actionable insights.*

</details>

## Key Findings

1. **Charges**: Higher churn rates among monthly users are attributed to **charges**. Customers with higher costs are more likely to churn. ![Charges and Churn Plot](https://github.com/d-sutariya/custmer_churn_prediction/blob/main/reports/visuals/chargs_vs_contract_type.png)

2. **Senior Citizens**: Senior citizens have a notably higher churn rate—approximately double that of younger customers. This is because they tend to be more cautious with their finances, leading them to reconsider non-essential services more frequently.![Senior Citizens and Churn Plot](https://github.com/d-sutariya/custmer_churn_prediction/blob/main/reports/visuals/age_vs_Churn.png)

3. **Automatic Payment Method**: Customers using automatic payment methods have a lower churn rate. The convenience of automatic payments reduces the likelihood of reconsidering their commitment to the service.![Automatic Payment and Churn Plot](https://github.com/d-sutariya/custmer_churn_prediction/blob/main/reports/visuals/p_method_vs_Churn.png)

4. **Fiber Optic Service**: Customers show a clear preference against fiber optic services, suggesting potential issues with reliability, speed, or customer support. Addressing these issues could help reduce churn and increase customer satisfaction.![Fiber Optic and Churn Plot](https://github.com/d-sutariya/custmer_churn_prediction/blob/main/reports/visuals/internet_service_vs_Churn.png)


**Follow the steps below to get the project up and running and uncover the secrets behind its success.**

## 1. Project Setup: Getting Started the Right Way

Before you dive into the data, let’s get your environment set up:

- **Clone the Project:**
  
    ```bash
    git clone https://github.com/d-sutariya/customer_churn_prediction.git
    ```
  
- **Create and Activate a Virtual Environment:**
  
    ```bash
    python -m venv env
    # On Unix: source env/bin/activate
    # On Windows: env\Scripts\activate
    ```

- **Install Dependencies:**
  
    ```bash
    pip install -r requirements.txt
    ```

- **Run the Setup Script:**

    ```bash
    cd customer_churn_prediction
    python src/config/setup_project.py 
  
    ```

*Now you're ready to transform raw data into valuable insights that could change business operations.*

## 2. Transforming Raw Data: See the Magic Happen

Transform raw customer data into a training-ready dataset with the following command:

- **Run the Transformation Script:**
  
    ```bash
    python src/data/make_dataset.py --input_file_path data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
    ```

***Curious to see the process in action?*** Explore my Jupyter notebooks for an in-depth look!

## 3. Jupyter Notebooks: Unlock the Insights

Here’s where the real magic happens. My Jupyter notebooks offer deep insights into customer churn predictions. Dive into them to see innovative approaches and results:

- **Explore the Notebook:**

    [Customer Churn EDA Notebook](https://github.com/d-sutariya/custmer_churn_prediction/tree/main/notebooks/0.01-deep-EDA.ipynb)

*These notebooks are not just scripts—they are a window into the detailed thought process behind every step.*

## 4. Production-Ready Scripts: Efficiency and Scalability

Head over to the `src/` directory to find core production scripts designed for efficiency and scalability:

- **ETL Pipeline Script:**
  
    - [`src/data/make_dataset.py`](https://github.com/d-sutariya/custmer_churn_prediction/tree/main/src/data/make_dataset.py)

- **Data Pipeline Configuration:**
  
    - [`src/pipeline/dvc.yaml`](https://github.com/d-sutariya/custmer_churn_prediction/tree/main/src/pipeline/dvc.yaml)

- **Hyperparameter Optimization:**
  
    - [`src/optimization/tuning_and_tracking.py`](https://github.com/d-sutariya/custmer_churn_prediction/tree/main/src/optimization/tuning_and_tracking.py)

*Imagine these scripts as part of your production pipeline. They are designed to be efficient and scalable.*

## 5. Post-Deployment Magic: Ongoing Success

The journey doesn’t end with deployment. The `post_deployment/` directory includes scripts for:

- Transforming new data.
- Periodically retraining the model.

Check the scripts here:

- [`post_deployment`](https://github.com/d-sutariya/custmer_churn_prediction/tree/main/post_deployment/)

*These scripts ensure your operations team stays ahead of potential issues and maintains model accuracy over time.*

## 6. Final Thoughts: Dive Deep into What Sets This Apart

I encourage you to explore this project thoroughly. From cutting-edge data transformations to production-ready pipelines, every piece has been crafted to address real-world problems.

*As you delve into the materials, I hope you see the value and potential of this project and how it could fit into your business.*


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

Feel free to reach out with any questions or feedback. I look forward to your thoughts!
