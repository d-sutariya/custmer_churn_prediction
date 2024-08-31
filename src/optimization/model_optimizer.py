import numpy as np
import pandas as pd
import optuna 
import time
import tensorflow as tf
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,f1_score,precision_score

# This class is optimizing the models on weighted recall score which is defined below
# weighted recall = 0.65 * recall + 0.35 * f1 
# I am adding various evaluation metrics like recall,precision,f1,weighted_recall,accuracy to evaluate better

class ModelOptimizer:
    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set
    
    def cat_objective(self, trial):
        param = {
            'objective': 'Logloss',  # Set objective to Logloss
            'eval_metric': 'AUC',  # Use AUC as evaluation metric
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),  # Suggest learning rate from log-uniform distribution
            'depth': trial.suggest_int('depth', 3, 12),  # Suggest tree depth
            'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-8, 10.0),  # Suggest L2 regularization parameter
            'bagging_temperature': trial.suggest_uniform('bagging_temperature', 0.0, 1.0),  # Suggest bagging temperature
            'border_count': trial.suggest_int('border_count', 1, 255),  # Suggest border count
            'scale_pos_weight': trial.suggest_loguniform('scale_pos_weight', 1e-3, 1.0),  # Suggest scale positive weight
            'verbose': 0,
            'early_stopping_rounds': 200,  # Set early stopping rounds
            'iterations': 3000  # Set maximum number of iterations
        }

        train_data = cb.Pool(data=self.train_set.drop('Churn', axis=1), label=self.train_set['Churn'])  # Prepare training data
        test_data = cb.Pool(data=self.test_set.drop('Churn', axis=1), label=self.test_set['Churn'])  # Prepare testing data

        cat = cb.CatBoostClassifier(**param)  # Initialize CatBoostClassifier with suggested parameters
        cat.fit(train_data, eval_set=test_data, use_best_model=True)  # Train the model with early stopping

        preds = cat.predict_proba(self.test_set.drop('Churn', axis=1))[:, 1]  # Get prediction probabilities
        preds_digits = [1 if pred >= 0.4 else 0 for pred in preds]  # Convert probabilities to binary predictions
        roc_auc = roc_auc_score(self.test_set['Churn'], preds)  # Calculate ROC AUC score
        f1 = f1_score(self.test_set['Churn'], preds_digits)  # Calculate F1 score
        recall = recall_score(self.test_set['Churn'], preds_digits)  # Calculate recall score
        accuracy = accuracy_score(self.test_set['Churn'], preds_digits)  # Calculate accuracy score
        weighted_recall = self.weighted_recall(self.test_set['Churn'], preds_digits)  # Calculate weighted recall score
        prec = precision_score(self.test_set['Churn'], preds_digits)  # Calculate precision score
        trial.set_user_attr('roc', roc_auc)  # Log ROC AUC score in Optuna trial attributes
        trial.set_user_attr('f1', f1)  # Log F1 score in Optuna trial attributes
        trial.set_user_attr('accuracy', accuracy)  # Log accuracy score in Optuna trial attributes
        trial.set_user_attr('recall', recall)  # Log recall score in Optuna trial attributes
        trial.set_user_attr('precision', prec)  # Log precision score in Optuna trial attributes
        return weighted_recall  # Return weighted recall score for optimization

    def optimize_catboost(self, n_trials=100):
        cat_study = optuna.create_study(direction='maximize',
                                        sampler=optuna.samplers.TPESampler(),  # Use TPE sampler for optimization
                                        pruner=optuna.pruners.HyperbandPruner(
                                            min_resource=5,  # Minimum resource for pruning
                                            max_resource=10,  # Maximum resource for pruning
                                            reduction_factor=2  # Reduction factor for pruning
                                        ))
        start_time = time.time()  # Record start time
        cat_study.optimize(self.cat_objective, n_trials=n_trials , n_jobs=-1)  # Start optimization
        print(f'time taken is {time.time() - start_time}')  # Print time taken for optimization
        cat_study_df = cat_study.trials_dataframe()  # Convert study trials to DataFrame
        return cat_study_df  # Return DataFrame with trial results

    def nn_objective(self, trial):
        model = Sequential()  # Initialize a Sequential model
        model.add(Dense(trial.suggest_int('units_layer1', 32, 512), input_dim=self.train_set.drop('Churn', axis=1).shape[1]))  # Add first dense layer with suggested units
        model.add(LeakyReLU(alpha=0.01))  # Add LeakyReLU activation
        model.add(Dropout(trial.suggest_uniform('dropout_layer1', 0.2, 0.5)))  # Add Dropout with suggested rate

        model.add(Dense(trial.suggest_int('units_layer2', 32, 512)))  # Add second dense layer with suggested units
        model.add(LeakyReLU(alpha=0.01))  # Add LeakyReLU activation
        model.add(Dropout(trial.suggest_uniform('dropout_layer2', 0.2, 0.5)))  # Add Dropout with suggested rate

        model.add(Dense(1, activation='sigmoid'))  # Add output layer with sigmoid activation

        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)  # Suggest learning rate from log-uniform distribution
        optimizer = Adam(learning_rate=learning_rate)  # Use Adam optimizer with suggested learning rate

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])  # Compile the model with binary crossentropy loss

        model.fit(self.train_set.drop('Churn', axis=1), self.train_set['Churn'],
                  validation_data=(self.test_set.drop('Churn', axis=1), self.test_set['Churn']),
                  batch_size=trial.suggest_int('batch_size', 32, 128),  # Suggest batch size
                  epochs=50,  # Set number of epochs
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],  # Use early stopping
                  verbose=0)

        preds = model.predict(self.test_set.drop('Churn', axis=1)).ravel()  # Get prediction probabilities
        preds_digits = [1 if pred >= 0.4 else 0 for pred in preds]  # Convert probabilities to binary predictions
        roc_auc = roc_auc_score(self.test_set['Churn'], preds)  # Calculate ROC AUC score
        f1 = f1_score(self.test_set['Churn'], preds_digits)  # Calculate F1 score
        recall = recall_score(self.test_set['Churn'], preds_digits)  # Calculate recall score
        accuracy = accuracy_score(self.test_set['Churn'], preds_digits)  # Calculate accuracy score
        weighted_recall = self.weighted_recall(self.test_set['Churn'], preds_digits)  # Calculate weighted recall score
        prec = precision_score(self.test_set['Churn'], preds_digits)  # Calculate precision score
        trial.set_user_attr('roc', roc_auc)  # Log ROC AUC score in Optuna trial attributes
        trial.set_user_attr('f1', f1)  # Log F1 score in Optuna trial attributes
        trial.set_user_attr('accuracy', accuracy)  # Log accuracy score in Optuna trial attributes
        trial.set_user_attr('recall', recall)  # Log recall score in Optuna trial attributes
        trial.set_user_attr('precision', prec)  # Log precision score in Optuna trial attributes
        return weighted_recall  # Return weighted recall score for optimization

    def optimize_nn(self, n_trials=100):
        nn_study = optuna.create_study(direction='maximize',
                                       sampler=optuna.samplers.TPESampler(),  # Use TPE sampler for optimization
                                       pruner=optuna.pruners.HyperbandPruner(
                                           min_resource=5,  # Minimum resource for pruning
                                           max_resource=20,  # Maximum resource for pruning
                                           reduction_factor=2  # Reduction factor for pruning
                                       ))
        start_time = time.time()  # Record start time
        nn_study.optimize(self.nn_objective, n_trials=n_trials , n_jobs=-1)  # Start optimization
        print(f'time taken is {time.time() - start_time}')  # Print time taken for optimization
        nn_study_df = nn_study.trials_dataframe()  # Convert study trials to DataFrame
        return nn_study_df  # Return DataFrame with trial results
    
    def weighted_recall(self, y_true, y_pred):
        recall = recall_score(y_true, y_pred)  # Calculate recall score
        f1 = f1_score(y_true, y_pred)  # Calculate F1 score
        weighted_metric = 0.65 * recall + 0.35 * f1  # Calculate weighted recall score
        return weighted_metric  # Return weighted recall score
    
    def lgb_objective(self, trial):
        param = {
            'objective': 'binary',  # Set objective to binary classification
            'metric': 'auc',  # Use AUC as evaluation metric
            'verbosity': -1,
            'boosting_type': 'gbdt',  # Use GBDT boosting type
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),  # Suggest L1 regularization parameter
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),  # Suggest L2 regularization parameter
            'num_leaves': trial.suggest_int('num_leaves', 2, 256),  # Suggest number of leaves
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),  # Suggest feature fraction
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),  # Suggest bagging fraction
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),  # Suggest bagging frequency
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),  # Suggest minimum child samples
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),  # Suggest learning rate from log-uniform distribution
        }
        dtrain = lgb.Dataset(self.train_set.drop('Churn', axis=1), label=self.train_set['Churn'])  # Prepare training data
        dvalid = lgb.Dataset(self.test_set.drop('Churn', axis=1), label=self.test_set['Churn'])  # Prepare validation data

        gbm = lgb.train(param, dtrain, valid_sets=[dvalid], num_boost_round=10000,  # Train model with early stopping
                        early_stopping_rounds=100, verbose_eval=False)

        preds = gbm.predict(self.test_set.drop('Churn', axis=1))  # Get prediction probabilities
        preds_digits = [1 if pred >= 0.4 else 0 for pred in preds]  # Convert probabilities to binary predictions
        roc_auc = roc_auc_score(self.test_set['Churn'], preds)  # Calculate ROC AUC score
        f1 = f1_score(self.test_set['Churn'], preds_digits)  # Calculate F1 score
        recall = recall_score(self.test_set['Churn'], preds_digits)  # Calculate recall score
        accuracy = accuracy_score(self.test_set['Churn'], preds_digits)  # Calculate accuracy score
        weighted_recall = self.weighted_recall(self.test_set['Churn'], preds_digits)  # Calculate weighted recall score
        prec = precision_score(self.test_set['Churn'], preds_digits)  # Calculate precision score
        trial.set_user_attr('roc', roc_auc)  # Log ROC AUC score in Optuna trial attributes
        trial.set_user_attr('f1', f1)  # Log F1 score in Optuna trial attributes
        trial.set_user_attr('accuracy', accuracy)  # Log accuracy score in Optuna trial attributes
        trial.set_user_attr('recall', recall)  # Log recall score in Optuna trial attributes
        trial.set_user_attr('precision', prec)  # Log precision score in Optuna trial attributes
        return weighted_recall  # Return weighted recall score for optimization

    def optimize_lgb(self, n_trials=100):
        lgb_study = optuna.create_study(direction='maximize',
                                        sampler=optuna.samplers.TPESampler(),  # Use TPE sampler for optimization
                                        pruner=optuna.pruners.HyperbandPruner(
                                            min_resource=5,  # Minimum resource for pruning
                                            max_resource=100,  # Maximum resource for pruning
                                            reduction_factor=2  # Reduction factor for pruning
                                        ))
        start_time = time.time()  # Record start time
        lgb_study.optimize(self.lgb_objective, n_trials=n_trials , n_jobs=-1)  # Start optimization
        print(f'time taken is {time.time() - start_time}')  # Print time taken for optimization
        lgb_study_df = lgb_study.trials_dataframe()  # Convert study trials to DataFrame
        return lgb_study_df  # Return DataFrame with trial results

    def xgb_objective(self, trial):
        param = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'verbosity': 0,
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_loguniform('gamma', 1e-8, 10.0),
            'lambda': trial.suggest_loguniform('lambda', 1e-8, 10.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-8, 10.0),
            'subsample': trial.suggest_uniform('subsample', 0.4, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.4, 1.0),
            'early_stopping_rounds': 200,
            'num_boost_round':3000,
        }

        train_data = xgb.DMatrix(self.train_set.drop('Churn', axis=1), label=self.train_set['Churn'])
        test_data = xgb.DMatrix(self.test_set.drop('Churn', axis=1), label=self.test_set['Churn'])

        gbm = xgb.train(param, train_data, evals=[(test_data, 'eval')], verbose_eval=False)

        preds = gbm.predict(test_data)
        preds_digits = [1 if pred >= 0.4 else 0 for pred in preds]
        roc_auc = roc_auc_score(self.test_set['Churn'], preds)
        f1 = f1_score(self.test_set['Churn'],preds_digits)
        recall = recall_score(self.test_set['Churn'],preds_digits)
        accuracy = accuracy_score(self.test_set['Churn'],preds_digits)
        weighted_recall = self.weighted_recall(self.test_set['Churn'],preds_digits)
        prec = precision_score(self.test_set['Churn'],preds_digits)
        trial.set_user_attr('roc',roc_auc)
        trial.set_user_attr('f1',f1)
        trial.set_user_attr('accuracy',accuracy)
        trial.set_user_attr('recall',recall)
        trial.set_user_attr('precision',prec)
        return weighted_recall

    def optimize_xgb(self, n_trials=100):
        xgb_study = optuna.create_study(direction='maximize',
                                        sampler=optuna.samplers.TPESampler(),
                                        pruner=optuna.pruners.HyperbandPruner(
                                            min_resource=5,
                                            max_resource=20,
                                            reduction_factor=2
                                        ))
        start_time = time.time()
        xgb_study.optimize(self.xgb_objective, n_trials=n_trials , n_jobs = -1)
        print(f'time taken is {time.time() - start_time}')
        xgb_study_df = xgb_study.trials_dataframe()
        return xgb_study_df
    
# save different results to kaggle working directory
def save_results(study_dfs, file_paths):
    for df, path in zip(study_dfs, file_paths):
        df.to_csv(path)

# load the results 
def load_results(file_paths):
    return [pd.read_csv(path).drop('Unnamed: 0', axis=1).reset_index(drop=True) for path in file_paths]
