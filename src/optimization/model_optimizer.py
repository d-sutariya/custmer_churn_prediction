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
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, precision_score

class ModelOptimizer:
    def __init__(self, train_set: pd.DataFrame, test_set: pd.DataFrame):
        """
        Initialize the ModelOptimizer.

        Parameters
        ----------
        train_set : pd.DataFrame
            Training dataset.
        test_set : pd.DataFrame
            Testing dataset.
        """
        self.train_set = train_set
        self.test_set = test_set

    def cat_objective(self, trial: optuna.trial.Trial) -> float:
        """
        Objective function for optimizing CatBoostClassifier using Optuna.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial object for suggesting hyperparameters.

        Returns
        -------
        float
            Weighted recall score for the trial.
        """
        param = {
            'objective': 'Logloss',
            'eval_metric': 'AUC',
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
            'depth': trial.suggest_int('depth', 3, 12),
            'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-8, 10.0),
            'bagging_temperature': trial.suggest_uniform('bagging_temperature', 0.0, 1.0),
            'border_count': trial.suggest_int('border_count', 1, 255),
            'scale_pos_weight': trial.suggest_loguniform('scale_pos_weight', 1e-3, 1.0),
            'verbose': 0,
            'early_stopping_rounds': 200,
            'iterations': 3000
        }

        train_data = cb.Pool(data=self.train_set.drop('Churn', axis=1), label=self.train_set['Churn'])
        test_data = cb.Pool(data=self.test_set.drop('Churn', axis=1), label=self.test_set['Churn'])

        cat = cb.CatBoostClassifier(**param)
        cat.fit(train_data, eval_set=test_data, use_best_model=True)

        preds = cat.predict_proba(self.test_set.drop('Churn', axis=1))[:, 1]
        preds_digits = [1 if pred >= 0.4 else 0 for pred in preds]
        roc_auc = roc_auc_score(self.test_set['Churn'], preds)
        f1 = f1_score(self.test_set['Churn'], preds_digits)
        recall = recall_score(self.test_set['Churn'], preds_digits)
        accuracy = accuracy_score(self.test_set['Churn'], preds_digits)
        weighted_recall = self.weighted_recall(self.test_set['Churn'], preds_digits)
        prec = precision_score(self.test_set['Churn'], preds_digits)
        trial.set_user_attr('roc', roc_auc)
        trial.set_user_attr('f1', f1)
        trial.set_user_attr('accuracy', accuracy)
        trial.set_user_attr('recall', recall)
        trial.set_user_attr('precision', prec)
        return weighted_recall

    def optimize_catboost(self, n_trials: int = 100) -> pd.DataFrame:
        """
        Optimizes CatBoostClassifier using Optuna.

        Parameters
        ----------
        n_trials : int, optional
            Number of trials for optimization (default is 100).

        Returns
        -------
        pd.DataFrame
            DataFrame containing the results of the optimization trials.
        """
        cat_study = optuna.create_study(direction='maximize',
                                        sampler=optuna.samplers.TPESampler(),
                                        pruner=optuna.pruners.HyperbandPruner(
                                            min_resource=5,
                                            max_resource=10,
                                            reduction_factor=2
                                        ))
        start_time = time.time()
        cat_study.optimize(self.cat_objective, n_trials=n_trials, n_jobs=-1)
        print(f'time taken is {time.time() - start_time}')
        cat_study_df = cat_study.trials_dataframe()
        return cat_study_df

    def nn_objective(self, trial: optuna.trial.Trial) -> float:
        """
        Objective function for optimizing a neural network using Optuna.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial object for suggesting hyperparameters.

        Returns
        -------
        float
            Weighted recall score for the trial.
        """
        model = Sequential()
        model.add(Dense(trial.suggest_int('units_layer1', 32, 512), input_dim=self.train_set.drop('Churn', axis=1).shape[1]))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(trial.suggest_uniform('dropout_layer1', 0.2, 0.5)))

        model.add(Dense(trial.suggest_int('units_layer2', 32, 512)))
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(trial.suggest_uniform('dropout_layer2', 0.2, 0.5)))

        model.add(Dense(1, activation='sigmoid'))

        learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-2)
        optimizer = Adam(learning_rate=learning_rate)

        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])

        model.fit(self.train_set.drop('Churn', axis=1), self.train_set['Churn'],
                  validation_data=(self.test_set.drop('Churn', axis=1), self.test_set['Churn']),
                  batch_size=trial.suggest_int('batch_size', 32, 128),
                  epochs=50,
                  callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)],
                  verbose=0)

        preds = model.predict(self.test_set.drop('Churn', axis=1)).ravel()
        preds_digits = [1 if pred >= 0.4 else 0 for pred in preds]
        roc_auc = roc_auc_score(self.test_set['Churn'], preds)
        f1 = f1_score(self.test_set['Churn'], preds_digits)
        recall = recall_score(self.test_set['Churn'], preds_digits)
        accuracy = accuracy_score(self.test_set['Churn'], preds_digits)
        weighted_recall = self.weighted_recall(self.test_set['Churn'], preds_digits)
        prec = precision_score(self.test_set['Churn'], preds_digits)
        trial.set_user_attr('roc', roc_auc)
        trial.set_user_attr('f1', f1)
        trial.set_user_attr('accuracy', accuracy)
        trial.set_user_attr('recall', recall)
        trial.set_user_attr('precision', prec)
        return weighted_recall

    def optimize_nn(self, n_trials: int = 100) -> pd.DataFrame:
        """
        Optimizes a neural network using Optuna.

        Parameters
        ----------
        n_trials : int, optional
            Number of trials for optimization (default is 100).

        Returns
        -------
        pd.DataFrame
            DataFrame containing the results of the optimization trials.
        """
        nn_study = optuna.create_study(direction='maximize',
                                       sampler=optuna.samplers.TPESampler(),
                                       pruner=optuna.pruners.HyperbandPruner(
                                           min_resource=5,
                                           max_resource=20,
                                           reduction_factor=2
                                       ))
        start_time = time.time()
        nn_study.optimize(self.nn_objective, n_trials=n_trials, n_jobs=-1)
        print(f'time taken is {time.time() - start_time}')
        nn_study_df = nn_study.trials_dataframe()
        return nn_study_df

    def weighted_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates a weighted recall score.

        Parameters
        ----------
        y_true : np.ndarray
            True labels.
        y_pred : np.ndarray
            Predicted labels.

        Returns
        -------
        float
            Weighted recall score.
        """
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        weighted_metric = 0.65 * recall + 0.35 * f1
        return weighted_metric

    def lgb_objective(self, trial: optuna.trial.Trial) -> float:
        """
        Objective function for optimizing LightGBM using Optuna.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial object for suggesting hyperparameters.

        Returns
        -------
        float
            Weighted recall score for the trial.
        """
        param = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
            'num_leaves': trial.suggest_int('num_leaves', 2, 255),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
            'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.5, 1.0),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.5, 1.0),
            'max_depth': trial.suggest_int('max_depth', -1, 15),
            'num_threads': -1,
            'early_stopping_round': 200,
            'n_estimators': 3000
        }

        train_data = lgb.Dataset(data=self.train_set.drop('Churn', axis=1), label=self.train_set['Churn'])
        valid_data = lgb.Dataset(data=self.test_set.drop('Churn', axis=1), label=self.test_set['Churn'])

        lgb_model = lgb.train(param, train_data, valid_sets=[valid_data], verbose_eval=0)

        preds = lgb_model.predict(self.test_set.drop('Churn', axis=1))
        preds_digits = [1 if pred >= 0.4 else 0 for pred in preds]
        roc_auc = roc_auc_score(self.test_set['Churn'], preds)
        f1 = f1_score(self.test_set['Churn'], preds_digits)
        recall = recall_score(self.test_set['Churn'], preds_digits)
        accuracy = accuracy_score(self.test_set['Churn'], preds_digits)
        weighted_recall = self.weighted_recall(self.test_set['Churn'], preds_digits)
        prec = precision_score(self.test_set['Churn'], preds_digits)
        trial.set_user_attr('roc', roc_auc)
        trial.set_user_attr('f1', f1)
        trial.set_user_attr('accuracy', accuracy)
        trial.set_user_attr('recall', recall)
        trial.set_user_attr('precision', prec)
        return weighted_recall

    def optimize_lightgbm(self, n_trials: int = 100) -> pd.DataFrame:
        """
        Optimizes LightGBM using Optuna.

        Parameters
        ----------
        n_trials : int, optional
            Number of trials for optimization (default is 100).

        Returns
        -------
        pd.DataFrame
            DataFrame containing the results of the optimization trials.
        """
        lgb_study = optuna.create_study(direction='maximize',
                                        sampler=optuna.samplers.TPESampler(),
                                        pruner=optuna.pruners.HyperbandPruner(
                                            min_resource=5,
                                            max_resource=10,
                                            reduction_factor=2
                                        ))
        start_time = time.time()
        lgb_study.optimize(self.lgb_objective, n_trials=n_trials, n_jobs=-1)
        print(f'time taken is {time.time() - start_time}')
        lgb_study_df = lgb_study.trials_dataframe()
        return lgb_study_df

    def xgb_objective(self, trial: optuna.trial.Trial) -> float:
        """
        Objective function for optimizing XGBoost using Optuna.

        Parameters
        ----------
        trial : optuna.trial.Trial
            A trial object for suggesting hyperparameters.

        Returns
        -------
        float
            Weighted recall score for the trial.
        """
        param = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-4, 1e-1),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            'lambda': trial.suggest_loguniform('lambda', 1e-8, 10.0),
            'alpha': trial.suggest_loguniform('alpha', 1e-8, 10.0),
            'scale_pos_weight': trial.suggest_loguniform('scale_pos_weight', 1e-3, 1.0),
            'n_estimators': 3000
        }

        train_data = xgb.DMatrix(data=self.train_set.drop('Churn', axis=1), label=self.train_set['Churn'])
        valid_data = xgb.DMatrix(data=self.test_set.drop('Churn', axis=1), label=self.test_set['Churn'])

        xgb_model = xgb.train(param, train_data, evals=[(valid_data, 'validation')], early_stopping_rounds=200, verbose_eval=0)

        preds = xgb_model.predict(xgb.DMatrix(self.test_set.drop('Churn', axis=1)))
        preds_digits = [1 if pred >= 0.4 else 0 for pred in preds]
        roc_auc = roc_auc_score(self.test_set['Churn'], preds)
        f1 = f1_score(self.test_set['Churn'], preds_digits)
        recall = recall_score(self.test_set['Churn'], preds_digits)
        accuracy = accuracy_score(self.test_set['Churn'], preds_digits)
        weighted_recall = self.weighted_recall(self.test_set['Churn'], preds_digits)
        prec = precision_score(self.test_set['Churn'], preds_digits)
        trial.set_user_attr('roc', roc_auc)
        trial.set_user_attr('f1', f1)
        trial.set_user_attr('accuracy', accuracy)
        trial.set_user_attr('recall', recall)
        trial.set_user_attr('precision', prec)
        return weighted_recall
