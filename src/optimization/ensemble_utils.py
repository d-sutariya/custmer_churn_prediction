import numpy as np
import pandas as pd
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential

# this function drop the columns which is not parameters from the study 
def drop_unnessesary_columns(df):
    """
    Drops columns from the DataFrame that do not contain the substring 'params' in their names.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame from which columns will be dropped.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with only the columns that contain 'params' in their names.
    """
    drop_cols = []
    for column in df.columns:
        if 'params' not in column:
            drop_cols.append(column)
    df = df.drop(columns=drop_cols)
    return df

def get_predictions(models, X):
    """
    Generate predictions from a list of models.

    Parameters
    ----------
    models : list
        A list of trained models that have a `predict` method.
    X : array-like
        The input data to be used for generating predictions.

    Returns
    -------
    numpy.ndarray
        An array of predictions from each model, where each row corresponds to the predictions of a single model.
    """
    predictions = []
    for model in models:
        pred = model.predict(X).ravel()
        predictions.append(pred)
    return np.array(predictions)

# Function to perform soft voting
def soft_voting(models, X):
    """
    Perform soft voting ensemble method on the given models and input data.

    Soft voting averages the predicted probabilities of each model to make a final prediction.

    Parameters
    ----------
    models : list
        A list of trained models that support the predict_proba method.
    X : array-like
        The input data to predict on.

    Returns
    -------
    numpy.ndarray
        The averaged predicted probabilities from all models.
    """
    predictions = get_predictions(models, X)
    avg_predictions = np.mean(predictions, axis=0)
    return avg_predictions

def clean_hyperparameters(hyperparameters):
    """
    Cleans the hyperparameters dictionary by removing specific prefixes from keys and converting certain float values to integers.

    This function processes a dictionary of hyperparameters by:
    - Removing the 'params_' prefix from each key.
    - Converting float values to integers if the float value is actually an integer.

    Parameters
    ----------
    hyperparameters : dict
        A dictionary where keys are hyperparameter names (potentially with 'params_' prefix) and values are hyperparameter values.

    Returns
    -------
    dict
        A cleaned dictionary with updated keys and values.
    """
    clean_params = {}
    for key, value in hyperparameters.items():
        clean_key = key.replace('params_', '')  # Remove 'params_' prefix from the key
        if isinstance(value, float) and value.is_integer():  # Check if float value is actually an integer
            clean_params[clean_key] = int(value)  # Convert to integer if float is an integer
        else:
            clean_params[clean_key] = value  # Otherwise, keep the value as it is
    return clean_params  # Return the cleaned hyperparameters dictionary



def weighted_voting(predictions, weights):
    """
    Combine predictions from multiple models using weighted voting.

    Parameters
    ----------
    predictions : dict
        A dictionary where keys are model names and values are arrays of predictions from each model.
    weights : dict
        A dictionary where keys are model names and values are the corresponding weights for each model.

    Returns
    -------
    np.ndarray
        An array of combined predictions, weighted and normalized by the sum of weights.
    """
    combined_predictions = np.zeros_like(next(iter(predictions.values())))  # Initialize array for combined predictions
    for model_name, model_preds in predictions.items():
        combined_predictions += weights[model_name] * model_preds  # Weight and sum predictions from each model
    combined_predictions /= sum(weights.values())  # Normalize the combined predictions by the sum of weights
    return combined_predictions  # Return the final weighted prediction

def create_nn_model(params, input_shape):
    model = Sequential()
    model.add(Dense(params['units_layer1'], input_dim=input_shape, activation='relu'))  # First dense layer
    model.add(Dropout(params['dropout_layer1']))  # Dropout layer after first dense layer
    model.add(Dense(params['units_layer2'], activation='relu'))  # Second dense layer
    model.add(Dropout(params['dropout_layer2']))  # Dropout layer after second dense layer
    model.add(Dense(1, activation='sigmoid'))  # Output layer with sigmoid activation for binary classification
    optimizer = Adam(learning_rate=params['learning_rate'])  # Adam optimizer with specified learning rate
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['AUC'])  # Compile model with binary crossentropy loss and AUC metric
    return model  # Return the compiled model


