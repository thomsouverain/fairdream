import os
import wget
import pickle
import random
import statistics
from math import floor
from pathlib import Path
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import torch
import xgboost
from sklearn import ensemble
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.augmentations import ClassificationSMOTE
import torch
import scipy
from xgboost import XGBClassifier
from xgboost import XGBRegressor

from fairdream.compute_scores import compute_best_fscore
from fairdream.compute_scores import get_auc
# import shap
from sklearn.neural_network import MLPClassifier

# Prepares data, selects important features, and trains baseline XGB models
# & Add predictions of a specified model (uncorrected or trained with a fairness purpose) to the initial dataset, to then compare models (fair / stat performances)


def automatic_preprocessing(dataset_name: str) -> pd.DataFrame:
    """According to a specified dataset, returns the dataset pre-processed to allow more performant training.
    For the moment, only available with dataset_name == "housing_nights_dataset".

    Parameters
    ----------
    dataset_name : str

    Returns
    -------
    pd.DataFrame
    """
    if dataset_name == "housing_nights_dataset":

        requests_train = pd.read_csv(
            filepath_or_buffer="housing_nights_dataset/requests_train.csv",
            sep=",",
            low_memory=False,
            error_bad_lines=False,
        )

        requests_test = pd.read_csv(
            filepath_or_buffer="housing_nights_dataset/requests_test.csv",
            sep=",",
            low_memory=False,
            error_bad_lines=False,
        )

        composition = requests_train["group_composition_label"]
        nights = requests_train["granted_number_of_nights"]

        individuals_train = pd.read_csv(
            filepath_or_buffer="housing_nights_dataset/individuals_train.csv",
            sep=",",
            low_memory=False,
            error_bad_lines=False,
        )

        individuals_test = pd.read_csv(
            filepath_or_buffer="housing_nights_dataset/individuals_test.csv",
            sep=",",
            low_memory=False,
            error_bad_lines=False,
        )

        train_set = requests_train.loc[
            :,
            (
                "granted_number_of_nights",
                "child_situation",
                "district",
                "group_composition_id",
                "housing_situation_id",
                "number_of_underage",
                "request_id",
            ),
        ].set_index("request_id")

        # Step 1: Dataset with categorical features to be one-hot encoded
        list_train_set = [
            (individuals_train, "individual_role", "child"),
            (individuals_train, "gender", "female"),
            (requests_train, "victim_of_violence", "t"),
            (individuals_train, "individual_role", "isolated parent"),
            (requests_train, "child_to_come", "t"),
            (individuals_train, "individual_role_2_label", "child/underage with family"),
            (
                individuals_train,
                "housing_situation_2_label",
                "hotel paid by the emergency structure",
            ),
            (individuals_train, "housing_situation_2_label", "on the street"),
            (individuals_train, "housing_situation_2_label", "emergency accomodation"),
            (requests_train, "housing_situation_label", "hotel paid by an association"),
            (requests_train, "housing_situation_label", "mobile or makeshift shelter"),
            (
                requests_train,
                "housing_situation_label",
                "religious place (church, mosque, synogogue)",
            ),
            (requests_train, "housing_situation_label", "inclusion structure"),
            (requests_train, "housing_situation_label", "other"),
            (requests_train, "housing_situation_label", "emergency structure"),
            (requests_train, "housing_situation_label", "public hospital"),
        ]

        for previous_dataset, individual_column, criteria in list_train_set:
            train_set = new_dataset_column(train_set, previous_dataset, individual_column, criteria)

        # Step 2: add time data (to consider potentially vulnerable people, with min or max age)
        individuals_train["birth_year"] = individuals_train["birth_year"].fillna(
            individuals_train["birth_year"].mean()
        )
        train_set["min_birth_year"] = (
            individuals_train.loc[:, ["birth_year", "request_id"]].groupby("request_id").min()
        )
        train_set["max_birth_year"] = (
            individuals_train.loc[:, ["birth_year", "request_id"]].groupby("request_id").max()
        )

        # Step 3 : add the dtypes int64 from individuals_train (groupby -> min)
        train_set["housing_situation_2_id"] = (
            individuals_train.loc[:, ["housing_situation_2_id", "request_id"]]
            .groupby("request_id")
            .min()
        )
        train_set["individual_role_2_id"] = (
            individuals_train.loc[:, ["individual_role_2_id", "request_id"]]
            .groupby("request_id")
            .min()
        )
        train_set["marital_status_id"] = (
            individuals_train.loc[:, ["marital_status_id", "request_id"]]
            .groupby("request_id")
            .min()
        )

        requests_train["answer - group creation date"] = pd.to_datetime(
            requests_train["answer_creation_date"]
        ).values.astype(np.int64) - pd.to_datetime(
            requests_train["group_creation_date"]
        ).values.astype(
            np.int64
        )

        train_set["answer - group creation date"] = (
            requests_train.loc[:, ["answer - group creation date", "request_id"]]
            .groupby("request_id")
            .min()
        )
        # replace negative values of 'answer - group creation date' with -0.5, i.e. indicate a special category to XGBoost
        train_set.loc[
            train_set["answer - group creation date"] < 0, "answer - group creation date"
        ] = -1

    return train_set


def new_dataset_column(
    train_set: pd.DataFrame, previous_dataset: pd.DataFrame, individual_column: str, criteria: str
) -> pd.DataFrame:
    """Add a column "criteria" of the individuals_dataset to the train_set (grouped by households), by joining.
    For the moment, only available with "housing_nights_dataset".

    Parameters
    ----------
    train_set : pd.DataFrame
        The dataset to be augmented with data concerning individuals
    previous_dataset : pd.DataFrame
    individual_column : str
    criteria : str

    Returns
    -------
    pd.DataFrame
    """
    new_column = (
        previous_dataset[["request_id"]]
        .join(pd.get_dummies(previous_dataset[individual_column]))
        .groupby("request_id")
        .sum()
        .loc[:, criteria]
    )
    # If the new feature corresponds to the name of the previous_dataset column (binary, takes 2 values: true 't'==1 or false 't'==0)
    if criteria == "t":
        train_set[individual_column] = new_column
    # If the new feature corresponds to a sub-criteria of the individuals_train_column
    else:
        # we add "nb" because we grouped individuals of the household meeting this criteria
        train_set["nb_" + criteria] = new_column
    return train_set

def label_encode_categorical_features(X: pd.DataFrame):

    dict_categorical_mapping = {}
    #X_encoded=X.copy()

    for col in X.columns:
        # label-encode the categorical features
        if X[col].dtypes in ['category','object']:
            le=LabelEncoder()
            X[col] = le.fit_transform(X[col])
            #X_encoded[col] = le.fit_transform(X[col])
            # get the associated mapping with feature's value for user's understanding 
            dict_categorical_mapping[col] = dict(zip(le.transform(le.classes_), le.classes_))
    
    # save the dict in a path # TODO the user can choose the path to find it again?
    # pickle_save_model(dict_categorical_mapping, "/work/data/dict_categorical_mapping.pkl")

    return X, dict_categorical_mapping # X_encoded, dict_categorical_mapping

def train_valid_test_split(X: pd.DataFrame, Y: pd.DataFrame, model_task: str) -> pd.DataFrame:
    """Splits data into train, valid, and test set -> to train a model with cross-validation
    & train_valid set -> to inspect if the model is discriminant on the data it trained

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame with all features (entered as columns) concerning individuals
    Y : pd.DataFrame
        Target to be predicted by the model (1 column for binary classification: int in {0,1})
    model_task: str
        Goal the user wants to achieve with the model: either classify, or regress...
        Must be set to value in {"regression", "classification"}

    Returns
    -------
    pd.DataFrame
        The initial data is splitted in 8 DataFrames for training and unfairness detection purposes:
        X_train, X_valid, X_train_valid, X_test, Y_train, Y_valid, Y_train_valid, Y_test

    """

    SEED = 7
    VALID_SIZE = 0.15

    if model_task in {"classification", "multiclass"}:

        # use label encoder instead of one-hot-encoder, to better capture the data (if not already label-encoded)
        if isinstance(X,pd.DataFrame):
            X,dict_categorical_mapping = label_encode_categorical_features(X)

        # Keep test values to ensure model is behaving properly
        X_model, X_test, Y_model, Y_test = train_test_split(
            X, Y, test_size=VALID_SIZE, random_state=SEED, stratify=Y
        )

        # Split valid set for early stopping & model selection
        X_train, X_valid, Y_train, Y_valid = train_test_split(
            X_model, Y_model, test_size=VALID_SIZE, random_state=SEED, stratify=Y_model
        )

        # assess model's predictions (discriminant?) on the set it was trained (train&valid)
        X_train_valid = X_train.append(X_valid)
        Y_train_valid = Y_train.append(Y_valid)

    elif model_task == "regression":

        # use label encoder instead of one-hot-encoder, to better capture the data (if not already label-encoded)
        if isinstance(X,pd.DataFrame):
            X,_ = label_encode_categorical_features(X)        
        
        # Keep test values to ensure model is behaving properly
        X_model, X_test, Y_model, Y_test = train_test_split(
            X,
            Y,
            test_size=VALID_SIZE,
            random_state=SEED,  # stratify=Y # TODO stratify or stratify K-fold with regression?
        )

        # Split valid set for early stopping & model selection
        X_train, X_valid, Y_train, Y_valid = train_test_split(
            X_model,
            Y_model,
            test_size=VALID_SIZE,
            random_state=SEED,  # stratify=Y_model
        )

        # assess model's predictions (discriminant?) on the set it was trained (train&valid)
        X_train_valid = X_train.append(X_valid)
        Y_train_valid = Y_train.append(Y_valid)

    else:
        raise NotImplementedError(
            f"The {model_task} task you want the model to perform is not implemented. Must be set to value in {'regression','classification','multiclass'}"
        )

    return X_train, X_valid, X_train_valid, X_test, Y_train, Y_valid, Y_train_valid, Y_test


def prediction_train_valid_by_task(
    model: xgboost,
    X_valid: pd.DataFrame,
    X_train_valid: pd.DataFrame,
    Y_valid: pd.DataFrame,
    Y_train_valid: pd.DataFrame,
    model_task: str,
) -> np.ndarray:
    """Returns in a np.ndarray the predicted labels (for classification) of values (for regression) of the model for X_train_valid.
    For multiclass: returns in a np.ndarray of shape(nb_individuals, nb_labels) the predicted probas by label, better to compute stat score.

    Parameters
    ----------
    model : xgboost
        Model built depending on model_task, in {xgboost.XGBClassifier, xgboost.XGBRegressor},
        already fitted with (X_train, Y_train) and cross-validated on X_train & X_valid
    X_valid : pd.DataFrame
    X_train_valid : pd.DataFrame
    Y_valid : pd.DataFrame
    Y_train_valid : pd.DataFrame
    model_task : str
        Goal the user wants to achieve with the model: either classify, or regress...
        Must be set to value in {"regression", "classification", "multiclass"}

    Returns
    -------
    Y_pred_train_valid : np.ndarray
        If model_task in {"classification", "regression"}, returns the predicted class or value: shape(nb_individuals,)
        If model_task == "multiclass", returns predict_proba by label: shape(nb_individuals, nb_labels)

    Raises
    ------
    NotImplementedError
    """
    if model_task == "classification":

        proba_valid = model.predict_proba(X_valid)[:, 1]
        proba_train_valid = model.predict_proba(X_train_valid)[:, 1]

        ## set y predicted with optimised thresholds
        best_threshold, best_fscore = compute_best_fscore(Y_valid, proba_valid)

        Y_pred_train_valid = (proba_train_valid >= best_threshold).astype(int)

    elif model_task == "multiclass":

        # in case of "multiclass", returns predict_proba (by label: shape(nb_individuals, nb_labels))
        # => will enable to compute different types of stat_scores (based on Y_pred -> auc, or on predict_proba -> log_loss)
        multi_predict_proba_train_valid = model.predict_proba(X_train_valid)
        Y_pred_train_valid = multi_predict_proba_train_valid

    elif model_task == "regression":

        Y_pred_train_valid = model.predict(X_train_valid)

        mse = mean_squared_error(Y_train_valid, Y_pred_train_valid)
        print("The mean squared error (MSE) on train_valid set: {:.4f}\n".format(mse))

    else:
        raise NotImplementedError(
            f"The {model_task} task you want the model to perform is not implemented. Must be set to value in {'regression','classification','multiclass'}"
        )

    return Y_pred_train_valid


def pickle_save_model(uncorrected_model: xgboost, uncorrected_model_path: str = None):
    """Creates a directory "/work/data/models" and store the model in the model_path (here, "/work/data/models/uncorrected_model.pkl").
    Useful to re-use the uncorrected_model, and compute features importances.

    Parameters
    ----------
    uncorrected_model : xgboost
    uncorrected_model_path : str, by default None
        Where the uncorrected_model is saved (by default, "/work/data/models/uncorrected_model.pkl")
    """

    # if the path does not already exist, creates the directory where the uncorrected_model is stored
    if uncorrected_model_path is None:
        uncorrected_model_path = "/work/data/models/uncorrected_model.pkl"

    models_directory = "/".join(uncorrected_model_path.split("/")[:-1])

    Path(models_directory).mkdir(parents=True, exist_ok=True)

    with open(uncorrected_model_path, "wb") as outp:  # Overwrites any existing file
        pickle.dump(uncorrected_model, outp, pickle.HIGHEST_PROTOCOL)


def pickle_load_model(uncorrected_model_path: str)->xgboost:
    """Load a model which has previously been saved in uncorrected_model_path.

    Args:
        uncorrected_model_path (str)
            Example: "/work/data/models/uncorrected_model.pkl"

    Returns:
        xgboost: the model previously trained
    """
    file = open(uncorrected_model_path, "rb")
    uncorrected_model = pickle.load(file)
    return uncorrected_model


def features_importances_from_pickle(
    augmented_train_valid_set: pd.DataFrame,
    X_train_valid: pd.DataFrame,
    model_task: str,
    uncorrected_model_path: str = None,
) -> list:
    """Loads the uncorrected_model and returns a list of the features whose SHAP influence on the output is > 1%.

    Parameters
    ----------
    augmented_train_valid_set : pd.DataFrame
        The train_valid_set augmented with model's prediction, from which columns are divided into groups of individuals *
        and inspected to detect gaps of fair_scores.
    X_train_valid : pd.DataFrame
    model_task: str
        Goal the user wants to achieve with the model: either classify, or regress...
        Must be set to value in {"regression", "classification", "multiclass"}
    uncorrected_model_path : str, by default None
        Where the uncorrected_model is saved (by default, "/work/data/models/uncorrected_model.pkl")

    Returns
    -------
    list
    """
    if uncorrected_model_path is None:
        uncorrected_model_path = "/work/data/models/uncorrected_model.pkl"

    with open(uncorrected_model_path, "rb") as inp:
        uncorrected_model = pickle.load(inp)

    # explainer = shap.DeepExplainer(uncorrected_model, X_train_valid)
    explainer = shap.TreeExplainer(uncorrected_model)
    shap_values = np.array(explainer.shap_values(X_train_valid))

    # in case of multiclass, return for all individual only the SHAP values corresponding to one's predicted class
    if model_task == "multiclass":
        multi_Y_pred_train_valid_uncorrected = augmented_train_valid_set[
            "multi_predicted_uncorrected"
        ].to_numpy()

        # for multiclass, reduce shape of shap_values_test: from shape (nb_classes, nb_individuals, nb_features) -> (nb_individuals, nb_features)
        shap_values_test = np.transpose(shap_values, (1, 0, 2))
        # to take the shap values which correspond to argmax class (i.e. predicted class), expand the Y_pred labels (multi_Y_pred_train_valid_uncorrected)
        # => same shape than shap_values_test
        pred_indices = np.expand_dims(multi_Y_pred_train_valid_uncorrected, axis=(1, 2))

        # then: for all individual, select the shap_values corresponding to the indice of the predicted label (e.g. 1 housing night)
        shap_values_by_argmax = np.take_along_axis(shap_values_test, pred_indices, axis=1)
        # finally, drop the now useless array of (nb_classes): from shape (predicted_class=1, nb_individuals, nb_features) -> (nb_individuals, nb_features)
        shap_values = np.squeeze(shap_values_by_argmax, axis=1)

    # compute the mean of shap values on predicted output
    # from shap_values of shape (nb_individuals, nb_features) -> shap_sum of shape (nb_features,)
    shap_sum = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame([X_train_valid.columns.tolist(), shap_sum.tolist()]).T
    importance_df.columns = ["column_name", "shap_importance"]
    importance_df = importance_df.sort_values("shap_importance", ascending=False)

    # select only features on which the uncorrected_model's output is influenced > 1% => on which the uncorrected_model really has a discriminant impact
    important_features_df = importance_df.loc[importance_df["shap_importance"] > 0.01]
    important_features_list = important_features_df["column_name"]

    # for the user, plot of the features whose importance > 1%
    X_train_valid_with_important_features = X_train_valid.loc[:, important_features_list]

    print(
        "\n Features whose influence is > 1%\n ''' Fairness analysis on these important features on which predictions are based ''' \n"
    )
    shap.summary_plot(
        shap_values, X_train_valid, max_display=X_train_valid_with_important_features.shape[1]
    )

    return important_features_list


def select_important_features(
    augmented_train_valid_set: pd.DataFrame,
    model_name: str,
    model_task: str,
    uncorrected_model_path: str = None,
) -> list:
    """Returns a list with the important features for the model's prediction (i.e. SHAP influence on output > 1%).

    Parameters
    ----------
    augmented_train_valid_set : pd.DataFrame
        The train_valid_set augmented with model's prediction, from which columns are divided into groups of individuals *
        and inspected to detect gaps of fair_scores.
    model_name : str
        Name of the model whose results are inspected.
        Must be set to value in {'uncorrected', 'fair_1', ..., 'fair_n'}
        depending on the number (n) of fairer models in competition (n == grid_size fixed by the user)
    model_task : str
        Goal the user wants to achieve with the model: either classify, or regress...
        Must be set to value in {"regression", "classification", "multiclass"}
    uncorrected_model_path : str, by default None
        Where the uncorrected_model is saved (by default, "/work/data/models/uncorrected_model.pkl")

    Returns
    -------
    list
    """
    features = list(
        set(augmented_train_valid_set.columns)
        - set(
            [
                "target_train_valid",
                f"proba_{model_name}",
                f"predicted_{model_name}",
                f"true_positive_{model_name}",
                f"false_positive_{model_name}",
                f"true_negative_{model_name}",
                f"false_negative_{model_name}",
            ]
        )
    )

    # eliminate columns not to be inspected (because not features) for multiclass
    if model_task == "multiclass":
        features = list(
            set(features)
            - set(
                [
                    "multi_target_train_valid",
                    f"multi_proba_{model_name}",
                    f"multi_predicted_{model_name}",
                ]
            )
        )

    # select list of features on which to raise discrimination alerts, based on features importances
    X_train_valid = augmented_train_valid_set.loc[:, features]

    # TODO make explicit the uncorrected_model_path
    uncorrected_model_path = "/work/data/models/uncorrected_model.pkl"

    important_features_list = features_importances_from_pickle(
        augmented_train_valid_set, X_train_valid, model_task, uncorrected_model_path
    )

    return important_features_list

class CustomDataset_for_nn(Dataset):
    def __init__(self, X, Y):
        self.X = np.array(X)
        self.Y = np.array(Y)
        
    def __len__(self):
        return self.X.shape[0]   
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(14, 1000),
            nn.ReLU(),
            nn.Linear(1000, 230),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(230, 2),
            #nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x

def compute_binary_loss_nn(y_hat, y):
    binary_loss = nn.BCELoss()
    return binary_loss(y_hat, y)

def train_loop_nn(dataloader, model, optimizer, adjusted_sample_weight_train:np.ndarray, get_model:bool=False):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    list_roc_auc = []
    proba_train = None

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        probas_pred = model(X.float())
        
        # sample weights to rebalance classification
        if np.unique(adjusted_sample_weight_train).shape[0] == 1: # get in balanced classification for unique values of weights
        # if torch.unique(adjusted_sample_weight_train).size()[0] == 1: # get in balanced classification for unique values of weights
            adjusted_sample_weight_train = None

        sample_weight = y.detach().clone()
        coeff_rebalanced = y.shape[0]/y.sum()
        dict_class_weights = {1:coeff_rebalanced*5, 0:1} if adjusted_sample_weight_train is None else {1:1, 0:1}
        # dict_class_weights = {0:1, 1:5} TODO FairDream tests
        for np_class in dict_class_weights.keys():
            sample_weight[sample_weight==np_class]=dict_class_weights[np_class]
    
        # TODO final vector with the 2 classes to get argmax => better predictions
        loss=torch.nn.CrossEntropyLoss()
        train_loss=loss(probas_pred, y)
        train_loss = train_loss* sample_weight
                
        # Backpropagation
        train_loss.mean().backward()
        #loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss, current = train_loss.mean(), (batch + 1) * len(X)
        #loss, current = loss.item(), (batch + 1) * len(X)
        #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        y_true=y.detach().numpy()
        probas_pred=probas_pred.detach().numpy()

        proba_train = probas_pred if proba_train is None else np.concatenate((proba_train, probas_pred))
            
    print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    # TODO find more optimal way than 2 return...
    if get_model == True:
        return proba_train, model
    
    elif get_model == False:
        return proba_train

def test_loop_nn(dataloader, model):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    proba_test = None

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            probas_pred = model(X.float())           
            # sample weights to rebalance classification
            sample_weight = y.detach().clone()
            coeff_rebalanced = y.shape[0]/y.sum()
            dict_class_weights = {1:coeff_rebalanced*5, 0:1} 
            # dict_class_weights = {0:1, 1:5} TODO FairDream tests
            for np_class in dict_class_weights.keys():
                sample_weight[sample_weight==np_class]=dict_class_weights[np_class]

            loss=torch.nn.CrossEntropyLoss()
            sample_loss=loss(probas_pred, y)
            sample_loss = sample_loss* sample_weight
            test_loss = test_loss + sample_loss.mean()
            
            correct += (probas_pred.argmax(1) == y).type(torch.float).sum().item()

            y_true=y.detach().numpy()
            probas_pred=probas_pred.detach().numpy()
        
            proba_test = probas_pred if proba_test is None else np.concatenate((proba_test, probas_pred))
            
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return proba_test

def train_naive_tabnet(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_train_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_train: pd.DataFrame,
    Y_valid: pd.DataFrame,
    Y_train_valid: pd.DataFrame,
    Y_test: pd.DataFrame,
    model_task: str,
    stat_criteria: str,
    adjusted_sample_weight_train: np.ndarray = None,
    save_model: bool = False,
) -> np.ndarray:
    """Quickly train a neural network model (with TabNet) to test FairDream in a notebook.
    For the moment, only implemented for binary classification. 
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = f"1"
    clf = TabNetClassifier(#**tabnet_params
                      )

    # This illustrates the behaviour of the model's fit method using Compressed Sparse Row matrices
    sparse_X_train = X_train.values #scipy.sparse.csr_matrix(X_train)  # Create a CSR matrix from X_train
    sparse_X_valid = X_valid.values #scipy.sparse.csr_matrix(X_valid)  # Create a CSR matrix from X_valid

    aug = ClassificationSMOTE(p=0.2)

    # Fitting the model
    # here, passe the wample_weights as argument for "weights"!
    if adjusted_sample_weight_train is not None:
        #adjusted_sample_weight_train=np.random.rand(Y_train.shape[0],1)
        adjusted_sample_weight_train=np.array(adjusted_sample_weight_train)
        weights=adjusted_sample_weight_train.squeeze()
    elif adjusted_sample_weight_train is None:
        weights=1 # TabNet option to handle imbalanced classification

    # This illustrates the warm_start=False behaviour
    save_history = []

    clf.fit(
        X_train=sparse_X_train, y_train=Y_train,
        eval_set=[(sparse_X_train, Y_train), (sparse_X_valid, Y_valid)],
        #eval_set=[(X_train, y_train), (X_valid, y_valid)], # TODO to avoid overfitting ; but slower? 
        eval_name=['train', 'valid'],
        eval_metric=['auc'],
        max_epochs=50 , patience=1,
        batch_size=1024, virtual_batch_size=128,
        num_workers=0,
        weights=weights,
        drop_last=False,
        augmentations=aug, #aug, None
        compute_importance=False
    )
    save_history.append(clf.history["valid_auc"])

    # Now, return vector of probabilities and predictions on X_train_valid
    proba_train=clf.predict_proba(sparse_X_train)
    proba_valid=clf.predict_proba(sparse_X_valid)
    # from vector of probabilities to predictions
    proba_train_valid = np.concatenate((proba_train, proba_valid))
    print(f"ROC-AUC on train&valid data")
    get_auc(y_true=Y_train_valid, probas_pred=proba_train_valid, plot=True)
    ## set y predicted with optimised thresholds
    probas_pred_class_1_train_valid = proba_train_valid[:,1]

    # to set the threshold according to the best Fscore, get probas for class 1 on valid set
    probas_pred_class_1_valid = np.argmax(proba_valid, axis=1)
    best_threshold, best_fscore = compute_best_fscore(Y_valid, probas_pred_class_1_valid)
    # best_threshold, best_fscore = compute_best_fscore(Y_valid, proba_valid)

    # Y_pred_train_valid = (proba_train_valid >= best_threshold).astype(int)
    Y_pred_train_valid = (probas_pred_class_1_train_valid >= best_threshold).astype(int)

    return Y_pred_train_valid, probas_pred_class_1_train_valid

def train_naive_nn(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_train_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_train: pd.DataFrame,
    Y_valid: pd.DataFrame,
    Y_train_valid: pd.DataFrame,
    Y_test: pd.DataFrame,
    model_task: str,
    stat_criteria: str,
    adjusted_sample_weight_train: np.ndarray = None,
    save_model: bool = False,
) -> np.ndarray:
    """Quickly train a neural network model (with Torch) to test FairDream in a notebook.
    For the moment, only implemented for binary classification. 
    """
    batch_size=500
    epochs = 100
    cv_step = 30 # number of steps in which a stagnation or decrease in performance of the model (on train & valid data) is accepted before stopping training 

    # insert a dropout layer as a form of regularization which will help reduce overfitting by randomly setting (here 30%) of the input unit values to zero
    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.fit_transform(X_valid)
    X_train_valid = scaler.fit_transform(X_train_valid)
    X_test = scaler.fit_transform(X_test)

    # transform data into a Torch-accessible format for neural network training
    training_data = CustomDataset_for_nn(X_train, Y_train)
    valid_data = CustomDataset_for_nn(X_valid, Y_valid)
    train_valid_data = CustomDataset_for_nn(X_train_valid, Y_train_valid)
    test_data = CustomDataset_for_nn(X_test, Y_test)

    train_dataloader = DataLoader(training_data, batch_size)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    # for training data, apply group weights if FairDream model
    if adjusted_sample_weight_train is not None:

        # print("adjusted_sample_weight_train before")
        # print(adjusted_sample_weight_train)

        adjusted_sample_weight_train=np.array(adjusted_sample_weight_train)
        # rescale the weights to give more importance to harmed groups during neural network training (normalized between 0 and 25, here)
        # adjusted_sample_weight_train *= (25/adjusted_sample_weight_train.max())
        samples_weight = torch.from_numpy(adjusted_sample_weight_train)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
        train_dataloader = DataLoader(training_data, batch_size=batch_size, sampler=sampler)

        # print("adjusted_sample_weight_train now")
        # print(adjusted_sample_weight_train)

    learning_rate = 1e-3

    device="cpu"
    model = NeuralNetwork().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # training with epochs 
    list_train_losses = []
    list_valid_losses = []
    epoch_nb = epochs

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        proba_train = train_loop_nn(train_dataloader, model, optimizer, adjusted_sample_weight_train) 
        proba_valid = test_loop_nn(valid_dataloader, model)

        roc_auc_train, _, _, _ = get_auc(y_true=Y_train, probas_pred=proba_train)
        roc_auc_valid, _, _, _ = get_auc(y_true=Y_valid, probas_pred=proba_valid)
        print(f"roc_auc_train: {roc_auc_train:>3f}")
        print(f"roc_auc_valid: {roc_auc_valid:>3f}")
        list_train_losses.append(roc_auc_train)
        list_valid_losses.append(roc_auc_valid)

        # cross-validation
        if len(list_train_losses) > cv_step: # enable initialisation of the losses 
            min_loss_train_registered = min(list_train_losses[:-cv_step])
            min_loss_train_current = min(list_train_losses)
            min_loss_valid_registered = min(list_valid_losses[:-cv_step])
            min_loss_valid_current = min(list_valid_losses)

            if min_loss_train_current >= min_loss_train_registered or min_loss_valid_current >= min_loss_valid_registered:
                print(f"Training no more improved over the past {cv_step} epochs")
                epoch_nb = epoch + 1
                break 

    # from vector of probabilities to predictions
    proba_train_valid = np.concatenate((proba_train, proba_valid))
    print(f"ROC-AUC on train&valid data, after {epoch_nb} epochs")
    get_auc(y_true=Y_train_valid, probas_pred=proba_train_valid, plot=True)
    ## set y predicted with optimised thresholds
    probas_pred_class_1_train_valid = np.argmax(proba_train_valid, axis=1)

    # to set the threshold according to the best Fscore, get probas for class 1 on valid set
    probas_pred_class_1_valid = np.argmax(proba_valid, axis=1)
    best_threshold, best_fscore = compute_best_fscore(Y_valid, probas_pred_class_1_valid)
    # best_threshold, best_fscore = compute_best_fscore(Y_valid, proba_valid)

    # Y_pred_train_valid = (proba_train_valid >= best_threshold).astype(int)
    Y_pred_train_valid = (probas_pred_class_1_train_valid >= best_threshold).astype(int)

    return Y_pred_train_valid, probas_pred_class_1_train_valid


def get_scaled_df(X:pd.DataFrame)->pd.DataFrame:
    """Generates the scaled version of the dataframe in input. Used for linear regression. 

    Args:
        X (pd.DataFrame): the dataframe input
        
    Returns:
        pd.DataFrame: the scaled dataframe
    """
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    return X_scaled_df

def train_naive_sklearn(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_train_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_train: pd.DataFrame,
    Y_valid: pd.DataFrame,
    Y_train_valid: pd.DataFrame,
    Y_test: pd.DataFrame,
    model_task: str,
    stat_criteria: str,
    model_type: str,
    adjusted_sample_weight_train: np.ndarray = None,
    save_model: bool = False,
) -> Tuple[np.ndarray]:
    """Quickly train a sklearn model (logistic regression, or multi-layer-perceptron) to test FairDream in a notebook.
    For the moment, only implemented for binary classification. 
    """

    X_train = get_scaled_df(X_train)
    X_valid = get_scaled_df(X_valid)
    X_train_valid = get_scaled_df(X_train_valid)
    X_test = get_scaled_df(X_test)

    if model_task == "classification":
        # reweight the under-represented class in baseline model, in case of unbalanced classification 
        if adjusted_sample_weight_train is None:
            one_to_zero = Y_train.sum()/Y_train.shape[0]
            balanced_weight_dict = {0:one_to_zero, 1:1}
            adjusted_sample_weight_train = Y_train.map(balanced_weight_dict)
        
        if model_type == 'log_reg':
            model = LogisticRegression(solver="liblinear", fit_intercept=True)
        elif model_type == 'mlp':
          model = MLPClassifier(random_state=1, max_iter=300)
        elif model_type == 'svm':
            model = svm.SVC(gamma=1, probability=True)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)

    else:
        raise NotImplementedError("Logistic Regression model is only implemented for model_task=='classification'")


    model.fit(
    X_train,
    Y_train,
    sample_weight=adjusted_sample_weight_train,
    #eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
    )

    Y_pred_train_valid = prediction_train_valid_by_task(
        model, X_valid, X_train_valid, Y_valid, Y_train_valid, model_task
    )


    probas_pred_train_valid = model.predict_proba(X_train_valid)
    probas_pred_class_1_train_valid = probas_pred_train_valid[:,1]
    get_auc(y_true=Y_train_valid, probas_pred=probas_pred_train_valid, plot=True)

    # in case of detection (for the first "uncorrected" model)
    # pre-select features where it is relevant to detect discriminations (i.e. important features for the model)
    # save the model on disk storage, to then: use it, and compute features influences
    if save_model == True:
        pickle_save_model(model)

    return Y_pred_train_valid, probas_pred_class_1_train_valid

def train_naive_xgb(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_train_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_train: pd.DataFrame,
    Y_valid: pd.DataFrame,
    Y_train_valid: pd.DataFrame,
    Y_test: pd.DataFrame,
    model_task: str,
    stat_criteria: str,
    adjusted_sample_weight_train: np.ndarray = None,
    save_model: bool = False,
) -> Tuple[np.ndarray]:
    """Train a baseline XGBoost, to get a vector of probabilities for the class 1 and the predicted events 0 and 1 (optimised for Fscore)

    Parameters
    ----------
    X_train : pd.DataFrame
    X_valid : pd.DataFrame
    X_train_valid : pd.DataFrame
    X_test : pd.DataFrame
    Y_train : pd.DataFrame
    Y_valid : pd.DataFrame
    Y_train_valid : pd.DataFrame
    Y_test : pd.DataFrame
    model_task: str
        Goal the user wants to achieve with the model: either classify, or regress...
        Must be set to value in {"regression", "classification", "multiclass"}
    stat_criteria : str
        Metrics of statistic performance the user wants to optimise
        -> For classification, must be set to value in {'auc','aucpr','mix_auc_aucpr'}
        -> For multiclass, must be set to value in {'merror','mlogloss','auc','f1_score'}
        -> For regression, must be set to value in {'rmse', 'mape'}, i.e. root mean square error and mean absolute percentage error
        https://xgboost.readthedocs.io/en/stable/parameter.html
    adjusted_sample_weight_train: np.ndarray, by default None
        Array with the weight of the error for each individual (row_number),
        i.e. during learning, how much the model is penalised when it goes away of the target for each individual
        Will be used during correction, to set higher weights to protected individuals
        shape : X_train.shape[0]
    save_model : bool, by default False
        If True, enables to save the model in path "/work/data/models/uncorrected_model.pkl" and computes a list with the best important features (i.e. importance > 1%).
        Used in detection phase, to select only relevant features to launch discrimination alerts (i.e. features on which the uncorrected_model is really discriminant)

    Returns
    -------
    Tuple[np.ndarray] of shape == Y_train_valid.shape:
        Y_pred_train_valid: Vector of classes (0,1) returned by the model, given a threshold optimised for Fscore (PR AUC)
        probas_pred_class_1_train_valid: Vector of probability for belonging to the class (1) returned by the model 
    """

    ## fixed parameters

    SEED = 7
    VALID_SIZE = 0.15

    early_stopping_rounds = 20
    verbose = 100

    ## then training of model as a XGB (quick, but already fixed parameters)
    # print(
    #     f"Training model with {X_train.shape[1]} features, on {X_train.shape[0]} rows (valid {X_valid.shape[0]} rows, test {X_test.shape[0]} rows) "
    # )

    if model_task == "classification":

        xgb_classif_params = {
            "seed": SEED,
            "objective": "binary:logistic",
            "n_estimators": 1000,
            "max_depth": 3,
            "importance_type": "gain",
            "use_label_encoder": True,

        }

        # reweight the under-represented class in baseline model, in case of unbalanced classification # TODO generalize it for multiclass 
        if adjusted_sample_weight_train is None:
            one_to_zero = Y_train.sum()/Y_train.shape[0]
            balanced_weight_dict = {0:one_to_zero, 1:1}
            adjusted_sample_weight_train = Y_train.map(balanced_weight_dict)

        model = XGBClassifier(**xgb_classif_params)

        # map user's preferences of stat evaluation with xgboost eval_metrics for better training of the model
        if stat_criteria in {"auc", "aucpr"}:
            xgb_eval_metric = stat_criteria
        elif stat_criteria == "mix_auc_aucpr":
            xgb_eval_metric = "aucpr"  # TODO add custom eval metrics of mix_auc_aucpr for uncorrected model
        else:
            raise NotImplementedError(
                f"stat_criteria {stat_criteria} is not implemented."
                f"\n For classification, must be set to value in {'auc','aucpr','mix_auc_aucpr'}."
                f"\n See https://xgboost.readthedocs.io/en/stable/parameter.html "
            )

    elif model_task == "multiclass":

        xgb_multiclass_params = {
            "seed": SEED,
            "objective": "multi:softptob",
            "n_estimators": 500,
            "max_depth": 3,
            "max_delta_step": 5,
            "learning_rate": 0.3,
    }

        model = XGBClassifier(**xgb_multiclass_params)

        # map user's preferences of stat evaluation with xgboost eval_metrics for better training of the model
        if stat_criteria in {"merror", "mlogloss", "auc", "f1_score"}:
            xgb_eval_metric = stat_criteria

        else:
            raise NotImplementedError(
                f"stat_criteria {stat_criteria} is not implemented."
                f"\n For multiclass, must be set to value in {'merror','mlogloss','auc','f1_score'}"
                f"\n See https://xgboost.readthedocs.io/en/stable/parameter.html "
            )

    elif model_task == "regression":

        xgb_reg_params = {
            "seed": SEED,
            "objective": "reg:squarederror",
            "n_estimators": 1000,
            "max_depth": 3,
            "importance_type": "gain",
            "use_label_encoder": False,
        }

        model = XGBRegressor(**xgb_reg_params)

        # map user's preferences of stat evaluation with xgboost eval_metrics for better training of the model
        if stat_criteria in {"rmse", "mape"}:
            xgb_eval_metric = stat_criteria

        else:
            raise NotImplementedError(
                f"stat_criteria {stat_criteria} is not implemented."
                f"\n For regression, must be set to value in {'rmse', 'mape'}, i.e. root mean square error and mean absolute percentage error"
                f"\n See https://xgboost.readthedocs.io/en/stable/parameter.html "
            )

    else:
        raise NotImplementedError(
            f"The {model_task} task you want the model to perform is not implemented. Must be set to value in {'regression','classification','multiclass'}"
        )

    model.fit(
        X_train,
        Y_train,
        sample_weight=adjusted_sample_weight_train,
        eval_metric=xgb_eval_metric,
        early_stopping_rounds=early_stopping_rounds,
        eval_set=[(X_train, Y_train), (X_valid, Y_valid)],
        verbose=False#verbose,
    )
    probas_pred_class_1_train_valid = model.predict_proba(X_train_valid)[:,1] # TODO delete probability mode after calibration tests!

    Y_pred_train_valid = prediction_train_valid_by_task(
        model, X_valid, X_train_valid, Y_valid, Y_train_valid, model_task
    )

    # in case of detection (for the first "uncorrected" model)
    # pre-select features where it is relevant to detect discriminations (i.e. important features for the model)
    # save the model on disk storage, to then: use it, and compute features influences
    if save_model == True:
        pickle_save_model(model)

    return Y_pred_train_valid, probas_pred_class_1_train_valid


def train_naive_model(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_train_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    Y_train: pd.DataFrame,
    Y_valid: pd.DataFrame,
    Y_train_valid: pd.DataFrame,
    Y_test: pd.DataFrame,
    model_task: str,
    stat_criteria: str,
    adjusted_sample_weight_train: np.ndarray = None,
    save_model: bool = False,
    model_type: str = "xgboost"
) -> Tuple[np.ndarray]:
    """Quickly trains a model of the desired type (by default gradient boosted trees).
     Will be to train a baseline model, and FairDream models in competition for fairer results.

     Returns 2 arrays for the train&valid sample: 
        - the predicted events (0 and 1, with a threshold optimised for the Fscore => maximising the PR-AUC) ;
        - the associated probabilities.

    Parameters
    ----------
    X_train : pd.DataFrame
    X_valid : pd.DataFrame
    X_train_valid : pd.DataFrame
    X_test : pd.DataFrame
    Y_train : pd.DataFrame
    Y_valid : pd.DataFrame
    Y_train_valid : pd.DataFrame
    Y_test : pd.DataFrame
    model_task: str
        Goal the user wants to achieve with the model: either classify, or regress...
        Must be set to value in {"regression", "classification", "multiclass"}
    stat_criteria : str
        Metrics of statistic performance the user wants to optimise
        -> For classification, must be set to value in {'auc','aucpr','mix_auc_aucpr'}
        -> For multiclass, must be set to value in {'merror','mlogloss','auc','f1_score'}
        -> For regression, must be set to value in {'rmse', 'mape'}, i.e. root mean square error and mean absolute percentage error
        https://xgboost.readthedocs.io/en/stable/parameter.html
    adjusted_sample_weight_train: np.ndarray, by default None
        Array with the weight of the error for each individual (row_number),
        i.e. during learning, how much the model is penalised when it goes away of the target for each individual
        Will be used during correction, to set higher weights to protected individuals
        shape : X_train.shape[0]
    save_model : bool, by default False
        If True, enables to save the model in path "/work/data/models/uncorrected_model.pkl" and computes a list with the best important features (i.e. importance > 1%).
        Used in detection phase, to select only relevant features to launch discrimination alerts (i.e. features on which the uncorrected_model is really discriminant)
    model_type : str, by default "xgboost"
        Type of the machine-learning model to be trained, either in trees (XGBoost) or neural networks family ('neural_net' implementation from Torch).
        The model can also entail less complexity in learning - we also implemented a logistic regression and simple neural network (multi-layer-perceptron) method. 
        This will be the root model to train a baseline model (not specifically trained for fairness) and FairDream models in competition for fairer results. 
        Must be set to a value in {'xgboost', 'neural_net', 'tabnet', 'log_reg', 'mlp'}

    Returns
    -------
    Tuple[np.ndarray] of shape == Y_train_valid.shape:
        Y_pred_train_valid: Vector of classes (0,1) returned by the model, given a threshold optimised for Fscore (PR AUC)
        probas_pred_class_1_train_valid: Vector of probability for belonging to the class (1) returned by the model 
    """
    if model_type=='xgboost':
        Y_pred_train_valid, probas_pred_class_1_train_valid=train_naive_xgb(X_train=X_train, 
                                           X_valid=X_valid, 
                                           X_train_valid=X_train_valid, 
                                           X_test=X_test, 
                                           Y_train=Y_train, 
                                           Y_valid=Y_valid, 
                                           Y_train_valid=Y_train_valid,
                                           Y_test=Y_test, 
                                           model_task=model_task, 
                                           stat_criteria=stat_criteria, 
                                           save_model=save_model,
                                           adjusted_sample_weight_train=adjusted_sample_weight_train)
    elif model_type=='neural_net':
        Y_pred_train_valid, probas_pred_class_1_train_valid=train_naive_nn(X_train=X_train, 
                                           X_valid=X_valid, 
                                           X_train_valid=X_train_valid, 
                                           X_test=X_test, 
                                           Y_train=Y_train, 
                                           Y_valid=Y_valid, 
                                           Y_train_valid=Y_train_valid,
                                           Y_test=Y_test, 
                                           model_task=model_task, 
                                           stat_criteria=stat_criteria,
                                           save_model=save_model,
                                           adjusted_sample_weight_train=adjusted_sample_weight_train)
    elif model_type=='tabnet':
        Y_pred_train_valid, probas_pred_class_1_train_valid=train_naive_tabnet(X_train=X_train, 
                                           X_valid=X_valid, 
                                           X_train_valid=X_train_valid, 
                                           X_test=X_test, 
                                           Y_train=Y_train, 
                                           Y_valid=Y_valid, 
                                           Y_train_valid=Y_train_valid,
                                           Y_test=Y_test, 
                                           model_task=model_task, 
                                           stat_criteria=stat_criteria,
                                           save_model=save_model,
                                           adjusted_sample_weight_train=adjusted_sample_weight_train)    
    
    elif model_type in {'log_reg', 'mlp', 'svm', 'random_forest'}:
        Y_pred_train_valid, probas_pred_class_1_train_valid=train_naive_sklearn(X_train=X_train, 
                                           X_valid=X_valid, 
                                           X_train_valid=X_train_valid, 
                                           X_test=X_test, 
                                           Y_train=Y_train, 
                                           Y_valid=Y_valid, 
                                           Y_train_valid=Y_train_valid,
                                           Y_test=Y_test, 
                                           model_task=model_task, 
                                           stat_criteria=stat_criteria, 
                                           model_type=model_type,
                                           save_model=save_model,
                                           adjusted_sample_weight_train=adjusted_sample_weight_train,
                                           )

    else:
        raise NotImplementedError("The model_type of the model you want to train is not implemented. Must be set to a value in {'xgboost', 'neural_net', 'tabnet', 'log_reg', 'mlp'}")
    
    return Y_pred_train_valid, probas_pred_class_1_train_valid

def augment_train_valid_set_with_results(
    model_name: str,
    previous_train_valid_set: pd.DataFrame,
    Y_train_valid: pd.DataFrame,
    Y_pred_train_valid: np.ndarray,
    probas_pred_class_1_train_valid:np.ndarray,
    model_task: str,
    multi_Y_train_valid: np.ndarray = None,
    multi_predict_proba_train_valid: np.ndarray = None,
) -> pd.DataFrame:
    """Add predictions and statistics to a model to the dataset train&valid,
    to enable further comparison and selection between the initial and fair train models.

    Parameters
    ----------
    model_name : str
        Name of the model whose results will be integrated.
        Must be set to value in {'uncorrected', 'fair_1', ..., 'fair_n'}
        depending on the number (n) of fairer models in competition (n == grid_size fixed by the user)
    previous_train_valid_set : pd.DataFrame
        Dataset extracted from X_train_valid, of shape (X_train_valid.shape)
    Y_train_valid : pd.DataFrame
        Target, ie true labels of Y on train_valid set.
    Y_pred_train_valid : np.ndarray
        Vector of classes (0,1) returned by the model given a threshold optimised for Fscore (shape == Y_train_valid.shape)
    probas_pred_class_1_train_valid: np.ndarray
        Vector of probability for belonging to the class (1) returned by the model 
    model_task: str
        Goal the user wants to achieve with the model: either classify, or regress...
        Must be set to value in {"regression", "classification"}
    multi_Y_train_valid: np.ndarray, by default None
        Only when model_task == "multiclass", to inspect stat performances of the model on different classes
        shape(Y_train_valid,): true labels, to further compare with the models' predicted labels
    multi_predict_proba_train_valid: np.ndarray, by default None
        Only when model_task == "multiclass", to inspect stat performances of the model on different classes
        shape(Y_train_valid, nb_labels)

    Returns
    -------
    pd.DataFrame
        Previous train_valid set augmented with probabilities, predicted labels, (true/false) (positive/negative)
        for further computation of fair_score of the model.
    """
    # do not modify the previous (uncorrected) dataset, create a new with true / false positive / negative of the new models

    # in case label encoding was used, recover the values of categorical columns
    if os.path.exists("/work/data/dict_categorical_mapping.pkl"):
        dict_categorical_mapping = pickle_load_model("/work/data/dict_categorical_mapping.pkl")
        previous_train_valid_set = previous_train_valid_set.replace({col: dict_categorical_mapping[col] for col in dict_categorical_mapping.keys()})

    new_train_valid_set = previous_train_valid_set.copy()

    if model_task in {"classification", "multiclass"}:

        new_train_valid_set["target_train_valid"] = Y_train_valid
        new_train_valid_set[f"predicted_{model_name}"] = Y_pred_train_valid
        new_train_valid_set[f"probas_pred_class_1_train_valid_{model_name}"] = probas_pred_class_1_train_valid

        new_train_valid_set=get_confusion_matrix_by_indiv_df(model_name=model_name,new_train_valid_set=new_train_valid_set)

        if model_task == "multiclass":
            # add predict_probas by label, to further compute the model's stat perf
            # pack the initial vector of predict_probas with a vector column: for all individual (i.e. line), list of probabilities by label
            new_train_valid_set[f"multi_proba_{model_name}"] = np.array(
                pd.DataFrame({0: multi_predict_proba_train_valid.tolist()})
            )
            # add multi_pred_train_valid to inspect stat performances on different classes
            multi_Y_pred_train_valid = multi_predict_proba_train_valid.argmax(axis=-1)

            new_train_valid_set[f"multi_predicted_{model_name}"] = multi_Y_pred_train_valid
            new_train_valid_set[f"multi_target_train_valid"] = multi_Y_train_valid

    elif model_task == "regression":

        new_train_valid_set["target_train_valid"] = Y_train_valid
        new_train_valid_set[f"predicted_{model_name}"] = Y_pred_train_valid

    return new_train_valid_set

def get_confusion_matrix_by_indiv_df(model_name:str,new_train_valid_set:pd.DataFrame)->pd.DataFrame:
    """From a DataFrame containing the event predicted by any binary classifier in the f"predicted_{model_name}" column, 
    And a column of true labels named "target_train_valid", returns 4 columns indicating with (1) or (0) 
     The case matched by the indiv (i.e. row):
        False Positive, True Positive, True Negative, False Negative.

    Args:
    model_name : str
        Name of the model whose results will be integrated.
        Must be set to value in {'uncorrected', 'fair_1', ..., 'fair_n'}
        depending on the number (n) of fairer models in competition (n == grid_size fixed by the user)
    previous_train_valid_set : pd.DataFrame
        Dataset extracted from X_train_valid, of shape (X_train_valid.shape)

    Returns:
        pd.DataFrame: the new train valid set with 4 new columns, for the 4 events of the confusion matrix:
            new_train_valid_set[f"true_positive_{model_name}"]
            new_train_valid_set[f"false_positive_{model_name}"]
            new_train_valid_set[f"true_negative_{model_name}"]
            new_train_valid_set[f"false_negative_{model_name}"]

    """
    new_train_valid_set[f"true_positive_{model_name}"] = np.where(
        (
            new_train_valid_set[f"predicted_{model_name}"]
            == new_train_valid_set["target_train_valid"]
        )
        & (new_train_valid_set[f"predicted_{model_name}"] == 1),
        1,
        0,
    ) 

    new_train_valid_set[f"false_positive_{model_name}"] = np.where(
        (
            new_train_valid_set[f"predicted_{model_name}"]
            != new_train_valid_set["target_train_valid"]
        )
        & (new_train_valid_set[f"predicted_{model_name}"] == 1),
        1,
        0,
    )        

    new_train_valid_set[f"true_negative_{model_name}"] = np.where(
        (
            new_train_valid_set[f"predicted_{model_name}"]
            == new_train_valid_set["target_train_valid"]
        )
        & (new_train_valid_set[f"predicted_{model_name}"] == 0),
        1,
        0,
    )

    new_train_valid_set[f"false_negative_{model_name}"] = np.where(
        (
            new_train_valid_set[f"predicted_{model_name}"]
            != new_train_valid_set["target_train_valid"]
        )
        & (new_train_valid_set[f"predicted_{model_name}"] == 0),
        1,
        0,
    )

    return new_train_valid_set

def set_marketing_treatment_effect(
        df_response_if_treated: pd.DataFrame,
        df_no_response_if_treated: pd.DataFrame,
        df_response_if_control: pd.DataFrame,
        df_no_response_if_control: pd.DataFrame,
        len_dataset: int,
        is_treated: str,
        percentage_treated: int,
        is_response: str,
        percentage_response_if_treated: int,
        percentage_response_if_control: int):

    # set treatment effect for the marketing real dataset (!) max len dataset is 40_00, due to the few indivs control that respond (4000)
        
    # illustration here: say 40% indivs are treated and 80% of them respond, else without treatment 10% response
    # 80% of 40% indivs respond if treated
    ix_size_treated_resp = floor(len_dataset*percentage_treated/100*percentage_response_if_treated/100)
    print(f"len_dataset: {len_dataset}")
    print(f"nb indivs treated with response: {ix_size_treated_resp}")
    ix_treated_resp = np.random.choice(len(df_response_if_treated), size=ix_size_treated_resp, replace=False)
    df_response_if_treated = df_response_if_treated.iloc[ix_treated_resp]

    # (100-80)% of 40% indivs do not respond if treated
    ix_size_treated_no_resp = floor(len_dataset*percentage_treated/100*(100-percentage_response_if_treated)/100)
    print(f"nb indivs treated with no response: {ix_size_treated_no_resp}")
    ix_treated_no_resp = np.random.choice(len(df_no_response_if_treated), size=ix_size_treated_no_resp, replace=False)
    df_no_response_if_treated = df_no_response_if_treated.iloc[ix_treated_no_resp]

    percentage_control = 100 - percentage_treated

    # 10% of (100-40)% indivs respond if control
    ix_size_control_resp = floor(len_dataset*percentage_control/100*percentage_response_if_control/100)
    print(f"nb indivs control with response: {ix_size_control_resp}")
    ix_control_resp = np.random.choice(len(df_response_if_control), size=ix_size_control_resp, replace=False)
    df_response_if_control = df_response_if_control.iloc[ix_control_resp]

    # (1-10)% of (100-40)% indivs do not respond if control
    ix_size_control_no_resp = floor(len_dataset*percentage_control/100*(100-percentage_response_if_control)/100)
    print(f"nb indivs control with no response: {ix_size_control_no_resp}")
    ix_control_no_resp = np.random.choice(len(df_no_response_if_control), size=ix_size_control_no_resp, replace=False)
    df_no_response_if_control = df_no_response_if_control.iloc[ix_control_no_resp]

    df = df_response_if_treated.append([df_no_response_if_treated, df_response_if_control, df_no_response_if_control])
    
    return df

def set_age_effect(dataset:pd.DataFrame, age_young_max:int, age_old_max:int, 
                   nb_old_wealthy: int, nb_old_unwealthy: int,
                   nb_young_wealthy: int, nb_young_unwealthy:int) -> pd.DataFrame:
    """Selects the desired numbers of true "wealthy" and "unwealthy" (i.e. revenue > or < $50_000) individuals in the Census dataset, according to a younger and older group of age.
    Serves to illustrate the tough interpretation of algorithmic fairness results in case of imbalanced labels distributions across ages
    (In particular, through the lens of the Simpson's paradox.)
    TODO Give the option of percentages (not only numbers) to inspect the effects of the model on a different data distribution (age, wealthiness)?

    Args:
        dataset (pd.DataFrame): initial Census Dataset. Must encompass the columns "age" (int) and "target" (bool: > (1) or <(0) $50_000)
        age_young_max (int): Maximal age to select the group of younger people.
        age_old_max (int): Maximal age to select the group of older people.
        nb_old_wealthy (int): Number of selected people in the older group of age with the label "wealthy" (1).
        nb_old_unwealthy (int): Number of selected people in the older group of age with the label "unwealthy" (0).
        nb_young_wealthy (int): Number of selected people in the younger group of age with the label "wealthy" (1).
        nb_young_unwealthy (int): Number of selected people in the younger group of age with the label "unwealthy" (0).

    Returns:
        pd.DataFrame: new dataset with the desired numbers for wealthiness distribution according to "younger" and "older" age groups 
    """

    # select the "young" and "old" clients in 2 dataframes for further selection
    df_young = dataset.loc[dataset['age']<=age_young_max]

    mask_older = (dataset['age'] > age_young_max) & (dataset['age'] <= age_old_max)
    df_old = dataset.loc[mask_older]

    # fix nb of old wealthy people
    df_old_wealthy = df_old.loc[df_old['target']==1]

    ix_old_wealthy = np.random.choice(len(df_old_wealthy), size=nb_old_wealthy, replace=False)
    df_old_wealthy_new = df_old_wealthy.iloc[ix_old_wealthy]
    print(f"Nb of old wealthy people in the dataset: {df_old_wealthy_new.shape[0]}")

    # fix nb of old unwealthy people
    df_old_unwealthy = df_old.loc[df_old['target']==0]

    ix_old_unwealthy = np.random.choice(len(df_old_unwealthy), size=nb_old_unwealthy, replace=False)
    df_old_unwealthy_new = df_old_unwealthy.iloc[ix_old_unwealthy]
    print(f"Nb of old unwealthy people in the dataset: {df_old_unwealthy_new.shape[0]}")

    # fix nb of young wealthy people
    df_young_wealthy = df_young.loc[df_young['target']==1]

    ix_young_wealthy = np.random.choice(len(df_young_wealthy), size=nb_young_wealthy, replace=False)
    df_young_wealthy_new = df_young_wealthy.iloc[ix_young_wealthy]
    print(f"Nb of young wealthy people in the dataset: {df_young_wealthy_new.shape[0]}")

    # fix nb of young unwealthy people
    df_young_unwealthy = df_young.loc[df_young['target']==0]

    ix_young_unwealthy = np.random.choice(len(df_young_unwealthy), size=nb_young_unwealthy, replace=False)
    df_young_unwealthy_new = df_young_unwealthy.iloc[ix_young_unwealthy]
    print(f"Nb of young unwealthy people in the dataset: {df_young_unwealthy_new.shape[0]}")


    # TODO? reset indexes for further operations on data
    df_old_wealthy_new.reset_index(drop=True, inplace=True)
    df_old_unwealthy_new.reset_index(drop=True, inplace=True)
    df_young_wealthy_new.reset_index(drop=True, inplace=True)
    df_young_unwealthy_new.reset_index(drop=True, inplace=True)
    # new_dataset = df_old_wealthy_new.append([df_old_unwealthy_new, df_young_wealthy_new, df_young_unwealthy_new])
    # new_dataset.reset_index(drop=True, inplace=True)

    # Tand then decompose the new dataset according to incomes and age groups, with all indexes being reset
    # df_old_wealthy_new = new_dataset.loc[(new_dataset["target"]==1) & mask_older]
    # df_old_unwealthy_new = new_dataset.loc[(new_dataset["target"]==0) & mask_older]
    # df_young_wealthy_new = new_dataset.loc[(new_dataset["target"]==1) & (new_dataset['age']<=age_young_max)]
    # df_young_unwealthy_new = new_dataset.loc[(new_dataset["target"]==0) & (new_dataset['age']<=age_young_max)]

    return df_old_wealthy_new, df_old_unwealthy_new, df_young_wealthy_new, df_young_unwealthy_new

def set_wealthiness_prediction(model_name: str, df_positive:pd.DataFrame, df_negative:pd.DataFrame, nb_true_positive:int, nb_true_negative:int)->pd.DataFrame:
    """According to given numbers of true positives and true negatives, provides a DataFrame with the reconstituted predictions of a model (age, prediction of wealthiness).
    TODO Give the option of percentages (not only numbers) to inspect the effects of the model on a different data distribution (age, wealthiness)?

    Args:
        model_name (str): name of the model whose predictions are reconstituted 
        df_positive (pd.DataFrame): DataFrame containing the positive (i.e. wealthy) individuals
        df_negative (pd.DataFrame): DataFrame containing the negative (i.e. not wealthy) individuals
        nb_true_positive (int): number of "predicted wealthy" that are truly wealthy, fixed by the user.
            Must be <= df_positive.shape[0]. TODO add error if this number is >
        nb_true_negative (int): number of "predicted not wealthy" that are truly not wealthy, fixed by the user.
            Must be <= df_negative.shape[0]. TODO add error if this number is >

    Returns:
        pd.DataFrame: DataFrame with the columns "age", "target" and f"predicted_{model_name}" with the reconstituted model's predictions
    """

    # fix nb of wealthy & predicted wealthy (true positive)
    ix_true_positive = np.random.choice(len(df_positive), size=nb_true_positive, replace=False)
    df_true_positive_new = df_positive.iloc[ix_true_positive]
    df_true_positive_new[f"predicted_{model_name}"] = 1
    print(f"Nb of old wealthy & predicted wealthy people in the dataset: {df_true_positive_new.shape[0]}")

    # accordingly, compute nb of wealthy & predicted not wealthy (false negative)
    ix_false_negative = list(set(df_positive.index) - set(ix_true_positive))
    df_false_negative_new = df_positive.iloc[ix_false_negative]
    df_false_negative_new[f"predicted_{model_name}"] = 0
    print(f"Nb of old wealthy & predicted non wealthy people in the dataset: {df_false_negative_new.shape[0]}")

    # fix nb of unwealthy & predicted unwealthy (true negative)
    ix_true_negative = np.random.choice(len(df_negative), size=nb_true_negative, replace=False)
    df_true_negative_new = df_negative.iloc[ix_true_negative]
    df_true_negative_new[f"predicted_{model_name}"] = 0
    print(f"Nb of old unwealthy & predicted unwealthy people in the dataset: {df_true_negative_new.shape[0]}")

    # accordingly, compute nb of unwealthy & predicted wealthy (false positive)
    ix_false_positive = list(set(df_negative.index) - set(ix_true_negative))
    df_false_positive_new = df_negative.iloc[ix_false_positive]
    df_false_positive_new[f"predicted_{model_name}"] = 1
    print(f"Nb of old unwealthy & predicted wealthy people in the dataset: {df_false_positive_new.shape[0]}")

    # re-order all indexes for futher model's comparisons of differents predictions for the same index 
    # df_true_positive_new.sort_index(inplace=True)
    # df_false_negative_new.sort_index(inplace=True)
    # df_true_negative_new.sort_index(inplace=True)
    # df_false_positive_new.sort_index(inplace=True)
    
    df_predictions = df_true_positive_new.append([df_false_negative_new, df_true_negative_new, df_false_positive_new])
    df_predictions.sort_index(inplace=True)
    
    return df_predictions
