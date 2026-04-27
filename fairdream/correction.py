import math
import os
import wget
import random
from math import exp
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import torch
from fairlearn.reductions import AbsoluteLoss
from fairlearn.reductions import BoundedGroupLoss
from fairlearn.reductions import ClassificationMoment
from fairlearn.reductions import DemographicParity
from fairlearn.reductions import EqualizedOdds
from fairlearn.reductions import ErrorRateParity
from fairlearn.reductions import FalsePositiveRateParity
from fairlearn.reductions import GridSearch
from fairlearn.reductions import Moment
from fairlearn.reductions import SquareLoss
from fairlearn.reductions import TruePositiveRateParity
from fairlearn.reductions import UtilityParity
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.augmentations import ClassificationSMOTE
import torch
import scipy 
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from xgboost import XGBClassifier
from xgboost import XGBRegressor

from fairdream.compute_scores import fair_score
from fairdream.compute_scores import is_fairness_purpose_to_maximise
from fairdream.compute_scores import stat_score
from fairdream.compute_scores import sum_scores_gains_by_groups
from fairdream.data_preparation import label_encode_categorical_features
from fairdream.data_preparation import augment_train_valid_set_with_results
from fairdream.data_preparation import CustomDataset_for_nn
from fairdream.data_preparation import get_auc
from fairdream.data_preparation import NeuralNetwork
from fairdream.data_preparation import prediction_train_valid_by_task
from fairdream.data_preparation import train_loop_nn
from fairdream.data_preparation import train_naive_model
from fairdream.data_preparation import train_valid_test_split
from fairdream.multiclass_fair_preparation import compute_binary_Y
from fairdream.multiclass_fair_preparation import get_frontier_label



def stat_fair_tradeoff(stat_score: int, fair_score: int, tradeoff: str = "moderate") -> int:
    """Scores a model based on the user's tradeoff: how one wants to combine stat and fair performances?
    Will serve to select the best model according to the user's preferences

    Parameters
    ----------
    stat_score : int
        Value of stat score corresponding to the user's vision of statistic performance
    fair_score : int
        Value of fair score aggregated by groups of the inspected column, corresponding to the user's fairness purpose
    tradeoff : str, optional
        How the user wants to (un)balance stat and fair performances to compute the model's score, by default "moderate"
        For the moment, 3 choices (coefficients stress on the user's preferences regarding stat or fair score):
        "moderate": equal coefficients
        "fair_preferred": coefficients stress on fair_score (=2/3) to establish the tradeoff_score
        "stat_preferred": coefficients stress on stat_score (=2/3) to establish the tradeoff_score

    Returns
    -------
    int
        Tradeoff score of the model (computation based on stat and fair scores)

    Raises
    ------
    ValueError
    """
    if tradeoff == "moderate":
        tradeoff_score = (stat_score + fair_score) / 2

    elif tradeoff == "fair_preferred":
        tradeoff_score = stat_score / 3 + fair_score * 2 / 3

    elif tradeoff == "stat_preferred":
        tradeoff_score = stat_score * 2 / 3 + fair_score / 3

    else:
        raise ValueError(
            "tradeoff must be set to value in {'moderate', 'fair_preferred', 'stat_preferred'}"
        )

    return tradeoff_score


def model_selection(models_df: pd.DataFrame) -> dict:
    """Select the best model based on dataframe informations and users' preferences.

    Parameters
    ----------
    models_df : pd.DataFrame
    For each line (i.e. model), DataFrame with columns = ['model_name', 'fair_score_value', 'stat_score_value', 'tradeoff_score',
    'fair_scores_df', 'selected']
    Serves here to select the best model according to the user's preferences, and then plot it to compare its fair/stat score with the initial "uncorrected" model
        "model_name": str,
        "fair_score_value": int,
        "stat_score_value": int,
        "tradeoff_score": int,
        "fair_scores_df"
    Returns
    -------
    dict
        Dictionary containing the best models scores (stat&fair),
        the model to be re-used for better predictions,
        and a DataFrame with scores of the initial "uncorrected" model to be challenged ('fair_scores_df_uncorrected')
        best_model_dict.columns = ['model_name', 'fair_score_value', 'stat_score_value', 'tradeoff_score', 'fair_scores_df', 'fair_scores_df_uncorrected']

    """
    # returns a dictionary with model_name, tradeoff_score and model (to test and re-use) of best model
    best_model_index = models_df["tradeoff_score"].idxmax()
    best_model_dict = models_df.iloc[best_model_index].to_dict()
    # for comparison, add fair_scores_df of the uncorrected_model (corresponding to index 0 of models_df)
    best_model_dict["fair_scores_df_uncorrected"] = models_df.loc[:, "fair_scores_df"][0]

    return best_model_dict

class NeuralNet_for_GridSearch():
    def __init__(self):
        self.batch_size=500
        self.epochs=100
        self.cv_step=30

    def fit(self, X, y, sample_weight):
        # insert a dropout layer as a form of regularization which will help reduce overfitting by randomly setting (here 30%) of the input unit values to zero
        scaler  = MinMaxScaler()
        X = scaler.fit_transform(X)
        # transform data into a Torch-accessible format for neural network training
        training_data = CustomDataset_for_nn(X, y)
        train_dataloader = DataLoader(training_data, self.batch_size)

        # for training data, apply group weights if FairDream model
        if sample_weight is not None:
            adjusted_sample_weight_train=np.array(sample_weight)
            samples_weight = torch.from_numpy(adjusted_sample_weight_train)
            sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
            train_dataloader = DataLoader(training_data, batch_size=self.batch_size, sampler=sampler)

        learning_rate = 1e-3

        device="cpu"
        model = NeuralNetwork().to(device)

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        # training with epochs TODO add cross-validation
        for t in range(self.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            proba_train, self.model = train_loop_nn(train_dataloader, model, optimizer, adjusted_sample_weight_train, get_model = True)

        # training with epochs 
        list_train_losses = []
        epoch_nb = self.epochs

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            proba_train, self.model = train_loop_nn(train_dataloader, model, optimizer, adjusted_sample_weight_train, get_model = True)

            roc_auc_train, _, _, _ = get_auc(y_true=y, probas_pred=proba_train)
            print(f"roc_auc_train: {roc_auc_train:>3f}")
            list_train_losses.append(roc_auc_train)

            # cross-validation (here only if results no more improve in train set)
            if len(list_train_losses) > self.cv_step: # enable initialisation of the losses 
                min_loss_train_registered = min(list_train_losses[:-self.cv_step])
                min_loss_train_current = min(list_train_losses)

                if min_loss_train_current >= min_loss_train_registered:
                    print(f"Training no more improved over the past {self.cv_step} epochs")
                    get_auc(y_true=y, probas_pred=proba_train, plot=True)
                    epoch_nb = epoch + 1
                    break 

        print(f"ROC-AUC on train&valid data, after {epoch_nb} epochs")


    def predict_proba(self, X):
        X_to_array = np.array(X)
        X_to_tensor = torch.tensor(X_to_array)
        y_proba_tensor = self.model(X_to_tensor.float())
        y_proba = np.array(y_proba_tensor.detach().numpy())
        return y_proba

    def predict(self, X):
        y_proba = self.predict_proba(X)
        y_pred = np.argmax(y_proba, axis=1)
        return y_pred


class TabNet_for_GridSearch():
    def __init__(self):
        np.random.seed(0)
        os.environ['CUDA_VISIBLE_DEVICES'] = f"1"
        self.batch_size=500
        self.max_epochs=50
        self.cv_step=30

    def fit(self, X, y, sample_weight):

        self.clf = TabNetClassifier(#**tabnet_params
                        )

        # This illustrates the behaviour of the model's fit method using Compressed Sparse Row matrices
        sparse_X = X.values #scipy.sparse.csr_matrix(X)  # Create a CSR matrix from X_train # TODO cut the weights of X to enable cv-learning, with weights_train!

        aug = ClassificationSMOTE(p=0.2)

        # Fitting the model
        # here, passe the wample_weights as argument for "weights"!
        if sample_weight is not None:
            sample_weight=np.array(sample_weight)
            weights=sample_weight.squeeze()
        elif sample_weight is None:
            weights=1 # TabNet option to handle imbalanced classification

        # This illustrates the warm_start=False behaviour
        save_history = []

        self.clf.fit(
            X_train=sparse_X, y_train=y,
            # eval_set=[(sparse_X_train, Y_train), (sparse_X_valid, Y_valid)],
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
        save_history.append(self.clf.history["valid_auc"])

    def predict_proba(self, X):
        sparse_X = X.values #scipy.sparse.csr_matrix(X)
        y_proba = self.clf.predict_proba(sparse_X)
        y_proba_class_1 = y_proba[:,1]
        return y_proba_class_1

    def predict(self, X):
        sparse_X = X.values #scipy.sparse.csr_matrix(X)
        y_pred = self.predict(sparse_X)
        return y_pred

def grid_search_fair_train(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    train_valid_set_with_uncorrected_results: pd.DataFrame,
    protected_attribute: List[str],
    fairness_purpose: str,
    model_task: str,
    stat_criteria: str,
    fairness_constraint: str = None,
    tradeoff: str = "moderate",
    nb_grid_search_models: int = 4,
    model_type: str = None,
) -> list:
    """Trains (nb_grid_search_models ) fairer models with GridSearch to counterbalance the discrimination of protected_attribute regarding fairness purpose,
    and returns their predictions on X_train_valid in a list.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame with all features (entered as columns) concerning individuals
    Y : pd.DataFrame
        Target to be predicted by the model (1 column for binary classification: int in {0,1})
    train_valid_set_with_uncorrected_results : pd.DataFrame
        Dataset extracted from X_train_valid, of shape (X_train_valid.shape)
        Eventually augmented with probabilities, predicted labels, (true/false) (positive/negative) of the first "uncorrected" model
    protected_attribute : List[str]
        Columns of the train_valid_set where the user wants to erase discrimination,
        based on discrimination alerts.
    fairness_purpose : str
        Metrics by which the user wants to measure the gap between groups : basis of evaluation to compute the fair_score.
        Will serve in detection: detects inequalities of selection by the model (between subgroups and the mean) regarding fairness_purpose
        Will serve in correction: selects the best model satisfying stat/fair tradeoff regarding fairness_purpose
        -> For classification (binary or multi-classes), must be set to value in 
        {"overall_positive_rate", "nb_positive", "false_positive_rate", "false_negative_rate", "true_positive_rate", "true_negative_rate}
        -> For regression, must be set to a value in {'distribution_gap','mean_squared_error', 'mean_absolute_percentage_error', 'r2_score'}
    model_task : str
        Goal the user wants to achieve with the model: either classify, or regress...
        Must be set to value in {"regression", "classification"}
    stat_criteria : str
        Metrics of statistic performance the user wants to optimise
        -> For classification, must be set to value in {'auc','aucpr','mix_auc_aucpr'}
        -> For multiclass, must be set to value in {'merror','mlogloss','auc','f1_score'}
        -> For regression, must be set to value in {'rmse', 'mape'}, i.e. root mean square error and mean absolute percentage error
        https://xgboost.readthedocs.io/en/stable/parameter.html
    fairness_constraint : str, optional
        Setting of fairness constraint to train fairer models, by default None
        The user has the choice to add a fairness constraint
        By default, we choose a fairness constraint (eg FalsePositiveRateParity()) to achieve the fairness_purpose (eg false_positive_rate) the user has already chosen
        Fairer models are trained by weights distorsions, according to GridSearch implemented by Fairlearn:
        https://github.com/fairlearn/fairlearn/blob/main/fairlearn/reductions/__init__.py
    tradeoff : str, optional
        How the user wants to (un)balance stat and fair performances to compute the model's score, by default "moderate"
        For the moment, 3 choices (coefficients stress on the user's preferences regarding stat or fair score):
        "moderate": equal coefficients
        "fair_preferred": coefficients stress on fair_score (=2/3) to establish the tradeoff_score
        "stat_preferred": coefficients stress on stat_score (=2/3) to establish the tradeoff_score
    nb_grid_search_models : int, optional
        Number of models for fair training in competition with the initial "uncorrected" model, by default 4
    model_type : str, by default "xgboost"
        Type of the machine-learning model to be trained, either in trees (XGBoost) or neural networks family ('neural_net' implementation from Torch).
        The model can also entail less complexity in learning - we also implemented a logistic regression and simple neural network (multi-layer-perceptron) method. 
        This will be the root model to train a baseline model (not specifically trained for fairness) and FairDream models in competition for fairer results. 
        Must be set to a value in {'xgboost', 'neural_net', 'tabnet', 'log_reg', 'mlp'}

    Returns
    -------
    List()
        Returns a list of (nb_grid_search_models ) fair_models to be compared

    """

    # returns predictions on train&valid set, to assess if the model has trained as discrimining

    # first, extract the protected attribute in a separate column (to permit further Grid Search)
    predictions_dict = {}

    protected_df, _ = label_encode_categorical_features(X)#pd.get_dummies(X)
    print(protected_df.columns)
    protected_df = protected_df.loc[:, protected_attribute]

    (
        X_train,
        X_valid,
        X_train_valid,
        X_test,
        Y_train,
        Y_valid,
        Y_train_valid,
        Y_test,
    ) = train_valid_test_split(X, Y, model_task)

    (
        protected_df_train,
        protected_df_valid,
        protected_df_train_valid,
        protected_df_test,
        Y_train,
        Y_valid,
        Y_train_valid,
        Y_test,
    ) = train_valid_test_split(protected_df, Y, model_task)

    # uncorrected model (regardless of fairness) trained as a basis for fair training

    # build the GridSearch basis model, depending on model_task (classifier, regressor...)
    if model_task == "classification":

        # setting of fairness constraint to train fairer models

        print(
            f"\n --- TRAINING of competing fair models for {model_task} task, \n to minimise gaps of {fairness_purpose} on {protected_attribute} --- \n"
        )

        if fairness_constraint is not None:
            fairness_constraint = fairness_constraint

        # to compare fairness metrics optimized according to labels (equalized odds) or looking at overall selection rates (demographic parity)
        elif fairness_purpose in {"overall_positive_rate", "nb_positive", "fscore"}:
            fairness_constraint = DemographicParity()

        elif fairness_purpose in {"false_positive_rate", "true_negative_rate", "true_positive_rate", "false_negative_rate"}:
            fairness_constraint = EqualizedOdds()
        #     fairness_constraint = FalsePositiveRateParity()

        # elif fairness_purpose in {"true_positive_rate", "false_negative_rate"}:
        #     fairness_constraint = TruePositiveRateParity()

        else:
            raise ValueError(
                "for classification, fairness_purpose must be set to a value in {'fscore', 'false_positive_rate', 'false_negative_rate',"
                "'true_positive_rate','true_negative_rate','overall_positive_rate','nb_positive'}"
            )

        if model_type == 'xgboost':

            xgb_classif_params = {
                "seed": 7,
                "objective": "binary:logistic",
                "n_estimators": 1000,
                "max_depth": 3,
                "importance_type": "gain",
                "use_label_encoder": False,
                "eval_metric": stat_criteria,
            }

            model = XGBClassifier(**xgb_classif_params)
        
        elif model_type == 'neural_net':
            model = NeuralNet_for_GridSearch()

        elif model_type == 'tabnet':
            model = TabNet_for_GridSearch()

        elif model_type == 'log_reg':
            model = LogisticRegression(solver="liblinear", fit_intercept=True)
        
        elif model_type == 'mlp':
            model = MLPClassifier(random_state=1, max_iter=300)

        elif model_type == 'svm':
            model = svm.SVC(gamma=1, probability=True)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=0)

        else:
            raise NotImplementedError("The model_type of the model you want to train is not implemented. Must be set to a value in {'xgboost', 'neural_net', 'tabnet', 'log_reg', 'mlp'}")
        

        grid_search_estimator = GridSearch(
            model,
            constraints=fairness_constraint,
            #grid_size=nb_grid_search_models,
        )

    elif model_task == "regression":

        xgb_reg_params = {
            "seed": 7,
            "objective": "reg:squarederror",
            "n_estimators": 1000,
            "max_depth": 3,
            "importance_type": "gain",
            "use_label_encoder": False,
            "eval_metric": stat_criteria,
        }

        # setting of fairness constraint to train fairer models

        print(
            f"\n --- TRAINING of competing fair models for {model_task} task, \n to minimise gaps of {fairness_purpose} on {protected_attribute} --- \n"
        )

        protected_attribute_values = X_train[protected_attribute].value_counts()

        if fairness_constraint is not None:
            fairness_constraint = fairness_constraint

        elif fairness_purpose in {"mean_squared_error", "r2_score", "distribution_gap"}:
            fairness_constraint = BoundedGroupLoss(
                SquareLoss(min(protected_attribute_values), max(protected_attribute_values)),
                upper_bound=0.1,
            )

        elif fairness_purpose == "mean_absolute_percentage_error":
            fairness_constraint = BoundedGroupLoss(
                AbsoluteLoss(min(protected_attribute_values), max(protected_attribute_values)),
                upper_bound=0.1,
            )

        else:
            raise ValueError(
                "for regression, fairness_purpose must be set to a value in {'distribution_gap','mean_squared_error', 'mean_absolute_percentage_error', 'r2_score'}"
            )

        grid_search_estimator = GridSearch(
            XGBRegressor(**xgb_reg_params),
            constraints=fairness_constraint,
            grid_size=nb_grid_search_models,
        )

    else:
        raise NotImplementedError(
            f"The {model_task} task you want the model to perform is not implemented with grid_search. Must be set to value in {'regression','classification'}"
        )

    # grid_search_estimator: class 'fairlearn.reductions._grid_search.grid_search.GridSearch'
    # Estimator to perform a grid search given a blackbox estimator algorithm.
    # The approach used is taken from section 3.4 of
    # `Agarwal et al. (2018) <https://arxiv.org/abs/1803.02453>`_ [1]_.

    # Example of grid_search_estimator, for classification (given fairness_constraint = ErrorRateParity() and grid_size=4):
    # GridSearch(constraints=<fairlearn.reductions._moments.utility_parity.ErrorRateParity object at 0x7f9b2164c310>,
    #    estimator=XGBClassifier(base_score=None, booster=None,
    #                            colsample_bylevel=None,
    #                            colsample_bynode=None, colsample_bytree=None,
    #                            enable_categorical=False,
    #                            eval_metric='error', gamma=None, gpu_id=None,
    #                            importance_type='gain',
    #                            interaction_constraints=None,
    #                            learning_rate=None, max_delta_step=None,
    #                            max_depth=3, min_child_weight=None,
    #                            missing=nan, monotone_constraints=None,
    #                            n_estimators=1000, n_jobs=None,
    #                            num_parallel_tree=None, predictor=None,
    #                            random_state=None, reg_alpha=None,
    #                            reg_lambda=None, scale_pos_weight=None,
    #                            seed=7, subsample=None, tree_method=None,
    #                            use_label_encoder=False,
    #                            validate_parameters=None, ...),
    #    grid_size=4)

    grid_search_estimator.fit(X_train, Y_train, sensitive_features=protected_df_train)

    # grid_search_estimator.fit: method to run the grid search.
    # This will result in multiple copies of the
    # estimator being made, with sample_weights each time different to train fairer models,
    # and the :code:`fit(X)` method of each one called.

    predictors = grid_search_estimator.predictors_

    for model_number, model in enumerate(predictors):

        # print(f"\n --- TRAINING of grid_search model n° {model_number} --- \n")
        model_name=f"grid_search_fair_{model_number}"
        predictions_dict[model_name]={}

        probas_pred_class_1_train_valid=model.predict_proba(X_train_valid)[:,1]

        Y_pred_train_valid = prediction_train_valid_by_task(
            model, X_valid, X_train_valid, Y_valid, Y_train_valid, model_task
        )

        predictions_dict[model_name]["Y_pred_train_valid"]=Y_pred_train_valid
        predictions_dict[model_name]["probas_pred_class_1_train_valid"]=probas_pred_class_1_train_valid

    return predictions_dict


def weighted_groups_fair_train(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    train_valid_set_with_uncorrected_results: pd.DataFrame,
    protected_attribute: List[str],
    fairness_purpose: str,
    model_task: str,
    stat_criteria: str,
    fairness_constraint: str = None,
    tradeoff: str = "moderate",
    nb_weighted_groups_models: int = 4,
    model_type: str = None,
) -> list:
    """Trains fairer models with sample weights to counterbalance the discrimination of protected_attribute regarding fairness purpose,
    and returns their predictions on X_train_valid in a list.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame with all features (entered as columns) concerning individuals
    Y : pd.DataFrame
        Target to be predicted by the model (1 column for binary classification: int in {0,1})
    train_valid_set_with_uncorrected_results : pd.DataFrame
        Dataset extracted from X_train_valid, of shape (X_train_valid.shape)
        Eventually augmented with probabilities, predicted labels, (true/false) (positive/negative) of the first "uncorrected" model
    protected_attribute : List[str]
        Columns of the train_valid_set where the user wants to erase discrimination,
        based on discrimination alerts.
    fairness_purpose : str
        Metrics by which the user wants to measure the gap between groups : basis of evaluation to compute the fair_score.
        Will serve in detection: detects inequalities of selection by the model (between subgroups and the mean) regarding fairness_purpose
        Will serve in correction: selects the best model satisfying stat/fair tradeoff regarding fairness_purpose
        -> For classification (binary or multi-classes), must be set to value in 
        {"overall_positive_rate", "nb_positive", "false_positive_rate", "false_negative_rate", "true_positive_rate", "true_negative_rate}
        -> For regression, must be set to a value in {'distribution_gap','mean_squared_error', 'mean_absolute_percentage_error', 'r2_score'}
    model_task : str
        Goal the user wants to achieve with the model: either classify, or regress...
        Must be set to value in {"regression", "classification", "multiclass"}
    stat_criteria : str
        Metrics of statistic performance the user wants to optimise
        -> For classification, must be set to value in {'auc','aucpr','mix_auc_aucpr'}
        -> For multiclass, must be set to value in {'merror','mlogloss','auc','f1_score'}
        -> For regression, must be set to value in {'rmse', 'mape'}, i.e. root mean square error and mean absolute percentage error
        https://xgboost.readthedocs.io/en/stable/parameter.html
    fairness_constraint : str, optional
        Setting of fairness constraint to train fairer models, by default None
        The user has the choice to add a fairness constraint
        By default, we choose a fairness constraint (eg FalsePositiveRateParity()) to achieve the fairness_purpose (eg false_positive_rate) the user has already chosen
        Fairer models are trained by weights distorsions, according to GridSearch implemented by Fairlearn:
        https://github.com/fairlearn/fairlearn/blob/main/fairlearn/reductions/__init__.py
    tradeoff : str, optional
        How the user wants to (un)balance stat and fair performances to compute the model's score, by default "moderate"
        For the moment, 3 choices (coefficients stress on the user's preferences regarding stat or fair score):
        "moderate": equal coefficients
        "fair_preferred": coefficients stress on fair_score (=2/3) to establish the tradeoff_score
        "stat_preferred": coefficients stress on stat_score (=2/3) to establish the tradeoff_score
    nb_weighted_groups_models : int, optional
        Number of models for fair training in competition with the initial "uncorrected" model, by default 4
    model_type : str, by default "xgboost"
        Type of the machine-learning model to be trained, either in trees (XGBoost) or neural networks family ('neural_net' implementation from Torch).
        The model can also entail less complexity in learning - we also implemented a logistic regression and simple neural network (multi-layer-perceptron) method. 
        This will be the root model to train a baseline model (not specifically trained for fairness) and FairDream models in competition for fairer results. 
        Must be set to a value in {'xgboost', 'neural_net', 'tabnet', 'log_reg', 'mlp'}

    Returns
    -------
    List()
        Returns a list of (nb_weighted_groups_models) fair_models predictions on X_train_valid to be compared

    """

    # returns predictions on train&valid set, to assess if the model has trained as discrimining

    # first, extract the protected attribute in a separate columnweights_method (to permit further Grid Search)
    predictions_dict = {}

    protected_df = X.copy()
    protected_df = protected_df.loc[:, protected_attribute]

    (
        X_train,
        X_valid,
        X_train_valid,
        X_test,
        Y_train,
        Y_valid,
        Y_train_valid,
        Y_test,
    ) = train_valid_test_split(X, Y, model_task)

    (
        protected_df_train,
        protected_df_valid,
        protected_df_train_valid,
        protected_df_test,
        Y_train,
        Y_valid,
        Y_train_valid,
        Y_test,
    ) = train_valid_test_split(protected_df, Y, model_task)

    # # assess model's predictions (discriminant?) on the set it was trained (train&valid)

    # link between fairness purpose and fairness constraint? TODO ?

    # get fair scores corresponding to uncorrected model's discriminant predictions
    fair_scores_df_uncorrected = fair_score(
        train_valid_set_with_uncorrected_results,
        "uncorrected",
        fairness_purpose,
        model_task,
        protected_attribute,
        fairness_mode="correction",
    )

    # and compute the "uncorrected" model stat/fair scores, to be compared with the future fair models
    scores_by_model_cumulator = []

    fair_score_value_uncorrected = sum_scores_gains_by_groups(
        fair_scores_df_uncorrected, fairness_purpose
    )

    stat_score_value_uncorrected = stat_score(
        train_valid_set_with_uncorrected_results, "uncorrected", model_task, stat_criteria
    )

    tradeoff_score = stat_fair_tradeoff(
        stat_score_value_uncorrected, fair_score_value_uncorrected, tradeoff
    )

    model_scores_dict_uncorrected = {
        "model_name": "uncorrected",
        "fair_score_value": fair_score_value_uncorrected,
        "stat_score_value": stat_score_value_uncorrected,
        "tradeoff_score": tradeoff_score,
        # "weights_by_group": "uncorrected",
        # "new_sample_weight": None, TODO save sample_weights in the dict: element of model's explainability?
        # e.g. with these weights distorsions, the model grow error weights on protected individuals => trains according to the user's fairness vision
        "fair_scores_df": fair_scores_df_uncorrected,
    }

    scores_by_model_cumulator.append(model_scores_dict_uncorrected)

    # training of new models
    # then for each "fairer" predictor: return Y_pred_train_valid and add results to the augmented train&valid set
    for fair_model_number in range(nb_weighted_groups_models):

        model_name = f"weighted_fair_{fair_model_number}"
        predictions_dict[model_name]={}
        new_weight_by_group_dict = {}

        max_fair_score = fair_scores_df_uncorrected[fairness_purpose].max()

        for col_nb in range(len(fair_scores_df_uncorrected.index)):
            # set ascending weights of loss during training: weights of error grow with the disadvantage of the group
            group_fair_score = fair_scores_df_uncorrected.iloc[col_nb][fairness_purpose]
            # set the weights of gaps, depending on the fairness_purpose (to minimise e.g. false negative rate, to maximise e.g. true positive rate)
            if is_fairness_purpose_to_maximise(fairness_purpose)==True:
                
                best_fair_score = fair_scores_df_uncorrected[fairness_purpose].max()
                gap_fair_scores = best_fair_score - group_fair_score 

            elif is_fairness_purpose_to_maximise(fairness_purpose)==False:
                best_fair_score = fair_scores_df_uncorrected[fairness_purpose].min()
                gap_fair_scores = group_fair_score - best_fair_score

            nb_indivs_disadvantaged = gap_fair_scores*fair_scores_df_uncorrected['nb_individuals_by_group'].iloc[col_nb]
            # normalize the number of people disadvantaged (relatively to other groups), to avoid the sample weights to explode...
            relative_nb_indivs_disadvantaged = nb_indivs_disadvantaged/fair_scores_df_uncorrected['nb_individuals_by_group'].sum()
            # coeffs_disadvantaged_df = (coeffs_disadvantaged_df-coeffs_disadvantaged_df.min())/(coeffs_disadvantaged_df.max()-coeffs_disadvantaged_df.min())
            coeff_disadvantaged = fair_model_number * relative_nb_indivs_disadvantaged if fair_model_number%2!=0 else 1

            # group_fair_score = fair_scores_df_uncorrected.iloc[col_nb][fairness_purpose]
            # # introduce coeff of people disadvantaged in 1/2 sample weights
            # coeff_disadvantaged = fair_model_number*(max_fair_score - group_fair_score)*math.log(coeffs_disadvantaged_df.iloc[col_nb]) if fair_model_number%2!=0 else 1
            # new weight of error increases exponentially when the group fair_score is far from the maximum 
                # the more people are disadvantaged (in gap), the more the exponential term is high
            # coeff_disadvantaged is a coefficient which stresses the number of people disadvantaged inside the group 
                # the more people are disadvantaged (in number), the more the multiplicative term is high 
                # the number of people (e.g. 700, 3000...) are normalized to scale the sample weight across reasonable values, 
                # comparing the number of people to the other groups (e.g. 0.1, 0.4...)
            # inside each term, fair_model_number (and the random apparition of the coeff_disadvantaged) serves to introduce variety 
                # when Weighted Group is in search of new weights to find 'fairer' results
            new_weight = coeff_disadvantaged * exp(fair_model_number * (max_fair_score - group_fair_score))       

            new_weight_by_group_dict[group_fair_score] = new_weight

            print(f"{fair_scores_df_uncorrected.index[col_nb]}: weight == {new_weight}")

        # reconstitute X_train individuals with their fair scores
        individuals_fair_scores_uncorrected_train_valid_set = (
            train_valid_set_with_uncorrected_results[f"{fairness_purpose}_uncorrected"]
        )
        individuals_fair_scores_uncorrected_train_set = (
            individuals_fair_scores_uncorrected_train_valid_set.iloc[: X_train.shape[0]]
        )
        individuals_fair_scores_uncorrected_train_set

        # then map each fair score with its new sample weight
        adjusted_sample_weight_train = individuals_fair_scores_uncorrected_train_set.map(
            new_weight_by_group_dict
        )

        # and new predictions with the sample weight
        Y_pred_train_valid, probas_pred_class_1_train_valid = train_naive_model(
            X_train=X_train,
            X_valid=X_valid,
            X_train_valid=X_train_valid,
            X_test=X_test,
            Y_train=Y_train,
            Y_valid=Y_valid,
            Y_train_valid=Y_train_valid,
            Y_test=Y_test,
            model_task=model_task,
            stat_criteria=stat_criteria,
            adjusted_sample_weight_train=adjusted_sample_weight_train,
            model_type=model_type,
        )

        predictions_dict[model_name]["Y_pred_train_valid"]=Y_pred_train_valid
        predictions_dict[model_name]["probas_pred_class_1_train_valid"]=probas_pred_class_1_train_valid

    return predictions_dict


def fair_train(
    X: pd.DataFrame,
    Y: pd.DataFrame,
    train_valid_set_with_uncorrected_results: pd.DataFrame,
    protected_attribute: List[str],
    fairness_purpose: str,
    model_task: str,
    stat_criteria: str,
    tradeoff: str = "moderate",
    weight_method: str = "weighted_groups",
    nb_fair_models: int = 4,
    fairness_constraint: str = None,
    sorted_labels_list: list = None,
    distribution_frontier: str = None,
    frontier_label: Union[str, int] = None,
    model_type: str = 'xgboost',
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Trains fairer models with weight_method to counterbalance the discrimination of protected_attribute regarding fairness purpose,
    and selects the best model based on the user's stat/fair tradeoff.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame with all features (entered as columns) concerning individuals
    Y : pd.DataFrame
        Target to be predicted by the model (1 column for binary classification: int in {0,1})
    train_valid_set_with_uncorrected_results : pd.DataFrame
        Dataset extracted from X_train_valid, of shape (X_train_valid.shape)
        Eventually augmented with probabilities, predicted labels, (true/false) (positive/negative) of the first "uncorrected" model
    protected_attribute : List[str]
        Columns of the train_valid_set where the user wants to erase discrimination,
        based on discrimination alerts.
    fairness_purpose : str
        Metrics by which the user wants to measure the gap between groups : basis of evaluation to compute the fair_score.
        Will serve in detection: detects inequalities of selection by the model (between subgroups and the mean) regarding fairness_purpose
        Will serve in correction: selects the best model satisfying stat/fair tradeoff regarding fairness_purpose
    -> For classification (binary or multi-classes), must be set to value in 
    {"overall_positive_rate", "nb_positive", "false_positive_rate", "false_negative_rate", "true_positive_rate", "true_negative_rate}
    -> For regression, must be set to a value in {'distribution_gap','mean_squared_error', 'mean_absolute_percentage_error', 'r2_score'}
    model_task : str
        Goal the user wants to achieve with the model: either classify, or regress...
        Must be set to value in {"regression", "classification", "multiclass"}
    stat_criteria : str
        Metrics of statistic performance the user wants to optimise
        -> For classification, must be set to value in {'auc','aucpr','mix_auc_aucpr'}
        -> For multiclass, must be set to value in {'merror','mlogloss','auc','f1_score'}
        -> For regression, must be set to value in {'rmse', 'mape'}, i.e. root mean square error and mean absolute percentage error
        https://xgboost.readthedocs.io/en/stable/parameter.html
    tradeoff : str, optional
        How the user wants to (un)balance stat and fair performances to compute the model's score, by default "moderate"
        For the moment, 3 choices (coefficients stress on the user's preferences regarding stat or fair score):
        "moderate": equal coefficients
        "fair_preferred": coefficients stress on fair_score (=2/3) to establish the tradeoff_score
        "stat_preferred": coefficients stress on stat_score (=2/3) to establish the tradeoff_score
    weight_method : str
        How to generate fair_models
        Must be set to values in {'weighted_groups','grid_search','grid_and_weighted_groups'}
        If weight_method == 'grid_and_weighted_groups':
            1/2 models trained with weights distorsion ('weighted_groups')
            1/2 models trained with 'grid_search'
    nb_fair_models : int, optional
        Number of models for fair training in competition with the initial "uncorrected" model, by default 4
    fairness_constraint : str, optional
        Setting of fairness constraint to train fairer models, by default None
        The user has the choice to add a fairness constraint
        By default, we choose a fairness constraint (eg FalsePositiveRateParity()) to achieve the fairness_purpose (eg false_positive_rate) the user has already chosen
        Fairer models are trained by weights distorsions, according to GridSearch implemented by Fairlearn:
        https://github.com/fairlearn/fairlearn/blob/main/fairlearn/reductions/__init__.py
    model_type : str, by default "xgboost"
        Type of the machine-learning model to be trained, either in trees (XGBoost) or neural networks family ('neural_net' implementation from Torch).
        The model can also entail less complexity in learning - we also implemented a logistic regression and simple neural network (multi-layer-perceptron) method. 
        This will be the root model to train a baseline model (not specifically trained for fairness) and FairDream models in competition for fairer results. 
        Must be set to a value in {'xgboost', 'neural_net', 'tabnet', 'log_reg', 'mlp'}
        
    --- Optional parameters for model_task == "multiclass" ---
    For "multiclass", the fairness is evaluated like in binary classification (!) valid only when Y labels are independant.
    To transform multiclass into 2 classes, the user ranks the labels by ascending order (ex : [0,1,2,3] housing nights).
    Then, one has 2 choices:
        (1) To set manually the 'frontier_label' (ex: one chooses that individuals > label "2" nights are privileged)
        (2) To set a % of individuals distribution, i.e. 'distribution_frontier' (median, quartiles) which will automatically determine the 'frontier_label'

    sorted_labels_list : list, optional
        When model_task == "multiclass", list of labels with the desired ascending ranking of the user.
        Ex: when labels are number of housing nights and the user wants to maximise it,
        sorted_labels_list = [0,1,2,3]
    distribution_frontier : str, optional
        When model_task == "multiclass", the user chooses which % of individuals (ranked by ascending labels) will be considered as 'privileged' (1) or not (0).
        Must be set to values in {'Q1','median','Q3'}:
            'Q1': 25% of (0), 75% of "privileged" (1). Recommended if you want to intervene for the few worse-off individuals.
            'median': 50% (0), 50% (1), default choice
            'Q3': 75% of (0), 25% of "privileged" (1). Recommended if you want to intervene against the few better-off individuals.
    frontier_label : Union[str, int]
        Here, the user can set manually
        the label corresponding to the 'distribution_frontier', depending on its type: "blue" (str), 1 night (int)...
        Must be set to values in np.unique(multi_Y_pred_train_valid)
        Ex: must be set to existing label values, in {0, 1, 2, 3} number of nights

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, dict]
        train_valid_set_with_corrected_results: pd.DataFrame
            X_train_valid augmented with probabilities, predicted labels, (true/false) (positive/negative)
            for all model ("uncorrected", "fair_1", ..., "fair_n", with (n) = grid_size)
            Will serve to attribute to each individual one's new fair score,
            to inspect if previously discriminated individuals (with "uncorrected" model) are now discriminated
        models_df: pd.DataFrame
            For each line (i.e. model), DataFrame with columns = ['model_name', 'fair_score_value', 'stat_score_value', 'tradeoff_score',
            'fair_scores_df', 'selected']
            Will serve to select the best model according to the user's preferences, and then plot it to compare its fair/stat score with the initial "uncorrected" model
        best_model_dict:
            Dictionary containing the best models scores (stat&fair),
            the model to be re-used for better predictions,
            and a DataFrame with scores of the initial "uncorrected" model to be challenged ('fair_scores_df_uncorrected')
            Columns of best_model_dict :
                'model_name': str
                'fair_score_value': int
                'stat_score_value': int
                'tradeoff_score': int
                'fair_scores_df': pd.DataFrame
                'model':XGBClassifier
                'fair_scores_df_uncorrected': pd.DataFrame

    """

    # returns predictions on train&valid set, to assess if the model has trained as discrimining
    # fair_models are generated depending on the weight_method the user chooses:

    # TODO futher step: change the form of the fair model, depending on the type of uncorrected_model entered by the user (xgb, neural networks...)
    if weight_method == "grid_search":

        nb_grid_search_models = nb_fair_models

        predictions_dict = grid_search_fair_train(
            X,
            Y,
            train_valid_set_with_uncorrected_results,
            protected_attribute,
            fairness_purpose,
            model_task,
            stat_criteria,
            fairness_constraint,
            tradeoff,
            nb_grid_search_models,
            model_type=model_type,
        )

    elif weight_method == "weighted_groups":

        nb_weighted_groups_models = nb_fair_models

        predictions_dict = weighted_groups_fair_train(
            X,
            Y,
            train_valid_set_with_uncorrected_results,
            protected_attribute,
            fairness_purpose,
            model_task,
            stat_criteria,
            fairness_constraint,
            tradeoff,
            nb_weighted_groups_models,
            model_type=model_type,
        )

    elif weight_method == "grid_and_weighted_groups":

        # generate 2 types of models depends on nb_fair_models (even / odd)
        half_nb_models = int((nb_fair_models + nb_fair_models % 2) / 2)

        nb_weighted_groups_models = half_nb_models
        nb_grid_search_models = half_nb_models

        grid_search_predictions_dict = grid_search_fair_train(
            X,
            Y,
            train_valid_set_with_uncorrected_results,
            protected_attribute,
            fairness_purpose,
            model_task,
            stat_criteria,
            fairness_constraint,
            tradeoff,
            model_type=model_type,
        )

        weighted_groups_predictions_dict = weighted_groups_fair_train(
            X,
            Y,
            train_valid_set_with_uncorrected_results,
            protected_attribute,
            fairness_purpose,
            model_task,
            stat_criteria,
            fairness_constraint,
            tradeoff,
            nb_weighted_groups_models,
            model_type=model_type,
        )

        predictions_dict = dict(grid_search_predictions_dict, **weighted_groups_predictions_dict)

    else:
        raise NotImplementedError(
            f"weight_method {weight_method} not implemented. Must be set to values in {'weighted_groups','grid_search','grid_and_weighted_groups'}"
        )

    model_names_list = ["uncorrected"]
    scores_by_model_cumulator = []
    Y_train_valid = train_valid_set_with_uncorrected_results["target_train_valid"]

    # then for each "fairer" predictor: return Y_pred_train_valid and add results to the augmented train&valid set
    # initialize the dataset with new models' results
    train_valid_set_with_corrected_results = train_valid_set_with_uncorrected_results.copy()
    # for model_number, Y_pred_train_valid in enumerate(predictions_dict):
        # model_name = f"fair_{model_number}"
    for model_name in predictions_dict:

        # convert multiclass results into binary results for fair analysis
        # and store multiclass results to compute stat score
        if model_task == "multiclass":

            # get multi_Y_train and median label to transform predictions vector into binary vector (> or < median label)
            multi_Y_pred_train_valid_uncorrected = train_valid_set_with_uncorrected_results[
                "multi_predicted_uncorrected"
            ].to_numpy()

            if frontier_label is None:
                print(f"\n distribution_frontier: {distribution_frontier} \n")
                sorted_labels_list=[0,1,2,3] # TODO pass as user's criterion
                frontier_label = get_frontier_label(
                    multi_Y_pred_train_valid_uncorrected, sorted_labels_list, distribution_frontier
                )

            elif frontier_label not in np.unique(multi_Y_pred_train_valid_uncorrected):
                raise ValueError(
                    f"The frontier_label {frontier_label} you manually set is not the correct writing of a target label. \n"
                    f"Must be set to values in {np.unique(multi_Y_pred_train_valid_uncorrected)} \n"
                    "Ex: must be set to existing label values, in {0, 1, 2, 3} number of nights"
                )

            multi_Y_train_valid = train_valid_set_with_uncorrected_results[
                "multi_target_train_valid"
            ]
            multi_predict_proba_train_valid = Y_pred_train_valid
            multi_Y_pred_train_valid = multi_predict_proba_train_valid.argmax(axis=-1)

            Y_pred_train_valid = compute_binary_Y(
                multi_Y_pred_train_valid, sorted_labels_list, frontier_label
            )

        elif model_task in {"regression", "classification"}:
            multi_Y_train_valid = None
            multi_predict_proba_train_valid = None

            Y_pred_train_valid=predictions_dict[model_name]["Y_pred_train_valid"]
            probas_pred_class_1_train_valid=predictions_dict[model_name]["probas_pred_class_1_train_valid"]

            # TODO after experiments, replace "FairDream" and "GridSearch" by the real model names
            # smarter solution: keep the name, but replace only after, if str contains 'weighted' / 'grid'? 
            if weight_method=="weighted_groups":
                model_name="FairDream"
            elif weight_method=="grid_search":
                model_name="GridSearch"

        # add new informations to a train_valid set with corrected results, for model selection
        train_valid_set_with_corrected_results = augment_train_valid_set_with_results(
            model_name=model_name,
            previous_train_valid_set=train_valid_set_with_corrected_results,
            Y_train_valid=Y_train_valid,
            Y_pred_train_valid=Y_pred_train_valid,
            probas_pred_class_1_train_valid=probas_pred_class_1_train_valid,
            model_task=model_task,
            multi_Y_train_valid=multi_Y_train_valid,
            multi_predict_proba_train_valid=multi_predict_proba_train_valid,
        )

        model_names_list.append(model_name)

    # now, compute fair_scores by model and store them in a DataFrame
    for model_name in model_names_list:

        fair_scores_df = fair_score(
            train_valid_set_with_corrected_results,
            model_name,
            fairness_purpose,
            model_task,
            protected_attribute,
            fairness_mode="correction",
        )

        # then transform df into value aggregated by groups (here, sum) # TODO technical user can choose how to aggregate fair scores by groups
        fair_score_value = sum_scores_gains_by_groups(fair_scores_df, fairness_purpose)

        stat_score_value = stat_score(
            train_valid_set_with_corrected_results, model_name, model_task, stat_criteria
        )

        tradeoff_score = stat_fair_tradeoff(stat_score_value, fair_score_value, tradeoff)

        # then store scores in a dict -> future line of a models_df
        model_scores_dict = {
            "model_name": model_name,
            "fair_score_value": fair_score_value,
            "stat_score_value": stat_score_value,
            "tradeoff_score": tradeoff_score,
            "fair_scores_df": fair_scores_df,
        }

        # # specify fair_scores_df for the initial "uncorrected" model to be compared with the other fairer models:
        # model_scores_dict["fair_scores_df_uncorrected"] = fair_scores_df

        scores_by_model_cumulator.append(model_scores_dict)

    # transform these data about models: from dict -> DataFrame to compare it
    models_df = pd.DataFrame(scores_by_model_cumulator)

    # selection of the best model according to the stat/fair tradeoff of the user
    best_model_dict = model_selection(models_df)

    # add to models_df a column with selected values for color plots of selected models
    models_df["selected"] = "Not selected"

    models_df.loc[models_df["model_name"] == "uncorrected", "selected"] = "Baseline"#"Uncorrected"

    models_df.loc[
        models_df["model_name"] == best_model_dict["model_name"], "selected"
    ] = "Best model"

    print(f"Best model with your fair / stat tradeoff is {best_model_dict['model_name']}")

    return train_valid_set_with_corrected_results, models_df, best_model_dict
