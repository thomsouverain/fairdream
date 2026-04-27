import logging
import os
import pickle 
import shutil
from pathlib import Path
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
import torch
from matplotlib import pyplot
from matplotlib.pylab import plt
from numpy import argmax
from pandas.api.types import is_numeric_dtype
from scipy import integrate 
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import auc
from sklearn.metrics import brier_score_loss
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.preprocessing import LabelEncoder
# from fairdream.data_preparation import get_confusion_matrix_by_indiv_df
# from fairdream.data_preparation import label_encode_categorical_features

logger = logging.getLogger(__name__)

def tensor_to_numpy(tensor:torch.tensor)->np.array:
    """Converts a tensor into numpy (if tensor).
     Serves to compute and plot metrics & avoids waste of GPU memory (CPU device).

    Args:
        tensor (torch.tensor) 

    Returns:
        np.array
    """
    if torch.is_tensor(tensor):
        numpy_vector = tensor.cpu().detach().numpy()
    else:
        raise NotImplementedError("The vector you want to convert to numpy is not a tensor")
    return numpy_vector

def pickle_save_model(uncorrected_model, uncorrected_model_path: str = None):
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


def compute_best_fscore(Y_splitted_set: pd.DataFrame, proba_splitted_set: pd.DataFrame) -> pyplot:
    """Based on fscore, optimises the threshold of the model.
    It will permit to convert predicted probabilities into labels (Y predicted).

    Parameters
    ----------
    Y_splitted_set : pd.DataFrame
        Target, ie true labels of Y.
    proba_splitted_set : pd.DataFrame
        Vector of probabilities predicted by the model = p(Y==1) for binary classification.

    Returns
    -------
    pyplot
        A plot of precision / recall curve for the model showing the best threshold.
    """
    precision_valid, recall_valid, thresholds = precision_recall_curve(
        Y_splitted_set, proba_splitted_set
    )
    # convert to f score
    fscore = (2 * precision_valid * recall_valid) / (precision_valid + recall_valid)
    # locate the index of the largest f score
    ix = argmax(fscore)

    best_threshold = thresholds[ix]
    best_fscore = fscore[ix]

    print("Best Threshold=%f, with F-Score=%.2f" % (best_threshold, best_fscore))

    # plot the roc curve for the model
    no_skill = len(Y_splitted_set[Y_splitted_set == 1]) / len(Y_splitted_set)
    train_fig = pyplot.plot([0, 1], [no_skill, no_skill], linestyle="--", label="No Skill")
    train_fig = pyplot.plot(recall_valid, precision_valid, label="Model")
    train_fig = pyplot.scatter(
        recall_valid[ix], precision_valid[ix], marker="o", color="black", label="Best"
    )
    # axis labels
    train_fig = pyplot.title("Statistical performance on valid set (PR AUC)")
    train_fig = pyplot.xlabel("Recall")
    train_fig = pyplot.ylabel("Precision")
    train_fig = pyplot.legend()
    # show the plot
    pyplot.show(train_fig)

    return best_threshold, best_fscore


def compute_fscore_by_group(
    true_positive: pd.DataFrame, false_positive: pd.DataFrame, false_negative: pd.DataFrame
) -> pd.DataFrame:
    """Based on columns of a DataFrame (true positive, false positive and false negative),
    computes for each group (ie line of the DataFrame) its mean fscore.

    Parameters
    ----------
    true_positive : pd.DataFrame
    false_positive : pd.DataFrame
    false_negative : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        New DataFrame column with mean fscore per line (group)
    """
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    fscore_by_group = 2 * (precision * recall) / (precision + recall)
    return fscore_by_group


def distribution_gap_kolmogorov_smirnov(target_df, predicted_df):
    kolmogorov_smirnov_gap, kolmogorov_smirnov_pvalue = stats.ks_2samp(
        target_df, predicted_df, alternative="two-sided"
    )

    # only consider statistically significant results, here at a confidence level of 95%
    if kolmogorov_smirnov_pvalue > 0.05:

        logger.warning(
            "\n---Warning--- Kolmogorov-Smirnov distribution gap should not be used for the group below,\n"
            f"because confidence level < 95%.\n"
            "Please use an other fairness_purpose for regression, \n"
            "in {'mean_squared_error', 'mean_absolute_percentage_error', 'r2_score'}"
        )

    return kolmogorov_smirnov_gap


def split_inspected_column_in_groups(
    augmented_train_valid_set: pd.DataFrame,
    copy_set: pd.DataFrame,
    inspected_column: str,
    inspected_column_values: np.array,
    num_quantile: int = 5,
    n_bins: int = None,
    bins_space: int = None,
) -> pd.Series:
    """Generates a new column of copy_set where each individual of the train_valid_set is assigned to a group,
    based on the inspected_column values (using pd.cut).

    Parameters
    ----------
    augmented_train_valid_set : pd.DataFrame
        The train_valid_set augmented with model's prediction, from which the inspected_column
        is extracted to separate groups into this column, and compute fair_scores by groups.
    copy_set: pd.DataFrame
        A copy of augmented_train_valid_set, to add groups regarding fairness_purpose
    inspected_column : str
        Name of the column of augmented_train_valid_set to be inspected, i.e. from which groups and their fair_scores will be extracted
        (protected attribute in correction phase)
    inspected_column_values: np.array
        Array containing the different values of inspected_column, necessary to split them into intervals and then attribute groups to individuals
    num_quantile: int, optional
        If the user wants to set a precise number of quantiles to build groups for all inspected column,
        by default 5
    n_bins : int, optional
        If the user wants to set a precise number of groups for all inspected_column,
        by default None
    bins_space : int, optional
        If the user wants to set a precise interval of values taken by inspected_column to create the groups of inspected_column,
        by default None

    Returns
    -------
    pd.Series
        copy_set[inspected_column]: pd.Series corresponding to the groups (as column) of individuals (as index) for the predicted_column

    Raises
    ------
    NotImplementedError
    """

    if not (is_numeric_dtype(augmented_train_valid_set[inspected_column])):
        copy_set[inspected_column] = copy_set[inspected_column]

    # same for inspected_columns only composed of 0 and 1 # TODO same when less than 10 categories 
    elif len(np.unique(inspected_column_values)) <= 10: #and (
    #     np.all(np.unique(inspected_column_values) == [0, 1])
    #     or np.all(np.unique(inspected_column_values) == [1, 0])
    # ):
        copy_set[inspected_column] = copy_set[inspected_column]

    # cut inspected_column values in intervals depending on min-max (to detect discriminations) in other cases
    elif n_bins is None:
        copy_set[inspected_column] = pd.qcut(
            copy_set[inspected_column], q=num_quantile, duplicates="drop"
        )

    elif not bins_space:
        min_val = copy_set[inspected_column].min()
        max_val = copy_set[inspected_column].max()
        bins_space = np.linspace(min_val - 0.001, max_val, num=n_bins)

        copy_set[inspected_column] = pd.cut(
            augmented_train_valid_set[inspected_column], bins=bins_space, duplicates="drop"
        )

    else:
        raise NotImplementedError(
            "fscore can not be computed, no dtype str and no bins number specified"
        )

    return copy_set[inspected_column]

def get_columns_for_confusion_matrix_fairness_purpose(
    copy_set: pd.DataFrame,
    model_name: str,
    inspected_column: str,
    fairness_purpose: str,
) -> Tuple[str]:

    """For model_task=='classification', returns the columns used to compute a fairness_purpose of the matrix confusion for the 'model_name', in             
            ['false_positive_rate',
            'false_negative_rate',
            'true_positive_rate',
            'true_negative_rate']
        Returns in a Tuple the names of the columns used to compute the fairness_purpose, (column_to_assign, column_complementary)
           e.g. (True Positives, False Negatives) to compute the True Positive Rate

    Parameters
    ----------
    copy_set : pd.DataFrame
        Copy of augmented_train_valid_set, with a new column where each individual of the train_valid_set is assigned to a group,
        based on the inspected_column values.
    model_name: str
        Name of the model whose results are integrated.
        Must be set to value in {'uncorrected', 'fair_1', ..., 'fair_n'}
        depending on the number (n) of fairer models in competition (n == grid_size fixed by the user)
    inspected_column : str
        Name of the column of augmented_train_valid_set to be inspected, i.e. from which groups and their fair_scores will be extracted
        (protected attribute in correction phase)
    fairness_purpose : str
        Metrics by which the user wants to measure the gap between groups : basis of evaluation to compute the fair_score.
        Will serve in detection: detects inequalities of selection by the model (between subgroups and the mean) regarding fairness_purpose
        Will serve in correction: selects the best model satisfying stat/fair tradeoff regarding fairness_purpose
        -> For classification (binary or multi-classes), must be set to value in 
        {"overall_positive_rate", "nb_positive", "false_positive_rate", "false_negative_rate", "true_positive_rate", "true_negative_rate}
        -> For regression, must be set to a value in {'distribution_gap','mean_squared_error', 'mean_absolute_percentage_error', 'r2_score'}

    Returns:
        Returns in a Tuple the names of the columns used to compute the fairness_purpose, (column_to_assign, column_complementary)
           e.g. (True Positives, False Negatives) to compute the True Positive Rate
    """

    columns_for_fair_score = [
        f"false_positive_{model_name}",
        f"false_negative_{model_name}",
        f"true_positive_{model_name}",
        f"true_negative_{model_name}"
    ]

    fair_scores_df = copy_set.groupby([inspected_column])[columns_for_fair_score].mean()

    if fairness_purpose == "false_positive_rate":
        column_to_assign = f"false_positive_{model_name}"
        column_complementary = f"true_negative_{model_name}"

    elif fairness_purpose == "false_negative_rate":
        column_to_assign = f"false_negative_{model_name}"
        column_complementary = f"true_positive_{model_name}"

    elif fairness_purpose == "true_positive_rate":
        column_to_assign = f"true_positive_{model_name}"
        column_complementary = f"false_negative_{model_name}"

    elif fairness_purpose == "true_negative_rate":
        column_to_assign = f"true_negative_{model_name}"
        column_complementary = f"false_positive_{model_name}"

    else:
        raise NotImplementedError(f"fairness_purpose {fairness_purpose} not implemented. For model_task=='classification', must be set to a value in ['false_positive_rate','false_negative_rate','true_positive_rate','true_negative_rate']")
    
    return column_to_assign, column_complementary

def compute_fair_score_by_group(
    copy_set: pd.DataFrame,
    model_name: str,
    inspected_column: str,
    fairness_purpose: str,
    model_task: str,
) -> pd.DataFrame:
    """Computes the mean by group of fair_score, based on groups of individuals and the user's fairness_purpose.

    Parameters
    ----------
    copy_set : pd.DataFrame
        Copy of augmented_train_valid_set, with a new column where each individual of the train_valid_set is assigned to a group,
        based on the inspected_column values.
    model_name: str
        Name of the model whose results are integrated.
        Must be set to value in {'uncorrected', 'fair_1', ..., 'fair_n'}
        depending on the number (n) of fairer models in competition (n == grid_size fixed by the user)
    inspected_column : str
        Name of the column of augmented_train_valid_set to be inspected, i.e. from which groups and their fair_scores will be extracted
        (protected attribute in correction phase)
    fairness_purpose : str
        Metrics by which the user wants to measure the gap between groups : basis of evaluation to compute the fair_score.
        Will serve in detection: detects inequalities of selection by the model (between subgroups and the mean) regarding fairness_purpose
        Will serve in correction: selects the best model satisfying stat/fair tradeoff regarding fairness_purpose
        -> For classification (binary or multi-classes), must be set to value in 
        {"overall_positive_rate", "nb_positive", "false_positive_rate", "false_negative_rate", "true_positive_rate", "true_negative_rate}
        -> For regression, must be set to a value in {'distribution_gap','mean_squared_error', 'mean_absolute_percentage_error', 'r2_score'}
    model_task: str
        Goal the user wants to achieve with the model: either classify, or regress...
        Must be set to value in {"regression", "classification", "multiclass}
        Note: for fairness analysis, "multiclass" is considered as binary classification (> or < the median_label)

    Returns
    -------
    pd.DataFrame
        DataFrame with values taken by groups of individuals for inspected_column as index (e.g. 'age': [17,26],[26,33]...),
        and as columns 'fair_score' (e.g. 'fscore') and 'nb_individuals_by_group'

    Raises
    ------
    ValueError
        _description_
    """
    if model_task in {"classification", "multiclass"}:
        # for classification (binary and multi-classes): when computation has to be done with several columns of fair_scores_df
        if fairness_purpose == "fscore":

            columns_for_fscore = [
                f"true_positive_{model_name}",
                f"false_positive_{model_name}",
                f"false_negative_{model_name}",
            ]

            fair_scores_df = copy_set.groupby(inspected_column)[columns_for_fscore].mean()

            fair_scores_df["fscore"] = compute_fscore_by_group(
                fair_scores_df[f"true_positive_{model_name}"],
                fair_scores_df[f"false_positive_{model_name}"],
                fair_scores_df[f"false_negative_{model_name}"],
            )

            # after computation, only keep fscore and nb_individuals_by_group
            fair_scores_df = fair_scores_df.drop(columns=columns_for_fscore)

        # for classification(binary and multi-classes): computation is directly linked with one column of fair_scores_df for other fairness_purposes
        elif fairness_purpose in {
            "false_positive_rate",
            "false_negative_rate",
            "true_positive_rate",
            "true_negative_rate",
        }:

            columns_for_fair_score = [
                f"false_positive_{model_name}",
                f"false_negative_{model_name}",
                f"true_positive_{model_name}",
                f"true_negative_{model_name}"
            ]

            fair_scores_df = copy_set.groupby([inspected_column])[columns_for_fair_score].mean()

            column_to_assign, column_complementary = get_columns_for_confusion_matrix_fairness_purpose(
                copy_set=copy_set, model_name=model_name, inspected_column=inspected_column, fairness_purpose=fairness_purpose
            )

            fair_scores_df[fairness_purpose] = fair_scores_df[column_to_assign]/(fair_scores_df[column_to_assign]+fair_scores_df[column_complementary])
            fair_scores_df = fair_scores_df.drop(columns=columns_for_fair_score+[column_complementary])

        elif fairness_purpose in {
            "overall_positive_rate",
            "nb_positive",
        }:

            columns_for_fair_score = [f"predicted_{model_name}"]

            if fairness_purpose == "overall_positive_rate":
                fair_scores_df = copy_set.groupby([inspected_column])[columns_for_fair_score].mean()
                column_to_assign = f"predicted_{model_name}"

            elif fairness_purpose == "nb_positive":
                fair_scores_df = copy_set.groupby([inspected_column])[columns_for_fair_score].sum()
                column_to_assign = f"predicted_{model_name}"

            else:
                raise NotImplementedError(f"fairness_purpose {fairness_purpose} not implemented.")

            fair_scores_df[fairness_purpose] = fair_scores_df[column_to_assign]
            fair_scores_df = fair_scores_df.drop(columns=columns_for_fair_score)

        else:
            raise ValueError(
                "for classification, fairness_purpose must be set to a value in {'fscore', 'false_positive_rate', 'false_negative_rate',"
                "'true_positive_rate','true_negative_rate','overall_positive_rate','nb_positive'}"
                )

        fair_scores_df["nb_individuals_by_group"] = copy_set.groupby([inspected_column])[
            inspected_column
        ].count()

    elif model_task == "regression":

        if fairness_purpose == "distribution_gap":
            # compute distribution gap between pred and target, according to Kolmogorov-Smirnov test
            fairness_metrics = distribution_gap_kolmogorov_smirnov

        elif fairness_purpose == "mean_squared_error":
            fairness_metrics = mean_squared_error

        elif fairness_purpose == "mean_absolute_percentage_error":
            fairness_metrics = mean_absolute_percentage_error

        elif fairness_purpose == "r2_score":
            fairness_metrics = r2_score

        else:
            raise ValueError(
                "For regression, fairness_purpose must be set to a value in {'distribution_gap', 'mean_absolute_percentage_error', 'r2_score'}"
            )

        fair_scores_by_group_cumulator = []
        nb_individuals_per_group_df = copy_set[inspected_column].value_counts()
        intervals_values = nb_individuals_per_group_df.index
        nb_intervals = len(intervals_values)

        for group_number in range(nb_intervals):

            group_interval = intervals_values[group_number]

            inspected_group_df = copy_set.loc[copy_set[inspected_column] == group_interval]

            group_fair_score = fairness_metrics(
                inspected_group_df["target_train_valid"],
                inspected_group_df[f"predicted_{model_name}"],
            )

            nb_individuals_in_group = nb_individuals_per_group_df.iloc[group_number]

            mean_fair_score = fairness_metrics(
                copy_set["target_train_valid"], copy_set[f"predicted_{model_name}"]
            )

            # for all group, join these data in a dict
            fair_scores_by_group_dict = {
                "group_interval": group_interval,
                fairness_purpose: group_fair_score,
                "nb_individuals_by_group": nb_individuals_in_group,
                "mean": mean_fair_score,
            }

            fair_scores_by_group_cumulator.append(fair_scores_by_group_dict)

        # then transform the dict with groups informations into a DataFrame
        fair_scores_df = pd.DataFrame(fair_scores_by_group_cumulator).set_index("group_interval")

    else:
        raise NotImplementedError(
            f"The {model_task} task you want the model to perform is not implemented. Must be set to value in {'regression','classification','multiclass'}"
        )

    return fair_scores_df


def fair_score(
    augmented_train_valid_set: pd.DataFrame,
    model_name: str,
    fairness_purpose: str,
    model_task: str,
    inspected_column: str,
    fairness_mode: str = None,
) -> pd.DataFrame:
    """For all inspected column, computes fair scores by groups of the column and returns them in a DataFrame.
    Will serve in detection: discrimination alert if the fair_score of a group in inspected_column < injustice_acceptance (set by the user) * mean fair_score
    Will serve in correction: for all fair trained model, its fair_score is computed to select the best model
    Will serve in detection & correction: for the user, plots of fair_scores by groups of inspected_column

        Parameters
        ----------
        augmented_train_valid_set : pd.DataFrame
            The train_valid_set augmented with model's prediction, from which the inspected_column
            is extracted to separate groups into this column, and compute fair_scores by groups.
        model_name : str
            Name of the model whose results are integrated.
            Must be set to value in {'uncorrected', 'fair_1', ..., 'fair_n'}
            depending on the number (n) of fairer models in competition (n == grid_size fixed by the user)
        fairness_purpose : str
            Metrics by which the user wants to measure the gap between groups : basis of evaluation to compute the fair_score.
            Will serve in detection: detects inequalities of selection by the model (between subgroups and the mean) regarding fairness_purpose
            Will serve in correction: selects the best model satisfying stat/fair tradeoff regarding fairness_purpose
            -> For classification (binary or multi-classes), must be set to value in 
            {"overall_positive_rate", "nb_positive", "false_positive_rate", "false_negative_rate", "true_negative_rate", "true_positive_rate"}
            -> For regression, must be set to a value in {'distribution_gap','mean_squared_error', 'mean_absolute_percentage_error', 'r2_score'}
        model_task: str
            Goal the user wants to achieve with the model: either classify, or regress...
            Must be set to value in {"regression", "classification","multiclass"}
        inspected_column : str
            Name of the column of augmented_train_valid_set to be inspected, i.e. from which groups and their fair_scores will be extracted
            (protected attribute in correction phase)
        fairness_mode : str, optional
            Must be set to value in {None, 'correction'}, by default None
            If 'correction', the DataFrame returns fair_score by indivs for protected attribute (to illustrate individual changes of predictions)


        Returns
        -------
        pd.DataFrame
            Returns a DataFrame with inspected_column groups (e.g. different ages), fair_score by group, nb_indivs by group, and mean fair_score.

        Raises
        ------
        NotImplementedError
    """

    copy_set = augmented_train_valid_set.copy()

    inspected_column_values = np.unique(copy_set[inspected_column])

    copy_set[inspected_column] = split_inspected_column_in_groups(
        augmented_train_valid_set, copy_set, inspected_column, inspected_column_values
    )

    fair_scores_df = compute_fair_score_by_group(
        copy_set, model_name, inspected_column, fairness_purpose, model_task
    )

    map_categories_dict = dict(zip(fair_scores_df.index, fair_scores_df[fairness_purpose]))
    copy_set[inspected_column] = copy_set[inspected_column].map(map_categories_dict)

    fair_scores_df["mean"] = copy_set[inspected_column].astype("float64").mean()

    # when fairness_mode == 'correction', the DataFrame saves the fair_score by indivs for protected attribute
    # to then illustrate individual changes of predictions
    if fairness_mode == "correction":

        augmented_train_valid_set[f"{fairness_purpose}_{model_name}"] = copy_set[
            inspected_column
        ].astype("float64")

    return fair_scores_df


def stat_score(
    augmented_train_valid_set: pd.DataFrame, model_name: str, model_task: str, stat_criteria: str
) -> int:
    """Computes a stat score value for all model.

    Parameters
    ----------
    augmented_train_valid_set : pd.DataFrame
        The train_valid_set augmented with model's prediction
    model_name : str
        Name of the model whose stat score is computed.
        Must be set to value in {'uncorrected', 'fair_1', ..., 'fair_n'}
        depending on the number (n) of fairer models in competition (n == grid_size fixed by the user)
    model_task : str
        Goal the user wants to achieve with the model: either classify, or regress...
        Must be set to value in {"regression", "classification", "multiclass"}
    stat_criteria : str
        Metrics of statistic performance the user wants to optimise
        -> For classification, must be set to value in {'auc','aucpr','mix_auc_aucpr'}
        -> For multiclass, must be set to value in {'merror','mlogloss','auc','f1_score'}
        -> For regression, must be set to value in {'rmse', 'mape'}, i.e. root mean square error and mean absolute percentage error
        https://xgboost.readthedocs.io/en/stable/parameter.html

    Returns
    -------
    int
    """

    Y_train_valid = augmented_train_valid_set[f"target_train_valid"].values.reshape(-1)
    pred_train_valid = augmented_train_valid_set[f"predicted_{model_name}"]

    if model_task == "multiclass":

        # get multi_Y_pred and multi_predict_proba to inspect stat performances on different labels
        # because depends on stat metrics for multiclass (either use labels or probabilities)

        # first unpack list of probabilities by label and reconstitute the initial vector of predict_probas
        packed_probas = augmented_train_valid_set[f"multi_proba_{model_name}"]
        multi_predict_proba_train_valid = np.array(packed_probas.values.tolist())
        # multi_predict_proba_train_valid = np.squeeze(unpacked, axis=1)

        multi_Y_pred_train_valid = augmented_train_valid_set[f"multi_predicted_{model_name}"]
        multi_Y_train_valid = augmented_train_valid_set[f"multi_target_train_valid"]

        # TODO compute stat_score for all labels (vectors of probabilities), not only one vs all approach
        # For the moment, eval = 0 if wrong label, 1 if right label -> then score of error / by nb_indivs
        if stat_criteria == "merror":
            # wrong cases / all cases
            total_indivs_nb = multi_Y_train_valid.shape[0]
            stat_score_value = (
                multi_Y_train_valid != multi_Y_pred_train_valid
            ).sum() / total_indivs_nb

        elif stat_criteria == "mlogloss":
            stat_score_value = log_loss(multi_Y_train_valid, multi_predict_proba_train_valid)

        elif stat_criteria == "auc":

            stat_score_value = roc_auc_score(
                multi_Y_train_valid,
                multi_predict_proba_train_valid,
                multi_class="ovr",
                average="weighted",
            )

        elif stat_criteria == "f1_score":
            stat_score_value = f1_score(
                multi_Y_train_valid, multi_predict_proba_train_valid, average="weighted"
            )

        else:
            raise NotImplementedError(
                f"stat_criteria {stat_criteria} is not implemented."
                f"\n For multiclass, must be set to value in {'merror','mlogloss','auc','f1_score'}."
            )

    elif model_task == "classification":

        if stat_criteria in {"aucpr", "mix_auc_aucpr"}:

            precision_train_valid, recall_train_valid, _ = precision_recall_curve(
                Y_train_valid, pred_train_valid
            )

            if stat_criteria == "aucpr":
                stat_score_value = auc(recall_train_valid, precision_train_valid)

            elif stat_criteria == "mix_auc_aucpr":
                stat_score_value = (
                    roc_auc_score(Y_train_valid, pred_train_valid)
                    + auc(recall_train_valid, precision_train_valid)
                ) / 2

        elif stat_criteria == "auc":
            stat_score_value = roc_auc_score(Y_train_valid, pred_train_valid)

        else:
            raise NotImplementedError(
                f"stat_criteria {stat_criteria} is not implemented."
                f"\n For classification, must be set to value in {'auc','aucpr','mix_auc_aucpr'}."
            )

    elif model_task == "regression":

        if stat_criteria == "rmse":
            stat_score_value = mean_squared_error(Y_train_valid, pred_train_valid)

        elif stat_criteria == "mape":
            stat_score_value = mean_absolute_percentage_error(Y_train_valid, pred_train_valid)

        else:
            raise NotImplementedError(
                f"stat_criteria {stat_criteria} is not implemented."
                f"\n For regression, must be set to value in {'rmse', 'mape'}, i.e. root mean square error and mean absolute percentage error"
            )

    # linked with the stat objectives the user wants to maximise (sum of scores) or minimise (-sum of errors to be reduced)
    stat_scores_to_minimise_set = {
        "rmse",
        "mape",
        "merror",
        "mlogloss",
    }

    if stat_criteria in stat_scores_to_minimise_set:
        stat_score_value = -stat_score_value

    return stat_score_value


def sum_scores_gains_by_groups(fair_scores_df: pd.DataFrame, fairness_purpose: str) -> int:
    """Computes a fair score value for all model,
    by summing fair scores of its inspected_column's groups. Close to the MiniMax fairness view: 
    - Sum of the gaps to the maximum score (if fairness_purpose must be maximised, e.g. 'overall_positive_rate' in loan granting),
    - Sum of the gaps to the minimum score (if fairness_purpose must be minimised, e.g. 'mean_squared_error' in loan granting).

    Parameters
    ----------
    fair_scores_df : pd.DataFrame
        For a specific model, contains its fair scores by groups of the inspected_column.
        DataFrame with inspected_column groups (e.g. different ages), fair_score by group, nb_indivs by group, and mean fair_score.
    fairness_purpose : str
        Metrics by which the user wants to measure the gap between groups : basis of evaluation to compute the fair_score.
        Will serve in detection: detects inequalities of selection by the model (between subgroups and the mean) regarding fairness_purpose
        Will serve in correction: selects the best model satisfying stat/fair tradeoff regarding fairness_purpose
        -> For classification (binary or multi-classes), must be set to value in 
        {"overall_positive_rate", "nb_positive", "false_positive_rate", "false_negative_rate", "true_positive_rate", "true_negative_rate}        
        -> For regression, must be set to a value in {'distribution_gap','mean_squared_error', 'mean_absolute_percentage_error', 'r2_score'}

    Returns
    -------
    int
    """
    # linked with the fair objectives the user wants to maximise (sum of scores) or minimise (-sum of errors to be reduced)
    is_to_maximise = is_fairness_purpose_to_maximise(fairness_purpose)

    if is_to_maximise == True:
        # decrease the gap to the maximum
        max_fair_score = fair_scores_df[fairness_purpose].max()
        gap_fair_scores_model_max = (fair_scores_df[fairness_purpose] - max_fair_score)
        fair_score_value = np.sum(gap_fair_scores_model_max)

    elif is_to_maximise == False:
        # decrease the gap to the minimum
        min_fair_score = fair_scores_df[fairness_purpose].min()
        gap_fair_scores_model_min = (fair_scores_df[fairness_purpose] - min_fair_score)
        fair_score_value = -np.sum(gap_fair_scores_model_min)

    return fair_score_value

def is_fairness_purpose_to_maximise(fairness_purpose:str)->bool:
    """To compute the fair score, indicates if the fairness purpose set by the user must be maximised (True) or minimised (False).

    Args:
        fairness_purpose : str
            Metrics by which the user wants to measure the gap between groups : basis of evaluation to compute the fair_score.
            Will serve in detection: detects inequalities of selection by the model (between subgroups and the mean) regarding fairness_purpose
            Will serve in correction: selects the best model satisfying stat/fair tradeoff regarding fairness_purpose
            -> For classification (binary or multi-classes), must be set to value in 
            {"overall_positive_rate", "nb_positive", "false_positive_rate", "false_negative_rate", "true_positive_rate", "true_negative_rate}        
            -> For regression, must be set to a value in {'distribution_gap','mean_squared_error', 'mean_absolute_percentage_error', 'r2_score'}

    Returns:
        bool: indicates if the fairness purpose set by the user must be maximised (True) or minimised (False)
    """
    fairness_purposes_to_maximise_set = {
        "overall_positive_rate",
        "nb_positive",
        "true_positive_rate",
        "false_positive_rate",
        "r2_score",
    }
    fairness_purposes_to_minimise_set = {
        "true_negative_rate",
        "false_negative_rate",
        "distribution_gap",
        "mean_squared_error",
        "mean_absolute_percentage_error",
    }

    if fairness_purpose in fairness_purposes_to_maximise_set:
        is_to_maximise = True

    elif fairness_purpose in fairness_purposes_to_minimise_set:
        is_to_maximise = False

    return is_to_maximise


def get_auc(y_true:Union[torch.tensor, np.array], probas_pred:Union[torch.tensor, np.array], plot:bool=False)->tuple:
    """Computes the ROC & PR AUCs, and false and true positive ratios, 
    given a vector of probabilities for 2 classes and true labels (binary classification).

    Args:
        probas_pred (Union[torch.tensor, np.array]): vector of probabilities for the 2 classes
            shape(n_indivs, 2)
        y_true (Union[torch.tensor, np.array]): true labels 
            Must be set to values in {0,1}
            shape (n_indivs, )
        plot (bool, optional):if True, plots the ROC AUC and PR AUC curves. Defaults to False.

    Returns:
        tuple: roc_auc, pr_auc, fpr_ratio, tpr_ratio
    """
    # ensure to use CPU and arrays -> accuracy and AUC computing need to separate values from gradients
    probas_pred = tensor_to_numpy(probas_pred) if torch.is_tensor(probas_pred) else probas_pred
    y_true = tensor_to_numpy(y_true) if torch.is_tensor(y_true) else y_true
    # in case of np.array or pandas objects, already in the expected format to plot AUC
    # only get probability of class 1, for AUCs computing
    probas_pred_class1 = probas_pred[:,1]
    precision, recall, _ = precision_recall_curve(y_true=y_true, probas_pred=probas_pred_class1)
    fpr, tpr, _ = roc_curve(y_true, probas_pred_class1)
    roc_auc = round(auc(fpr, tpr),2)
    pr_auc = round(auc(recall, precision),2)

    if plot==True:
        # calculate the no skill line as the proportion of the positive class
        no_skill = len(y_true[y_true == 1]) / len(y_true)
        # plot the no skill precision-recall curve
        plt.plot(
            [0, 1], [no_skill, no_skill], linestyle="--", label="No Skill PR AUC"
        )
        plt.plot([0, 1], [0, 1], linestyle="--", label="No Skill ROC AUC")
        # plot model precision-recall curve
        plt.plot(recall, precision, label=f"PR AUC ({pr_auc})")
        plt.plot(fpr, tpr, label=f"ROC AUC ({roc_auc})")
        # Add in a title and axes labels
        plt.title("Last Epoch Classifier - PR AUC and ROC AUC")
        plt.xlabel("Recall/FPR")
        plt.ylabel("Precision/TPR")
        # show the legend
        plt.legend()
        # show the plot
        plt.show()

    # also return false positive and true positive mean ratios TODO other aggregation than mean?
    fpr_ratio = round(fpr.mean(),2)
    tpr_ratio = round(tpr.mean(),2)
    return roc_auc, pr_auc, fpr_ratio, tpr_ratio

def get_auc_by_group(model_name:str,dict_y_true_predicted:dict,plot:bool=False,auc_to_plot:str=None,threshold:list=None,inspected_column:str=None)->pd.DataFrame:
    """Compares groups of a feature  for the predictions of the same model, returns a DataFrame. 
    For each group, computes the ROC & PR AUCs, and false and true positive ratios, 
    given a vector of probabilities for 2 classes and true labels (binary classification).

    Args:
        model_name (str)
            Name of the model whose predictions are analysed to get AUCs. 
            Must be set to a value in {"uncorrected", "weighted_fair_0", .. "weighted_fair_n", "grid_search_fair_0", .. "grid_search_fair_n}
            Depending on the number (n) of models trained during the FairDream process. 
        dict_y_true_predicted (Dict[str:List(y_true,probas_pred)]): dictionary with key "group_name" and the DataFrame of X_train_valid for each group
            KEY
            group_name (str)
            COLUMNS required inside the DataFrame
            f"probas_pred_class_1_train_valid_{model_name}" (Union[torch.tensor, np.array]): vector of probabilities for the class 1 (income > $50,000)
                shape(n_indivs, 1)
            "target_train_valid" (Union[torch.tensor, np.array]): true labels 
                Must be set to values in {0,1}
                shape (n_indivs, )
        plot (bool, optional):if True, plots the ROC AUC and PR AUC curves. Defaults to False.
        auc_to_plot (str, optional):if set to "roc_auc" (resp. "pr_auc"), plots only the ROC (resp. PR) AUC curves. Defaults to None.
        threshold (List[float,float]): if (FPR,TPR) of a threshold is provided, will plot the point on AUC curves

    Returns:
        pd.DataFrame: for each group_name with its roc_auc, pr_auc, fpr_ratio, tpr_ratio
    """
    dict_perfs={}
    for group_name in dict_y_true_predicted.keys():
        dict_perfs[group_name]={}
        # only get probability of class 1, for AUCs computing
        precision, recall, _ = precision_recall_curve(y_true=dict_y_true_predicted[group_name]["target_train_valid"], probas_pred=dict_y_true_predicted[group_name][f"probas_pred_class_1_train_valid_{model_name}"])
        fpr, tpr, _ = roc_curve(dict_y_true_predicted[group_name]["target_train_valid"], dict_y_true_predicted[group_name][f"probas_pred_class_1_train_valid_{model_name}"])
        dict_perfs[group_name]["roc_auc"] = round(auc(fpr, tpr),2)
        dict_perfs[group_name]["pr_auc"]  = round(auc(recall, precision),2)

        if plot==True:
            if auc_to_plot is None:
                # plot both ROC and PR AUCs if not specified
                plt.plot(fpr, tpr, label=f"ROC AUC {group_name}: ({dict_perfs[group_name]['roc_auc'] })")
                plt.plot(recall, precision, label=f"PR AUC {group_name}: ({dict_perfs[group_name]['pr_auc']})")
                title = "ROC AUC and PR AUC by group"
                xlabel = "Recall/FPR"
                ylabel = "Precision/TPR"
            elif auc_to_plot=="roc_auc":
                plt.plot(fpr, tpr, label=f"ROC AUC {group_name}: ({dict_perfs[group_name]['roc_auc'] })")
                title = "ROC AUC by group"
                xlabel = "False Positive Rate"
                ylabel = "True Positive Rate"
            elif threshold is None and auc_to_plot=="pr_auc":
                # add precision-recall curve
                plt.plot(recall, precision, label=f"PR AUC {group_name}: ({dict_perfs[group_name]['pr_auc']})")
                title = "PR AUC by group"
                xlabel = "Recall"
                ylabel = "Precision"
            else:
                raise NotImplementedError("If plot==True, auc_to_plot must be set to a value in [None, 'roc_auc', 'pr_auc']")

        # also return false positive and true positive mean ratios TODO other aggregation than mean?
        dict_perfs[group_name]["fpr_ratio"] = round(fpr.mean(),2)
        dict_perfs[group_name]["tpr_ratio"] = round(tpr.mean(),2)

    if threshold is not None:
        # only plot the AUC curve, and the defined threshold to get probabilities into predictions
        x = threshold[0]
        y = threshold[1]
        plt.plot([x, x], [0, y], 'r--')  # Dashed line from x-axis to point (x, y)
        plt.plot([0, x], [y, y], 'r--')  # Dashed line from y-axis to point (x, y)
        plt.scatter(x,y, marker="o", color="black", label="Threshold")

    # Add in a title and axes labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # show the legend
    plt.legend()
    # save the fig # TODO unify the function with Brier, and positive rates? 
    path_gap_plots_col=f"calibration_plots/group_comparison/{inspected_column}"

    if not os.path.isdir(path_gap_plots_col):
        os.makedirs(path_gap_plots_col)

    plt.savefig(os.path.join(path_gap_plots_col,f"{auc_to_plot}_{model_name}"))
    # show the plot
    plt.show()
    # show the plot
    plt.show()
    # compute the DataFrame with results
    df_perfs_by_group=pd.DataFrame(dict_perfs)

    return df_perfs_by_group

def get_deciles_scores_class1(dict_y_true_predicted:dict,percentage_mode:bool=False)->pd.DataFrame:
    """Compares groups of a feature  for the predictions of the same model, returns a DataFrame. 
    For each group, computes the individuals above the score (d) by decile (d) => enables to assess if the model is calibrated.

    Args:
        dict_y_true_predicted (Dict[str:List(y_true,probas_pred)]): dictionary with key "group_name" and the DataFrame of X_train_valid for each group
            KEY
            group_name (str)
            COLUMNS required inside the DataFrame
            f"probas_pred_class_1_train_valid_{model_name}" (Union[torch.tensor, np.array]): vector of probabilities for the class 1 (income > $50,000)
                shape(n_indivs, 1)
            "y_true" (Union[torch.tensor, np.array]): true labels 
                Must be set to values in {0,1}
                shape (n_indivs, )
        percentage_mode (bool, optional):if True, prints the percentage (else count) of individuals above the score (d) in each group of feature. Defaults to False.

    Returns:
        pd.DataFrame: for each group_name with its individuals above the score (d) by decile (d)
    """
    dict_percent_class1_by_decile={}
    
    for group_name in dict_y_true_predicted.keys():
        dict_percent_class1_by_decile[group_name]={}

        for decile_nb in range(1,11):

            df_nb_decile=dict_y_true_predicted[group_name].loc[dict_y_true_predicted[group_name]["decile_score_class1"]==decile_nb]
            df_nb_class1_in_decile=dict_y_true_predicted[group_name].loc[(dict_y_true_predicted[group_name]["decile_score_class1"]==decile_nb)&(dict_y_true_predicted[group_name]["y_true"]==1)]

            percent_class1_in_decile=0 if df_nb_decile.shape[0]==0 else round(df_nb_class1_in_decile.shape[0]/df_nb_decile.shape[0],2)
            dict_percent_class1_by_decile[group_name][decile_nb]=percent_class1_in_decile if percentage_mode==True else df_nb_class1_in_decile.shape[0]

    df_deciles_scores_class1 = pd.DataFrame(dict_percent_class1_by_decile)
    return df_deciles_scores_class1

# below are functions computing weights by group with the numbers of the FairDream paper 
# def get_fairdream_weight(group_str, group_fair_score, nb_group_indivs, max_fair_score, nb_total_indivs, nb_fair_model):
#     gap_fair_scores=np.abs(group_fair_score-max_fair_score)
#     rate_indivs_disadvantaged=gap_fair_scores*nb_group_indivs/nb_total_indivs
#     new_weight=rate_indivs_disadvantaged*np.exp(nb_fair_model*gap_fair_scores)
#     print(f"{group_str} : {new_weight}")
#     return new_weight

# def get_new_weights(nb_fair_model):
#     max_fair_score=0.91
#     nb_total_indivs=3616+3666+3294+3263+3161
#     get_fairdream_weight("17-29", 0.12, 3616, max_fair_score, nb_total_indivs, nb_fair_model)
#     get_fairdream_weight("29-37", 0.66, 3666, max_fair_score, nb_total_indivs, nb_fair_model)
#     get_fairdream_weight("37-44", 0.76, 3294, max_fair_score, nb_total_indivs, nb_fair_model)
#     get_fairdream_weight("37-44", 0.74, 3263, max_fair_score, nb_total_indivs, nb_fair_model)
#     get_fairdream_weight("52-90", 0.12, 3161, max_fair_score, nb_total_indivs, nb_fair_model)

def split_df_into_groups(
    augmented_train_valid_set: pd.DataFrame,
    inspected_column: str,
    X_not_encoded:pd.DataFrame=None,
    ) -> Dict[object,pd.DataFrame]:
   """Returns a dictionary with 1 df by group (key:interval, value:splitted df), splitted according to split_inspected_column_in_groups.

    Parameters
    ----------
    augmented_train_valid_set : pd.DataFrame
        Copy of augmented_train_valid_set, with a new column where each individual of the train_valid_set is assigned to a group,
        based on the inspected_column values.
    inspected_column : str
        Name of the column of augmented_train_valid_set to be inspected, i.e. from which groups and their fair_scores will be extracted
        (protected attribute in correction phase)

    Returns
    -------
    Dict[object,pd.DataFrame]
        Key (object): interval splitted according to split_inspected_column_in_groups
        Value (pd.DataFrame): augmented_train_valid_set, selected for the corresponding interval (key) of inspected_column

    """
   copy_set = augmented_train_valid_set.copy()
   
   inspected_column_values = np.unique(copy_set[inspected_column])
   
   copy_set[inspected_column] = split_inspected_column_in_groups(augmented_train_valid_set, copy_set, inspected_column, inspected_column_values)
   
   dict_train_valid_by_group={}
   # if initial values provided (initial DataFrame), give back their initial names to column values
   if X_not_encoded is not None: 
        _, dict_categorical_mapping=label_encode_categorical_features(X_not_encoded)
        if inspected_column in dict_categorical_mapping.keys(): # Transformation only for categorical features!
            dict_inv_categorical_mapping={group_interval : group_name for group_interval, group_name in dict_categorical_mapping[inspected_column].items()}
   else:
        dict_inv_categorical_mapping=None

   for group_interval in np.unique(copy_set[inspected_column]):
        group_name=get_initial_value_from_interval(group_interval=group_interval,dict_inv_categorical_mapping=dict_inv_categorical_mapping)
        dict_train_valid_by_group[f"{inspected_column} {group_name}"] = copy_set.loc[copy_set[inspected_column]==group_interval]
   
   return dict_train_valid_by_group    

def get_initial_value_from_interval(group_interval:str, dict_inv_categorical_mapping:dict=None)->str:
    """From a dictionary containing for a column {group_name:group_interval}, returns the initial group_name which was label-encoded for model training.

    Args:
        group_interval (str): label-encoded name of the group of the initial DataFrame (X_not_encoded)
        dict_inv_categorical_mapping (dict, optional):dictionary containing for a column {group_name:group_interval}. Defaults to None.

    Returns:
        str: the initial "group_name" which was label-encoded in "group_interval" for efficient model training
    """
    if dict_inv_categorical_mapping is None:
        group_name=group_interval
    elif dict_inv_categorical_mapping is not None:
        group_name=dict_inv_categorical_mapping[group_interval]
    return group_name



def get_max_gap_groups_auc(model_name:str,inspected_column:str,dict_train_valid_by_group:Dict[object,pd.DataFrame],auc_to_plot:str)->Tuple[float,float]:
    # -> Tuple: get gap to the maximum group, 
    # and absolute score for the minimum group (= if the model performs well, even on the worst predicted group)

    df_perfs_by_group=get_auc_by_group(model_name=model_name,dict_y_true_predicted=dict_train_valid_by_group,
                                       plot=True,auc_to_plot=auc_to_plot, inspected_column=inspected_column)

    df_group_auc = df_perfs_by_group.loc[df_perfs_by_group.index==auc_to_plot].transpose()
    array_group_auc = np.array(df_group_auc)

    max_gap_groups_auc = array_group_auc.max() - array_group_auc.min()
    min_group_auc = array_group_auc.min()
    
    max_gap_groups_auc=round(max_gap_groups_auc,2)
    min_group_auc=round(min_group_auc,2)
    
    return max_gap_groups_auc, min_group_auc

def get_max_gap_groups_opr_tpr_fpr(augmented_train_valid_set:pd.DataFrame, inspected_column:str,model_name:str,fairness_purpose:str,model_task:str="classification")->Tuple[float,float]:
    # Computes gaps of TPR, and add OPR and FPR? 
    # TODO compare with calibration area, AT the point of a particular threshold (plot on AUC vs calibration group curves)?
    # TODO docstring
    dict_opr_tpr_fpr={}
    dict_opr_tpr_fpr[fairness_purpose]={}
    copy_set=augmented_train_valid_set.copy()

    # for model_name in ["uncorrected","GridSearch","FairDream"]:
    # to enable plots of multiple experiments, rename the FairDream model into "FairDream" vs "Baseline"
    # TODO add vs "GridSearch"!
    #if model_name == best_model_dict['model_name']:
        #simplified_model_name = "FairDream"
    if model_name == "uncorrected":
        simplified_model_name = "Baseline"
    else:
        simplified_model_name = model_name
    
    dict_opr_tpr_fpr[fairness_purpose][simplified_model_name]={}

    fair_score_inspected_column=compute_fair_score_by_group( 
        # TODO fair_score vs stat_score, to plot and save? Or only (O/F/P)R bars to compare, FairDream vs GridSearch vs Baseline!! 
        # TODO Then: automatic counts of nb (gaps>, worst score>)
    copy_set=copy_set,
    model_name=model_name,
    inspected_column=inspected_column,
    fairness_purpose=fairness_purpose,
    model_task=model_task,
    )

    min_group_rate=round(fair_score_inspected_column[fairness_purpose].min(),2)
    max_group_rate=round(fair_score_inspected_column[fairness_purpose].max(),2)    
    max_min_group_rate=round(max_group_rate-min_group_rate,2)
            
    dict_opr_tpr_fpr[fairness_purpose][simplified_model_name]["max_gap_groups"]=max_min_group_rate
    dict_opr_tpr_fpr[fairness_purpose][simplified_model_name]["worst_group_score"]=min_group_rate

    # plot the bars of scores by groups to save the experiments
    dict_opr_tpr_fpr[fairness_purpose][simplified_model_name]["all_groups"] = fair_score(
        augmented_train_valid_set=augmented_train_valid_set,
        model_name=model_name,
        fairness_purpose=fairness_purpose,
        model_task=model_task,
        inspected_column=inspected_column,
        fairness_mode="correction",
    )

    stat_criteria="auc" # for a view of statistical performance (classically measured by ROC-AUC) on all groups
    dict_opr_tpr_fpr[fairness_purpose][simplified_model_name]["roc_auc"] = round(stat_score(
        augmented_train_valid_set, model_name, model_task, stat_criteria
    ), 2)

    return (dict_opr_tpr_fpr[fairness_purpose][simplified_model_name]["max_gap_groups"],
            dict_opr_tpr_fpr[fairness_purpose][simplified_model_name]["worst_group_score"],
            dict_opr_tpr_fpr[fairness_purpose][simplified_model_name]["all_groups"],
            dict_opr_tpr_fpr[fairness_purpose][simplified_model_name]["roc_auc"]
    )

def get_max_gap_groups_brier(model_name:str,inspected_column:str,dict_train_valid_by_group:Dict[object,pd.DataFrame])->Tuple[float,float]:

    dict_brier_loss={}
    
    for group_name in dict_train_valid_by_group.keys():
        
        dict_brier_loss[group_name]={}
        # only get probability of class 1, for AUCs computing
        y_true=dict_train_valid_by_group[group_name]["target_train_valid"]
        y_prob=dict_train_valid_by_group[group_name][f"probas_pred_class_1_train_valid_{model_name}"]

        #dict_brier_loss[group_name] = round(brier_score_loss(y_true, y_prob),2)

        x_p, y_p=calibration_curve(y_true,y_prob,n_bins=10)#,normalize=True)
        # Compute the area between the perfectly calibrated curve - the model calibration curve
        x_y_curve = np.linspace(0, 1, 100)
        # Interpolate y_p values for x_y_curve points
        interp_y_p = np.interp(x_y_curve, x_p, y_p)
        # Calculate the area between the two curves using the trapezoidal rule
        area = np.trapz(interp_y_p - x_y_curve, x_y_curve)
        area = round(np.abs(area), 2)

        # TODO get this area for groups comparison, instead of Brier loss (asymmetric for groups with different base rates)
        dict_brier_loss[group_name] = area#round(brier_score_loss(y_true, y_prob),2)
        # plt.plot(x_p, y_p, label=f"Brier Loss {group_name}: ({dict_brier_loss[group_name]}) ; area {area}")
        plt.plot(x_p, y_p, label=f"Distance with perfect calibration {group_name}: ({dict_brier_loss[group_name]})")

        #dict_brier_loss[group_name] = round(get_brier_calibration(y_true,y_prob),2)
    # Add perfect calibration for reference
    plt.plot([0, 1], [0, 1], "--", label = "Perfect Calibration")
    # Add in a title and axes labels
    plt.title("Calibration Loss by group")
    plt.xlabel("Mean predicted probability (Positive class: 1)")
    plt.ylabel("Fraction of positives (Positive class: 1)")
    # show the legend
    plt.legend()
    # save the fig (and delete if same name, to avoid any confusion with a previous experiment) 
    # TODO into a function for Brier, AUC, and positive rates? 
    # TODO at the beginning of the experiment: delete the root path calibration_plots 
    path_gap_plots_col=f"calibration_plots/group_comparison/{inspected_column}"

    if not os.path.isdir(path_gap_plots_col):
        os.makedirs(path_gap_plots_col)

    # plt.savefig(os.path.join(path_gap_plots_col,f"calibration_loss_{model_name}"))
    # show the plot
    plt.show()
            
    max_gap_groups_brier = dict_brier_loss[max(dict_brier_loss)] - dict_brier_loss[min(dict_brier_loss)]
    max_group_brier = dict_brier_loss[max(dict_brier_loss)]
    
    max_gap_groups_brier = abs(round(max_gap_groups_brier,2))
    # as Brier score is a loss, the worst score is the maximum Brier loss
    max_group_brier = abs(round(max_group_brier,2))
        
    return max_gap_groups_brier, max_group_brier


def plot_calibration_by_group_tpr_curves(augmented_train_valid_set:pd.DataFrame,list_models_in_competition:List[str],inspected_column:str)->Tuple[float,float]:

    dict_train_valid_by_group=split_df_into_groups(augmented_train_valid_set=augmented_train_valid_set, inspected_column=inspected_column)

    dict_brier_loss={}

    for group_name in dict_train_valid_by_group.keys(): 
        
        for model_name in list_models_in_competition:
            
            dict_brier_loss[group_name]={}
            # only get probability of class 1, for AUCs computing
            y_true=dict_train_valid_by_group[group_name]["target_train_valid"]
            y_prob=dict_train_valid_by_group[group_name][f"probas_pred_class_1_train_valid_{model_name}"]

            # compute OPR along thresholds (calibration)
            x_p, y_p=calibration_curve(y_true,y_prob,n_bins=10)#,normalize=True)
            # Compute the area between the perfectly calibrated curve - the model calibration curve
            x_y_curve = np.linspace(0, 1, 100)
            # Interpolate y_p values for x_y_curve points
            interp_y_p = np.interp(x_y_curve, x_p, y_p)
            # Calculate the area between the two curves using the trapezoidal rule
            area = np.trapz(interp_y_p - x_y_curve, x_y_curve)
            area = round(np.abs(area), 2)

            # compute TPR (and FPR) along thresholds
            fpr, tpr, thresholds = roc_curve(y_true, y_prob)

            dict_brier_loss[group_name] = area # TODO ensure area is correctly computed (negative and positive values)
            # now, plot calibration and TPR curves # TODO determine the threshold chosen by Fairdream - F1 score, harmonic mean of ..PR!
            plt.plot(x_p, y_p, label=f"Distance with perfect calibration {group_name} ({model_name}): ({dict_brier_loss[group_name]})")
            plt.plot(thresholds, tpr, label=f"TPR {group_name} ({model_name})")

            #dict_brier_loss[group_name] = round(get_brier_calibration(y_true,y_prob),2)
        # Add perfect calibration for reference
        # plt.plot([0, 1], [0, 1], "--", label = "Perfect Calibration")
        # Add in a title and axes labels
        plt.title(f"Calibration Loss and True Positive Rate for {group_name}")
        plt.xlabel("Threshold to predict Positive class (1)")
        plt.ylabel("Fraction of positives (Positive class: 1) / True Positives")
        # show the legend
        plt.legend()
        # save the fig (and delete if same name, to avoid any confusion with a previous experiment) 
        # TODO into a function for Brier, AUC, and positive rates? 
        # TODO at the beginning of the experiment: delete the root path calibration_plots 
        path_gap_plots_col=f"calibration_plots/group_comparison/{inspected_column}"

        if not os.path.isdir(path_gap_plots_col):
            os.makedirs(path_gap_plots_col)

        # plt.savefig(os.path.join(path_gap_plots_col,f"calibration_loss_{model_name}"))
        # show the plot
        plt.show()
            
    max_gap_groups_brier = dict_brier_loss[max(dict_brier_loss)] - dict_brier_loss[min(dict_brier_loss)]
    max_group_brier = dict_brier_loss[max(dict_brier_loss)]
    
    max_gap_groups_brier = abs(round(max_gap_groups_brier,2))
    # as Brier score is a loss, the worst score is the maximum Brier loss
    max_group_brier = abs(round(max_group_brier,2))
        
    return max_gap_groups_brier, max_group_brier

def plot_opr_tpr_fpr_by_threshold(X_not_encoded:pd.DataFrame,
                                  augmented_train_valid_set:pd.DataFrame,
                                  list_models_in_competition:List[str],
                                  inspected_column:str,
                                  list_fairness_purposes:List[str])->Tuple[float,float]:

    dict_train_valid_by_group=split_df_into_groups(augmented_train_valid_set=augmented_train_valid_set, 
                                                   inspected_column=inspected_column,
                                                   X_not_encoded=X_not_encoded)
    
    #list_fairness_purposes=['false_positive_rate','false_negative_rate','true_positive_rate','true_negative_rate','overall_positive_rate','precision','calibration_by_decile']
    dict_colors={'FairDream':'blue','uncorrected':'red','GridSearch':'grey'}

    dict_thresholds_max_fscore={}
    dict_max_fscores={}

    for model_name in list_models_in_competition:
    # TODO adapt threshold by group?
            # get back the threshold corresponding to the best F-Score (PR-AUC, balance recall->TPR vs precision->calibration on a threshold!!)
        # TODO get it only on valid set, vs train & valid set?
                    # TODO test differenciated thresholds maximizing PR-AUC on each group?
        print(f"--- {model_name} --- Getting the threshold corresponding to the Precision / Recall trade-off maximizing F-score")
        dict_thresholds_max_fscore[model_name], dict_max_fscores[model_name] = compute_best_fscore(augmented_train_valid_set["target_train_valid"], augmented_train_valid_set[f"probas_pred_class_1_train_valid_{model_name}"])

    dict_ratios={}

    for group_name in dict_train_valid_by_group.keys(): 

        dict_ratios[group_name]={}

        for model_name in list_models_in_competition:
            dict_ratios[group_name][model_name]={}

            for fairness_purpose in list_fairness_purposes:
                dict_ratios[group_name][model_name][fairness_purpose]={}
            
        for model_name in list_models_in_competition:
            
            for fairness_purpose in list_fairness_purposes:

                if fairness_purpose=='calibration_by_decile': # for calibration, register values only for deciles!
                    # calibration: check if for each decile of score (r%) i.e. new individuals which are predicted as Y==1 above the threshold (r),
                        # there are (r%) individuals of this decile which are truly Y==1
                        # <=> if the scores correspond to probabilities in each group 
                        # P(Y=1/R=r)=r (Barocas and Selbst 2023)
                        # here, we inspect it taking the mean score (r) by decile // COMPAS // Mayson 
                        # below, we compute and plot the distance from perfect calibration by decile (for each (r) decile) and area for each (r)
                    n_bins=10
                    # step 1: get deciles
                    dict_train_valid_by_group[group_name][f"decile_mean_score_{model_name}"]=pd.qcut(x=dict_train_valid_by_group[group_name][f"probas_pred_class_1_train_valid_{model_name}"],
                                                                                                     q=n_bins)

                    dict_train_valid_by_group[group_name][f"decile_mean_score_{model_name}"]=dict_train_valid_by_group[group_name].groupby(f"decile_mean_score_{model_name}")[f"probas_pred_class_1_train_valid_{model_name}"].transform("mean")

                    # step 2: compute the mean score (r) for each decile
                    dict_ratios[group_name][model_name][fairness_purpose]={0:0,0.001:0}

                    for decile_mean_score in np.unique(dict_train_valid_by_group[group_name][f"decile_mean_score_{model_name}"]):
                        df_score_decile=dict_train_valid_by_group[group_name].loc[dict_train_valid_by_group[group_name][f"decile_mean_score_{model_name}"]==decile_mean_score]

                        # get the mean of truly positive samples among the decile
                        nb_y_1_among_score_decile=df_score_decile[f"target_train_valid"].mean()
                        dict_ratios[group_name][model_name][fairness_purpose][round(decile_mean_score,3)]=round(nb_y_1_among_score_decile,3)
                    
                    dict_ratios[group_name][model_name][fairness_purpose][0.999]=1
                    dict_ratios[group_name][model_name][fairness_purpose][1]=1
                
                elif fairness_purpose in ['false_positive_rate','false_negative_rate','true_positive_rate','true_negative_rate','overall_positive_rate','precision']:
                    
                    for threshold in range(1,100):
                        threshold=threshold/100

                        # compute the predicted "1" event by the model "model_name", above the provided threshold
                        dict_train_valid_by_group[group_name][f"predicted_{model_name}"] = (dict_train_valid_by_group[group_name][f"probas_pred_class_1_train_valid_{model_name}"]>threshold).astype(int)
                        dict_train_valid_by_group[group_name]=get_confusion_matrix_by_indiv_df(model_name=model_name,new_train_valid_set=dict_train_valid_by_group[group_name])

                        # compute FPR, FNR, TPR, TNR matching the given threshold
                        if fairness_purpose in ['false_positive_rate','false_negative_rate','true_positive_rate','true_negative_rate']:
                            column_to_assign, column_complementary = get_columns_for_confusion_matrix_fairness_purpose(
                                copy_set=dict_train_valid_by_group[group_name], model_name=model_name, inspected_column=inspected_column, 
                                fairness_purpose=fairness_purpose)
                            
                            dict_ratios[group_name][model_name][fairness_purpose][threshold]=(dict_train_valid_by_group[group_name][column_to_assign].sum()
                                                                    /(dict_train_valid_by_group[group_name][column_to_assign].sum()+dict_train_valid_by_group[group_name][column_complementary].sum()))
                        # compute OPR matching the given threshold
                        elif fairness_purpose=='overall_positive_rate':
                            dict_ratios[group_name][model_name][fairness_purpose][threshold]=dict_train_valid_by_group[group_name][f"predicted_{model_name}"].sum()/dict_train_valid_by_group[group_name].shape[0]

                        # for calibration, compute by threshold the share of (Y_true==1) among (Y_pred==1)
                        elif fairness_purpose=='precision':
                            # here, we get the % of right predictions among the (Y_pred=1) ; but we do not get it compared to perfect calibration !
                            # TODO plot thresholds -> % (deciles?) of Y_pred=1 [as x_axis] -> % among them of Y_true=1 [as y_axis] ; for now we missed a step...
                            # TODO hence get mean probability for each decile = f(threshold)
                            # perfect calibration would be along thresholds, Y_pred(t)=Y_true (currently, we only have the plot t->(Y_true/Y_pred(t)))
                            df_y_pred_1 = dict_train_valid_by_group[group_name].loc[dict_train_valid_by_group[group_name][f"predicted_{model_name}"]==1]
                            df_y_1_among_y_pred_1=df_y_pred_1.loc[df_y_pred_1[f"true_positive_{model_name}"]==1]

                            y_1_among_y_pred_1=1 if df_y_pred_1.shape[0]==0 else df_y_1_among_y_pred_1.shape[0]/df_y_pred_1.shape[0]

                            dict_ratios[group_name][model_name][fairness_purpose][threshold] = y_1_among_y_pred_1

                else:
                    raise NotImplementedError("The fairness_purpose you want to compare between groups is not implemented. Must be set to a value in ['false_positive_rate','false_negative_rate','true_positive_rate','true_negative_rate','overall_positive_rate','precision','calibration_by_decile']")

    # Now, plot the metrics comparing groups - or reverse commented below, to inspect all metrics for each group ? # TODO in a specific plot function?
    # for group_name in dict_train_valid_by_group.keys():     
    #     for fairness_purpose in list_fairness_purposes:     
    for fairness_purpose in list_fairness_purposes:
        for group_name in dict_train_valid_by_group.keys():
            for model_name in list_models_in_competition:

                plt.plot(dict_ratios[group_name][model_name][fairness_purpose].keys(), dict_ratios[group_name][model_name][fairness_purpose].values(),
                label=f"{fairness_purpose} ({model_name})",
                color=dict_colors[model_name]
                )

                # get the intersection meaning on selection rates
                intersection_y = None
                x_values=np.array(list(dict_ratios[group_name][model_name][fairness_purpose].keys()))
                y_values=np.array(list(dict_ratios[group_name][model_name][fairness_purpose].values()))
                for i in range(x_values.size - 1):
                    if x_values[i] <= dict_thresholds_max_fscore[model_name] <= x_values[i + 1]:
                        slope = (y_values[i + 1] - y_values[i]) / (x_values[i + 1] - x_values[i])
                        intersection_y = y_values[i] + slope * (dict_thresholds_max_fscore[model_name] - x_values[i])
                        break

                # Plot the horizontal line from the intersection point
                if intersection_y is not None:
                    plt.plot([x_values[0], dict_thresholds_max_fscore[model_name]], [intersection_y, intersection_y], color=dict_colors[model_name], linestyle='--', label=f'{model_name} for Best FScore ({round(dict_max_fscores[model_name],2)})')
                # Plot the vertical line from the intersection point
                if intersection_y is not None:
                    min_y_value = min(y_values)
                    vertical_line_y_end = 0 if intersection_y > min_y_value else max(y_values)
                    plt.plot([dict_thresholds_max_fscore[model_name], dict_thresholds_max_fscore[model_name]], [intersection_y, vertical_line_y_end], color=dict_colors[model_name], linestyle='--')

                if fairness_purpose in ['calibration_by_decile']: #compute and plot area from perfect calibration (sum of positive and negative trapezoidal areas between x=y line)
                #if fairness_purpose in ['precision','calibration_by_decile']: #compute and plot area from perfect calibration (sum of positive and negative trapezoidal areas between x=y line)
                    # Compute the absolute difference between the calibration curve and x=y line ("perfect calibration")
                    difference = np.abs(y_values - x_values)
                    # Separate the points where the curve is above and below the x=y line
                    above_line = y_values > x_values
                    below_line = y_values < x_values
                    # Compute the area below the curve
                    area_below = integrate.trapz(difference[below_line], x=x_values[below_line])
                    # Compute the area above the curve
                    area_above = integrate.trapz(difference[above_line], x=x_values[above_line])
                    total_area = round(area_below + area_above,2) 
                    plt.plot(x_values, x_values, linestyle='dotted', color='gray', label=f"Perfect {fairness_purpose}".replace('_by_decile',''))   
                    # Fill the area below the curve
                    plt.fill_between(x_values[below_line], y_values[below_line], x_values[below_line], color=dict_colors[model_name], alpha=0.3, interpolate=True)
                    plt.fill_between(x_values[above_line], y_values[above_line], x_values[above_line], color=dict_colors[model_name], alpha=0.3, interpolate=True, 
                                     label=f"{model_name} - Area to Perfect {fairness_purpose} ({total_area})".replace('_by_decile','')+f" sum by decile == {round(difference.sum(),2)}")
                    if fairness_purpose=='calibration_by_decile':
                        plt.scatter(dict_ratios[group_name][model_name][fairness_purpose].keys(), dict_ratios[group_name][model_name][fairness_purpose].values(),
                            color=dict_colors[model_name]
                            )

            # Set x and y limits to ensure that the axes touch the left and bottom frame
            plt.xlim(left=0)
            plt.ylim(bottom=0)
            plt.title(f"Along Thresholds, {fairness_purpose} for {group_name}")
            plt.xlabel("Threshold to predict Positive class (1)")
            plt.ylabel(f"Fraction of {fairness_purpose}".replace("_rate"," predicted").replace(
                "precision","True Positive among Predicted Positive").replace("calibration", "True Positive among Predicted Positive"))
            # plt.legend()
            plt.savefig(f"example_{inspected_column}_{fairness_purpose}_{group_name}")
            plt.show()

    return dict_ratios


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

def label_encode_categorical_features(X: pd.DataFrame):

    dict_categorical_mapping = {}
    X_encoded=X.copy()

    for col in X.columns:
        # label-encode the categorical features
        if X[col].dtypes in ['category','object']:
            le=LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col])
            # get the associated mapping with feature's value for user's understanding 
            dict_categorical_mapping[col] = dict(zip(le.transform(le.classes_), le.classes_))
    
    # save the dict in a path # TODO the user can choose the path to find it again?
    # pickle_save_model(dict_categorical_mapping, "/work/data/dict_categorical_mapping.pkl")

    return X_encoded, dict_categorical_mapping

def plot_positive_rate(dict_opr_tpr_fpr:dict, fairness_purpose:str, inspected_column:str)->plt:
    # TODO docstring, and add into plots.py module => current functions into experiments.py!
    fig = go.Figure(
        data=[go.Bar(
                name=f"{simplified_model_name},\n ROC-AUC={dict_opr_tpr_fpr[fairness_purpose][simplified_model_name]['roc_auc']}",
                x=dict_opr_tpr_fpr[fairness_purpose][simplified_model_name]["all_groups"].index.astype("str"),
                y=dict_opr_tpr_fpr[fairness_purpose][simplified_model_name]["all_groups"][fairness_purpose],
                text=dict_opr_tpr_fpr[fairness_purpose][simplified_model_name]["all_groups"]['nb_individuals_by_group'],
            )
                for simplified_model_name in ["Baseline","FairDream","GridSearch"]]
    )

    fig.update_layout(
        title=f"New {fairness_purpose} by group of {inspected_column}",
        yaxis_title=f"{fairness_purpose}",
        legend_title_text=f"Models optimized for ROC-AUC and {fairness_purpose}",
    )

    path_gap_plots_col=f"calibration_plots/group_comparison/{inspected_column}"

    if not os.path.isdir(path_gap_plots_col):
        os.makedirs(path_gap_plots_col)

    fig.write_html(f"{path_gap_plots_col}_{fairness_purpose}.html")

    plotly.io.write_image(fig, f"{path_gap_plots_col}_{fairness_purpose}.pdf", format='pdf')

    fig.show()


def get_dfs_gaps_brier_auc(augmented_train_valid_set:pd.DataFrame, inspected_column:str)->Tuple[pd.DataFrame]:
    # For each model trained (FairDream vs GridSearch vs Baseline, *features, *model types), 
    # Returns the gaps between groups treated min-max and the "worst" (max ratio, min calibration)
    # In one DataFrame by metric: calibration (distance area to perfect calibration), # TODO rename Brier -> area with perfect calibration
        # AUC-ROC, AUC-PR, Overall Positive Rate (OPR), False Positive Rate (FPR), True Positive Rate (TPR)

    dict_train_valid_by_group=split_df_into_groups(augmented_train_valid_set=augmented_train_valid_set, inspected_column=inspected_column)

    # TODO unify the results into one dict, dict_gap_metric? Then, only keys would vary...
    dict_gap_auc={}
    dict_gap_auc["roc_auc"]={}
    dict_gap_auc["pr_auc"]={}
    dict_gap_brier={}
    dict_opr_tpr_fpr={}
    for fairness_purpose in ["overall_positive_rate", "false_positive_rate" ,"true_positive_rate"]:
        dict_opr_tpr_fpr[fairness_purpose]={}
    # when several metrics, compute DataFrames into a dict combining the metrics
    dict_df_gap_auc={}
    dict_df_gap_opr_tpr_fpr={}

    # save the fig # TODO unify the function with Brier, and positive rates? 
    path_gap_plots=f"calibration_plots/group_comparison"#/{inspected_column}/{model_name}"

    if os.path.isdir(path_gap_plots):
        shutil.rmtree(path_gap_plots)

    for model_name in ["uncorrected","GridSearch","FairDream"]:#,best_model_dict['model_name']]:

        # to enable plots of multiple experiments, rename the FairDream model into "FairDream" vs "Baseline"
        # TODO add vs "GridSearch"!
        #if model_name == best_model_dict['model_name']:
            #simplified_model_name = "FairDream"
        if model_name == "uncorrected":
            simplified_model_name = "Baseline"
        else:
            simplified_model_name = model_name

        print(f"Calibration gaps between groups for {simplified_model_name} model")
        dict_gap_brier[simplified_model_name] = {}
        dict_gap_brier[simplified_model_name]["max_gap_groups"], dict_gap_brier[simplified_model_name]["worst_group_score"] = get_max_gap_groups_brier(
            model_name=model_name, 
            inspected_column=inspected_column,
            dict_train_valid_by_group=dict_train_valid_by_group) 

        for auc_to_plot in ["roc_auc","pr_auc"]:
            print(f"AUC-{auc_to_plot} gaps between groups for {simplified_model_name} model")
            # dict_gap_auc[auc_to_plot]={}
            dict_gap_auc[auc_to_plot][simplified_model_name] = {}
            dict_gap_auc[auc_to_plot][simplified_model_name]["max_gap_groups"], dict_gap_auc[auc_to_plot][simplified_model_name]["worst_group_score"] = get_max_gap_groups_auc(
                model_name=model_name,
                inspected_column=inspected_column,
                dict_train_valid_by_group=dict_train_valid_by_group,
                auc_to_plot=auc_to_plot)
            
            # dict_df_gap_auc[auc_to_plot]=pd.DataFrame(dict_gap_auc[auc_to_plot]).transpose()

        for fairness_purpose in ["overall_positive_rate", "false_positive_rate" ,"true_positive_rate"]:
            # dict_opr_tpr_fpr[fairness_purpose]={}
            dict_opr_tpr_fpr[fairness_purpose][simplified_model_name] = {}
            (dict_opr_tpr_fpr[fairness_purpose][simplified_model_name]["max_gap_groups"],
                        dict_opr_tpr_fpr[fairness_purpose][simplified_model_name]["worst_group_score"],
                        dict_opr_tpr_fpr[fairness_purpose][simplified_model_name]["all_groups"],
                        dict_opr_tpr_fpr[fairness_purpose][simplified_model_name]["roc_auc"]
            ) = get_max_gap_groups_opr_tpr_fpr(
                augmented_train_valid_set=augmented_train_valid_set, 
                inspected_column=inspected_column,
                model_name=model_name,
                fairness_purpose=fairness_purpose)

    # finally transpose to enable agregation of future model types and protected attributes, 
            # On the basis (rows) of "Baseline" vs "GridSearch" vs "FairDream" rows
    df_gap_brier = pd.DataFrame(dict_gap_brier).transpose()
    for auc_to_plot in ["roc_auc","pr_auc"]:
        dict_df_gap_auc[auc_to_plot]=pd.DataFrame(dict_gap_auc[auc_to_plot]).transpose()
    for fairness_purpose in ["overall_positive_rate", "false_positive_rate" ,"true_positive_rate"]:
        dict_df_gap_opr_tpr_fpr[fairness_purpose]=pd.DataFrame(dict_opr_tpr_fpr[fairness_purpose]).transpose()
        # We then plot the bars of positive rates and save them // In my previous experiments, into a public repository? 
        # TODO graphical part in the plots.py module? 
        plot_positive_rate(dict_opr_tpr_fpr=dict_opr_tpr_fpr, 
                           fairness_purpose=fairness_purpose, 
                           inspected_column=inspected_column)

    return df_gap_brier, dict_df_gap_auc["roc_auc"], dict_df_gap_auc["pr_auc"], dict_df_gap_opr_tpr_fpr["overall_positive_rate"], dict_df_gap_opr_tpr_fpr["false_positive_rate"], dict_df_gap_opr_tpr_fpr["true_positive_rate"]
