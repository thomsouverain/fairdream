import os
import random
from typing import Dict
from typing import List

import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot
from matplotlib.axes import Axes

from fairdream.compute_scores import fair_score

def plot_fair_scores(
    fair_scores_df: pd.DataFrame, inspected_column: str, fairness_purpose: str
) -> px:
    """For all group of the inspected column, plot bars with % and nb individuals disadvantaged when alerts are raised,
    depending on the user's fairness_purpose.

    Parameters
    ----------
    fair_scores_df : pd.DataFrame
        For a specific model, contains its fair scores by groups of the inspected_column.
        DataFrame with inspected_column groups (e.g. different ages), fair_score by group, nb_indivs by group, and mean fair_score.
    inspected_column : str
        Name of the column of augmented_train_valid_set to be inspected
    fairness_purpose : str
        Metrics by which the user wants to measure the gap between groups : basis of evaluation to compute the fair_score.
        -> For classification (binary or multi-classes), must be set to value in 
        {"overall_positive_rate", "nb_positive", "false_positive_rate", "false_negative_rate", "true_positive_rate", "true_negative_rate}
        -> For regression, must be set to a value in {'distribution_gap','mean_squared_error', 'mean_absolute_percentage_error', 'r2_score'}

    Returns
    -------
    plotly.express
    """
    fig = px.bar(
        fair_scores_df,
        x=fair_scores_df.index.astype("str"),
        y=fair_scores_df[fairness_purpose],
        text=fair_scores_df["nb_individuals_by_group"].astype("str"),
        width=600,
        height=400,
        title=f"% individuals disadvantaged for {inspected_column}:",
        labels={"x": "Groups", "text": "nb individuals"},
    )

    fig.update_layout(yaxis_title=f"{fairness_purpose} %")

    fig.show()


def plot_all_scores(models_df: pd.DataFrame) -> plt:
    """Plots the tradeoff between stat score (x) and fair score (y) of all models on a scatter,
    showing the initial "uncorrected" and the selected "best" model (according to the user's preferences).

    Parameters
    ----------
    models_df : pd.DataFrame
        For each line (i.e. model), DataFrame with columns = ['model_name', 'fair_score_value', 'stat_score_value', 'tradeoff_score',
        'fair_scores_df', 'selected']
        Will serve to select the best model according to the user's preferences, and then plot it to compare its fair/stat score with the initial "uncorrected" model

    Returns
    -------
    plt
    """
    df = models_df.copy()

    for color, label in zip("brg", ["Not selected", "Baseline", "Best model"]):
        subset = df[df["selected"] == label]
        fig_scores = plt.scatter(
            subset["stat_score_value"], subset["fair_score_value"], s=50, c=color, label=str(label)
        )
        fig_scores = plt.legend()

    fig_scores = pyplot.title("Stat and fair scores for models")
    fig_scores = pyplot.xlabel("Stat score")
    fig_scores = pyplot.ylabel("Fair score")

    uncorrected_model_line_df = models_df.loc[models_df["selected"] == "Baseline"]
    best_model_line_df = models_df.loc[models_df["selected"] == "Best model"]

    plt.show()


def plot_best_uncorrected_fair_scores(
    best_model_dict: dict, inspected_column: str, fairness_purpose: str
) -> go:
    """Plots bar charts with mean (line), before ("uncorrected") and after (best_model) fair_scores by groups.
    Parameters
    ----------

    best_model_dict: dict
        Dictionary containing the best models scores (stat&fair),
        the model to be re-used for better predictions,
        and a DataFrame with scores of the initial "uncorrected" model to be challenged ('fair_scores_df_uncorrected')
        best_model_dict.columns = ['model_name', 'fair_score_value', 'stat_score_value', 'tradeoff_score', 'fair_scores_df', 'model', 'fair_scores_df_uncorrected']
    inspected_column : str
        Name of the column of augmented_train_valid_set to be inspected, i.e. from which groups and their fair_scores are extracted
        (protected attribute in correction phase)
    fairness_purpose : str
        Metrics by which the user wants to measure the gap between groups : basis of evaluation to compute the fair_score.
        Here after correction, shows the best model satisfying stat/fair tradeoff regarding fairness_prupose.
        -> For classification (binary or multi-classes), must be set to value in 
        {"overall_positive_rate", "nb_positive", "false_positive_rate", "false_negative_rate", "true_positive_rate", "true_negative_rate}
        -> For regression, must be set to a value in {'distribution_gap','mean_squared_error', 'mean_absolute_percentage_error', 'r2_score'}
    Returns
    -------
    graph_objects

    """
    fair_scores_df_uncorrected = best_model_dict["fair_scores_df_uncorrected"]
    fair_scores_df = best_model_dict["fair_scores_df"]

    fig = go.Figure(
        data=[
            go.Bar(
                name="uncorrected",
                x=fair_scores_df_uncorrected.index.astype("str"),
                y=fair_scores_df_uncorrected[fairness_purpose],
            ),
            # text=fair_scores_df_uncorrected["nb_individuals_by_group"].astype("str")), # TODO to print nb indivs concerned?
            go.Bar(
                name=f"{best_model_dict['model_name']}",
                x=fair_scores_df.index.astype("str"),
                y=fair_scores_df[fairness_purpose],
            ),
            # text=fair_scores_df["nb_individuals_by_group"].astype("str")) # TODO to print nb indivs concerned?
        ]
    )

    fig.add_traces(
        [
            go.Scatter(
                name="mean",
                visible="legendonly",
                x=fair_scores_df.index.astype("str"),
                y=fair_scores_df["mean"],
            )
        ]
    )

    fig.update_layout(
        title=f"New {fairness_purpose} by group of {inspected_column}",
        yaxis_title=f"{fairness_purpose} %",
    )

    fig.show()


def individual_results(
    train_valid_set_with_corrected_results: pd.DataFrame, best_model_dict: dict, model_task: str
) -> pd.DataFrame:
    """Returns a selection of individuals of the train_valid_set that were disadvantaged
    before (with "uncorrected" model), and are now more (with "selected" best model)

    Parameters
    ----------
    train_valid_set_with_corrected_results : pd.DataFrame
        X_train_valid augmented with probabilities, predicted labels, (true/false) (positive/negative)
        for all model ("uncorrected", "fair_1", ..., "fair_n", with (n) = grid_size)
        Will serve to attribute to each individual one's new fair score,
        to inspect if previously disadvantaged individuals (with "uncorrected" model) are now disadvantaged
    best_model_dict : dict
        Dictionary containing the best models scores (stat&fair),
        the model to be re-used for better predictions,
        and a DataFrame with scores of the initial "uncorrected" model to be challenged ('fair_scores_df_uncorrected')
        best_model_dict.columns = ['model_name', 'fair_score_value', 'stat_score_value', 'tradeoff_score', 'fair_scores_df', 'model', 'fair_scores_df_uncorrected']
    model_task : str
        Goal the user wants to achieve with the model: either classify, or regress...
        Must be set to value in {"regression", "classification", "multiclass"}

    Returns
    -------
    pd.DataFrame
        Individuals that were unfair disadvantaged, and are no more

    """
    if model_task in {"classification", "multiclass"}:
        # individuals before misclassified (Y^_uncorrected ==O), now well classified (Y^_best_model ==Y==1)
        new_fair_treated_indivs = train_valid_set_with_corrected_results.loc[
            (train_valid_set_with_corrected_results["target_train_valid"] == 1)
            & (train_valid_set_with_corrected_results["predicted_uncorrected"] == 0)
            & (
                train_valid_set_with_corrected_results[f"predicted_{best_model_dict['model_name']}"]
                == 1
            )
        ]

    elif model_task == "regression":
        # individuals before under-valued, now closer to their real target
        # |Y - Y^_best_model| < |Y - Y^_uncorrected|
        train_valid_set_with_corrected_results["abs_gap_uncorrected"] = np.absolute(
            train_valid_set_with_corrected_results["target_train_valid"]
            - train_valid_set_with_corrected_results["predicted_uncorrected"]
        )

        train_valid_set_with_corrected_results["abs_gap_best_model"] = np.absolute(
            train_valid_set_with_corrected_results[f"predicted_{best_model_dict['model_name']}"]
            - train_valid_set_with_corrected_results["predicted_uncorrected"]
        )

        new_fair_treated_indivs = train_valid_set_with_corrected_results.loc[
            train_valid_set_with_corrected_results["abs_gap_best_model"]
            < train_valid_set_with_corrected_results["abs_gap_uncorrected"]
        ]

    else:
        raise NotImplementedError(
            f"The {model_task} task you want the model to perform is not implemented. Must be set to value in {'regression','classification','multiclass'}"
        )

    # TODO plot previous and new fair results of inspected_column for individual results?

    return new_fair_treated_indivs


def fair_model_results(
    train_valid_set_with_corrected_results: pd.DataFrame,
    models_df: pd.DataFrame,
    best_model_dict: dict,
    protected_attribute: str,
    fairness_purpose: str,
    model_task: str,
) -> pd.DataFrame:
    """Summary of results comparing the previous "uncorrected" model and the new "selected" best model:

        -> Scatter plot of uncorrected and fairer models, showing the stat/fair tradeoff and the selection of the best model
        -> Plots bar charts with mean (line), before ("uncorrected") and after (best_model) fair_scores by groups
        -> Selection of individuals of the train_valid_set that were disadvantaged
        before (with "uncorrected" model), and are now more (with "selected" best model)

    Parameters
    ----------
    train_valid_set_with_corrected_results : pd.DataFrame
        X_train_valid augmented with probabilities, predicted labels, (true/false) (positive/negative)
        for all model ("uncorrected", "fair_1", ..., "fair_n", with (n) = grid_size)
        Will serve to attribute to each individual one's new fair score,
        to inspect if previously disadvantaged individuals (with "uncorrected" model) are now disadvantaged
    models_df : pd.DataFrame
        For each line (i.e. model), DataFrame with columns = ['model_name', 'fair_score_value', 'stat_score_value', 'tradeoff_score',
        'fair_scores_df', 'selected']
        Will serve to select the best model according to the user's preferences, and then plot it to compare its fair/stat score with the initial "uncorrected" model
    best_model_dict : dict
        Dictionary containing the best models scores (stat&fair),
        the model to be re-used for better predictions,
        and a DataFrame with scores of the initial "uncorrected" model to be challenged ('fair_scores_df_uncorrected')
        best_model_dict.columns = ['model_name', 'fair_score_value', 'stat_score_value', 'tradeoff_score', 'fair_scores_df', 'model', 'fair_scores_df_uncorrected']
    protected_attribute : str
        In correction phase,
        name of the column of augmented_train_valid_set to be inspected, i.e. from which groups and their fair_scores are extracted
    fairness_purpose : str
        Metrics by which the user wants to measure the gap between groups : basis of evaluation to compute the fair_score.
        Will serve in detection: detects inequalities of selection by the model (between subgroups and the mean) regarding fairness_purpose
        Will serve in correction: selects the best model satisfying stat/fair tradeoff regarding fairness_purpose
        -> For classification (binary or multi-classes), must be set to value in 
        {"overall_positive_rate", "nb_positive", "false_positive_rate", "false_negative_rate", "true_positive_rate", "true_negative_rate}
        -> For regression, must be set to a value in {'distribution_gap','mean_squared_error', 'mean_absolute_percentage_error', 'r2_score'}
    model_task : str
        Goal the user wants to achieve with the model: either classify, or regress...
        Must be set to value in {"regression", "classification","multiclass"}

    Returns
    -------
    pd.DataFrame
        Returns a selection of individuals of the train_valid_set that were disadvantaged
        before (with "uncorrected" model), and are now more (with "selected" best model)
    """
    print(
        f"\n--- Results of fair train with fairness purpose {fairness_purpose} to protect {protected_attribute} --- \n"
    )

    print(
        f"\n Following your fair / stat tradeoff, {best_model_dict['model_name']} was selected as best model \n"
    )
    plot_all_scores(models_df)

    print(
        f"\n Differences of {fairness_purpose} before and now : groups of {protected_attribute} are more fairly treated \n"
    )
    plot_best_uncorrected_fair_scores(best_model_dict, protected_attribute, fairness_purpose)

    # TODO add a selection of individuals considering their new re-evaluated fscore (not random...)
    print(f"\n Individual differences : individuals that were disadvantaged are now integrated \n")
    new_fair_treated_indivs = individual_results(
        train_valid_set_with_corrected_results, best_model_dict, model_task
    )
    # Here we take an example of individual with maximal value of the protected column (e.g. old with "age", or "1" when sex = Female)
    # To select and show individuals whose prediction was unfair and changed due to fairdream
    # (but we could have taken examples on other values, it is just to select specific individuals whose prediction changed)
    max_inspected_column_value = new_fair_treated_indivs[protected_attribute].min()#.max()

    max_inspected_column_indiv = new_fair_treated_indivs.loc[
        new_fair_treated_indivs[protected_attribute] == max_inspected_column_value
    ]

    max_inspected_column_indiv = max_inspected_column_indiv.head()

    if model_task in "classification":
        print(
            f"\n Example : an individual of {protected_attribute} {max_inspected_column_value} is now integrated \n"
        )

    elif model_task in "multiclass":
        print(
            f"\n Example : an individual of {protected_attribute} {max_inspected_column_value} is now > median value \n"
        )

    elif model_task == "regression":
        print(
            f"\n Example : an individual of {protected_attribute} {max_inspected_column_value} is now closer to one's real target \n"
        )

    else:
        raise NotImplementedError(
            f"The {model_task} task you want the model to perform is not implemented. Must be set to value in {'regression','classification','multiclass'}"
        )

    # print(f"{max_inspected_column_indiv}")
    return max_inspected_column_indiv

def plot_compared_metrics(
                            train_valid_set_with_corrected_results:pd.DataFrame, 
                            protected_attribute: List[str],
                            fairness_purpose: str,
                            model_task: str,
                            best_model_dict:dict,
                            list_compared_metrics:List[str],
                            )->go:
    """For the new model selected by FairDream, plot the comparison baseline / FairDream model 
    for a list of selected fairness_purposes (list_compared_metrics).

    Args:

    train_valid_set_with_corrected_results : pd.DataFrame
        X_train_valid augmented with probabilities, predicted labels, (true/false) (positive/negative)
        for all model ("uncorrected", "fair_1", ..., "fair_n", with (n) = grid_size)
        Will serve to attribute to each individual one's new fair score,
        to inspect if previously disadvantaged individuals (with "uncorrected" model) are now disadvantaged
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
    best_model_dict : dict
        Dictionary containing the best models scores (stat&fair),
        the model to be re-used for better predictions,
        and a DataFrame with scores of the initial "uncorrected" model to be challenged ('fair_scores_df_uncorrected')
        best_model_dict.columns = ['model_name', 'fair_score_value', 'stat_score_value', 'tradeoff_score', 'fair_scores_df', 'model', 'fair_scores_df_uncorrected']
    list_compared_metrics : List[str]
        List of the fairness metrics to plot the difference baseline / FairDream model.
        -> For classification (binary or multi-classes), each fairness metric must be set to value in 
        {"overall_positive_rate", "nb_positive", "false_positive_rate", "false_negative_rate", "true_positive_rate", "true_negative_rate}
        -> For regression, each fairness metric must be set to a value in {'distribution_gap','mean_squared_error', 'mean_absolute_percentage_error', 'r2_score'}
    Returns:
        go: bar plots comparing baseline / FairDream model for each specified fairness metric
    """

    model_name = best_model_dict['model_name']

    print(f"Comparison of fairness metrics between the baseline uncorrected / new model {model_name} optimised for {fairness_purpose}")

    for fairness_comparison_metric in list_compared_metrics:

        check_valid_fairness_purpose(fairness_purpose=fairness_comparison_metric, model_task=model_task)

        fair_scores_df_uncorrected = fair_score(
            train_valid_set_with_corrected_results,
            'uncorrected',
            fairness_comparison_metric,
            model_task,
            protected_attribute,
            fairness_mode="correction",
        )


        fair_scores_df = fair_score(
            train_valid_set_with_corrected_results,
            model_name,
            fairness_comparison_metric,
            model_task,
            protected_attribute,
            fairness_mode="correction",
        )

        best_model_comparison_metric_dict = {'model_name':model_name, 
                                             'fair_scores_df_uncorrected':fair_scores_df_uncorrected, 
                                             'fair_scores_df':fair_scores_df}

        plot_best_uncorrected_fair_scores(
            best_model_dict=best_model_comparison_metric_dict, 
            inspected_column=protected_attribute, 
            fairness_purpose=fairness_comparison_metric)

def check_valid_fairness_purpose(fairness_purpose:str, model_task:str):
    """Raise NotImplementedError if the specified fairness_purpose is not implemented in FairDream, according to the model's task (classification, multiclass, regression).

    Args:
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

    """
    classification_metrics = {'fscore', 'false_positive_rate', 'false_negative_rate','true_positive_rate','true_negative_rate','overall_positive_rate','nb_positive'}
    regression_metrics = {'distribution_gap', 'mean_absolute_percentage_error', 'r2_score'}
    
    if model_task in {'classification', 'multiclass'} and fairness_purpose not in classification_metrics:
        raise NotImplementedError(
            f"for classification, fairness_purpose must be set to a value in {classification_metrics}"
            )

    elif model_task in {'regression'} and fairness_purpose not in regression_metrics:
        raise NotImplementedError(
            f"For regression, fairness_purpose must be set to a value in {regression_metrics}"
        )