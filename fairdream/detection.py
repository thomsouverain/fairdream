import os
import random
from collections import defaultdict
from typing import Set

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from fairdream.compute_scores import fair_score
from fairdream.compute_scores import is_fairness_purpose_to_maximise
from fairdream.data_preparation import select_important_features
from fairdream.plots import plot_fair_scores

# prepare functions to compute perfs (fair and stat), and then trade_off_score of models
# illustration : "how" did we get it? Distorsion of sample weights...


def discrimination_alert(
    augmented_train_valid_set: pd.DataFrame,
    model_name: str,
    fairness_purpose: str,
    model_task: str,
    injustice_acceptance: int = 3,
    min_individuals_disadvantaged: float = 0.03,
    uncorrected_model_path: str = None,
) -> Set[str]:
    """Alerts if groups of the train_valid_set columns are disadvantaged,
    i.e. discrimination alert if the fair_score of a group in inspected_column < injustice_acceptance (set by the user) * mean fair_score.

    Also print alerts by groups disadvantaged, with their fair_scores gap to the mean and plots illustrating the gap.

    You should remember that discriminations are detected on TRAIN & VALID set:
    The goal is, to identify if the model became discriminant by learning, to correct it more efficiently.

    The model is considered as discriminant if it selects in average really more (depending on user's sensitivity to injustice) individuals
    than in a disadvantaged group as positive (% or nb), positive that should not be (false positive),
    negative that should be (true negative), or regarding fscore.

    Parameters
    ----------
    augmented_train_valid_set : pd.DataFrame
        The train_valid_set augmented with model's prediction, from which columns are divided into groups of individuals *
        and inspected to detect gaps of fair_scores.
    model_name : str
        Name of the model whose results are inspected.
        Must be set to value in {'uncorrected', 'fair_1', ..., 'fair_n'}
        depending on the number (n) of fairer models in competition (n == grid_size fixed by the user)
    fairness_purpose : str
        Metrics by which the user wants to measure the gap between groups : basis of evaluation to compute the fair_score.
        Will serve in detection: detects inequalities of selection by the model (between subgroups and the mean) regarding fairness_purpose
        Will serve in correction: selects the best model satisfying stat/fair tradeoff regarding fairness_purpose
        -> For classification (binary or multi-classes), must be set to value in 
        {"overall_positive_rate", "nb_positive", "false_positive_rate", "false_negative_rate", "true_positive_rate", "true_negative_rate}
        -> For regression, must be set to a value in {'distribution_gap','mean_squared_error', 'mean_absolute_percentage_error', 'r2_score'}
    model_task: str
        Goal the user wants to achieve with the model: either classify, or regress...
        Must be set to value in {"regression", "classification", "multiclass"}
    injustice_acceptance : int, optional
        High injustice acceptance means that user wants AI to detect high inequalities between groups before alerting
        Medium by default (3), low tolerance to injustice in range(1,3) and higher injustice acceptance >= 4    min_individuals_disadvantaged : float, optional
    min_individuals_disadvantaged : float, optional
        Minimal % of the population concertned to launch discrimination alerts, by default 0.03
    uncorrected_model_path : str, by default None
        Where the uncorrected_model is saved (by default, "/work/data/models/uncorrected_model.pkl")

    Returns
    -------
    Set[str]
        Set of columns of augmented_train_valid_set in which subgroups are disadvantage (according to users' preferences)
    """

    is_to_maximise = is_fairness_purpose_to_maximise(fairness_purpose)

    set_alerts = set()

    # inspect if the model discriminates only on the features it really impacts (i.e. SHAP influence on output > 1%)
    # before, ensure the matrix passed for features' importance only contains label-encoded values 
    d = defaultdict(LabelEncoder)
    label_encoded_augmented_train_valid_set = augmented_train_valid_set.apply(lambda x: d[x.name].fit_transform(x))
    # important_features_list = select_important_features(
    #     label_encoded_augmented_train_valid_set, model_name, model_task, uncorrected_model_path
    # )

    # for the moment, do not sort important features for neural networks - but TODO 
    features_list = list(
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

    important_features_list = features_list

    predicted_column = f"predicted_{model_name}"

    print(f"fairness_purpose : {fairness_purpose}\n")
    print(f"predicted_column : {predicted_column}")

    for inspected_column in important_features_list:

        fair_scores_df = fair_score(
            augmented_train_valid_set,
            model_name,
            fairness_purpose,
            model_task,
            inspected_column,
        )

        global_selection_rate = fair_scores_df["mean"].iloc[0]
        for inspected_column_selection_rate, nb_individuals_by_group in zip(
            fair_scores_df[fairness_purpose], fair_scores_df["nb_individuals_by_group"]
        ):

            discriminant_interval = fair_scores_df.index[
                fair_scores_df[fairness_purpose] == inspected_column_selection_rate
            ].tolist()

            # conditions to launch alerts:
            # only if a minimal number of individuals are disadvantage
            if (
                nb_individuals_by_group
                > min_individuals_disadvantaged * augmented_train_valid_set.shape[0]
            ):

                # if fairness_purpose is an objective to maximise, alert if mean selection >= group selection
                if is_to_maximise==True:

                    if inspected_column_selection_rate == 0:

                        print(
                            f"inspected_column: '{inspected_column}' \n"
                            f"--Discrimination Alert--"
                            f"\n the group of {nb_individuals_by_group} persons is disadvantaged"
                            f"\n for {discriminant_interval} "
                            f"0 persons of the group are selected by the model"
                        )

                        plot_fair_scores(fair_scores_df, inspected_column, fairness_purpose)

                    elif (
                        global_selection_rate
                        >= injustice_acceptance * inspected_column_selection_rate
                    ):

                        print(
                            f"inspected_column: '{inspected_column}' \n"
                            f"--Discrimination Alert--"
                            f"\n the group of {nb_individuals_by_group} persons is disadvantaged"
                            f"\n for {discriminant_interval} "
                            f"mean {fairness_purpose} = {round(global_selection_rate/inspected_column_selection_rate,1)} * group {fairness_purpose} \n"
                        )

                        plot_fair_scores(fair_scores_df, inspected_column, fairness_purpose)

                        set_alerts.add(inspected_column)

                # if fairness_purpose is an objective to minimise (an error), alert if group error >= mean error
                elif is_to_maximise==False:

                    if (
                        inspected_column_selection_rate
                        >= injustice_acceptance * global_selection_rate
                    ):

                        print(
                            f"inspected_column: '{inspected_column}' \n"
                            f"--Discrimination Alert--"
                            f"\n the group of {nb_individuals_by_group} persons is disadvantaged"
                            f"\n for {discriminant_interval} "
                            f"group {fairness_purpose} = {round(inspected_column_selection_rate/global_selection_rate,1)} * mean {fairness_purpose} \n"
                        )

                        plot_fair_scores(fair_scores_df, inspected_column, fairness_purpose)

                        set_alerts.add(inspected_column)

                else:
                    raise ValueError(
                        "-> For classification, fairness_purpose must be set to a value in {'fscore', 'false_positive_rate', 'true_negative_rate','overall_positive_rate','nb_positive'}\n"
                        "-> For regression, fairness_purpose must be set to a value in {'distribution_gap', 'mean_absolute_percentage_error', 'r2_score'}"
                    )

    return set_alerts
