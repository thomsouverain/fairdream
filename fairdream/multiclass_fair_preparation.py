from typing import Union

import numpy as np
import pandas as pd


def get_indivs_by_label_dict(multi_Y_pred_train_valid: np.ndarray) -> dict:
    """Returns a dictionary of {label: nb_individuals} predicted by the inspected model.
    Will be used to get the label corresponding to the 'distribution frontier' (Q1, median, or Q3) between individuals chosen by the user,

    Parameters
    ----------
    multi_Y_pred_train_valid : np.ndarray, of shape (nb_individuals, nb_labels)
        Vector with the predicted label for all individual (corresponding to the argmax of all labels probabilities)
        Ex: array(1,0,3,...,1)

    Returns
    -------
    dict
        Dictionary of {label: nb_individuals} predicted by the inspected model
    """

    indivs_by_label_dict = {}
    pred_labels_counts = np.unique(multi_Y_pred_train_valid, return_counts=True)

    for label, count in zip(pred_labels_counts[0], pred_labels_counts[1]):
        indivs_by_label_dict[label] = count

    return indivs_by_label_dict


def get_frontier_label(
    multi_Y_pred_train_valid: np.ndarray,
    sorted_labels_list: list,
    distribution_frontier: str = "median",
) -> Union[str, int]:
    """Returns the label corresponding to the user's 'distribution_frontier' between individuals, and according to labels sorted by the user.

    Parameters
    ----------
    multi_Y_pred_train_valid : np.ndarray, of shape (nb_individuals, nb_labels)
        Vector with the predicted label for all individual (corresponding to the argmax of all labels probabilities)
        Ex: array(1,0,3,...,1)
    sorted_labels_list : list
        List of labels with the desired ascending ranking of the user.
        Ex: when labels are number of housing nights and the user wants to maximise it,
        sorted_labels_list = [0,1,2,3]
    distribution_frontier : str
        The user chooses which % of individuals (ranked by ascending labels) will be considered as 'privileged' (1) or not (0).
        Must be set to values in {'Q1','median','Q3'}:
            'Q1': 25% of (0), 75% of "privileged" (1). Recommended if you want to intervene for the few worse-off individuals.
            'median': 50% (0), 50% (1), default choice
            'Q3': 75% of (0), 25% of "privileged" (1). Recommended if you want to intervene against the few better-off individuals.

    Returns
    -------
    Union[str, int]
        The label corresponding to the 'distribution_frontier', depending on its type: "blue" (str), 1 night (int)...
    """
    indivs_by_label_dict = get_indivs_by_label_dict(multi_Y_pred_train_valid)

    total_indivs_nb = sum(indivs_by_label_dict.values())
    indivs_summed_by_label_nb = 0

    if distribution_frontier == "Q1":
        frontier_indiv_nb = int(total_indivs_nb * 0.25)
    elif distribution_frontier == "median":
        frontier_indiv_nb = int(total_indivs_nb * 0.5)
    elif distribution_frontier == "Q3":
        frontier_indiv_nb = int(total_indivs_nb * 0.75)
    else:
        raise ValueError(
            f"distribution_frontier {distribution_frontier} is not implemented. Must must be set to values in {'Q1','median','Q3'}."
        )

    for sorted_label in sorted_labels_list:

        indivs_summed_by_label_nb = indivs_summed_by_label_nb + indivs_by_label_dict[sorted_label]

        if indivs_summed_by_label_nb >= frontier_indiv_nb:
            frontier_label = sorted_label

            return frontier_label


def compute_binary_Y(
    multi_Y_pred_train_valid: np.ndarray,
    sorted_labels_list: list,
    frontier_label: Union[str, int],
) -> np.ndarray:
    """From multi_Y_train_valid, computes a binary Y_train_valid (1 if label is ranked by the user > frontier_label, else 0)

    Parameters
    ----------
    multi_Y_pred_train_valid : np.ndarray, of shape (nb_individuals, nb_labels)
        Vector with the predicted label for all individual (corresponding to the argmax of all labels probabilities)
        Ex: array(1,0,3,...,1)
    sorted_labels_list : list
        List of labels with the desired ascending ranking of the user.
        Ex: when labels are number of housing nights and the user wants to maximise it,
        sorted_labels_list = [0,1,2,3]
    frontier_label : Union[str, int]
        The label corresponding to the 'distribution_frontier', depending on its type: "blue" (str), 1 night (int)...
        Must be set to values in np.unique(multi_Y_pred_train_valid)
        Ex: must be set to existing label values, in {0, 1, 2, 3} number of nights

    Returns
    -------
    np.ndarray
    """
    set_preds = set(np.unique(multi_Y_pred_train_valid)) 
    set_labels = set(sorted_labels_list)

    if set_preds.issubset(set_labels)==False:
        raise ValueError(
            f"The sorted_labels_list with {set_labels} labels you manually set does not match the correct writing of all the target labels. \n"
            f"Must contain all the values in {set_preds} \n"
            "Ex: if label values = {0, 1, 2, 3} number of nights, sorted_labels_list = [3,2,0,1] or sorted_labels_list = [0,1,2,3] would be valid"
        )

    binary_labels_dict = {}

    frontier_position = sorted_labels_list.index(frontier_label)

    for label in sorted_labels_list:

        label_position = sorted_labels_list.index(label)

        if label_position <= frontier_position:
            binary_label = 0
        elif label_position > frontier_position:
            binary_label = 1

        binary_labels_dict[label] = binary_label

    # Then create a np.ndarray with the new binary label (> or < frontier_label) for all individual
    binary_Y_train_valid = np.array(pd.Series(multi_Y_pred_train_valid).map(binary_labels_dict))

    return binary_Y_train_valid


def multi_to_binary_Y_pred(
    multi_Y_train_valid: np.ndarray,
    multi_Y_pred_train_valid: np.ndarray,
    sorted_labels_list: list,
    distribution_frontier: str = None,
    frontier_label: Union[str, int] = None,
) -> tuple():
    """Transforms multiclass into binary, according to the user's 'distribution_frontier' between individuals, and to labels sorted by the user.
    Returns a tuple with 2 vectors (target, predicted) on Y_train_valid.

    Parameters
    ----------
    multi_Y_train_valid : pd.DataFrame
        Target, ie true multi-labels of Y on train_valid set.
        Ex: array(2,3,3,...,1)
    multi_Y_pred_train_valid : np.ndarray, of shape (nb_individuals, nb_labels)
        Vector with the predicted label for all individual (corresponding to the argmax of all labels probabilities)
        Ex: array(1,0,3,...,1)
    distribution_frontier : str, by default None
        The user chooses which % of individuals (ranked by ascending labels) will be considered as 'privileged' (1) or not (0).
        Must be set to values in {'Q1','median','Q3'}:
            'Q1': 25% of (0), 75% of "privileged" (1). Recommended if you want to intervene for the few worse-off individuals.
            'median': 50% (0), 50% (1)
            'Q3': 75% of (0), 25% of "privileged" (1). Recommended if you want to intervene against the few better-off individuals.
    frontier_label : Union[str, int], by default None
        Here, the user can set manually
        the label corresponding to the 'distribution_frontier', depending on its type: "blue" (str), 1 night (int)...
        Must be set to values in np.unique(multi_Y_pred_train_valid)
        Ex: must be set to existing label values, in {0, 1, 2, 3} number of nights

    Returns
    -------
    Tuple(np.ndarray)
        Tuple with 2 vectors (target, predicted) on Y_train_valid
        (binary_Y_train_valid, binary_Y_pred_train_valid)
    """
    # compute median label with the uncorrected model's predictions
    if frontier_label is None:
        frontier_label = get_frontier_label(
            multi_Y_pred_train_valid, sorted_labels_list, distribution_frontier
        )

        # clarification message: new binary frontier
        if distribution_frontier == "Q1":
            print(
                f"25% of individuals are > {frontier_label} :\n"
                "Recommended if you want to intervene for the few worse-off individuals\n ''' \n"
                f"For fairness detection, people <= {frontier_label} will be considered in class 0 and others in class 1"
            )
        elif distribution_frontier == "median":
            print(
                f"50% of individuals are > {frontier_label} :\n"
                f"For fairness detection, people <= {frontier_label} will be considered in class 0 and others in class 1"
            )
        elif distribution_frontier == "Q3":
            print(
                f"75% of individuals are > {frontier_label} :\n"
                "Recommended if you want to intervene against the few better-off individuals\n ''' \n"
                f"For fairness detection, people <= {frontier_label} will be considered in class 0 and others in class 1"
            )
        else:
            raise ValueError(
                f"distribution_frontier {distribution_frontier} is not implemented. Must be set to values in {'Q1','median','Q3'}."
            )

    elif frontier_label not in np.unique(multi_Y_pred_train_valid):
        raise ValueError(
            f"The frontier_label {frontier_label} you manually set does not match the correct writing of a target label. \n"
            f"Must be set to a value in {np.unique(multi_Y_pred_train_valid)} \n"
            "Ex: if label values = {0, 1, 2, 3} number of nights, frontier_label = 2 or sorted_labels_list = 1 would be valid"
        )

    # then, transform vectors wich labels are > or < frontier_label into binary vectors

    binary_Y_train_valid = compute_binary_Y(multi_Y_train_valid, sorted_labels_list, frontier_label)
    binary_Y_pred_train_valid = compute_binary_Y(
        multi_Y_pred_train_valid, sorted_labels_list, frontier_label
    )

    return binary_Y_train_valid, binary_Y_pred_train_valid
