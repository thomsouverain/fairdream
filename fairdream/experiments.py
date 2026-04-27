import os
import shutil
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot
from matplotlib.axes import Axes
from sklearn.datasets import fetch_openml

from fairdream.compute_scores import fair_score
from fairdream.compute_scores import get_dfs_gaps_brier_auc
from fairdream.compute_scores import stat_score
from fairdream.correction import fair_train
from fairdream.data_preparation import augment_train_valid_set_with_results
from fairdream.data_preparation import train_naive_model
from fairdream.data_preparation import train_naive_xgb
from fairdream.data_preparation import train_valid_test_split
from fairdream.detection import discrimination_alert
#import plotly as plt

def baseline_training_and_detection(
                            X: pd.DataFrame,
                            Y: pd.DataFrame):
    # we fix the stat_criterion
    stat_criteria = "auc"
    model_task = "classification"

    X_train, X_valid, X_train_valid, X_test, Y_train, Y_valid, Y_train_valid, Y_test = train_valid_test_split(X,Y, model_task)

    # save the uncorrected model, to then sort its features by importances
    save_model=True
    Y_pred_train_valid, probas_pred_class_1_train_valid = train_naive_xgb(X_train, X_valid, X_train_valid, X_test, Y_train, Y_valid, Y_train_valid, Y_test, model_task, stat_criteria, save_model=save_model)

    train_valid_set_with_uncorrected_results = augment_train_valid_set_with_results("uncorrected", X_train_valid, Y_train_valid, Y_pred_train_valid, probas_pred_class_1_train_valid, model_task)

    augmented_train_valid_set = train_valid_set_with_uncorrected_results
    model_name = "uncorrected"

    dict_set_disadvantaging_features = {}
    list_fairness_purposes = ['overall_positive_rate', 'true_positive_rate', 'false_positive_rate']

    for fairness_purpose in list_fairness_purposes:

        # we fix a minimum for injustice_acceptance
        injustice_acceptance=3
        min_individuals_discriminated=0.01

        dict_set_disadvantaging_features[fairness_purpose] = discrimination_alert(augmented_train_valid_set,
                                                                                  model_name, 
                                                                                  fairness_purpose, 
                                                                                  model_task, 
                                                                                  injustice_acceptance, 
                                                                                  min_individuals_discriminated)
    return dict_set_disadvantaging_features

def compare_fairness_methods(
                            X: pd.DataFrame,
                            Y: pd.DataFrame,    
                            train_valid_set_with_uncorrected_results:pd.DataFrame, 
                            list_protected_attributes: List[str],
                            model_type: str,
                            )->go:
    """On several protected attributes (provided by the user), launches GridSearch (vs) Weighted Groups models. 
    Compares their results according to fairness metrics: unconditional to true labels (e.g. overall_positive_rate equality, i.e. demographic parity)
    (vs) conditional to true labels (e.g. true positive rate and false positive rates equality, i.e. equalized odds). 
    """
    # we fix the stat_criterion
    stat_criteria = "auc"
    model_task = "classification"
    # in this experiment, we fix the desired balance between stat and fair performances, and numbers of models in competition to achieve fairer results 
    tradeoff = "fair_preferred"
    nb_fair_models = 10

    list_fairness_purposes = ['overall_positive_rate', 'true_positive_rate', 'false_positive_rate']

    # begin with erasing the previous plots to avoid confusion (if a previous experiment was launched)
    if os.path.isdir("comparison_plots"):
        shutil.rmtree("comparison_plots")

    for protected_attribute in list_protected_attributes:

        print(f"\n \n ''' FEATURE for correction == {protected_attribute} ''' ")

        for fairness_purpose in list_fairness_purposes:

            print(f"Fairness purpose to optimize == {fairness_purpose}")

            dict_fair_scores_df = {}
            dict_fair_scores_df['uncorrected'] = {} 
            dict_fair_scores_df['weighted_groups'] = {}
            dict_fair_scores_df['grid_search'] = {}

            for weight_method in ['grid_search', 'weighted_groups']:

                train_valid_set_with_corrected_results, models_df, best_model_dict = fair_train(
                    X=X,
                    Y=Y,
                    train_valid_set_with_uncorrected_results=train_valid_set_with_uncorrected_results,
                    protected_attribute=protected_attribute,
                    fairness_purpose=fairness_purpose,
                    model_task=model_task,
                    stat_criteria=stat_criteria,
                    tradeoff=tradeoff,
                    weight_method=weight_method,
                    nb_fair_models=nb_fair_models,
                    model_type=model_type,
                )

                for fairness_comparison_metric in list_fairness_purposes:

                    dict_fair_scores_df['uncorrected'][fairness_comparison_metric] = fair_score(
                        augmented_train_valid_set=train_valid_set_with_uncorrected_results,
                        model_name='uncorrected',
                        fairness_purpose=fairness_comparison_metric,
                        model_task=model_task,
                        inspected_column=protected_attribute,
                        fairness_mode="correction",
                    )

                    dict_fair_scores_df['uncorrected'][stat_criteria] = round(stat_score(
                        train_valid_set_with_corrected_results, 'uncorrected', model_task, stat_criteria
                    ), 2)

                    model_name = best_model_dict['model_name']

                    dict_fair_scores_df[weight_method][fairness_comparison_metric] = fair_score(
                        augmented_train_valid_set=train_valid_set_with_corrected_results,
                        model_name=model_name,
                        fairness_purpose=fairness_comparison_metric,
                        model_task=model_task,
                        inspected_column=protected_attribute,
                        fairness_mode="correction",
                    )

                    dict_fair_scores_df[weight_method][stat_criteria] = round(stat_score(
                        train_valid_set_with_corrected_results, model_name, model_task, stat_criteria
                    ), 2)


            print(f"Comparison of fairness metrics between the baseline uncorrected / new model {model_name} optimized for {fairness_purpose}")

            for fairness_comparison_metric in list_fairness_purposes:

                dict_model_types = {'uncorrected':f'Baseline (only optimized for {stat_criteria})', 'grid_search':'Grid Search', 'weighted_groups':'FairDream'}

                fig = go.Figure(
                    data=[go.Bar(
                            name=f"{dict_model_types[model_name]},\n{stat_criteria}={dict_fair_scores_df[model_name][stat_criteria]}",
                            x=dict_fair_scores_df[model_name][fairness_comparison_metric].index.astype("str"),
                            y=dict_fair_scores_df[model_name][fairness_comparison_metric][fairness_comparison_metric],
                            text=dict_fair_scores_df[model_name][fairness_comparison_metric]['nb_individuals_by_group'],
                        )
                          for model_name in dict_fair_scores_df.keys()]
                )

                # fig.add_traces(
                #     [
                #         go.Scatter(
                #             name="mean",
                #             visible="legendonly",
                #             x=dict_fair_scores_df['uncorrected'][fairness_comparison_metric].index.astype("str"),
                #             y=dict_fair_scores_df['uncorrected'][fairness_comparison_metric]["mean"],
                #         )
                #     ]
                # )

                fig.update_layout(
                    title=f"New {fairness_comparison_metric} by group of {protected_attribute}",
                    yaxis_title=f"{fairness_comparison_metric}",
                    legend_title_text=f"Models optimized for {stat_criteria} and {fairness_purpose}",
                )

                if not os.path.isdir(f"comparison_plots/{protected_attribute}/correction_{fairness_purpose}"):
                    os.makedirs(f"comparison_plots/{protected_attribute}/correction_{fairness_purpose}")

                fig.write_html(f"comparison_plots/{protected_attribute}/correction_{fairness_purpose}/evaluation_{fairness_comparison_metric}.html")

                plt.io.write_image(fig, f"comparison_plots/{protected_attribute}/correction_{fairness_purpose}/evaluation_{fairness_comparison_metric}.pdf", format='pdf')

                fig.show()
    
    # finish with preparing the file "comparison_plots" to be downloaded
    # go to the root of the jupyter notebook, and then from new->terminal (right side)... https://github.com/jupyter/notebook/issues/3092
    # $ tar czf comparison_plots.tar.gz notebooks/comparison_plots

def get_fairness_metrics_by_group(model_type, dict_protected_attributes, dict_df_gaps): 
    t_init=time.time() 
    # save the uncorrected model, to then sort its features by importances
    save_model=True
    uncorrected_model_path = "/work/data/models/uncorrected_model.pkl"

    # step 1, BASELINE: train a Baseline model, regardless of fairness
    model_task="classification"
    data = fetch_openml(data_id=1590, as_frame=True)
    X = data.data
    Y = (data.target == '>50K') * 1

    X_train, X_valid, X_train_valid, X_test, Y_train, Y_valid, Y_train_valid, Y_test = train_valid_test_split(X,Y, model_task)

    Y_pred_train_valid, probas_pred_class_1_train_valid = train_naive_model(X_train, X_valid, X_train_valid, X_test, Y_train, Y_valid, Y_train_valid, Y_test, model_task, stat_criteria, save_model=save_model,
                                        model_type=model_type)
    
    train_valid_set_with_uncorrected_results = augment_train_valid_set_with_results("uncorrected", X_train_valid, Y_train_valid, Y_pred_train_valid, probas_pred_class_1_train_valid, model_task)

    # step 2, DETECTION: get a list of protected attributes, based on Demographic Parity (overall_positive_rate between groups)
    augmented_train_valid_set = train_valid_set_with_uncorrected_results
    model_name = "uncorrected"

    fairness_purpose = "overall_positive_rate"
    injustice_acceptance=3
    min_individuals_discrimined=0.01

    list_protected_attributes=discrimination_alert(augmented_train_valid_set, model_name, fairness_purpose, model_task, injustice_acceptance, min_individuals_discrimined)
    
    # step 3, CORRECTION: for each protected attribute, launch 5 models of FairDream vs GridSearch vs Baseline for fairer results (according to Demographic Parity)
    tradeoff = "fair_preferred" # TODO balanced? Or stat_preferred?
    nb_fair_models = 10

    for protected_attribute in list_protected_attributes:
        print(f"{model_type}: {protected_attribute} - {nb_fair_models} models of FairDream, GridSearch and Baseline in competition for fairer results on Demographic Parity \n")
        
        # GridSearch training - store results
        weight_method = 'grid_search'
        train_valid_set_with_uncorrected_results, models_df, best_model_dict = fair_train(
            X=X,
            Y=Y,
            train_valid_set_with_uncorrected_results=train_valid_set_with_uncorrected_results,
            protected_attribute=protected_attribute,
            fairness_purpose=fairness_purpose,
            model_task=model_task,
            stat_criteria=stat_criteria,
            tradeoff=tradeoff,
            weight_method=weight_method,
            nb_fair_models=nb_fair_models,
            model_type=model_type,
        )

        # then FairDream training - store results
        weight_method = 'weighted_groups'
        train_valid_set_with_corrected_results, models_df, best_model_dict = fair_train(
            X=X,
            Y=Y,
            train_valid_set_with_uncorrected_results=train_valid_set_with_uncorrected_results,
            protected_attribute=protected_attribute,
            fairness_purpose=fairness_purpose,
            model_task=model_task,
            stat_criteria=stat_criteria,
            tradeoff=tradeoff,
            weight_method=weight_method,
            nb_fair_models=nb_fair_models,
            model_type=model_type,
        )

        augmented_train_valid_set = train_valid_set_with_corrected_results

        #dict_df_gap_brier[model_type][protected_attribute], dict_df_gap_roc_auc[model_type][protected_attribute],dict_df_gap_pr_auc[model_type][protected_attribute], dict_df_gap_opr[model_type][protected_attribute], dict_df_gap_fpr[model_type][protected_attribute], dict_df_gap_tpr[model_type][protected_attribute]
        (dict_df_gaps["calibration_error"][model_type][protected_attribute],
         dict_df_gaps["roc_auc"][model_type][protected_attribute],
         dict_df_gaps["pr_auc"][model_type][protected_attribute],
         dict_df_gaps["overall_positive_rate"][model_type][protected_attribute],
         dict_df_gaps["false_positive_rate"][model_type][protected_attribute],
         dict_df_gaps["true_positive_rate"][model_type][protected_attribute])=get_dfs_gaps_brier_auc(
                augmented_train_valid_set=augmented_train_valid_set, 
                inspected_column=protected_attribute)
        
    # step 4: STORE the results *nb protected attributes on calibration, positive rates, and AUCs for the (model_type, protected_attributes, list_models_in_competition) 
    # TODO into an experiment function
    dict_protected_attributes[model_type]=list_protected_attributes

    return (dict_protected_attributes[model_type],
         dict_df_gaps["calibration_error"][model_type],
         dict_df_gaps["roc_auc"][model_type],
         dict_df_gaps["pr_auc"][model_type],
         dict_df_gaps["overall_positive_rate"][model_type],
         dict_df_gaps["false_positive_rate"][model_type],
         dict_df_gaps["true_positive_rate"][model_type])

def get_list_colors_by_model_type(list_model_types, dict_df_results_to_plot, dict_color_by_model_type, gap_to_measure, x_to_plot):

    df_color_by_model_type = dict_df_results_to_plot[gap_to_measure][x_to_plot].copy()

    df_color_by_model_type["model_type"]='grey'

    for model_type in list_model_types:
        df_color_by_model_type.loc[df_color_by_model_type.index.str.contains(model_type), "model_type"]=dict_color_by_model_type[model_type]
        #df_color_by_model_type[df_color_by_model_type.index.str.contains(model_type)]["model_type"]=model_type

    return df_color_by_model_type["model_type"]

def get_list_colors_by_(list_models_for_color, dict_df_results_to_plot, dict_color_by_, gap_to_measure, x_to_plot):

    df_color_by_ = dict_df_results_to_plot[gap_to_measure][x_to_plot].copy()

    df_color_by_["color"]='grey'

    for model_for_color in list_models_for_color:
        df_color_by_.loc[df_color_by_.index.str.contains(model_for_color), "color"]=dict_color_by_[model_for_color]

    return df_color_by_["color"]

def get_list_shapes_by_(list_models_for_shape, dict_df_results_to_plot, dict_shape_by_, gap_to_measure, x_to_plot):

    df_shape_by_ = dict_df_results_to_plot[gap_to_measure][x_to_plot].copy()

    df_shape_by_["shape"]='.'

    for model_for_shape in list_models_for_shape:
        df_shape_by_.loc[df_shape_by_.index.str.contains(model_for_shape), "shape"]=dict_shape_by_[model_for_shape]

    return df_shape_by_["shape"]

def plot_gap_groups(color,
        dict_df_results_to_plot, x_to_plot, y_to_plot, gap_to_measure, list_model_types, list_models_in_competition):
    # gap_to_measure: Must be set to a value in {"max_gap_groups","worst_group_score"}
    # fairness_gap_metric: either in Positive Rates, AUCs, or Calibration... 
        # Must be set to a value in ["calibration_error","roc_auc","pr_auc","overall_positive_rate","false_positive_rate","true_positive_rate"]
    # set text for plots, depending on the gap_to_measure
    # "shape" (str): which indicator sets the shapes of the points. Must be set to a value in {"model_name","model_type"} 
    # "color" (str): which indicator sets the colors of the points. Must be set to a value in {"protected_attribute","model_name","model_type"} 
    dict_text_gap_to_measure={"max_gap_groups":"Gap between min-max groups", "worst_group_score": "Worst"}
    dict_shape_by_model_name = {"FairDream":"*","Baseline":"o", "GridSearch":"v"}

    # dict_list_models = {"model_name": list_models_in_competition, "model_types": list_model_types}

    # dict_shape_by = {"model_name": {"FairDream":"*","Baseline":"o", "GridSearch":"v"},
    #                  "model_type": {"xgboost":"X","random_forest":"^", "log_reg":".", "neural_net":"P"},
    #     }

    # dict_color_by_model_type = {'random_forest':'green', 'log_reg':'grey','xgboost':'r'}

    dict_color_by_ = {"model_name": {"FairDream":"cyan","Baseline":"black", "GridSearch":"grey"},
                     "model_type": {'xgboost':'red', 'log_reg':'grey', 'random_forest':'green', 'neural_net':'blue','tabnet':'magenta'},
                     "protected_attribute": {} # TODO extensible list, from protected attributes of dict_df_results_to_plot!
                     } # TODO extract the protected attribute and set in a list, then colours (or shapes?)
    
    #list_models_by_protected_attribute=[f"{model_type}_{protected_attribute}" for protected_attribute in dict_protected_attributes[model_type] for model_type in list_model_types]
    list_models_by_protected_attribute=dict_df_results_to_plot[gap_to_measure][x_to_plot].index
    # 5 shapes for each model_type, to compare them visually 
        # else, link the points of FairDream / GridSearch / Baseline in a curve, for every model?
    # Plot each point individually with a different marker shape
    marker_shapes = ['o', 's', '^', 'D', 'v', '>', '<', 'p', '*', 'H'] # markers shapes for each protected_attribute
    for model_name in list_models_in_competition:
        # for protected_attribute in dict_protected_attributes[model_type]:
        for i, (x, y) in enumerate(zip(dict_df_results_to_plot[gap_to_measure][x_to_plot][model_name],dict_df_results_to_plot[gap_to_measure][y_to_plot][model_name])):
            #plt.scatter(x, y, marker=marker_shape, color=dict_color_by_model_name[model_name], label=list_models_by_protected_attribute[i])
            # add color by model type (TODO mark the name of the model FairDream... ?)
            # add shape by model name
            # list_colors_by_model_type=get_list_colors_by_model_type(list_model_types, dict_df_results_to_plot, dict_color_by_model_type, gap_to_measure, x_to_plot)
            
            # list_colors=get_list_colors_by_(dict_list_models[color], dict_df_results_to_plot, dict_color_by_[color], gap_to_measure, x_to_plot)
            # plt.scatter(x, y, marker=dict_shape_by_model_name[model_name], color=list_colors[i], label=list_models_by_protected_attribute[i])

            if color=="model_name": # then shape == protected_attribute # TODO enable other shape
                shape = "feature"
                marker_shape = marker_shapes[i % len(marker_shapes)]  # Cycle through marker shapes
                plt.scatter(x, y, marker=marker_shape, color=dict_color_by_[color][model_name], label=list_models_by_protected_attribute[i])
            elif color=="model_type": # i.e. color of XGBoost, neural nets... Then shape == model_name (FairDream, Baseline...)
                shape = "model_name" # TODO enable other shape 
                list_colors_by_model_type=get_list_colors_by_model_type(list_model_types, dict_df_results_to_plot, dict_color_by_[color], gap_to_measure, x_to_plot)
                plt.scatter(x, y, marker=dict_shape_by_model_name[model_name], color=list_colors_by_model_type[i], label=list_models_by_protected_attribute[i])


    plt.title(f"{list_models_in_competition} models, {y_to_plot} vs {x_to_plot}: {gap_to_measure}")
    plt.xlabel(f"{dict_text_gap_to_measure[gap_to_measure]} {x_to_plot}")
    plt.ylabel(f"{dict_text_gap_to_measure[gap_to_measure]} {y_to_plot}")
    # heavy legend! Add it only in the last plot ("roc_auc" vs "roc_auc")
    if x_to_plot==y_to_plot:
        plt.legend()

    path_gap_plots=f"calibration_plots/{list_models_in_competition}/{gap_to_measure}"
    if not os.path.isdir(path_gap_plots):
        os.makedirs(path_gap_plots)

    plt.savefig(os.path.join(path_gap_plots, f"{gap_to_measure}_{x_to_plot}_vs_{y_to_plot}_{list_models_in_competition}_color_{color}_shape_{shape}"))
    plt.show()
