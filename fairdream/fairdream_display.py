import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
sys.path.append("../")

from fairdream.correction import fair_train
from fairdream.detection import discrimination_alert
from fairdream.plots import fair_model_results

from sklearn.datasets import fetch_openml

st.title("FairDream")
st.subheader("Alert-Based Correction of Discriminations for Income Prediction (income > or < $50,000)")

# default parameters (not seen by the final user)
model_name = "uncorrected"
model_task = "classification"

stat_criteria = "auc" # TODO later set as user's choice
model_type = 'random_forest' # TODO add option for the user to fix the models structure to (1) train => augmented train valid set (2) correct 


def translate_fairness_purpose(fairness_objective:str)->str:
    """ Translates fairness objective of the user into fairness metrics to analyse AI
    '''
    fairness_objective : str
        Objective of fairness the user wants AI to achieve in the banking context
        Must be set to a value in {"% of clients selected as earning over $50,000",
        "Total number of clients selected as earning over $50,000", 
        "% of clients selected as earning over $50,000, but who do not earn over $50,000", 
        "% of clients selected as not earning over $50,000, and who do not earn over $50,000"}
    '''
    returns
    '''
    fairness_purpose : str
        Metrics by which the user wants to measure the gap between groups : basis of evaluation to compute the fair_score.
        Will serve in detection: detects inequalities of selection by the model (between subgroups and the mean) regarding fairness_purpose
        Will serve in correction: selects the best model satisfying stat/fair tradeoff regarding fairness_purpose
        -> For classification (binary or multi-classes), must be set to value in {"overall_positive_rate", "nb_positive", "false_positive_rate", "true_negative_rate"}
        -> For regression, must be set to a value in {'distribution_gap','mean_squared_error', 'mean_absolute_percentage_error', 'r2_score'}
    
    """
    dict_map_objective_purpose = {
        "% of clients selected as earning over $50,000 (overall positive rate)":"overall_positive_rate",
        "Total number of clients selected as earning over $50,000 (overall positive count)":"nb_positive", 
        "% of clients selected as earning over $50,000, but who do not earn over $50,000 (false positive rate)":"false_positive_rate", 
        "% of clients selected as not earning over $50,000, and who do not earn over $50,000 (true negative rate)":"true_negative_rate"}

    if fairness_objective in dict_map_objective_purpose.keys():
        fairness_purpose = dict_map_objective_purpose[fairness_objective]
    else:
        raise NotImplementedError("The fairness objective you want to achieve is not implemented. Must be set to a value in"
        "\n {'% of clients selected as earning over $50,000 (overall positive rate)',"
        "\n 'Total number of clients selected as earning over $50,000 (overall positive count)',"
        "\n '% of clients selected as earning over $50,000, but who do not earn over $50,000 (false positive rate)',"
        "\n '% of clients selected as not earning over $50,000, and who do not earn over $50,000 (true negative rate)'}")

    return fairness_purpose 

def translate_correction_method(correction_idea:str)->str:
    """ Translates the user's idea of a correction method into a weight method to train the new models
    '''
    correction_idea : str
        Idea of a method to correct the new AI integrating fairness objective
        Must be set to a value in {
        'Optimize the new weights of errors on individuals, growing exponentially with their previous disadvantage (FairDream)',
        'Optimize the new weights of errors on individuals, using Lagrange multipliers (GridSearch)',
        'Mix reweighting penalisation and search for different individual weights'}

    '''
    returns
    weight_method : str
        How to generate fair_models
        Must be set to values in {'weighted_groups','grid_search','grid_and_weighted_groups'}
        If weight_method == 'grid_and_weighted_groups':
            1/2 models trained with weights distorsion ('weighted_groups')
            1/2 models trained with 'grid_search'
    """
    dict_map_correction_method = {
        'Optimize the new weights of errors on individuals, growing exponentially with their previous disadvantage (FairDream)': 'weighted_groups',
        'Optimize the new weights of errors on individuals, using Lagrange multipliers (GridSearch)': 'grid_search',
        'Mix reweighting penalisation and search for different individual weights': 'grid_and_weighted_groups'}

    if correction_idea in dict_map_correction_method.keys():
        weight_method = dict_map_correction_method[correction_idea]
    else:
        raise NotImplementedError("The correction method you want to achieve is not implemented. Must be set to a value in"
        "\n {'Optimize the new weights of errors on individuals, growing exponentially with their previous disadvantage (FairDream)',"
        "\n 'Optimize the new weights of errors on individuals, using Lagrange multipliers (GridSearch)',"
        "\n 'Mix reweighting penalisation and search for different individual weights'}")

    return weight_method

DATA_PATH = ('uncorrected_augmented_train_valid_set_label_encoded.csv')
uncorrected_model_path = 'uncorrected_model.pkl'

@st.cache(allow_output_mutation=True)
def load_data(DATA_PATH: str)->pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df 

augmented_train_valid_set = load_data(DATA_PATH)

if st.checkbox('Show Clients Data'):
    st.write(augmented_train_valid_set)

st.subheader(f"Detection - AI Model that was trained for {model_task} of loan applicants is analysed\n")

# taking user inputs
fairness_objective = st.selectbox(label='What is your fairness objective to compare groups?',
        options=('% of clients selected as earning over $50,000 (overall positive rate)',
        'Total number of clients selected as earning over $50,000 (overall positive count)',
        '% of clients selected as earning over $50,000, but who do not earn over $50,000 (false positive rate)',
        '% of clients selected as not earning over $50,000, and who do not earn over $50,000 (true negative rate)'))

fairness_purpose = translate_fairness_purpose(fairness_objective)

st.write("")
st.write("In your opinion, to consider there is a discriminated group...")

col1, buff, col2 = st.columns([5, 1, 7])

with col1:
    # insert pictures (big / smaller bars)
    injustice_acceptance = st.slider("What is the minimal gap with other groups?", 1, 5, value=3, step=1)

    fig = go.Figure(data=[
        go.Bar(name='discriminated', x=[injustice_acceptance], y= [1]),
        go.Bar(name='advantaged', x=[injustice_acceptance], y= [injustice_acceptance])
    ])
    # Change the bar mode
    fig.update_layout(barmode='group',yaxis_visible=False, yaxis_showticklabels=False)
    # Set figure size
    fig.update_layout(width=800, height=400)

    st.plotly_chart(fig, use_container_width=True)

with col2:
    # insert more visible %
    min_percentage_discriminated = st.slider("What is the minimal % of clients concerned?", 0, 100, 7,)
    min_individuals_disadvantaged = min_percentage_discriminated/100

    fig = go.Figure()

    # Create scatter trace of text labels
    fig.add_trace(go.Scatter(
        x=[min_individuals_disadvantaged+1.7, 3],
        y=[min_individuals_disadvantaged+0.2, 2.5],
        text=["Discriminated group size",
            "Clients total"],
        mode="text",
    ))

    # Set axes properties
    fig.update_xaxes(range=[0, 4.5], zeroline=False)
    fig.update_yaxes(range=[0, 4.5])

    x0=0.1
    y0=0.1
    x1=4
    y1=4

    # Add circles
    fig.add_shape(type="circle",
        xref="x", yref="y",
        x0=x0, y0=y0, x1=x1, y1=y1,
        line_color="LightSeaGreen",
    )
    fig.add_shape(type="circle",
        xref="x", yref="y",
        fillcolor="PaleTurquoise",
        x0=min_individuals_disadvantaged*x0, y0=min_individuals_disadvantaged*y0, 
        x1=min_individuals_disadvantaged*x1, y1=min_individuals_disadvantaged*y1,
        #x0=1, y0=1, x1=3, y1=3,
        line_color="LightSeaGreen",
    )

    # Set figure size
    fig.update_layout(width=800, height=400, xaxis_visible=False, yaxis_visible=False)

    fig.update_traces(textfont_size=14)

    st.plotly_chart(fig, use_container_width=True)

# initialise protected_attribute and nb_fair_models
if 'set_alerts' not in st.session_state:
    st.session_state.set_alerts = []
if 'protected_attribute' not in st.session_state:
    st.session_state.protected_attribute = None 
if 'nb_fair_models' not in st.session_state:
    st.session_state.nb_fair_models = 0 
if "tradeoff" not in st.session_state:
    st.session_state.tradeoff = "moderate"
if "weight_method" not in st.session_state:
    st.session_state.weight_method = "grid_and_weighted_groups"

detection_button = st.button("Detect if AI discriminates against groups")

if detection_button:
    st.session_state.set_alerts = discrimination_alert(augmented_train_valid_set=augmented_train_valid_set, model_name=model_name, 
        fairness_purpose=fairness_purpose, model_task=model_task, injustice_acceptance=injustice_acceptance, 
        min_individuals_disadvantaged=min_individuals_disadvantaged,uncorrected_model_path = uncorrected_model_path)

st.subheader("Correction - Do you want to launch new AIs in competition for fairer results?")

protected_attribute = st.selectbox("On which feature do you want to bridge the gaps between groups?",
         st.session_state.set_alerts)

nb_fair_models = st.number_input("How many models do you put in competition for fairer results?", 
        min_value=4, 
        max_value=12) 

tradeoff = st.selectbox("What is your tradeoff between fairness and statistical performances of the new model?",
        ["moderate","fair_preferred","stat_preferred"])


correction_idea = st.radio(label = "Which method for fair correction of AI?",
        options = [
        'Optimize the new weights of errors on individuals, growing exponentially with their previous disadvantage (FairDream)',
        'Optimize the new weights of errors on individuals, using Lagrange multipliers (GridSearch)',
        ])
        # 'Mix FairDream and GridSearch'])

weight_method = translate_correction_method(correction_idea)

correction_button = st.button("Correct")

if correction_button:
    # st.session_state.protected_attribute = protected_attribute
    # st.session_state.nb_fair_models = nb_fair_models
    # st.session_state.tradeoff = tradeoff
    # st.session_state.weight_method = weight_method

    # st.write(f"Training of {st.session_state.nb_fair_models} AI for correction on {st.session_state.protected_attribute} groups")

    st.write(f"Training of {nb_fair_models} models for correction on {protected_attribute} groups")

    # preparing the dataset on clients for binary classification
    data = fetch_openml(data_id=1590, as_frame=True)

    X = data.data
    Y = (data.target == '>50K') * 1

    train_valid_set_with_corrected_results, models_df, best_model_dict = fair_train(
        X=X,
        Y=Y,
        train_valid_set_with_uncorrected_results=augmented_train_valid_set,
        protected_attribute=protected_attribute,
        fairness_purpose=fairness_purpose,
        model_task=model_task,
        stat_criteria=stat_criteria,
        tradeoff=tradeoff,
        weight_method=weight_method,
        nb_fair_models=nb_fair_models,
        model_type=model_type)
    
    fair_model_results(
        train_valid_set_with_corrected_results=train_valid_set_with_corrected_results, 
        models_df=models_df, 
        best_model_dict=best_model_dict,
        protected_attribute=protected_attribute,
        fairness_purpose=fairness_purpose,
        model_task=model_task)
