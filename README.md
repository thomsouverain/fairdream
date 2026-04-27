# FairDream

FairDream is an experimental package to **detect** and **mitigate** inequalities in binary classification for tabular data, for example credit scoring, recidivism scoring, income prediction, or other risk-scoring use cases.

The package focuses on an explainable fairness workflow:

1. train or bring a baseline model;
2. detect disparities across groups;
3. choose the protected attribute and the fairness metric to correct;
4. train fairer candidate models through transparent reweighting;
5. select the model that best balances statistical performance and fairness.

The reweighting approach is described in the paper [Fairness Interventions: A Study in AI Explainability](https://arxiv.org/abs/2407.14766). The hands-on example is available in [`notebooks/Fair_detection_correction_xgb_binary_classification_census.ipynb`](notebooks/Fair_detection_correction_xgb_binary_classification_census.ipynb).

## Installation

CPU mode:

```shell
make install
```

GPU mode:

```shell
make install-gpu
```

## Running the project

Start the container in CPU mode:

```shell
make start
```

Start the container in GPU mode:

```shell
make start-gpu
```

Launch Jupyter Notebook inside the container:

```shell
make notebook
```

Launch JupyterLab inside the container:

```shell
make lab
```

## Fairness by reweighting: user guide

### 1. Start from a baseline model

FairDream first needs a baseline model trained without any fairness correction. In the Census notebook, the baseline is an XGBoost binary classifier trained to predict whether an individual earns more than `$50K`.

The baseline model is evaluated on the train-validation set and its predictions are appended to the original rows:

```python
train_valid_set_with_uncorrected_results = augment_train_valid_set_with_results(
    'uncorrected',
    X_train_valid,
    Y_train_valid,
    Y_pred_train_valid,
    model_task,
)
```

This table becomes the common object used for detection and correction.

### 2. Detect whether the model creates group gaps

The detection step asks whether the model treats groups differently according to a chosen fairness purpose.

Example:

```python
fairness_purpose = 'overall_positive_rate'
injustice_acceptance = 3
min_individuals_discrimined = 0.01

discrimination_alert(
    train_valid_set_with_uncorrected_results,
    'uncorrected',
    fairness_purpose,
    model_task,
    injustice_acceptance,
    min_individuals_discrimined,
)
```

With `overall_positive_rate`, FairDream measures the share of positive predictions in each group. A group is flagged as disadvantaged when its score is much lower than another group according to the user-defined alert threshold.

### 3. Choose the correction target

After detection, the user chooses:

```python
protected_attribute = 'sex'
fairness_purpose = 'overall_positive_rate'
tradeoff = 'fair_preferred'
weight_method = 'weighted_groups'
nb_fair_models = 4
```

These choices mean:

- correct disparities measured on `sex`;
- focus on positive prediction rates;
- prefer fairness in the final trade-off score;
- use FairDream’s group reweighting method;
- train four fair candidate models.

### 4. How the reweighting mechanism works

The correction is implemented in [`fairdream/correction.py`](fairdream/correction.py), especially in `weighted_groups_fair_train` and `fair_train`.

For each group, FairDream computes the baseline fairness score. For `overall_positive_rate`, the disadvantaged group is the one with the lower positive prediction rate.

The current reweighting logic can be read as:

```text
score_gap_to_best_group = best_group_score - group_score
impacted_individuals = score_gap_to_best_group * nb_individuals_in_group
relative_impacted_share = impacted_individuals / total_nb_individuals
candidate_weight = coefficient * exp(candidate_id * score_gap_to_best_group)
```

The coefficient introduces diversity across candidate models. In the current implementation, even-numbered candidates use coefficient `1`, while odd-numbered candidates use a coefficient proportional to the relative share of impacted individuals.

The practical meaning is simple: **errors on a previously disadvantaged group become more expensive during training**. The model is therefore encouraged to fit that group better.

This is an in-processing correction: the intervention happens during model training through `sample_weight`. It is not pre-processing, because labels are not rewritten and rows are not duplicated. It is not post-processing, because thresholds are not adjusted after the classifier has been trained.

### 5. Train and select the corrected model

Use `fair_train` to run the full correction workflow:

```python
train_valid_set_with_corrected_results, models_df, best_model_dict = fair_train(
    X=X,
    Y=Y,
    train_valid_set_with_uncorrected_results=train_valid_set_with_uncorrected_results,
    protected_attribute=protected_attribute,
    fairness_purpose=fairness_purpose,
    model_task='classification',
    stat_criteria='auc',
    tradeoff=tradeoff,
    weight_method=weight_method,
    nb_fair_models=nb_fair_models,
    model_type='xgboost',
)
```

`models_df` compares the baseline and the corrected candidates. `best_model_dict` contains the selected model name, the statistical score, the fairness score, the trade-off score, and the detailed group-level fairness table.

### 6. Interpret the output

FairDream does not only answer “is the new model fair?”. It exposes the trade-off:

- Did `overall_positive_rate` improve across groups?
- Did statistical performance, such as ROC-AUC, remain acceptable?
- Did true-positive and false-positive rates also move closer across groups?
- Is the final model consistent with the true-label distribution?

This last point is central in the article. Because FairDream reweights errors inside the usual supervised loss, the model remains sensitive to true labels. As a result, a correction initiated with a Demographic-Parity-style metric may also reveal movement toward Equalized Odds, where true-positive and false-positive rates become closer across groups.

## Recommended notebook

Run the pedagogical notebook:

[`notebooks/Fair_detection_correction_xgb_binary_classification_census.ipynb`](notebooks/Fair_detection_correction_xgb_binary_classification_census.ipynb)

It shows:

- how to train the uncorrected Census classifier;
- how to trigger and read discrimination alerts;
- how to compute the group weights before correction;
- how individual rows inherit group sample weights;
- how to train fair candidates;
- how to compare the baseline and the selected corrected model.

## Notes for maintainers

The original repository template commands are still useful when creating a new package from scratch:

1. change the package folder name;
2. update the package name in `pyproject.toml`;
3. update `IMAGE_NAME`, `DOCKER_NAME`, and `DOCKER_NAME_GPU` in `Makefile`;
4. update logger paths if the package name changes;
5. update `INSTALL_PYTHON` in `pre-commit/pre-commit.py`;
6. create and configure the development branch if needed.
