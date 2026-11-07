# Late Deliveries

This AIGC 5005 Lab 05 project will predict whether an order arrives late using information available when the order is placed. Stage 1 contains repository and environment scaffolding only.

## Setup

```sh
conda env create -f environment.yml
conda activate lab05-late-deliveries
pip install -r requirements.txt
pip install -e .
```

## Run

The required entry point will be implemented in a later approved stage:

```sh
python -m latedeliveries.train
```

## Check it

```sh
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

There are no tests yet at Stage 1; pytest exit code 5 is not a passing suite.

## Decisions

Approved numeric features are distance_km, items and hour. Approved categorical features are weather and day_type. They describe the placed order, current conditions and calendar.

Excluded columns:

- order_id: an identifier, not a legitimate predictive attribute.
- prep_minutes: actual preparation time is unknown at order placement.
- customer_rating: post-delivery feedback and target-derived information.
- vehicle: no evidence a courier/vehicle is assigned at placement.
- courier_months: no evidence the courier is assigned at placement.
- late: the target, never a feature.

The future pipeline will ignore prep_minutes even when it is missing in raw input. A separate test will verify median imputation for a retained numeric feature, alongside the required unseen-weather test.

Planned candidates are DummyClassifier, LogisticRegression, RandomForestClassifier and DecisionTreeClassifier(class_weight="balanced"), using identical five-fold stratified training-only cross-validation. Comparison results, model choice, tuning and cost-based threshold remain pending. No model, hyperparameters or threshold may change after test results are viewed.

## Result

Pending: no model has been trained and no test evaluation has been run. The final report will contain the confusion matrix, metrics, all three policy costs and a statement about confidence in the difference.

## What the tests cover

Pending: the behavioral test suite starts in later approved stages. No tests have been omitted from an implemented suite because none exists yet.

## Known limitations

Only repository/environment scaffolding and the unchanged instructor data generator exist. Pipeline, training entry point, reports, saved model and notebook remain unimplemented. Courier-related fields are conservatively excluded because assignment timing is unspecified.
