# Late Deliveries

This AIGC 5005 Lab 05 project predicts whether a QuickBite order will arrive late using only information available when the order is placed. A scikit-learn Pipeline supports a voucher decision whose threshold is chosen from training-only out-of-fold probabilities and the company's costs.

## Setup

Run these commands from the repository root with Conda installed:

```sh
conda env create -f environment.yml
conda activate lab05-late-deliveries
pip install -r requirements.txt
pip install -e .
```

The environment uses Python 3.11 from conda-forge. Pinned packages include scikit-learn 1.9.1, pandas, matplotlib, pytest, ruff and ipykernel. Select this environment's Python interpreter/kernel when opening `notebooks/analysis.ipynb` in VS Code. The notebook reads the saved CSV/JSON reports and preserves its executed tables and plots; it does not train or evaluate the model again.

## Run

```sh
python -m latedeliveries.train
```

This recreates the comparison, tuning, threshold and leakage reports, fits the frozen final pipeline, evaluates the held-out set once per run, and writes `reports/metrics.json`, `models/late_orders.joblib` and `models/model_info.json`. The pipeline includes preprocessing; metadata records the library version, seed, threshold, features, parameters and test results. Only load joblib files you created or trust, using the recorded scikit-learn version.

For result inspection, read the committed reports or run the notebook instead. All model and threshold decisions were fixed before the original test evaluation. Repeated runs are for identical frozen reproduction, never for changing settings until the test score improves; a reproduced best setting or threshold that differs from approval causes the command to stop before test prediction.

## Check it

```sh
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

All three checks pass in the project environment, including 91 pytest cases. Ruff includes `notebooks/analysis.ipynb`.

## Decisions

**Order-time features and preprocessing.** Numeric features are `distance_km`, `items` and `hour`; categorical features are `weather` and `day_type`. They describe the placed order, current weather and calendar. Features must be available at placement: later information would answer an easier problem than the deployed prediction must solve.

| Excluded column | Reason |
|---|---|
| `order_id` | An identifier, not a legitimate predictive attribute. |
| `prep_minutes` | Actual kitchen preparation duration is unknown at placement. |
| `customer_rating` | Post-delivery feedback influenced by the outcome. |
| `vehicle` | No evidence that the courier/vehicle has already been assigned at placement. |
| `courier_months` | No evidence that the courier has already been assigned at placement. |
| `late` | The target, never an input feature. |

`split_orders(orders)` returns X_train, X_test, y_train and y_test using the unchanged 3,000-order generator, stratified 75/25 split and SEED=7: 2,250 training and 750 test orders. Training data contain 351 late orders (15.6%). No preprocessing statistics are fitted before splitting or CV.

`build_pipeline(model)` accepts a scikit-learn estimator. Its ColumnTransformer applies SimpleImputer(strategy="median") then StandardScaler to numeric features, and OneHotEncoder(handle_unknown="ignore") to categories, before the model. These steps are learned inside each training fold. Extra raw columns are dropped. A raw-order test combines missing `prep_minutes` with unseen `weather`; this verifies ignoring the excluded column and unknown-category handling. Separate tests inject missing values into each retained numeric feature and verify the fitted median is used: ignoring `prep_minutes` is not evidence of median imputation.

**Baseline and model selection.** The four full pipelines share five shuffled stratified training folds with SEED=7. Below are mean +/- population fold SD (`ddof=0`); precision, recall and F1 refer to the late class under each estimator's default `predict()` rule. Real binary classifiers use the 0.5 decision boundary, with exact ties assigned to class 0; Dummy predicts the majority class. Undefined precision is reported as zero. ROC-AUC uses continuous scores rather than thresholded labels.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| DummyClassifier | 0.844000 +/- 0.000889 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 | 0.000000 +/- 0.000000 | 0.500000 +/- 0.000000 |
| Logistic Regression | 0.857778 +/- 0.005963 | 0.643570 +/- 0.067079 | 0.199356 +/- 0.019314 | 0.304262 +/- 0.029339 | 0.772782 +/- 0.035244 |
| Random Forest | 0.847556 +/- 0.013003 | 0.528010 +/- 0.079856 | 0.270543 +/- 0.039496 | 0.356063 +/- 0.048863 | 0.709683 +/- 0.035158 |
| Balanced Decision Tree | 0.779111 +/- 0.018249 | 0.292070 +/- 0.039569 | 0.284789 +/- 0.031390 | 0.287356 +/- 0.032076 | 0.578922 +/- 0.019925 |

Accuracy measures all correct decisions; precision measures how often a flag is truly late; recall measures how many actual late orders are found; F1 balances precision and recall; AUC measures score-ranking discrimination. Dummy's 0.844 accuracy hides zero recall, so accuracy alone is insufficient.

Logistic Regression was selected with AUC 0.772782 and precision 0.643570, the highest among the real candidates. Its accuracy, recall and F1 SDs are smaller than the other real models'; its AUC spread is similar to Random Forest's. Default-rule recall is only 0.199356. Random Forest has higher recall and F1, which remains a real advantage. ROC-AUC was used as the primary model-selection metric because it measures discrimination across thresholds, allowing the final threshold to be selected separately based on business costs. Default-threshold recall alone therefore does not decide the model, and high ROC-AUC does not guarantee the lowest business cost. Fold SD is not a confidence interval or proof of significant superiority.

Every flagged order costs $4.00, including correctly flagged late orders; every unflagged late order costs $15.00. The selection therefore considers finding costly delays while avoiding excessive voucher volume, with the eventual trade-off determined explicitly by the cost rule rather than by a single classification metric.

**Hyperparameter tuning.** GridSearchCV searches `model__C=[0.01, 0.1, 1.0, 10.0]` and `model__class_weight=[None, "balanced"]`: two hyperparameters, eight combinations, five training folds. C controls regularization strength (lower C means stronger regularization), so testing several values helps balance underfitting and overfitting. Balanced weights test whether emphasizing the minority class helps ranking; they do not replace threshold optimization or guarantee better probabilities/costs. ROC-AUC is the scoring metric because the final threshold is not fixed at 0.5 and is selected separately from training OOF probabilities.

After cross-validation, `GridSearchCV(refit=True)` automatically refits the best configuration on all 2,250 training orders without using the test set. This search refit is separate from the final production pipeline training in `train.py`.

The best settings are C=0.1, class_weight=None, with solver="lbfgs", max_iter=1000 and random_state=SEED unchanged. Untuned AUC is 0.772781796 +/- 0.035244026; tuned AUC is 0.772862258 +/- 0.035305596. The improvement is only 0.000080462, much smaller than the fold SD of about 0.035. Tuning did not produce a clear improvement beyond fold-to-fold variation. Balanced settings were evaluated but not selected by AUC; that does not establish that they would always have higher business cost. All eight configurations and fold scores are in `reports/tuning_results.csv`.

**Cost-based threshold.** The approved C=0.1 pipeline generates class-1 training OOF probabilities with `cross_val_predict(method="predict_proba")` on the same five stratified folds. Preprocessing/model fitting is repeated inside each training fold. We compare 19 thresholds from 0.05 to 0.95 by 0.05, flag when probability >= threshold, and choose the lowest threshold among equal minimum costs. The cost function is $4.00 per flag plus $15.00 per missed late order.

Threshold 0.25 uniquely minimizes this grid's cost: 374 vouchers and 180 missed delays, costing $1,496.00 + $2,700.00 = $4,196.00.

| Training OOF policy (2,250 orders) | Total cost |
|---|---:|
| Flag nobody | $5,265.00 |
| Flag everybody | $9,000.00 |
| Model at 0.25 | $4,196.00 |

These are costs on the data used to choose the threshold, not independent test performance. Hyperparameters were also selected on training data, so OOF prediction does not eliminate selection optimism. All 19 counts/costs are in `reports/threshold_results.csv`.

**Leakage demonstration.** A separate training-only experiment adds just `customer_rating` to the numeric branch, imputing it within each fold. It repeats the Stage 4 candidates and folds without changing approved feature lists. For Stage 4 Logistic Regression (C=1.0), normal AUC 0.772782 becomes 0.905277, a rise of 0.132495. This is outcome-derived data leakage, not a usable prediction improvement: ratings are given after delivery and are influenced by late status. Fold-local preprocessing cannot make future information available at order placement. Results for all four candidates are in `reports/leakage_results.csv`; the final model remains C=0.1 and excludes rating.

## Result

The frozen model and threshold were evaluated once on the final 750-order test set, containing 117 late orders. Rows below are actual classes and columns are predicted decisions.

| Actual / predicted | On time (0) | Late / flagged (1) |
|---|---:|---:|
| On time (0) | 570 TN | 63 FP |
| Late (1) | 70 FN | 47 TP |

| Metric at threshold 0.25 | Value |
|---|---:|
| Precision | 0.427273 |
| Recall | 0.401709 |
| F1 | 0.414097 |
| ROC-AUC (from probabilities) | 0.752238 |
| Flagged orders | 110 |
| Missed late orders | 70 |

| Final test policy | Total cost |
|---|---:|
| Flag nobody | $1,755.00 |
| Flag everybody | $3,000.00 |
| Model | $1,490.00 |

Model cost is 110 x $4.00 + 70 x $15.00 = $1,490.00, $265.00 (15.10%) below flagging nobody. I treat this as an observed difference on one simulated holdout, not reliable evidence of the same savings in future operations. The model still missed 70 of 117 late orders. Full-precision results are in `reports/metrics.json` and agree with `models/model_info.json`; no settings were revised after seeing them.

## What the tests cover

The 91 cases cover approved feature exclusions, stratified/repeatable/disjoint splits and target alignment; median imputation, scaling, one-hot and unseen-category behavior; valid probabilities; shared CV folds, fold-local statistics and correct metric/mean/SD calculations; the approved grid and its best-result selection; hand-calculated cost/threshold cases including nobody/everybody minima, equality and ties; experimental leakage isolation; fixed-threshold final metrics and AUC from probabilities; training-only orchestration, approval-mismatch guards and one final probability call; JSON schema and metadata consistency; and whole-pipeline joblib reload equivalence.

Parametrized cases cover excluded features, metric/grid combinations, missing numeric inputs, policy costs and boundary behaviors. Error tests check deliberate validation failures and messages. Persistence tests use `tmp_path` and synthetic raw rows, including missing values and unseen weather, to compare probabilities, default labels and threshold decisions before/after loading. Orchestration tests replace the real final test data with sentinels/mocks.

I deliberately do not assert that the model must beat baseline on every possible dataset or seed: that is an empirical result, not a guaranteed software invariant. Test data are never used to choose settings in tests, and the notebook reads saved results rather than performing new predictions.

## Known limitations

- Orders are simulated rather than operational observations. Distance, items, placement hour, current weather and day type must actually be available at prediction time; courier assignment is unsupported, so vehicle and experience are conservatively excluded.
- The model missed 70 of 117 test delays. The observed cost reduction does not establish acceptable service quality or guarantee future savings.
- The $4.00/$15.00 costs are assumed fixed and the scenario assumes vouchers always work. Different costs, customer behavior or probability calibration can change the appropriate decision rule.
- The threshold comes from a coarse 19-point grid on the same training data used for tuning. CV spread and threshold-selection costs are not independent final estimates; the tiny tuning gain does not prove superiority.
- Saved joblib artifacts depend on trusted provenance and compatible library versions. Cross-version identical results are not assumed.
