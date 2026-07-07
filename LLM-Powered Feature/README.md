# Part 2 — Predictive Modeling: Regression + Classification

Builds on the cleaned dataset from Part 1. Trains and evaluates a regression
model (predicting **annual income**) and a classification model (predicting
**loan default**) on a bank-customer dataset.

> **Note on the dataset:** no `cleaned_data.csv` was supplied with this
> request, so `generate_sample_data.py` creates a realistic synthetic
> stand-in (2,000 customers) so the whole pipeline runs today. To use your
> own Part 1 output: drop your real `cleaned_data.csv` into this folder and
> update the four lines in the `CONFIG` block at the top of `main.py`
> (`REG_TARGET`, `CLF_TARGET`, `ORDINAL_COLS`, `NOMINAL_COLS`) to match your
> actual column names. Everything downstream is column-name-driven and needs
> no other changes.

## How to run

```bash
pip install -r requirements.txt
python3 generate_sample_data.py   # only if you don't have your own cleaned_data.csv
python3 main.py
```

Outputs: metrics printed to stdout (also saved to `results.txt`),
`roc_curve.png`, `threshold_sensitivity.png`.

---

## 1. Label definitions

- **`y_reg` (regression target): `annual_income`** — a continuous numeric
  column.
- **`y_clf` (classification target): `loan_default`** — a *natural* binary
  column in the dataset (not a median split of `annual_income`). It was
  chosen instead of a median split specifically because a median split would
  force an artificial 50/50 balance, whereas the real business question
  ("will this customer default?") is naturally imbalanced — only ~14% of
  customers default in this data, which is realistic and lets the imbalance
  handling in Task 5 actually matter.

## 2. Encoding decisions

- **`education_level` → label/ordinal encoding** (`High School`=0 <
  `Bachelors`=1 < `Masters`=2 < `PhD`=3). Justified because education levels
  have a genuine, agreed-upon ranking — a PhD represents strictly more
  formal education than a Bachelors, so mapping to increasing integers
  preserves real information the model can use (e.g., "one step up the
  ladder") instead of discarding it.
- **`city` → one-hot encoding**, drop-first (`city_Houston`, `city_Los
  Angeles`, `city_New York`, `city_Phoenix`, with `Chicago` as the dropped
  baseline). Cities have no inherent order — labeling them 0–4 would tell
  the model "Phoenix > Chicago" numerically, a false ordinal relationship
  that doesn't exist in reality and would bias any coefficient or distance
  calculation. One-hot avoids this by giving each city its own independent
  0/1 indicator.

## 3. Leak-free split and scaling

`train_test_split(X, y, test_size=0.2, random_state=42)` was used, then
`StandardScaler` was **fit only on `X_train`** and used to `.transform()`
both `X_train` and `X_test`.

**Why fitting on the full dataset would leak:** `StandardScaler` learns the
mean and standard deviation of each feature. If it's fit on the combined
train+test data, the test set's statistics (its mean, its spread) get baked
into the numbers the model is trained on — the model implicitly "sees" a
summary of data it's supposed to be evaluated on later. That inflates
reported performance because the test set is no longer truly unseen. Fitting
only on the training set keeps the test set as a genuine, untouched holdout.

## 4. Regression: Linear Regression vs Ridge

| Model | MSE | R² |
|---|---|---|
| Linear Regression (OLS) | 81,810,809.75 | 0.8886 |
| Ridge (α=1.0) | 81,890,087.52 | 0.8885 |

**Top 3 coefficients by absolute value** (Linear Regression, on scaled
features):

| Feature | Coefficient |
|---|---|
| employment_years | +20,881.22 |
| education_level | +11,814.56 |
| credit_score | +3,541.34 |

**Interpreting coefficients:** because features are standardized, each
coefficient means "for a one-standard-deviation increase in this feature,
predicted annual income changes by this many dollars, holding other features
fixed." A large **positive** coefficient (e.g., `employment_years`) means
more of that feature is associated with higher predicted income. A large
**negative** coefficient (e.g., `num_dependents`, −1,495.57) means more of
that feature is associated with lower predicted income, all else equal.

**Ridge vs OLS:** Ridge adds an L2 penalty (`alpha`) to the loss function
that shrinks coefficients toward zero in proportion to their size, trading a
small amount of bias for reduced variance. This tends to matter most when
features are correlated or the model is close to overfitting — OLS can
assign large, unstable coefficients to correlated predictors, while Ridge
spreads the "credit" more evenly and shrinks everything slightly. Here MSE
and R² are nearly identical between the two models, which suggests the
training features aren't strongly collinear and OLS wasn't overfitting much
in the first place — `alpha=1.0` is a mild penalty on this particular
dataset. `alpha` itself controls the strength of that shrinkage: `alpha=0`
recovers plain OLS, and larger `alpha` shrinks coefficients more aggressively
(at the cost of some fit).

## 5. Classification: Logistic Regression

**Class balance before handling imbalance** (training set): 85.94% no-default
(1,375) vs 14.06% default (225) — minority class is well under the 35%
threshold, so imbalance handling was required.

**Strategy chosen: `class_weight='balanced'`.** SMOTE (oversampling the
training set) was the alternative, but `class_weight='balanced'` was chosen
because it re-weights the loss function using only real observations — no
synthetic rows are created, so every training example the model sees is an
actual customer, which is easier to justify and audit. Weights applied:
class 0 → 0.582, class 1 → 3.556 (the minority class is weighted roughly
6× more heavily per sample than the majority class).

**Confusion matrix** (rows = actual, columns = predicted; test set, n=400):

| | Predicted: No Default | Predicted: Default |
|---|---|---|
| **Actual: No Default** | 237 | 96 |
| **Actual: Default** | 23 | 44 |

**Classification report:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| 0 (No Default) | 0.9115 | 0.7117 | 0.7993 |
| 1 (Default) | 0.3143 | 0.6567 | 0.4251 |
| **Accuracy** | | | **0.7025** |

**AUC: 0.7610**

**Precision / Recall formulas:**
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)

**Which metric matters more here?** For loan default, **recall** on the
default class is more important than precision. A false negative (predicting
"won't default" for someone who actually defaults) directly costs the bank
the unpaid loan principal. A false positive (flagging a safe customer as
risky) just means extra manual review or a slightly more cautious offer —
annoying, but far cheaper than an unrecovered loan. This is exactly why
`class_weight='balanced'` was used: it deliberately trades some precision for
higher recall on the minority (default) class.

**What AUC = 0.7610 means:** if you picked one random defaulter and one
random non-defaulter from the test set, the model would rank the defaulter
as higher-risk about 76% of the time. That's meaningfully better than a coin
flip (0.50) but leaves real room for improvement — the model is a useful
risk-ranking signal, not a highly precise separator.

### Decision-threshold sensitivity

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.30 | 0.2287 | 0.8806 | 0.3631 |
| 0.40 | 0.2741 | 0.8060 | 0.4091 |
| 0.50 | 0.3143 | 0.6567 | 0.4251 |
| 0.60 | 0.4000 | 0.5075 | **0.4474** |
| 0.70 | 0.4792 | 0.3433 | 0.4000 |

- Precision = TP / (TP + FP); Recall = TP / (TP + FN)
- **F1-maximizing threshold: 0.60** (F1 = 0.4474)
- As argued above, **recall** is the more important metric for this task,
  since missed defaulters are the costly error.
- Given that, we would **lower** the threshold below the F1-optimal 0.60 —
  e.g., toward 0.30–0.40 — to catch more true defaulters. The cost of doing
  so is a lower precision (more false alarms: safe customers flagged as
  risky, at 0.30 that's precision of just 0.2287), meaning more manual
  reviews or declined applications for customers who would have repaid.
  That's the deliberate trade-off: accept more false positives to avoid
  missing actual defaults.

## 6. Regularization experiment: C=1.0 vs C=0.01

| Model | Precision | Recall | AUC |
|---|---|---|---|
| C=1.0 (baseline) | 0.3143 | 0.6567 | 0.7610 |
| C=0.01 (strong L2) | 0.3212 | 0.6567 | 0.7628 |

**What `C` controls:** in scikit-learn's `LogisticRegression`, `C` is the
**inverse** of the regularization strength — smaller `C` means a *stronger*
L2 penalty on the coefficients, pushing them closer to zero (more
regularization, simpler decision boundary); larger `C` means a weaker
penalty (closer to unregularized logistic regression, more flexibility to
fit the training data). Here, `C=0.01` (strong regularization) produced
almost identical precision, identical recall, and marginally *higher* AUC
than the baseline `C=1.0`. That suggests the baseline model wasn't
overfitting much to begin with — this dataset has few features and a modest
sample size, so heavier regularization neither helps nor hurts much; it
mainly shrinks coefficients without materially changing performance.

## 7. Bootstrap confidence interval for the AUC difference

500 bootstrap samples were drawn from the test set (`np.random.choice`,
sampling row indices with replacement), and for each sample the AUC of the
C=1.0 model minus the AUC of the C=0.01 model was recorded.

- **Mean AUC difference (C=1.0 − C=0.01): −0.0019**
- **95% CI: [−0.0099, 0.0056]**
- **The interval includes zero.**

**Interpretation:** because the confidence interval spans zero, the small
difference in AUC observed between the two models (C=1.0 slightly *lower*
than C=0.01 on the actual test set) is **not statistically reliable** — it's
consistent with there being no real difference in ranking performance
between the two regularization strengths on this dataset. In practice this
means either `C` setting could be chosen based on other considerations
(e.g., preferring the simpler, more-regularized C=0.01 model for
robustness) without meaningfully sacrificing discriminative power.

---

## Repository contents

- `generate_sample_data.py` — creates the synthetic `cleaned_data.csv` (swap
  for your real Part 1 output as described above)
- `cleaned_data.csv` — the dataset used (2,000 rows × 8 columns)
- `main.py` — all preprocessing, model training, and evaluation code
- `results.txt` — full captured console output from the run referenced above
- `results.json` — same results in structured form
- `roc_curve.png` — ROC curve for the baseline logistic regression
- `threshold_sensitivity.png` — precision/recall/F1 vs. decision threshold
- `requirements.txt` — Python dependencies
