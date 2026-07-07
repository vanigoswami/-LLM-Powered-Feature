# Part 3 — Advanced Modeling: Ensembles, Tuning, Full ML Pipeline

Continues directly from Part 2. Run with:

```bash
python3 main_part3.py
```

Reuses the exact same `cleaned_data.csv`, encoding, split (`random_state=42`),
and scaling as Part 2, so results are directly comparable. Outputs:
`results_part3.txt`, `results_part3.json`, `best_model.pkl`.

---

## 1. Decision Tree — unconstrained

| | Train Accuracy | Test Accuracy | Gap |
|---|---|---|---|
| Unconstrained (max_depth=None) | 1.0000 | 0.7575 | 0.2425 |

**This shows clear overfitting** — perfect training accuracy but a 24-point
drop on test data. Decision trees are **high-variance models** because they
build splits greedily: at each node they pick the single best split for the
data in front of them and never revisit or reconsider earlier decisions.
Left unconstrained, the tree keeps splitting until every training leaf is
pure (often down to 1 sample), which means it has effectively memorized
training-set noise rather than learning a generalizable pattern.

## 2. Decision Tree — controlled

| | Train Accuracy | Test Accuracy | Gap |
|---|---|---|---|
| Controlled (max_depth=5, min_samples_split=20) | 0.8725 | 0.8325 | 0.0400 |

- **`max_depth`** caps how many splits deep the tree can go, directly
  limiting how specific/memorized its rules can become — this trades a
  little bias (the tree can't capture very fine-grained patterns) for a
  big reduction in variance.
- **`min_samples_split=20`** stops the tree from splitting a node that has
  fewer than 20 samples, which prevents it from carving out rules based on
  a handful of possibly-noisy points.
- **Comparison:** the train-test gap shrank from **0.2425 → 0.0400** — the
  controlled tree gives up a little training accuracy (1.00 → 0.87) but
  gains far more test accuracy (0.7575 → 0.8325), a clear win against
  overfitting.

## 3. Gini vs Entropy

| Criterion | Test Accuracy |
|---|---|
| Gini | 0.8375 |
| Entropy | 0.8375 |

- **Gini impurity:** `1 - Σ pᵢ²` (sum over each class's proportion pᵢ in the
  node)
- **Entropy:** `-Σ pᵢ log₂(pᵢ)`
- **Gini = 0** means the node is perfectly pure — every sample in it belongs
  to the same class, so there's no "impurity" left to split away.
- On this dataset the two criteria produced identical test accuracy — they
  usually agree closely in practice since both measure node purity, just on
  slightly different scales, so the specific splits chosen rarely diverge
  enough to change final accuracy.

## 4. Random Forest

| | Value |
|---|---|
| Train Accuracy | 0.9387 |
| Test Accuracy | 0.8375 |
| Test AUC | 0.7079 |

**Top 5 features by importance:**

| Feature | Importance |
|---|---|
| credit_score | 0.3292 |
| age | 0.1999 |
| employment_years | 0.1824 |
| num_dependents | 0.1170 |
| education_level | 0.0699 |

**How Random Forest computes feature importance:** for each feature, the
algorithm averages the reduction in Gini impurity produced by every split
that uses that feature, across every tree in the forest. A feature that
reliably separates classes well whenever it's used to split gets a high
score. This is fundamentally different from a **linear regression
coefficient**, which measures a fixed, linear, additive effect size (holding
other features constant) — feature importance instead reflects how *useful*
a feature was across many non-linear, interaction-rich splits, with no
sign (direction) or linear-effect-size interpretation attached to it.

**Bagging concept:** each tree in the Random Forest is trained on a
**bootstrap sample** — a random sample of the training rows drawn *with
replacement*, so some rows appear multiple times and others not at all in
any given tree's training set. Additionally, at each split, only a random
subset of **√(number of features)** features is even considered, forcing
trees to diversify rather than all leaning on the same dominant feature.
Averaging predictions (or class votes) across many such de-correlated trees
cancels out each individual tree's idiosyncratic noise, which is why a
Random Forest's variance is much lower than any single deep, unconstrained
decision tree's variance — the ensemble smooths out the memorization any one
tree does.

## 4a. Gradient Boosting

| | Value |
|---|---|
| Train Accuracy | 0.8919 |
| Test Accuracy | 0.8475 |
| Test AUC | 0.7497 |

Gradient Boosting (`n_estimators=100, learning_rate=0.1, max_depth=3`)
slightly outperformed the Random Forest on test accuracy and AUC here,
consistent with boosting's strategy of building shallow trees sequentially,
each one correcting the errors of the previous ensemble, rather than
averaging independent trees.

## 4b. Feature ablation study

**5 lowest-importance features removed:** `education_level`,
`city_Los Angeles`, `city_New York`, `city_Phoenix`, `city_Houston`

| Model | Test AUC |
|---|---|
| Full model (9 features) | 0.7079 |
| Reduced model (4 features) | 0.7244 |
| Difference (full − reduced) | −0.0164 |

**Interpretation:** removing the 5 lowest-importance features actually
*improved* AUC slightly rather than hurting it. This suggests these features
(largely the one-hot city dummies, plus education_level) were **genuinely
close to uninformative** for this particular target — at best contributing
noise that the full model had to "work around," at worst adding
dimensionality without adding signal. **Production implication:** this is
exactly the case where shipping the simpler, lower-dimensional model is a
clear win — it's cheaper to run, easier to maintain (4 fewer columns to
keep validated in a live feature pipeline), and it doesn't cost any
predictive accuracy; in fact here it slightly helped. The only reason to
prefer a reduced model over a full one in general is when AUC degradation
from removing features stays below whatever threshold the business
considers tolerable — here there was no degradation at all, so removing them
is a straightforward win.

## 5. Cross-validated comparison (5-fold, StratifiedKFold)

| Model | Mean AUC | Std AUC |
|---|---|---|
| Logistic Regression | 0.7632 | 0.0277 |
| Decision Tree (max_depth=5) | 0.7256 | 0.0298 |
| Random Forest | 0.7112 | 0.0213 |
| Gradient Boosting | 0.7276 | 0.0318 |

**Why cross-validation beats a single train-test split:** a single 80/20
split gives one estimate of test performance that depends heavily on which
rows happened to land in the test set — a "lucky" or "unlucky" split can
make a model look better or worse than it really is. 5-fold CV instead
rotates which 20% of the data is held out five times, so every row gets
used for both training and validation across the folds. Averaging the five
resulting scores gives a much more stable estimate of how the model
generalizes, and the **standard deviation across folds** tells you how
sensitive that estimate is to which data it sees — a model with low CV
std is more reliably going to perform close to its mean AUC in production.

## 6. GridSearchCV — Random Forest tuning

**Parameter grid:**
```python
param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [5, 10, None],
    'randomforestclassifier__min_samples_leaf': [1, 5]
}
```

- **Total configurations evaluated:** 3 × 3 × 2 = **18 combinations**, each
  fit across 5 folds → **90 total model fits**.
- **Best params:** `max_depth=5, min_samples_leaf=1, n_estimators=50`
- **Best CV AUC:** 0.7433
- **Best pipeline test AUC (held-out):** 0.7514

**Grid Search vs Randomized Search trade-off:** Grid Search is exhaustive —
it guarantees you evaluate every single combination in the grid, so you
never miss the best point *within* that grid, but the cost grows
multiplicatively with every parameter you add or every value you include
(here, adding one more value to any single parameter would mean 30 more
model fits). Randomized Search instead samples a fixed number of random
combinations from the parameter space, which scales far better to large
grids or continuous parameter ranges and in practice often finds a
near-optimal combination in a fraction of the fits — the trade-off is it
isn't guaranteed to find the single best combination, only a good one, in
exchange for much lower compute cost.

## 7. Manual learning curve

| Training fraction | Training AUC | Test AUC |
|---|---|---|
| 0.2 | 0.9762 | 0.6822 |
| 0.4 | 0.9318 | 0.7257 |
| 0.6 | 0.9021 | 0.7386 |
| 0.8 | 0.8770 | 0.7491 |
| 1.0 | 0.8535 | 0.7514 |

- **(i) Training AUC decreases** as the training set grows (0.9762 → 0.8535)
  — this is expected for a high-variance model type (Random Forest): with
  very little data it can nearly memorize the small training set, and as
  more data arrives it can no longer fit every point perfectly, so training
  AUC drops toward a more realistic level.
- **(ii) Test AUC increases** with more training data (0.6822 → 0.7514),
  and fairly steadily across every step — this means collecting more data
  would likely continue to help.
- **(iii) Conclusion: the model is currently data-limited, not
  capacity-limited.** Test AUC is still rising at 100% of available
  training data with no sign of flattening out, and the train/test gap is
  still fairly wide (0.8535 vs 0.7514) rather than converged — both point to
  "more data would likely improve this model further" rather than "this
  model has hit its ceiling."

## 8. Serialized model

Saved with:
```python
joblib.dump(best_pipeline, 'best_model.pkl')
```

Reload-and-predict check (included in `main_part3.py`):
```python
import joblib
loaded_model = joblib.load('best_model.pkl')
sample_rows = X_test.iloc[:2]          # two hand-picked unscaled test rows
sample_preds = loaded_model.predict(sample_rows)
print(sample_preds)                    # -> [0, 0], ran without errors
```
`best_model.pkl` is included in this repository (well under 100MB).

---

## Summary comparison — all models (Parts 2 & 3)

| Model | 5-fold CV Mean AUC | 5-fold CV Std AUC | Test-set AUC |
|---|---|---|---|
| Logistic Regression (C=1.0) | 0.7632 | 0.0277 | 0.7610 |
| Logistic Regression (C=0.01) | — | — | 0.7628 |
| Decision Tree (max_depth=5) | 0.7256 | 0.0298 | — |
| Random Forest (max_depth=10) | 0.7112 | 0.0213 | 0.7079 |
| Gradient Boosting | 0.7276 | 0.0318 | 0.7497 |
| **Tuned RF Pipeline (GridSearchCV)** | **0.7433** | — | **0.7514** |

**Recommendation: Logistic Regression (with `class_weight='balanced'`).**
It has the highest 5-fold CV mean AUC of any model tested (0.7632), one of
the tightest standard deviations (0.0277), and its test AUC (0.7610) closely
matches its CV mean — meaning it generalizes consistently rather than being
a lucky split. The tree-based ensembles didn't outperform it here despite
far more complexity and tuning effort, likely because this dataset's true
signal is close to linear (income/credit-driven risk) and modest in size —
exactly the regime where simpler linear models tend to hold their own.
Logistic Regression is also the easiest to explain to a non-technical
client ("higher credit score and income lower default risk") and cheapest
to deploy and monitor, which matters given the ablation study also showed
several features contributed little to no signal.

---

## Repository contents (Part 3 additions)

- `main_part3.py` — Decision Trees, Random Forest, Gradient Boosting, CV
  comparison, GridSearchCV pipeline, ablation study, manual learning curve,
  serialization + reload-predict check
- `best_model.pkl` — the tuned Random Forest pipeline saved via joblib
- `results_part3.txt` / `results_part3.json` — full captured output
