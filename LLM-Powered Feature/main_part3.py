"""
Part 3 — Advanced Modeling: Ensembles, Tuning, Full ML Pipeline
==================================================================
Run with:  python3 main_part3.py
(Assumes cleaned_data.csv from Part 1/2 is in this folder, and reuses the
 exact same preprocessing/split/scaling as main.py from Part 2.)

Outputs: printed tables to stdout (also saved to results_part3.txt),
best_model.pkl, results_part3.json
"""

import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ------------------------------------------------------------------
# CONFIG -- identical to Part 2's main.py
# ------------------------------------------------------------------
DATA_PATH = "cleaned_data.csv"
REG_TARGET = "annual_income"
CLF_TARGET = "loan_default"
ORDINAL_COLS = {"education_level": ["High School", "Bachelors", "Masters", "PhD"]}
NOMINAL_COLS = ["city"]

results = {}

# ==================================================================
# REBUILD PART 2 PREPROCESSING (same split/scaling, so results line up)
# ==================================================================
df = pd.read_csv(DATA_PATH)
y_reg = df[REG_TARGET].copy()
y_clf = df[CLF_TARGET].copy()
X = df.drop(columns=[REG_TARGET, CLF_TARGET])

X_enc = X.copy()
for col, order in ORDINAL_COLS.items():
    mapping = {cat: i for i, cat in enumerate(order)}
    X_enc[col] = X_enc[col].map(mapping)
X_enc = pd.get_dummies(X_enc, columns=NOMINAL_COLS, drop_first=True)
feature_names = X_enc.columns.tolist()

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X_enc, y_reg, y_clf, test_size=0.2, random_state=RANDOM_STATE
)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

minority_pct = y_clf_train.value_counts(normalize=True).min()
class_weight = "balanced" if minority_pct < 0.35 else None

print(f"Loaded {DATA_PATH}. Train={X_train.shape[0]}, Test={X_test.shape[0]}, "
      f"Features={feature_names}")

# ==================================================================
# TASK 1: Decision Tree baseline (unconstrained)
# ==================================================================
print("\n" + "=" * 60)
print("TASK 1: Decision Tree — unconstrained (max_depth=None)")
print("=" * 60)

dt_full = DecisionTreeClassifier(random_state=RANDOM_STATE)
dt_full.fit(X_train_scaled, y_clf_train)
train_acc_full = accuracy_score(y_clf_train, dt_full.predict(X_train_scaled))
test_acc_full = accuracy_score(y_clf_test, dt_full.predict(X_test_scaled))
print(f"Train accuracy: {train_acc_full:.4f}   Test accuracy: {test_acc_full:.4f}")
print(f"Train-test gap: {train_acc_full - test_acc_full:.4f}")

results["dt_unconstrained"] = {"train_acc": train_acc_full, "test_acc": test_acc_full}

# ==================================================================
# TASK 2: Controlled Decision Tree
# ==================================================================
print("\n" + "=" * 60)
print("TASK 2: Decision Tree — controlled (max_depth=5, min_samples_split=20)")
print("=" * 60)

dt_ctrl = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=RANDOM_STATE)
dt_ctrl.fit(X_train_scaled, y_clf_train)
train_acc_ctrl = accuracy_score(y_clf_train, dt_ctrl.predict(X_train_scaled))
test_acc_ctrl = accuracy_score(y_clf_test, dt_ctrl.predict(X_test_scaled))
print(f"Train accuracy: {train_acc_ctrl:.4f}   Test accuracy: {test_acc_ctrl:.4f}")
print(f"Train-test gap: {train_acc_ctrl - test_acc_ctrl:.4f}")

results["dt_controlled"] = {"train_acc": train_acc_ctrl, "test_acc": test_acc_ctrl}

# ==================================================================
# TASK 3: Gini vs Entropy
# ==================================================================
print("\n" + "=" * 60)
print("TASK 3: Gini vs Entropy (both max_depth=5)")
print("=" * 60)

dt_gini = DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=RANDOM_STATE)
dt_gini.fit(X_train_scaled, y_clf_train)
test_acc_gini = accuracy_score(y_clf_test, dt_gini.predict(X_test_scaled))

dt_entropy = DecisionTreeClassifier(max_depth=5, criterion="entropy", random_state=RANDOM_STATE)
dt_entropy.fit(X_train_scaled, y_clf_train)
test_acc_entropy = accuracy_score(y_clf_test, dt_entropy.predict(X_test_scaled))

print(f"Gini test accuracy:    {test_acc_gini:.4f}")
print(f"Entropy test accuracy: {test_acc_entropy:.4f}")

results["gini_test_acc"] = test_acc_gini
results["entropy_test_acc"] = test_acc_entropy

# ==================================================================
# TASK 4: Random Forest
# ==================================================================
print("\n" + "=" * 60)
print("TASK 4: Random Forest (n_estimators=100, max_depth=10)")
print("=" * 60)

rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)
rf.fit(X_train_scaled, y_clf_train)
rf_train_acc = accuracy_score(y_clf_train, rf.predict(X_train_scaled))
rf_test_acc = accuracy_score(y_clf_test, rf.predict(X_test_scaled))
rf_auc = roc_auc_score(y_clf_test, rf.predict_proba(X_test_scaled)[:, 1])
print(f"Train accuracy: {rf_train_acc:.4f}   Test accuracy: {rf_test_acc:.4f}   Test AUC: {rf_auc:.4f}")

importances = pd.DataFrame({
    "feature": feature_names,
    "importance": rf.feature_importances_,
}).sort_values("importance", ascending=False)
print("\nTop 5 features by importance:")
print(importances.head(5).to_string(index=False))

results["rf"] = {"train_acc": rf_train_acc, "test_acc": rf_test_acc, "auc": rf_auc}
results["rf_importances"] = importances.to_dict(orient="records")

# ==================================================================
# TASK 4a: Gradient Boosting
# ==================================================================
print("\n" + "=" * 60)
print("TASK 4a: Gradient Boosting (n_estimators=100, lr=0.1, max_depth=3)")
print("=" * 60)

gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=RANDOM_STATE)
gb.fit(X_train_scaled, y_clf_train)
gb_train_acc = accuracy_score(y_clf_train, gb.predict(X_train_scaled))
gb_test_acc = accuracy_score(y_clf_test, gb.predict(X_test_scaled))
gb_auc = roc_auc_score(y_clf_test, gb.predict_proba(X_test_scaled)[:, 1])
print(f"Train accuracy: {gb_train_acc:.4f}   Test accuracy: {gb_test_acc:.4f}   Test AUC: {gb_auc:.4f}")

results["gb"] = {"train_acc": gb_train_acc, "test_acc": gb_test_acc, "auc": gb_auc}

# ==================================================================
# TASK 4b: Feature ablation study
# ==================================================================
print("\n" + "=" * 60)
print("TASK 4b: Feature ablation — remove 5 lowest-importance features")
print("=" * 60)

lowest5 = importances.tail(5)["feature"].tolist()
print(f"5 lowest-importance features: {lowest5}")

keep_idx = [i for i, f in enumerate(feature_names) if f not in lowest5]
X_train_reduced = X_train_scaled[:, keep_idx]
X_test_reduced = X_test_scaled[:, keep_idx]

rf_reduced = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)
rf_reduced.fit(X_train_reduced, y_clf_train)
auc_full = rf_auc
auc_reduced = roc_auc_score(y_clf_test, rf_reduced.predict_proba(X_test_reduced)[:, 1])

print(f"Full model AUC (all {len(feature_names)} features):     {auc_full:.4f}")
print(f"Reduced model AUC ({len(keep_idx)} features):            {auc_reduced:.4f}")
print(f"Difference (full - reduced): {auc_full - auc_reduced:.4f}")

results["ablation"] = {
    "removed_features": lowest5,
    "auc_full": auc_full,
    "auc_reduced": auc_reduced,
}

# ==================================================================
# TASK 5: Cross-validated comparison
# ==================================================================
print("\n" + "=" * 60)
print("TASK 5: 5-fold cross-validated AUC comparison")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

log_reg = LogisticRegression(max_iter=1000, class_weight=class_weight, C=1.0, random_state=RANDOM_STATE)

cv_models = {
    "Logistic Regression": log_reg,
    "Decision Tree (max_depth=5)": dt_ctrl,
    "Random Forest": rf,
    "Gradient Boosting": gb,
}

cv_results = []
for name, model in cv_models.items():
    scores = cross_val_score(model, X_train_scaled, y_clf_train, cv=cv, scoring="roc_auc")
    cv_results.append({"Model": name, "Mean AUC": scores.mean(), "Std AUC": scores.std()})
    print(f"{name:30s}  mean AUC = {scores.mean():.4f}   std = {scores.std():.4f}")

cv_table = pd.DataFrame(cv_results)
results["cv_comparison"] = cv_table.to_dict(orient="records")

# ==================================================================
# TASK 6: GridSearchCV hyperparameter tuning
# ==================================================================
print("\n" + "=" * 60)
print("TASK 6: GridSearchCV — Random Forest tuning")
print("=" * 60)

pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    StandardScaler(),
    RandomForestClassifier(random_state=RANDOM_STATE),
)

param_grid = {
    "randomforestclassifier__n_estimators": [50, 100, 200],
    "randomforestclassifier__max_depth": [5, 10, None],
    "randomforestclassifier__min_samples_leaf": [1, 5],
}

n_configs = 1
for v in param_grid.values():
    n_configs *= len(v)
print(f"Total parameter combinations: {n_configs}  x  5 folds = {n_configs * 5} model fits")

grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring="roc_auc", n_jobs=-1)
grid_search.fit(X_train, y_clf_train)   # unscaled -- pipeline handles it

print(f"\nBest params: {grid_search.best_params_}")
print(f"Best CV AUC: {grid_search.best_score_:.4f}")

best_pipeline = grid_search.best_estimator_
best_test_auc = roc_auc_score(y_clf_test, best_pipeline.predict_proba(X_test)[:, 1])
print(f"Best pipeline test AUC (held-out test set): {best_test_auc:.4f}")

results["grid_search"] = {
    "n_configs": n_configs,
    "total_fits": n_configs * 5,
    "best_params": {k: (v if v is not None else "None") for k, v in grid_search.best_params_.items()},
    "best_cv_auc": grid_search.best_score_,
    "best_test_auc": best_test_auc,
}

# ==================================================================
# TASK 7: Manual learning curve
# ==================================================================
print("\n" + "=" * 60)
print("TASK 7: Manual learning curve (best pipeline, 20%-100% of training data)")
print("=" * 60)

fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
learning_rows = []
for f in fractions:
    n_rows = int(f * len(X_train))
    X_sub = X_train.iloc[:n_rows]
    y_sub = y_clf_train.iloc[:n_rows]

    # clone-fresh pipeline with the SAME best hyperparameters
    lc_pipeline = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        RandomForestClassifier(random_state=RANDOM_STATE, **{
            k.split("__")[1]: v for k, v in grid_search.best_params_.items()
        }),
    )
    lc_pipeline.fit(X_sub, y_sub)

    train_auc = roc_auc_score(y_sub, lc_pipeline.predict_proba(X_sub)[:, 1])
    test_auc = roc_auc_score(y_clf_test, lc_pipeline.predict_proba(X_test)[:, 1])
    learning_rows.append({"Training fraction": f, "Training AUC": train_auc, "Test AUC": test_auc})

learning_table = pd.DataFrame(learning_rows)
print(learning_table.to_string(index=False))

results["learning_curve"] = learning_table.to_dict(orient="records")

# ==================================================================
# TASK 8: Serialize best model
# ==================================================================
print("\n" + "=" * 60)
print("TASK 8: Serialize best model to best_model.pkl")
print("=" * 60)

joblib.dump(best_pipeline, "best_model.pkl")
print("Saved best_model.pkl")

# reload-and-predict demo
loaded_model = joblib.load("best_model.pkl")
sample_rows = X_test.iloc[:2]
sample_preds = loaded_model.predict(sample_rows)
print(f"Reload-and-predict check on 2 hand-picked test rows: predictions = {sample_preds.tolist()}")

results["reload_predict_sample"] = sample_preds.tolist()

# ==================================================================
# SAVE ALL RESULTS
# ==================================================================
with open("results_part3.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\nAll done. Results saved to results_part3.json, model saved to best_model.pkl.")
