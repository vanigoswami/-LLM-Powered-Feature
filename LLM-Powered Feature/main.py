"""
Part 2 — Predictive Modeling: Regression + Classification
============================================================
Run with:  python3 main.py
Outputs:
  - Printed metrics/tables to stdout (also captured in results.txt)
  - roc_curve.png
  - threshold_sensitivity.png (bonus visual)
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score,
    confusion_matrix, classification_report,
    roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score,
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ------------------------------------------------------------------
# CONFIG  ---  edit these four lines to point at YOUR real cleaned_data.csv
# ------------------------------------------------------------------
DATA_PATH = "cleaned_data.csv"
REG_TARGET = "annual_income"      # y_reg  (continuous)
CLF_TARGET = "loan_default"       # y_clf  (binary, natural column -- NOT median-split)
ORDINAL_COLS = {                  # column -> ordered category list (low -> high)
    "education_level": ["High School", "Bachelors", "Masters", "PhD"],
}
NOMINAL_COLS = ["city"]           # one-hot, no natural order

results = {}  # collected for README auto-fill

# ==================================================================
# 1. LOAD + DEFINE LABELS
# ==================================================================
df = pd.read_csv(DATA_PATH)
print(f"Loaded {DATA_PATH}: {df.shape[0]} rows, {df.shape[1]} columns")

y_reg = df[REG_TARGET].copy()
y_clf = df[CLF_TARGET].copy()
X = df.drop(columns=[REG_TARGET, CLF_TARGET])

results["reg_target"] = REG_TARGET
results["clf_target"] = CLF_TARGET

# ==================================================================
# 2. ENCODING
# ==================================================================
X_enc = X.copy()

# Ordinal: map to integers preserving order
for col, order in ORDINAL_COLS.items():
    mapping = {cat: i for i, cat in enumerate(order)}
    X_enc[col] = X_enc[col].map(mapping)

# Nominal: one-hot, drop first to avoid multicollinearity
X_enc = pd.get_dummies(X_enc, columns=NOMINAL_COLS, drop_first=True)

feature_names = X_enc.columns.tolist()
print(f"\nEncoded feature matrix: {X_enc.shape[1]} columns -> {feature_names}")

# ==================================================================
# 3. LEAK-FREE SPLIT + SCALING
# ==================================================================
X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X_enc, y_reg, y_clf, test_size=0.2, random_state=RANDOM_STATE
)

scaler = StandardScaler()
scaler.fit(X_train)                       # fit ONLY on training data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain size: {X_train.shape[0]}   Test size: {X_test.shape[0]}")

# ==================================================================
# 4. REGRESSION -- Linear Regression
# ==================================================================
print("\n" + "=" * 60)
print("REGRESSION: Linear Regression")
print("=" * 60)

lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_reg_train)
y_pred_reg = lin_reg.predict(X_test_scaled)

mse_lr = mean_squared_error(y_reg_test, y_pred_reg)
r2_lr = r2_score(y_reg_test, y_pred_reg)
print(f"MSE: {mse_lr:,.2f}")
print(f"R^2: {r2_lr:.4f}")

coef_table = pd.DataFrame({
    "feature": feature_names,
    "coefficient": lin_reg.coef_,
}).sort_values("coefficient", key=np.abs, ascending=False)
print("\nCoefficients (sorted by |value|):")
print(coef_table.to_string(index=False))

top3 = coef_table.head(3)
results["mse_lr"] = mse_lr
results["r2_lr"] = r2_lr
results["coef_table"] = coef_table.to_dict(orient="records")
results["top3_features"] = top3.to_dict(orient="records")

# ---- Ridge Regression ----
print("\n" + "-" * 60)
print("REGRESSION: Ridge (alpha=1.0)")
print("-" * 60)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_scaled, y_reg_train)
y_pred_ridge = ridge.predict(X_test_scaled)

mse_ridge = mean_squared_error(y_reg_test, y_pred_ridge)
r2_ridge = r2_score(y_reg_test, y_pred_ridge)
print(f"MSE: {mse_ridge:,.2f}")
print(f"R^2: {r2_ridge:.4f}")

print("\nModel comparison:")
comparison = pd.DataFrame({
    "Model": ["Linear Regression (OLS)", "Ridge (alpha=1.0)"],
    "MSE": [mse_lr, mse_ridge],
    "R2": [r2_lr, r2_ridge],
})
print(comparison.to_string(index=False))

results["mse_ridge"] = mse_ridge
results["r2_ridge"] = r2_ridge

# ==================================================================
# 5. CLASSIFICATION -- Logistic Regression
# ==================================================================
print("\n" + "=" * 60)
print("CLASSIFICATION: Logistic Regression")
print("=" * 60)

before_counts = y_clf_train.value_counts()
before_pct = y_clf_train.value_counts(normalize=True)
print("Class balance BEFORE handling imbalance (train set):")
print(before_counts.to_string())
print(before_pct.round(4).to_string())

minority_pct = before_pct.min()
use_balanced = minority_pct < 0.35
print(f"\nMinority class share: {minority_pct:.2%} -> "
      f"{'imbalance handling required' if use_balanced else 'no handling needed'}")

# Chosen strategy: class_weight='balanced'
# (SMOTE from imblearn is the alternative but requires an extra dependency;
#  class_weight reweights the loss function without synthesizing rows, which
#  keeps every training example real and is simplest to justify to a reviewer.)
class_weight = "balanced" if use_balanced else None
results["minority_pct_before"] = float(minority_pct)
results["imbalance_strategy"] = "class_weight='balanced'" if use_balanced else "none needed"

log_reg = LogisticRegression(max_iter=1000, class_weight=class_weight, C=1.0, random_state=RANDOM_STATE)
log_reg.fit(X_train_scaled, y_clf_train)

# "after" view: effective weighting applied during training (not resampled rows,
# since we used class_weight rather than SMOTE) -- show the weights actually used
if class_weight == "balanced":
    classes = np.array([0, 1])
    n_samples = len(y_clf_train)
    weights = {c: n_samples / (2 * (y_clf_train == c).sum()) for c in classes}
    print(f"\nClass weights applied during training: {weights}")
    results["class_weights_applied"] = {int(k): float(v) for k, v in weights.items()}

y_pred_clf = log_reg.predict(X_test_scaled)
y_proba_clf = log_reg.predict_proba(X_test_scaled)[:, 1]

cm = confusion_matrix(y_clf_test, y_pred_clf)
print("\nConfusion Matrix (rows=actual, cols=predicted):")
print(cm)

report = classification_report(y_clf_test, y_pred_clf, digits=4)
print("\nClassification Report:")
print(report)

auc = roc_auc_score(y_clf_test, y_proba_clf)
print(f"\nAUC: {auc:.4f}")

results["confusion_matrix"] = cm.tolist()
results["classification_report"] = report
results["auc_baseline"] = auc

# ---- ROC curve plot ----
fpr, tpr, _ = roc_curve(y_clf_test, y_proba_clf)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.4f})")
plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Chance")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Logistic Regression (Loan Default)")
plt.annotate(f"AUC = {auc:.4f}", xy=(0.55, 0.15), fontsize=11,
             bbox=dict(boxstyle="round", fc="white", ec="gray"))
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.close()
print("\nSaved roc_curve.png")

# ==================================================================
# 5b. DECISION-THRESHOLD SENSITIVITY
# ==================================================================
print("\n" + "=" * 60)
print("THRESHOLD SENSITIVITY (0.30 -> 0.70)")
print("=" * 60)

thresholds = np.arange(0.30, 0.71, 0.10)
threshold_rows = []
for t in thresholds:
    preds_t = (y_proba_clf >= t).astype(int)
    p = precision_score(y_clf_test, preds_t, zero_division=0)
    r = recall_score(y_clf_test, preds_t, zero_division=0)
    f1 = f1_score(y_clf_test, preds_t, zero_division=0)
    threshold_rows.append({"Threshold": round(t, 2), "Precision": p, "Recall": r, "F1": f1})

threshold_table = pd.DataFrame(threshold_rows)
print(threshold_table.to_string(index=False))

best_row = threshold_table.loc[threshold_table["F1"].idxmax()]
print(f"\nBest F1 at threshold = {best_row['Threshold']:.2f} (F1 = {best_row['F1']:.4f})")

results["threshold_table"] = threshold_table.to_dict(orient="records")
results["best_threshold"] = float(best_row["Threshold"])
results["best_threshold_f1"] = float(best_row["F1"])

# bonus visual
plt.figure(figsize=(7, 5))
plt.plot(threshold_table["Threshold"], threshold_table["Precision"], marker="o", label="Precision")
plt.plot(threshold_table["Threshold"], threshold_table["Recall"], marker="s", label="Recall")
plt.plot(threshold_table["Threshold"], threshold_table["F1"], marker="^", label="F1")
plt.xlabel("Decision Threshold")
plt.ylabel("Score")
plt.title("Precision / Recall / F1 vs Decision Threshold")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("threshold_sensitivity.png", dpi=150)
plt.close()
print("Saved threshold_sensitivity.png")

# ==================================================================
# 6. REGULARIZATION EXPERIMENT (C=0.01 vs C=1.0)
# ==================================================================
print("\n" + "=" * 60)
print("REGULARIZATION: C=0.01 (strong) vs C=1.0 (baseline)")
print("=" * 60)

log_reg_strong = LogisticRegression(max_iter=1000, class_weight=class_weight, C=0.01, random_state=RANDOM_STATE)
log_reg_strong.fit(X_train_scaled, y_clf_train)
y_pred_strong = log_reg_strong.predict(X_test_scaled)
y_proba_strong = log_reg_strong.predict_proba(X_test_scaled)[:, 1]

precision_base = precision_score(y_clf_test, y_pred_clf, zero_division=0)
recall_base = recall_score(y_clf_test, y_pred_clf, zero_division=0)
auc_base = auc

precision_strong = precision_score(y_clf_test, y_pred_strong, zero_division=0)
recall_strong = recall_score(y_clf_test, y_pred_strong, zero_division=0)
auc_strong = roc_auc_score(y_clf_test, y_proba_strong)

reg_comparison = pd.DataFrame({
    "Model": ["C=1.0 (baseline)", "C=0.01 (strong L2)"],
    "Precision": [precision_base, precision_strong],
    "Recall": [recall_base, recall_strong],
    "AUC": [auc_base, auc_strong],
})
print(reg_comparison.to_string(index=False))

results["reg_comparison"] = reg_comparison.to_dict(orient="records")

# ==================================================================
# 7. BOOTSTRAP CONFIDENCE INTERVAL FOR AUC DIFFERENCE
# ==================================================================
print("\n" + "=" * 60)
print("BOOTSTRAP CI: AUC(C=1.0) - AUC(C=0.01)")
print("=" * 60)

n_boot = 500
y_clf_test_arr = y_clf_test.to_numpy()
diffs = np.empty(n_boot)

rng = np.random.RandomState(RANDOM_STATE)
for i in range(n_boot):
    idx = rng.choice(len(y_clf_test_arr), size=len(y_clf_test_arr), replace=True)
    y_sample = y_clf_test_arr[idx]
    # skip degenerate samples with only one class present (AUC undefined)
    if len(np.unique(y_sample)) < 2:
        diffs[i] = np.nan
        continue
    auc_base_i = roc_auc_score(y_sample, y_proba_clf[idx])
    auc_strong_i = roc_auc_score(y_sample, y_proba_strong[idx])
    diffs[i] = auc_base_i - auc_strong_i

diffs = diffs[~np.isnan(diffs)]
mean_diff = diffs.mean()
ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])

print(f"Mean AUC difference (C=1.0 minus C=0.01): {mean_diff:.4f}")
print(f"95% CI: [{ci_low:.4f}, {ci_high:.4f}]")
excludes_zero = (ci_low > 0) or (ci_high < 0)
print(f"CI excludes zero: {excludes_zero}")

results["bootstrap_mean_diff"] = float(mean_diff)
results["bootstrap_ci_low"] = float(ci_low)
results["bootstrap_ci_high"] = float(ci_high)
results["bootstrap_excludes_zero"] = bool(excludes_zero)

# ==================================================================
# SAVE RESULTS FOR README AUTO-FILL
# ==================================================================
with open("results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("\nAll done. Results saved to results.json, plots saved as .png files.")
