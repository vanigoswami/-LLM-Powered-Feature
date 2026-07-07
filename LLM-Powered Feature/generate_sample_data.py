"""
Generates a synthetic 'cleaned_data.csv' standing in for the output of Part 1.

Dataset story: a bank's customer book. We predict:
  - annual_income (regression target, continuous)
  - loan_default (classification target, binary, naturally imbalanced ~15%)

If you have your own cleaned_data.csv from Part 1, just drop it in this folder
(overwriting this one) and update the CONFIG block at the top of main.py to
point at your actual column names.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 2000

age = np.random.randint(21, 65, n)
employment_years = np.clip(age - 21 - np.random.randint(0, 8, n), 0, None)
education_level = np.random.choice(
    ["High School", "Bachelors", "Masters", "PhD"], n, p=[0.35, 0.4, 0.2, 0.05]
)
city = np.random.choice(["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"], n)
num_dependents = np.random.poisson(1.1, n)
credit_score = np.clip(np.random.normal(650, 80, n), 300, 850).round().astype(int)

edu_bonus = pd.Series(education_level).map(
    {"High School": 0, "Bachelors": 15000, "Masters": 28000, "PhD": 40000}
).values

annual_income = (
    28000
    + employment_years * 1800
    + edu_bonus
    + (credit_score - 650) * 40
    - num_dependents * 1200
    + np.random.normal(0, 9000, n)
).round(2)
annual_income = np.clip(annual_income, 18000, None)

# Natural imbalanced binary target: loan default (~15% positive), driven by
# credit score / income / dependents but with real noise so it isn't trivially
# separable — independent target from annual_income, not derived from its median.
default_logit = (
    -1.6
    - 0.008 * (credit_score - 650)
    - 0.000012 * (annual_income - 60000)
    + 0.30 * num_dependents
    - 0.04 * employment_years
)
default_prob = 1 / (1 + np.exp(-default_logit))
loan_default = np.random.binomial(1, default_prob)

df = pd.DataFrame({
    "age": age,
    "employment_years": employment_years,
    "education_level": education_level,
    "city": city,
    "num_dependents": num_dependents,
    "credit_score": credit_score,
    "annual_income": annual_income,
    "loan_default": loan_default,
})

df.to_csv("cleaned_data.csv", index=False)
print("Wrote cleaned_data.csv:", df.shape)
print(df["loan_default"].value_counts(normalize=True))
