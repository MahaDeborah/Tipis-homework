import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


# ===============================
# 1. Load and prepare data
# ===============================
df = pd.read_csv('jamb_exam_results.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')
df = df.drop(columns=['student_id']).fillna(0)

y = df['jamb_score']
X = df.drop(columns=['jamb_score'])

X_full_train, X_test, y_full_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

X_train, X_val, y_train, y_val = train_test_split(
    X_full_train, y_full_train, test_size=0.25, random_state=1
)

dv = DictVectorizer(sparse=True)
X_train_dv = dv.fit_transform(X_train.to_dict(orient='records'))
X_val_dv = dv.transform(X_val.to_dict(orient='records'))

print("\n===== Homework 5 Results =====\n")


# ===============================
# Question 1
# ===============================
dt = DecisionTreeRegressor(max_depth=1, random_state=1)
dt.fit(X_train_dv, y_train)

feature_index = dt.tree_.feature[0]
q1_feature = dv.feature_names_[feature_index]

print(f"Q1. Split feature (Decision Tree): {q1_feature}")


# ===============================
# Question 2
# ===============================
rf = RandomForestRegressor(
    n_estimators=10,
    random_state=1,
    n_jobs=-1
)

rf.fit(X_train_dv, y_train)
rmse_q2 = root_mean_squared_error(y_val, rf.predict(X_val_dv))

print(f"Q2. RMSE (n_estimators=10): {rmse_q2:.2f}")


# ===============================
# Question 3
# ===============================
best_rmse = float('inf')
best_n = None

for n in range(10, 201, 10):
    rf = RandomForestRegressor(
        n_estimators=n,
        random_state=1,
        n_jobs=-1
    )
    rf.fit(X_train_dv, y_train)
    rmse = root_mean_squared_error(y_val, rf.predict(X_val_dv))

    if rmse < best_rmse:
        best_rmse = rmse
        best_n = n

print(f"Q3. RMSE stops improving after n_estimators = {best_n}")


# ===============================
# Question 4
# ===============================
depths = [10, 15, 20, 25]
best_depth = None
best_mean_rmse = float('inf')

for depth in depths:
    rmses = []
    for n in range(10, 201, 20):  # faster: step 20 instead of 10
        rf = RandomForestRegressor(
            n_estimators=n,
            max_depth=depth,
            random_state=1,
            n_jobs=-1
        )
        rf.fit(X_train_dv, y_train)
        rmses.append(root_mean_squared_error(y_val, rf.predict(X_val_dv)))

    mean_rmse = np.mean(rmses)

    if mean_rmse < best_mean_rmse:
        best_mean_rmse = mean_rmse
        best_depth = depth

print(f"Q4. Best max_depth: {best_depth}")


# ===============================
# Question 5
# ===============================
rf = RandomForestRegressor(
    n_estimators=10,
    max_depth=20,
    random_state=1,
    n_jobs=-1
)

rf.fit(X_train_dv, y_train)

importances = rf.feature_importances_
features = dv.feature_names_

most_important = features[np.argmax(importances)]

print(f"Q5. Most important feature: {most_important}")


print("\n===== End =====\n")
