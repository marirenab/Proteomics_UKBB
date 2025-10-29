import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt
import lightgbm as lgb
import joblib
import random
import argparse
import os
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
from sklearn.base import BaseEstimator, ClassifierMixin, clone

# -------------------
# Parse arguments
# -------------------
parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Input CSV file")
parser.add_argument("--input_features", required=True, help="Input features list")
parser.add_argument("--input_clf", required=True, help="Classifier name (must be one of model_dict keys)")
args = parser.parse_args()

seed = 42
np.random.seed(seed)

# -------------------
# Load data
# -------------------
proteomics_df = pd.read_csv(args.input)
X = proteomics_df.drop(columns='Diagnosis')
y = proteomics_df['Diagnosis']
X.set_index("eid", inplace=True)

# -------------------
# Define param grids
# -------------------
lgb_params_prefixed = {
    'clf__n_estimators': [50, 100, 200, 500],
    'clf__learning_rate': [0.01, 0.1],
    'clf__max_depth': [1, 5, 7, 10, 15],
    'clf__subsample': [0.01, 1],
    'clf__colsample_bytree': [0.01, 0.1, 1.0],
    'clf__reg_alpha': [0, 0.1, 10],
    'clf__reg_lambda': [0, 0.1, 10]
}

lasso_grid_prefixed = [
    {'clf__penalty': ['l1'], 'clf__C': [0.01, 0.1, 1, 10], 'clf__solver': ['saga']},
]

# -------------------
# Define models
# -------------------
model_names = (
    "LightGBM_balanced",
    "LASSORegression"
)

classifiers = [
    lgb.LGBMClassifier(random_state=seed, class_weight='balanced', verbose=-1, n_jobs=4,max_bin=50),
    LogisticRegression(random_state=seed, max_iter=1000, class_weight='balanced',n_jobs=5)
]

param_spaces = [
    lgb_params_prefixed,
    lasso_grid_prefixed
]

model_dict = {name: clf for name, clf in zip(model_names, classifiers)}
param_dict = {name: grid for name, grid in zip(model_names, param_spaces)}

# -------------------
# Pick classifier
# -------------------
model_name = args.input_clf
if model_name not in model_dict:
    raise ValueError(f"Invalid model name: {model_name}. Must be one of: {list(model_dict.keys())}")

clf = model_dict[model_name]
param_grid_prefixed = param_dict[model_name]

# -------------------
# Nested GridSearch wrapper
# -------------------
class NestedGridSearchWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, base_pipeline, param_grid, cv_inner=2, scoring='recall', n_jobs=5):
        self.base_pipeline = base_pipeline
        self.param_grid = param_grid
        self.cv_inner = cv_inner
        self.scoring = scoring
        self.n_jobs = n_jobs
    
    def fit(self, X, y):
        self.grid_search_ = GridSearchCV(
            estimator=clone(self.base_pipeline),
            param_grid=self.param_grid,
            cv=self.cv_inner,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            refit=True
        )
        self.grid_search_.fit(X, y)
        self.best_estimator_ = self.grid_search_.best_estimator_
        self.classes_ = self.best_estimator_.classes_ 
        return self
    
    def predict(self, X):
        return self.best_estimator_.predict(X)
    
    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)
    
    def score(self, X, y):
        return self.best_estimator_.score(X, y)
    
    @property
    def feature_importances_(self):
        clf = self.best_estimator_.named_steps['clf']
        if hasattr(clf, 'feature_importances_'):
            return clf.feature_importances_
        elif hasattr(clf, 'coef_'):
            return np.abs(clf.coef_).flatten()
        else:
            raise AttributeError("Classifier has no feature_importances_ or coef_")

# -------------------
# Base pipeline
# -------------------
base_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', clf)
])

nested_estimator = NestedGridSearchWrapper(
    base_pipeline=base_pipeline,
    param_grid=param_grid_prefixed,
    cv_inner=3,
    scoring='recall',
    n_jobs=5
)

# -------------------
# RFECV feature selection
# -------------------
rfecv = RFECV(
    estimator=nested_estimator,
    step=1,
    cv=5,
    scoring='recall',
    importance_getter=lambda clf: clf.feature_importances_,
    n_jobs=3
)

with open(args.input_features) as f:
    features = [line.strip() for line in f if line.strip()]

sel = rfecv.fit(X[features], y)

feature_info = pd.DataFrame({
    'Feature': features,
    'Selected': sel.support_,
    'Ranking': sel.ranking_
})

# -------------------
# Save results
# -------------------
data_name = os.path.splitext(os.path.basename(args.input))[0]
out_file = f"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Feature_selection/Results/{model_name}_{data_name}_rfecv.csv"
feature_info.to_csv(out_file, index=False)
print(f"Saved feature info to {out_file}")
