import pandas as pd
import numpy as np
import seaborn as sns 
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
import argparse
import sys
import os
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import BalancedBaggingClassifier
from imblearn.ensemble import RUSBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.ensemble import EasyEnsembleClassifier
import lightgbm as lgb
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer , KNNImputer
from sklearn.model_selection import GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
import math
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score, recall_score, roc_auc_score, precision_score ,  accuracy_score, f1_score

seed = 42
np.random.seed(seed)

train_csv = sys.argv[1]
train_csv_baseline = sys.argv[2]

print(f"train_csv passed to Python: {train_csv}")
import os
print(f"Exists? {os.path.exists(train_csv)}")

df = pd.read_csv(train_csv)


df_baseline = pd.read_csv(train_csv_baseline)

dataset_name = os.path.splitext(os.path.basename(train_csv))[0]

all_protein_cols = np.loadtxt("/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/columns_30pd.txt", dtype=str).tolist()

X = df.set_index("eid")[all_protein_cols]
y = df.set_index("eid")["Diagnosis"]


X_baseline = df_baseline.set_index("eid")[all_protein_cols]
y_baseline = df_baseline.set_index("eid")["Diagnosis"]

y_weights = pd.concat([y_baseline, y]) 


class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y), y=y)
class_weights_dict = {0: class_weights[0], 1: class_weights[1]}
class_weights_dict_sqrt = {0: math.sqrt(class_weights[0]), 1: math.sqrt(class_weights[1])}
class_weights_dict_sqr = {0: (class_weights[0])**2, 1: (class_weights[1])**2}


n_pos = (y == 1).sum()  # count of positive samples
n_neg = (y == 0).sum()  # count of negative samples

ratio_pos_to_neg = n_neg / n_pos
ratio_pos_to_neg_sqrt = math.sqrt(ratio_pos_to_neg)
ratio_pos_to_neg_sqr = ratio_pos_to_neg **2

# Define the hyperparameter grid for Random Forest
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [1, 5, 10],
    'min_samples_split': [2,10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [0.1, 0.3, 1.0]
}

# Define the hyperparameter grid for XGBoost
xgb_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5],
    'max_depth': [1, 5,10, 15],
    'subsample': [ 0.3, 0.5,  1.0],
    'colsample_bytree': [0.1, 1.0],
    'gamma': [0, 0.2,  1.0],
    'reg_alpha': [0, 0.1, 1.0],
    'reg_lambda': [0, 0.1,  1.0]
}

adaboost_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.5]
}

# Define the hyperparameter grid for Logistic Regression
lr_grid = [
    {'penalty': ['l1'], 'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'saga']},
    {'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10], 'solver': ['lbfgs', 'liblinear', 'saga']},
    {'penalty': ['elasticnet'], 'C': [0.01, 0.1, 1, 10], 'solver': ['saga'], 'l1_ratio': [0, 0.25, 0.5, 0.75, 1]},
]

lgb_params = {
    'n_estimators': [50, 100,200,500],
    'learning_rate': [0.01, 0.1],
    'max_depth': [1,5,7,10,15],
    'subsample': [0.01, 1],
    'colsample_bytree': [0.01, 0.1, 1.0],
    'reg_alpha': [0,0.1, 10],
    'reg_lambda': [0, 0.1, 10]
}


# Random Forest
rf_params_prefixed = {
    'clf__n_estimators': [50, 100, 200],
    'clf__max_depth': [1, 5, 10],
    'clf__min_samples_split': [2, 10],
    'clf__min_samples_leaf': [1, 2, 4],
    'clf__max_features': [0.1, 0.3, 1.0]
}

# XGBoost
xgb_params_prefixed = {
    'clf__n_estimators': [50, 100, 200],
    'clf__learning_rate': [0.01, 0.1, 0.5],
    'clf__max_depth': [1, 5, 10, 15],
    'clf__subsample': [0.3, 0.5, 1.0],
    'clf__colsample_bytree': [0.1, 1.0],
    'clf__gamma': [0, 0.2, 1.0],
    'clf__reg_alpha': [0, 0.1, 1.0],
    'clf__reg_lambda': [0, 0.1, 1.0]
}

# AdaBoost
adaboost_params_prefixed = {
    'clf__n_estimators': [50, 100, 200],
    'clf__learning_rate': [0.01, 0.1, 0.5]
}

# Logistic Regression
lr_grid_prefixed = [
    {'clf__penalty': ['l1'], 'clf__C': [0.01, 0.1, 1, 10], 'clf__solver': ['liblinear', 'saga']},
    {'clf__penalty': ['l2'], 'clf__C': [0.01, 0.1, 1, 10], 'clf__solver': ['lbfgs', 'liblinear', 'saga']},
    {'clf__penalty': ['elasticnet'], 'clf__C': [0.01, 0.1, 1, 10], 'clf__solver': ['saga'], 'clf__l1_ratio': [0, 0.25, 0.5, 0.75, 1]},
]

# LightGBM
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
    {'clf__penalty': ['elasticnet'], 'clf__C': [0.01, 0.1, 1, 10], 'clf__solver': ['saga'], 'clf__l1_ratio': [1]},
]


protein_cols = []


def perform_kfold_evaluation(estimator, X, X_baseline, y, kf, params_space,metric_optimising):
    metrics_dict = {
        'Fold_no': [], 'Accuracy': [], 'F1 Score': [], 'Precision': [], 'Recall': [],
        'Specificity': [], 'PR AUC': [], 'ROC AUC': [], 'Model_name': [], 'TN': [], 'TP': [], 'FP': [], 'FN': []
    }
    feature_importances_list= []

    for fold_idx, (train_index, val_index) in enumerate(kf.split(X, y), 1):
        print(f"Starting Fold {fold_idx}/{kf.get_n_splits()}")
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        X_train = pd.concat([X_train, X_baseline])
        y_train = pd.concat([y_train, y_baseline])

        
        estimator_grid = GridSearchCV(estimator, params_space, cv=3, scoring=metric_optimising, n_jobs=20, refit=True)
        estimator_grid.fit(X_train, y_train)
        best_estimator = estimator_grid.best_estimator_

        y_pred = best_estimator.predict(X_val)
        y_prob = best_estimator.predict_proba(X_val)[:,1]

        report = classification_report(y_val, y_pred, output_dict=True, zero_division=1)
        cm = confusion_matrix(y_val, y_pred)
        accuracy = accuracy_score(y_val, y_pred)
        f1_score_ = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred, pos_label=1)
        specificity = recall_score(y_val, y_pred, pos_label=0)
        roc_auc = roc_auc_score(y_val, y_prob)
        pr_auc = average_precision_score(y_val, y_prob)
        TN, FP, FN, TP = cm.ravel()

        clf = best_estimator.named_steps['clf']
        metrics_dict['Fold_no'].append(f'Fold {fold_idx}')
        metrics_dict['Accuracy'].append(accuracy)
        metrics_dict['F1 Score'].append(f1_score_)
        metrics_dict['Precision'].append(precision)
        metrics_dict['Recall'].append(recall)
        metrics_dict['Specificity'].append(specificity)
        metrics_dict['PR AUC'].append(pr_auc)
        metrics_dict['ROC AUC'].append(roc_auc)
        metrics_dict['Model_name'].append(clf.__class__.__name__)
        metrics_dict['TN'].append(TN)
        metrics_dict['TP'].append(TP)
        metrics_dict['FP'].append(FP)
        metrics_dict['FN'].append(FN)

        # Feature importances
        feature_names = X.columns
        if hasattr(clf, 'feature_importances_'):
            importances = clf.feature_importances_
        elif hasattr(clf, 'coef_'):
            importances = np.abs(clf.coef_).flatten()
        else:
            importances = np.zeros(len(feature_names))

        for feat_name, imp in zip(feature_names, importances):
            feature_importances_list.append({'Fold': fold_idx, 'Feature': feat_name, 'Importance': imp})

    return pd.DataFrame(metrics_dict), pd.DataFrame(feature_importances_list)

kf = StratifiedKFold(n_splits=5)

param_spaces = [
    rf_params_prefixed, 
    rf_params_prefixed,
    rf_params_prefixed,
    rf_params_prefixed,
    xgb_params_prefixed,
    xgb_params_prefixed,
    xgb_params_prefixed,
    lgb_params_prefixed,
    lgb_params_prefixed,
    lgb_params_prefixed,
    rf_params_prefixed, 
    adaboost_params_prefixed,
    lr_grid_prefixed,
    lr_grid_prefixed,
    lr_grid_prefixed,
    lasso_grid_prefixed
]

model_names=(
    "RandomForest_sqrt",
    "RandomForest_sqr",
    "RandomForest_balanced",
    "RandomForest_subsample",
    "XGBoost_sqrt",
    "XGBoost_sqr",
    "XGBoost",
    "LightGBM_sqrt",
    "LightGBM_sqr",
    "LightGBM_balanced",
    "BalancedRandomForest",
    "RUSBoost",
    "LogisticRegression_sqrt",
    "LogisticRegression_sqr",
    "LogisticRegression",
    "LASSORegression"
   
)


classifiers = [
    RandomForestClassifier(random_state=seed, class_weight=class_weights_dict_sqrt),
    RandomForestClassifier(random_state=seed, class_weight=class_weights_dict_sqr),
    RandomForestClassifier(random_state=seed, class_weight='balanced'),
    RandomForestClassifier(random_state=seed, class_weight='balanced_subsample'),
    xgb.XGBClassifier(random_state=seed, scale_pos_weight=ratio_pos_to_neg_sqrt, n_jobs=4),
    xgb.XGBClassifier(random_state=seed, scale_pos_weight=ratio_pos_to_neg_sqr, n_jobs=4),
    xgb.XGBClassifier(random_state=seed, scale_pos_weight=ratio_pos_to_neg, n_jobs=4),
    lgb.LGBMClassifier(random_state=seed, class_weight=class_weights_dict_sqrt, verbose=-1, n_jobs=4),
    lgb.LGBMClassifier(random_state=seed, class_weight=class_weights_dict_sqr, verbose=-1, n_jobs=4),
    lgb.LGBMClassifier(random_state=seed, class_weight='balanced', verbose=-1, n_jobs=4),
    BalancedRandomForestClassifier(random_state=seed, n_jobs=8),
    RUSBoostClassifier(random_state=seed, algorithm='SAMME'), 
    LogisticRegression(random_state=seed, max_iter=1000, class_weight=class_weights_dict_sqrt),
    LogisticRegression(random_state=seed, max_iter=1000, class_weight=class_weights_dict_sqr),
    LogisticRegression(random_state=seed, max_iter=1000, class_weight='balanced'),
    LogisticRegression(random_state=seed, max_iter=1000, class_weight='balanced')
]

model_dict = {name: clf for name, clf in zip(model_names, classifiers)}

model_name = sys.argv[3]
print(model_name)

protein_file = sys.argv[4]


with open(protein_file, "r") as f:
    protein_cols_list = [line.strip() for line in f if line.strip()]


filename = os.path.basename(protein_file)         
name_no_ext = os.path.splitext(filename)[0]        
prefix = name_no_ext.split("_list")[0]  
print(prefix)
if model_name not in model_dict:
    raise ValueError(f"Invalid model name: {model_name}")

# Retrieve classifier and parameter space
classifier = model_dict[model_name]
params_space = param_spaces[list(model_dict.keys()).index(model_name)]

imputers = [
    KNNImputer(n_neighbors=5),
    KNNImputer(n_neighbors=10),
    SimpleImputer(strategy='median')
]

imputers_names = ['KNN5','KNN10','SIMPLEMEDIAN']

# create dictionary mapping names to imputers
imputer_dict = {name: imp for name, imp in zip(imputers_names, imputers)}

# pick imputer name from command-line argument
imputer_name = sys.argv[5]

if imputer_name not in imputer_dict:
    raise ValueError(f"Invalid imputer name: {imputer_name}")
    
imputer = imputer_dict[imputer_name]



# allowed metrics
allowed_metrics = ['f1', 'recall', 'average_precision', 'balanced_accuracy', 'precision']

# validate metric argument from sys.argv

metric_optimising = sys.argv[6]

if metric_optimising not in allowed_metrics:
    raise ValueError(f"Invalid metric '{metric_optimising}'. Must be one of: {allowed_metrics}")

print(f"Optimising metric: {metric_optimising}")


base_pipeline = Pipeline([
    ('imputer', imputer),
    ('scaler', StandardScaler()),
    ('clf', classifier)
])



df_metrics ,feature_importances= perform_kfold_evaluation(base_pipeline, X[protein_cols_list], X_baseline[protein_cols_list], y, kf,params_space,metric_optimising)
df_metrics.to_csv(f"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results_trainedbaselinemoreweighted/Predictions{model_name}_{prefix}_{imputer_name}_{metric_optimising}_{dataset_name}_metrics.csv", index=False)
feature_importances.to_csv(f"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Model/Results_trainedbaselinemoreweighted/Predictions{model_name}_{prefix}_{imputer_name}_{metric_optimising}_{dataset_name}_Featureimportance.csv", index=False)