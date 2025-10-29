import pandas as pd
import numpy as np
import sys, os

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

seed = 42

# --- Read inputs ---
train_csv_path = sys.argv[1]
model_name = sys.argv[2]

train_df = pd.read_csv(train_csv_path)
X = train_df.set_index("eid").drop(columns=train_df.columns[-1])
y = train_df.set_index("eid")[train_df.columns[-1]]

dataset_name = os.path.splitext(os.path.basename(train_csv_path))[0]

# --- Define classifiers and param grids ---
classifiers = [
    lgb.LGBMClassifier(random_state=seed, class_weight='balanced', verbose=-1, n_jobs=8,max_bin=50),
    LogisticRegression(random_state=seed, max_iter=1000, class_weight='balanced', solver='saga')
]

model_names = ["LightGBM_balanced", "LASSORegression"]

lgb_params = {
    'n_estimators': [50, 100, 200, 500],
    'learning_rate': [0.01, 0.1],
    'max_depth': [1,5,7,10,15],
    'subsample': [0.01, 1],
    'colsample_bytree': [0.01, 0.1, 1.0],
    'reg_alpha': [0,0.1,10],
    'reg_lambda': [0,0.1,10]
}

lasso_params = {
    'C': [0.01, 0.1, 1],
    'penalty' :['l1']
    
}

param_dict = {
    "LightGBM_balanced": lgb_params,
    "LASSORegression": lasso_params
}

model_dict = dict(zip(model_names, classifiers))
clf = model_dict[model_name]
param_grid = param_dict[model_name]

# --- Run GridSearchCV ---
grid_search = GridSearchCV(
    estimator=clf,
    param_grid=param_grid,
    cv=3,
    scoring="recall",
    n_jobs=4
)
grid_search.fit(X, y)

best_model = grid_search.best_estimator_
print("Best params:", grid_search.best_params_)
print("Best CV recall:", grid_search.best_score_)

# --- Feature importances ---
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
elif hasattr(best_model, "coef_"):
    importances = np.abs(best_model.coef_).flatten()
else:
    raise AttributeError("No feature importances found.")

feature_importances = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

feature_importances.to_csv(f"/rds/general/user/meb22/home/UKBB/Prediction_proteomics/New/Feature_selection/Results/Feature_importances_{model_name}_{dataset_name}.csv", index=False)
