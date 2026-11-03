"""Approach 2: Discrete Wavelet Transform features + Random Forest."""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from utils import SEED

PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [None, 20],
    "min_samples_leaf": [1, 3],
}


def train_rf(X_train, y_train, cv_splits):
    rf = RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1)
    search = GridSearchCV(rf, PARAM_GRID, cv=cv_splits, scoring="f1_macro", n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"[RF] best params: {search.best_params_}, best CV macro-F1: {search.best_score_:.4f}")
    return search.best_estimator_
