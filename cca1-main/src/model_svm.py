"""Approach 1: RR interval / HRV features + SVM."""
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils import SEED

PARAM_GRID = {
    "svc__C": [1, 10, 100],
    "svc__gamma": ["scale", 0.01, 0.1],
}


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", class_weight="balanced", random_state=SEED, probability=True)),
    ])


def train_svm(X_train, y_train, cv_splits):
    """cv_splits: iterable of (train_idx, val_idx) built from DS1 only, so the
    held-out DS2 test set never influences hyperparameter selection."""
    pipe = build_pipeline()
    search = GridSearchCV(pipe, PARAM_GRID, cv=cv_splits, scoring="f1_macro", n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"[SVM] best params: {search.best_params_}, best CV macro-F1: {search.best_score_:.4f}")
    return search.best_estimator_
