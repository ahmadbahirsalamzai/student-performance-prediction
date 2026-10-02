import os
import sys

import joblib
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        joblib.dump(obj, file_path)
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        return joblib.load(file_path)
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}
        fitted = {}

        for name, model in models.items():
            search = GridSearchCV(model, params[name], cv=3, n_jobs=-1)
            search.fit(X_train, y_train)

            # refit on the full training set with the parameters the search picked
            best = model.set_params(**search.best_params_)
            best.fit(X_train, y_train)

            report[name] = r2_score(y_test, best.predict(X_test))
            fitted[name] = best

        return report, fitted
    except Exception as e:
        raise CustomException(e, sys)
