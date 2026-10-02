import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object

MIN_ACCEPTABLE_R2 = 0.6

MODELS = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Linear Regression": LinearRegression(),
    "XGBRegressor": XGBRegressor(random_state=42),
    "CatBoosting Regressor": CatBoostRegressor(verbose=False, allow_writing_files=False, random_state=42),
    "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
}

PARAMS = {
    "Random Forest": {"n_estimators": [8, 16, 32, 64, 128, 256]},
    "Decision Tree": {"criterion": ["squared_error", "absolute_error", "poisson"]},
    "Gradient Boosting": {
        "learning_rate": [0.1, 0.05, 0.01],
        "subsample": [0.7, 0.8, 0.9],
        "n_estimators": [32, 64, 128],
    },
    "Linear Regression": {},
    "XGBRegressor": {"learning_rate": [0.1, 0.05, 0.01], "n_estimators": [32, 64, 128]},
    "CatBoosting Regressor": {
        "depth": [6, 8, 10],
        "learning_rate": [0.1, 0.05, 0.01],
        "iterations": [30, 50, 100],
    },
    "AdaBoost Regressor": {"learning_rate": [0.5, 0.1, 0.01], "n_estimators": [32, 64, 128]},
}


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            report, fitted = evaluate_models(X_train, y_train, X_test, y_test, MODELS, PARAMS)

            for name, score in sorted(report.items(), key=lambda item: item[1], reverse=True):
                print("{:<25} test R2 = {:.4f}".format(name, score))

            best_model_name = max(report, key=report.get)
            best_model_score = report[best_model_name]

            if best_model_score < MIN_ACCEPTABLE_R2:
                raise CustomException("No model reached the minimum acceptable R2", sys)

            logging.info(f"Best model: {best_model_name} with test R2 {best_model_score:.4f}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=fitted[best_model_name],
            )

            return best_model_name, r2_score(y_test, fitted[best_model_name].predict(X_test))
        except Exception as e:
            raise CustomException(e, sys)
