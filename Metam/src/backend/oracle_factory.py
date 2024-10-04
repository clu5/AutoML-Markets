import logging

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (SelectKBest, f_classif,
                                       mutual_info_classif)
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             r2_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tpot import TPOTClassifier, TPOTRegressor

logger = logging.getLogger(__name__)


class BaseOracle:
    def __init__(self, model_class, metric_func, config):
        self.model = model_class(
            n_estimators=config.get("max_trees", 100),
            random_state=config.get("random_state", 42),
        )
        self.metric_func = metric_func
        self.config = config

    def train(self, data, target_col):
        X = data.drop(columns=[target_col])
        y = data[target_col]
        logger.info(f"Training oracle with {X.shape[1]} features and {len(y)} samples")

        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Feature selection
        if self.config.get("use_best_features", True):
            selector = SelectKBest(
                (
                    mutual_info_classif
                    if isinstance(self.model, RandomForestClassifier)
                    else f_classif
                ),
                k=min(self.config.get("max_features", 20), X.shape[1]),
            )
            X_selected = selector.fit_transform(X_scaled, y)
        else:
            X_selected = X_scaled

        X_train, X_test, y_train, y_test = train_test_split(
            X_selected,
            y,
            test_size=self.config.get("test_size", 0.2),
            random_state=self.config.get("random_state", 42),
        )

        # Early stopping
        best_score = float("-inf")
        patience = 5
        counter = 0
        for i in range(1, self.config.get("max_trees", 100) + 1):
            self.model.n_estimators = i
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            score = self.metric_func(y_test, y_pred)
            if score > best_score:
                best_score = score
                counter = 0
            else:
                counter += 1
            if counter >= patience:
                break

        logger.info(f"Best score: {best_score}")
        return best_score


class RegressionOracle(BaseOracle):
    def __init__(self, config):
        super().__init__(RandomForestRegressor, r2_score, config)


class ClassificationOracle(BaseOracle):
    def __init__(self, config):
        super().__init__(RandomForestClassifier, f1_score, config)


class OracleFactory:
    @staticmethod
    def create(oracle_type, config=None):
        if oracle_type == "regression":
            return AutoMLOracle("regression", config)
        elif oracle_type == "classification":
            return AutoMLOracle("classification", config)
        else:
            raise ValueError(f"Unknown oracle type: {oracle_type}")


class AutoMLOracle:
    def __init__(self, problem_type="classification", config=None):
        self.problem_type = problem_type
        self.config = config or {}
        if problem_type == "classification":
            self.model = TPOTClassifier(
                generations=self.config.get("generations", 5),
                population_size=self.config.get("population_size", 20),
                cv=self.config.get("cv", 5),
                random_state=self.config.get("random_state", 42),
                verbosity=2,
                n_jobs=self.config.get("n_jobs", -1),
                max_time_mins=self.config.get("max_time_mins", 60),
                max_eval_time_mins=self.config.get("max_eval_time_mins", 5),
                config_dict="TPOT light",
            )
            self.metric_func = f1_score
        elif problem_type == "regression":
            self.model = TPOTRegressor(
                generations=self.config.get("generations", 5),
                population_size=self.config.get("population_size", 20),
                cv=self.config.get("cv", 5),
                random_state=self.config.get("random_state", 42),
                verbosity=2,
                n_jobs=self.config.get("n_jobs", -1),
                max_time_mins=self.config.get("max_time_mins", 60),
                max_eval_time_mins=self.config.get("max_eval_time_mins", 5),
                config_dict="TPOT light",
            )
            self.metric_func = r2_score
        else:
            raise ValueError("problem_type must be 'classification' or 'regression'")

    def train(self, data, target_col):
        X = data.drop(columns=[target_col])
        y = data[target_col]

        logger.info(
            f"Training AutoML oracle with {X.shape[1]} features and {len(y)} samples"
        )

        # Feature scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Feature selection
        if self.config.get("use_best_features", True):
            selector = SelectKBest(
                (
                    mutual_info_classif
                    if self.problem_type == "classification"
                    else f_classif
                ),
                k=min(self.config.get("max_features", 20), X.shape[1]),
            )
            X_selected = selector.fit_transform(X_scaled, y)
        else:
            X_selected = X_scaled

        X_train, X_test, y_train, y_test = train_test_split(
            X_selected,
            y,
            test_size=self.config.get("test_size", 0.2),
            random_state=self.config.get("random_state", 42),
        )

        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        score = self.metric_func(y_test, y_pred)

        logger.info(f"AutoML training completed. Score: {score:.4f}")

        return score
