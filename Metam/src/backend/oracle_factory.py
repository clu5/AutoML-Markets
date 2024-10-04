import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier, TPOTRegressor


class BaseOracle:
    def __init__(self, model_class, metric_func):
        self.model = model_class(n_estimators=100, random_state=42)
        self.metric_func = metric_func

    def train(self, data, target_col):
        X = data.drop(columns=[target_col])
        y = data[target_col]
        logger.info(f"Training oracle with {X.shape[1]} features and {len(y)} samples")


        # Feature selection
        selector = SelectKBest(f_classif, k=min(20, X.shape[1]))
        X_selected = selector.fit_transform(X, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42
        )

        # Early stopping
        best_score = float("inf") if self.metric_func == mean_absolute_error else 0
        patience = 5
        counter = 0
        for i in range(1, 101):  # Max 100 trees
            self.model.n_estimators = i
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            score = self.metric_func(y_test, y_pred)
            if (self.metric_func == mean_absolute_error and score < best_score) or (
                self.metric_func == accuracy_score and score > best_score
            ):
                best_score = score
                counter = 0
            else:
                counter += 1
            if counter >= patience:
                break

        # Normalize score to 0-1 range
        if self.metric_func == mean_absolute_error:
            max_possible_mae = np.abs(y_test).max()
            normalized_score = 1 - (best_score / max_possible_mae)
        else:
            normalized_score = best_score  # accuracy is already in 0-1 range

        logger.info(f"Normalized score: {normalized_score}")

        return normalized_score


class RegressionOracle(BaseOracle):
    def __init__(self):
        super().__init__(RandomForestRegressor, mean_absolute_error)


class ClassificationOracle(BaseOracle):
    def __init__(self):
        super().__init__(RandomForestClassifier, accuracy_score)


# class OracleFactory:
#    @staticmethod
#    def create(oracle_type):
#        if oracle_type == "regression":
#            return RegressionOracle()
#        elif oracle_type == "classification":
#            return ClassificationOracle()
#        else:
#            raise ValueError(f"Unknown oracle type: {oracle_type}")


class OracleFactory:
    @staticmethod
    def create(oracle_type, config=None):
        if oracle_type == "regression":
            return AutoMLOracle("regression", config)
        elif oracle_type == "classification":
            return AutoMLOracle("classification", config)
        else:
            raise ValueError(f"Unknown oracle type: {oracle_type}")


class AutoMLOracle(BaseOracle):
    def __init__(self, problem_type="classification", config=None):
        self.problem_type = problem_type
        self.config = config or {}
        if problem_type == "classification":
            self.model = TPOTClassifier(
                generations=self.config.get('generations', 5),
                population_size=self.config.get('population_size', 20),
                cv=self.config.get('cv', 5),
                random_state=42,
                verbosity=2,
                n_jobs=self.config.get('n_jobs', -1),
                max_time_mins=self.config.get('max_time_mins', 60),
                max_eval_time_mins=self.config.get('max_eval_time_mins', 5)
            )
        elif problem_type == "regression":
            self.model = TPOTRegressor(
                generations=self.config.get('generations', 5),
                population_size=self.config.get('population_size', 20),
                cv=self.config.get('cv', 5),
                random_state=42,
                verbosity=2,
                n_jobs=self.config.get('n_jobs', -1),
                max_time_mins=self.config.get('max_time_mins', 60),
                max_eval_time_mins=self.config.get('max_eval_time_mins', 5)
            )
        else:
            raise ValueError("problem_type must be 'classification' or 'regression'")

    def train(self, data, target_col):
        X = data.drop(columns=[target_col])
        y = data[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model.fit(X_train, y_train)
        score = self.model.score(X_test, y_test)

        return score
