import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score

class BaseOracle:
    def __init__(self, model_class, metric_func):
        self.model = model_class(n_estimators=100, random_state=42)
        self.metric_func = metric_func

    def train(self, data, target_col):
        X = data.drop(columns=[target_col])
        y = data[target_col]

        # Feature selection
        selector = SelectKBest(f_classif, k=min(20, X.shape[1]))
        X_selected = selector.fit_transform(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

        # Early stopping
        best_score = float('inf') if self.metric_func == mean_absolute_error else 0
        patience = 5
        counter = 0
        for i in range(1, 101):  # Max 100 trees
            self.model.n_estimators = i
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            score = self.metric_func(y_test, y_pred)
            if (self.metric_func == mean_absolute_error and score < best_score) or \
               (self.metric_func == accuracy_score and score > best_score):
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

        return normalized_score

class RegressionOracle(BaseOracle):
    def __init__(self):
        super().__init__(RandomForestRegressor, mean_absolute_error)

class ClassificationOracle(BaseOracle):
    def __init__(self):
        super().__init__(RandomForestClassifier, accuracy_score)

class OracleFactory:
    @staticmethod
    def create(oracle_type):
        if oracle_type == "regression":
            return RegressionOracle()
        elif oracle_type == "classification":
            return ClassificationOracle()
        else:
            raise ValueError(f"Unknown oracle type: {oracle_type}")
