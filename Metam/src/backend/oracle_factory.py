from .classifier_oracle import Oracle as ClassifierOracle
from .regression_oracle import Oracle as RegressionOracle

class OracleFactory:
    @staticmethod
    def create(oracle_type: str):
        if oracle_type == "classification":
            return ClassifierOracle("random forest")
        elif oracle_type == "regression":
            return RegressionOracle("random forest")
        else:
            raise ValueError(f"Unknown oracle type: {oracle_type}")
