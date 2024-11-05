from typing import Dict, Any
import optuna
from mlflow import MlflowClient

from steps.data_ingestion.data_module import Digit_Data_Module
config_SVM = dict(
    library = "sklearn",
    model_class = "SVC",
    model_params = dict(
        C = dict(
            param_type = "float",
            param_range = [0.1, 2.0],
        ),
        max_iter = dict(
            param_type = "int",
            param_range = [10, 20]
        ),
        kernel = dict(
            param_type = "categorical",
            param_range = ["linear", "poly", "rbf"]
        )
    )
)
class Objective:
    def __init__(self, config: Dict[str, Any], data_module: Digit_Data_Module, client: MlflowClient = None):
        self.config = config
        self.data_module = data_module
        self.client = client
    def __call__(self, trial: optuna.trial.Trial):
        pass
