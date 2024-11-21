import optuna
from mlflow import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from typing import Dict, Any

from core import Digit_Data_Module
from utils import generate_next_run_name
from core import Classifier

class Tuner:
    def __init__(
            self,
            model_config: Dict[str, Any],
            tuning_config: Dict[str, Any],
            data_module: Digit_Data_Module,
            mlflow_client: MlflowClient,
            experiment_id: str
        ) -> None:
        self.model_config = model_config
        self.tuning_config = tuning_config
        self.data_module = data_module
        self.mlflow_client = mlflow_client
        self.experiment_id = experiment_id
        parent_run_name = self.model_config["model_class"]
        self.parent_run = self.mlflow_client.create_run(
            experiment_id=self.experiment_id,
            run_name=parent_run_name
        )
    def get_objective(self, parent_run_id):
        child_run_name = generate_next_run_name(
            mlflow_client=self.mlflow_client,
            experiment_id=self.experiment_id,
            prefix = self.model_config["model_class"] + "_param_set"
        )
        child_run = self.mlflow_client.create_run(
            experiment_id=self.experiment_id,
            tags={
                MLFLOW_PARENT_RUN_ID: parent_run_id
            },
            run_name = child_run_name
        )
        def objective(trial: optuna.trial.Trial) -> Any:
            config = {k: v for k, v in self.model_config.items() if k != "model_params"}
            config["model_params"] = {}
            model_class = config["model_class"]
            for name, value in self.model_config["model_params"].items():
                if isinstance(value, dict):
                    if value["param_type"] == "float":
                        low, high = value["param_range"]
                        config["model_params"][name] = trial.suggest_float(
                            name=f"{model_class}_{name}",
                            low=low,
                            high=high
                        )
                    elif value["param_type"] == "int":
                        low, high = value["param_range"]
                        config["model_params"][name] = trial.suggest_int(
                            name=f"{model_class}_{name}",
                            low=low,
                            high=high
                        )
                    elif value["param_type"] == "categorical":
                        config["model_params"][name] = trial.suggest_categorical(
                            name=f"{model_class}_{name}",
                            choices=value["param_range"]
                        )
                    else:
                        raise "param_type should be in ['float', 'int', 'categorical']"
                else:
                    config["model_params"][name] = value
                if config["model_params"].get("random_state") is None:
                    config["model_params"]["random_state"] = 42
                self.mlflow_client.log_param(
                    run_id=child_run.info.run_id, 
                    key=name, 
                    value=config["model_params"][name]
                )
            dataset = self.data_module.get_training_dataset()
            train_dataset = dataset["train_dataset"]
            val_dataset = dataset["val_dataset"]
            preprocessor = self.data_module.get_preprocessor()
            clf = Classifier(config=config, preprocessor=preprocessor)
            clf.fit(train_dataset["data"], train_dataset["target"])
            acc = clf.score(val_dataset["data"], val_dataset["target"])
            self.client.log_metric(
                run_id=child_run.info.run_id,
                key="accuracy",
                value=acc
            )
            
            return acc
        return objective
    def tune(self):
        parent_run_id = self.parent_run.info.run_id
        objective = self.get_objective(parent_run_id)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, **self.tuning_config)

        # best model
        best_params = study.best_params
        for k, v in best_params.items():
            self.mlflow_client.log_param(
                run_id=parent_run_id,
                key=k.split("_")[-1],
                value=v
            )
        best_value = study.best_value
        self.mlflow_client.log_metric(
            run_id=parent_run_id,
            key="accuracy",
            value=best_value
        )

        