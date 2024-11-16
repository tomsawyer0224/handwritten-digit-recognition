import sys
if "." not in sys.path: sys.path.append(".")
from typing import Dict, Any
import optuna
from mlflow import MlflowClient

#from steps.data_ingestion.data_module import Digit_Data_Module
from core import Digit_Data_Module
from get_model import get_model
from wrapper import Classifier
from mlflow_utils import get_or_create_experiment
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
    def __init__(
            self, 
            config: Dict[str, Any], 
            data_module: Digit_Data_Module, 
            client: MlflowClient,
            experiment_id: str
        ) -> None:
        self.config = config
        self.data_module = data_module
        self.client = client
        self.experiment_id = experiment_id
    def __call__(self, trial: optuna.trial.Trial) -> Any:
        trial_run = self.client.create_run(experiment_id=self.experiment_id)
        config = {k: v for k, v in self.config.items() if k != "model_params"}
        config["model_params"] = {}
        model_class = config["model_class"]

        for name, value in self.config["model_params"].items():
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
            self.client.log_param(
                run_id=trial_run.info.run_id, 
                key=name, 
                value=config["model_params"][name]
            )
        #model = get_model(config=config)
        dataset = self.data_module.get_training_dataset()
        train_dataset = dataset["train_dataset"]
        val_dataset = dataset["val_dataset"]
        preprocessor = self.data_module.get_preprocessor()
        clf = Classifier(config=config, preprocessor=preprocessor)
        clf.fit(train_dataset["data"], train_dataset["target"])
        acc = clf.score(val_dataset["data"], val_dataset["target"])
        self.client.log_metric(
            run_id=trial_run.info.run_id,
            key="accuracy",
            value=acc
        )
        
        return acc

if __name__=="__main__":
    data_module = Digit_Data_Module()
    client = MlflowClient()
    experiment_id = get_or_create_experiment(
        experiment_name="find_best_from_config_1", client=client
    )
    study_run = client.create_run(experiment_id=experiment_id)
    study = optuna.create_study(direction="maximize")
    objective = Objective(
        config=config_SVM,
        data_module=data_module,
        client=client,
        experiment_id=experiment_id
    )
    study.optimize(objective, n_trials=3)
    client.log_metric(study_run.info.run_id, "best_accuracy", study.best_value)
