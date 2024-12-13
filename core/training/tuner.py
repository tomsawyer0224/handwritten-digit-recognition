import optuna
from mlflow import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from typing import Dict, Any, Callable
from sklearn.utils import Bunch
import logging
import lightgbm as lbg

from core import Digit_Data_Module, Classifier
from utils import (
    generate_next_run_name,
    name2id,
    id2name,
    get_fit_config,
    prepare_training_data,
    prepare_model_config,
    get_fixed_config
)

logger = logging.getLogger(__name__)

optuna.logging.set_verbosity(optuna.logging.ERROR)
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
        parent_run_name = generate_next_run_name(
            client=self.mlflow_client,
            experiment_id=self.experiment_id,
            prefix=self.model_config["model_class"] + "_test"
        )
        self.parent_run = self.mlflow_client.create_run(
            experiment_id=self.experiment_id,
            run_name=parent_run_name,
            tags={"candidate": "good"}
        )
    def get_objective(self, parent_run_id):
        def objective(trial: optuna.trial.Trial) -> Any:
            child_run_name = f"{self.model_config["model_class"]}_param_set_{trial.number}"
            child_run = self.mlflow_client.create_run(
                    experiment_id=self.experiment_id,
                    tags={
                        MLFLOW_PARENT_RUN_ID: parent_run_id
                    },
                    run_name = child_run_name
                )
            model_config = prepare_model_config(
                model_config=self.model_config,
                trial=trial,
                return_default_config=False
            )
            self.mlflow_client.log_param(
                run_id=child_run.info.run_id, 
                key="model_config",
                value=model_config
            )
            train_dataset = self.data_module.train_dataset
            val_dataset = self.data_module.val_dataset
            train_data, train_target, val_data, val_target = prepare_training_data(
                train_dataset=train_dataset, val_dataset=val_dataset
            )
            clf = Classifier(model_config=model_config)
            fit_config = get_fit_config(
                classifier=clf, val_data=val_data, val_target=val_target
            )
            clf.fit(train_data, train_target, **fit_config)

            # validate
            acc = clf.score(val_data, val_target)
            self.mlflow_client.log_metric(
                run_id=child_run.info.run_id,
                key="accuracy",
                value=acc
            )
            
            return acc
        return objective
    def _train_default_model(self):
        default_run_name = self.model_config["model_class"] + "_default_test"
        runs = self.mlflow_client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string=f"attributes.run_name = '{default_run_name}'"
        )
        if len(runs) == 0:
            default_run = self.mlflow_client.create_run(
                experiment_id=self.experiment_id,
                run_name=default_run_name,
                tags={"candidate": "good"}
            )
            default_config = prepare_model_config(
                model_config=self.model_config,
                trial=None,
                return_default_config=True
            )
            self.mlflow_client.log_param(
                run_id=default_run.info.run_id,
                key="model_config",
                value=default_config
            )
            clf = Classifier(model_config=default_config)
            train_dataset = self.data_module.train_dataset
            val_dataset = self.data_module.val_dataset
            train_data, train_target, val_data, val_target = prepare_training_data(
                train_dataset=train_dataset, val_dataset=val_dataset
            )
            fit_config = get_fit_config(
                classifier=clf, val_data=val_data, val_target=val_target
            )
            clf.fit(data=train_data, target=train_target, **fit_config)
            acc = clf.score(data=val_data, target=val_target)
            
            self.mlflow_client.log_metric(
                run_id=default_run.info.run_id,
                key="accuracy",
                value=acc
            )

    def tune(self):
        logger.info(f"start hyper prameters tuning on {self.model_config["model_class"]}")
        self._train_default_model()
        parent_run_id = self.parent_run.info.run_id
        objective = self.get_objective(parent_run_id)
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, **self.tuning_config)

        # best model
        best_params = study.best_params
        model_config = get_fixed_config(self.model_config)
        for k, v in best_params.items():
            _, param = k.split("-")
            model_config["model_params"][param] = v
        self.mlflow_client.log_param(
                run_id=parent_run_id,
                key="model_config",
                value=model_config
            )
        best_value = study.best_value
        self.mlflow_client.log_metric(
            run_id=parent_run_id,
            key="accuracy",
            value=best_value
        )

        