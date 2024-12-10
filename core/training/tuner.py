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
    get_tuning_config,
    prepare_training_data
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
        #self.datasets = data_module.get_training_dataset()
        #self.preprocessor = data_module.get_preprocessor()
        self.mlflow_client = mlflow_client
        self.experiment_id = experiment_id
        #logger.info("generate a new parent_run_name")
        parent_run_name = generate_next_run_name(
            client=self.mlflow_client,
            experiment_id=self.experiment_id,
            prefix=self.model_config["model_class"] + "_test"
        )
        self.parent_run = self.mlflow_client.create_run(
            experiment_id=self.experiment_id,
            run_name=parent_run_name,
            tags={"candidate": True}
        )
        #logger.info(f"created a new parent run with parent_run_name: {parent_run_name}")
    def get_objective(self, parent_run_id):
        def objective(trial: optuna.trial.Trial) -> Any:
            #logger.info("create a new child run")
            child_run_name = f"{self.model_config["model_class"]}_param_set_{trial.number}"
            child_run = self.mlflow_client.create_run(
                    experiment_id=self.experiment_id,
                    tags={
                        MLFLOW_PARENT_RUN_ID: parent_run_id
                    },
                    run_name = child_run_name
                )
            #logger.info(f"created a new child run with child_run_name: {child_run_name}")
            
            config = get_tuning_config(
                model_config=self.model_config, trial=trial
            )
            self.mlflow_client.log_param(
                run_id=child_run.info.run_id, 
                key="model_config", 
                value=config
            )
            """
            config = {k: v for k, v in self.model_config.items() if k != "model_params"}
            config["model_params"] = {}
            model_class = config["model_class"]
            for name, value in self.model_config["model_params"].items():
                if isinstance(value, dict):
                    if value["param_type"] == "float":
                        low, high = value["param_range"]
                        config["model_params"][name] = trial.suggest_float(
                            name=f"{model_class}-{name}",
                            low=low,
                            high=high
                        )
                    elif value["param_type"] == "int":
                        low, high = value["param_range"]
                        config["model_params"][name] = trial.suggest_int(
                            name=f"{model_class}-{name}",
                            low=low,
                            high=high
                        )
                    elif value["param_type"] == "categorical":
                        config["model_params"][name] = trial.suggest_categorical(
                            name=f"{model_class}-{name}",
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
            """

            #dataset = self.data_module.get_training_dataset()
            #train_dataset = self.datasets["train_dataset"]
            #val_dataset = self.datasets["val_dataset"]
            train_dataset = self.data_module.train_dataset
            val_dataset = self.data_module.val_dataset
            train_data, train_target, val_data, val_target = prepare_training_data(
                train_dataset=train_dataset, val_dataset=val_dataset
            )
            """
            train_data = train_dataset["data"]
            train_target = name2id(train_dataset["target"])
            val_data = val_dataset["data"]
            val_target = name2id(val_dataset["target"])
            """
            #preprocessor = self.data_module.get_preprocessor()
            #clf = Classifier(config=config, preprocessor=self.preprocessor)
            
            fit_config = get_fit_config(
                classifier=clf, val_data=val_data, val_target=val_target
            )
            """
            if clf.library == "xgboost":
                fit_config = dict(
                    eval_set=[(val_data, val_target)],
                    verbose=False
                )
            elif clf.library == "lightgbm":
                fit_config = dict(
                    eval_set=[(val_data, val_target)],
                    callbacks=[lbg.early_stopping(stopping_rounds=10)]
                )
            elif clf.library == "catboost":
                fit_config = dict(
                    eval_set=[(val_data, val_target)],
                )
            else:
                fit_config = {}
            """
            
            clf = Classifier(config=config, use_default=False, **fit_config) 
            #clf.fit(train_dataset["data"], train_dataset["target"])
            clf.fit(train_data, train_target, **fit_config)

            # validate
            #acc = clf.score(val_dataset["data"], val_dataset["target"])
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
                tags={"candidate": True}
            )
            default_config = {
                k: v for k, v in self.model_config.items()
            }
            default_config["model_params"] = dict(
                random_state=self.model_config["model_params"].get("random_state")
            )
            self.mlflow_client.log_param(
                run_id=default_run.info.run_id,
                key="model_config",
                value=default_config
            )
            clf = Classifier(config=self.model_config, use_default=True)
            train_dataset = self.data_module.train_dataset
            val_dataset = self.data_module.val_dataset
            train_data, train_target, val_data, val_target = prepare_training_data(
                train_dataset=train_dataset, val_dataset=val_dataset
            )
            
            clf.fit(data=train_dataset["data"], target=train_dataset["target"])
            acc = clf.score(data=val_dataset["data"], target=val_dataset["target"])
            
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
        model_config = dict(
            model_class = None,
            model_params = {}
        )
        for k, v in best_params.items():
            model_class, param = k.split("-")
            model_config["model_params"][param] = v
        model_config["model_class"] = model_class
        self.mlflow_client.log_param(
                run_id=parent_run_id,
                key="model_config",
                value=model_config
            )
        '''
        for k, v in best_params.items():
            self.mlflow_client.log_param(
                run_id=parent_run_id,
                key=k.split("-")[-1],
                value=v
            )
        '''
        best_value = study.best_value
        self.mlflow_client.log_metric(
            run_id=parent_run_id,
            key="accuracy",
            value=best_value
        )

        